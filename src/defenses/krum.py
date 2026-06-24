import torch
import torch.nn as nn
from typing import Dict, List, Optional
from torch.utils.data import DataLoader
import numpy as np

from ..fl.server import FedAvgAggregator
from .metrics_logger import DefenseMetrics

class MKrumServer(DefenseMetrics, FedAvgAggregator):
    """
    Implements the Multi-Krum (M-Krum) defense mechanism.
    
    It selects the 'm' clients with the lowest Krum scores and
    performs a standard FedAvg on only that subset.
    """
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        config = kwargs.get('defense_cfg')
        if config is None and len(args) >= 4:
            config = args[3]
        
        self.config = config if config is not None else {}
        self.num_byzantine = self.config.get('krum_f', 0)
        self.num_to_select = self.config.get('krum_m', 1)

        # --- Reputation ---
        self.client_reputation = {}
        self.client_rep_sum = {}
        self.client_rep_count = {}
        self.round = 0
        self.T_warmup = 50
        self.rejected_clients = set()
        
        print(f"Initialized MKrumServer to tolerate f={self.num_byzantine} Byzantine clients and select m={self.num_to_select} for aggregation.")

    def aggregate(self, current_round=0) -> Dict[str, torch.Tensor]:
        """
        Overrides the FedAvg aggregate method.
        
        1. Checks if Krum can run (n > 2f + 2).
        2. Calculates all pairwise distances between client *deltas*.
        3. Computes Krum scores for each client.
        4. Selects the 'm' clients with the lowest scores.
        5. Performs FedAvg (weighted average) on this subset.
        6. Updates the global model and clears the buffer.
        """
        num_updates = len(self.received_updates)
        if num_updates == 0:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        client_ids = list(self.received_updates.keys())
        self.round = current_round
        # M-Krum requires n > 2f + 2 to provide guarantees.
        # If not met, fall back to the parent's standard FedAvg.
        if num_updates <= 2 * self.num_byzantine + 2:
            print(f"Warning: Not enough clients ({num_updates}) for Krum with f={self.num_byzantine}. Falling back to standard FedAvg.")
            client_ids_received_set = set(client_ids_list)
            rejected_client_ids = set() 
            self.update_defense_metrics(
                client_ids_received=client_ids_received_set,
                rejected_client_ids=rejected_client_ids
            )
            
            # super().aggregate() will average all clients and clear the buffer.
            return super().aggregate()
        
        # 1. Create a stable mapping from index (0...n-1) to client_id
        client_ids_list = list(self.received_updates.keys())
        
        # 2. Get deltas (local - global)
        global_params = self.get_params() # On CPU
        client_deltas = []
        for client_id in client_ids_list:
            local_params = self.received_updates[client_id]['params'] # On CPU
            delta = {name: local_params[name] - global_params[name] for name in local_params}
            client_deltas.append(delta)

        # 3. Flatten deltas for distance calculation
        flat_deltas = [torch.cat([p.flatten() for p in delta.values()]) for delta in client_deltas]

        # 4. Compute pairwise squared Euclidean distances
        distances = torch.zeros((num_updates, num_updates))
        for i in range(num_updates):
            for j in range(i, num_updates):
                # Krum uses squared L2 norm
                dist = torch.linalg.norm(flat_deltas[i] - flat_deltas[j]) ** 2
                distances[i, j] = distances[j, i] = dist.item()

        # 5. For each client, find the sum of distances to its k nearest neighbors
        # k = n - f - 2 (number of "closest" clients to sum)
        num_neighbors = num_updates - self.num_byzantine - 2
        scores = torch.zeros(num_updates)

        for i in range(num_updates):
            d_sorted, _ = torch.sort(distances[i])
            scores[i] = torch.sum(d_sorted[1:num_neighbors + 1])    
        
        scores_np = scores.numpy()
        # 6. Select the 'm' clients with the lowest scores
        sorted_indices = np.argsort(scores_np)
        selected_indices = sorted_indices[:self.num_to_select]
        
        # 7. Map the selected list-indices back to their original client_ids
        selected_client_ids = [client_ids_list[i] for i in selected_indices]
        self.rejected_clients = set(client_ids_list) - set(selected_client_ids)
        
        print(f"Krum selected clients (by ID): {selected_client_ids}")

        client_ids_received_set = set(client_ids_list)
        selected_client_ids_set = set(selected_client_ids)
        # Rejected clients are those received minus those selected
        rejected_client_ids = client_ids_received_set - selected_client_ids_set

        self.update_defense_metrics(
            client_ids_received=client_ids_received_set,
            rejected_client_ids=rejected_client_ids
        )

        # 8. Aggregate only the selected clients using standard FedAvg logic
        
        total_samples = sum(self.received_updates[cid]['length'] for cid in selected_client_ids)
        
        if total_samples == 0:
            print("Warning: Krum selected clients with 0 total samples. Model will not be updated.")
            self.received_updates = {} # Clear buffer
            return self.get_params()

        averaged = {}
        first_client_id = selected_client_ids[0]
        first_params = self.received_updates[first_client_id]['params']
        
        for k in first_params.keys():
            acc = torch.zeros_like(first_params[k], dtype=torch.float32)
            for cid in selected_client_ids:
                client_params = self.received_updates[cid]['params']
                client_len = self.received_updates[cid]['length']
                weight = float(client_len) / float(total_samples)
                
                if k in client_params:
                   acc += client_params[k] * weight
                
            averaged[k] = acc # Final averaged param is on CPU


        # --------------------------------------------------
        # Update reputation 
        # --------------------------------------------------
        self.update_reputation(
          client_ids=client_ids,
          scores=scores
        )    

        self.set_params({k: v.to(self.device) for k, v in averaged.items()})

        self.received_updates = {}
        
        return {k: v.cpu().clone() for k, v in averaged.items()}
    


        # ------------------------------------------------------------
    # Reputation Update using Krum scores
    # ------------------------------------------------------------

    def update_reputation(self, client_ids, scores):

       eps = 1e-12

       # scores = Krum scores (lower is better)
       score_dict = {
        cid: scores[i].item()
        for i, cid in enumerate(client_ids)
       }

       s_vals = torch.tensor(list(score_dict.values()))

       median_s = s_vals.median()
       mad_s = (s_vals - median_s).abs().median() + eps

       tau = 2.0

       # --------------------------------------------------------
       # Reputation update
       # --------------------------------------------------------
       for cid in client_ids:

          s_i = score_dict[cid]

          # selected by Multi-Krum ?
          reliable = 1.0 if cid not in self.rejected_clients else 0.0

          # lower score => higher reputation
          scaled = (s_i - median_s) / mad_s

          C_i_t = torch.sigmoid(
            torch.tensor(-scaled / tau)
          ).item()
          
          alpha = 0.2
          # soft penalty for rejected clients
          C_i_t = alpha * C_i_t + (1 - alpha) * reliable

          # ----------------------------------------------------
          # initialize reputation buffers
          # ----------------------------------------------------
          if cid not in self.client_rep_sum:
             self.client_rep_sum[cid] = 0.0
             self.client_rep_count[cid] = 0
             self.client_reputation[cid] = 0.7

          # ----------------------------------------------------
          # EMA update
          # ----------------------------------------------------
          gamma_t = min(1.0, self.round / self.T_warmup)

          beta = 0.95 * (1 - gamma_t/2)  # start with 0.95, decay to 0.475

          self.client_reputation[cid] = (
            beta * self.client_reputation[cid]
            + (1 - beta) * C_i_t
          )

          self.client_rep_sum[cid] += C_i_t
          self.client_rep_count[cid] += 1

          print(
            f"Client {cid}: "
            f"Krum={s_i:.4f}, "
            f"C={C_i_t:.4f}, "
            f"L_rep={self.client_reputation[cid]:.4f}"
          )

       print(f"[Reputation] Updated for {len(client_ids)} clients")