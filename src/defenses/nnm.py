import torch
from typing import Dict, List, Tuple

from src.fl.server import FedAvgAggregator
from src.defenses.metrics_logger import DefenseMetrics


class NNMKrumServer(DefenseMetrics, FedAvgAggregator):
    """
    NNM (Nearest Neighbor Mixing) + Multi-Krum Aggregation
    WITH contribution-aware metrics

    Pipeline:
    1. Flatten updates
    2. Apply NNM (with neighbor tracking)
    3. Multi-Krum selection
    4. Compute contributing clients
    5. Aggregate
    6. Update metrics (CORRECTED)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = kwargs.get('config', {})
        defense_config = kwargs.get('defense_cfg', {})

        if not config and len(args) >= 1:
            config = args[0]

        if not defense_config:
            defense_config = config.get('defense_params', {})

        # --- Krum params ---
        self.krum_f = defense_config.get('krum_f', 5)
        self.krum_m = defense_config.get('krum_m', 14)

        # --- NNM params ---
        self.nnm_k = defense_config.get('nnm_k', 3)


        # --- Reputation ---
        self.client_reputation = {}
        self.client_rep_sum = {}
        self.client_rep_count = {}
        self.round = 0
        self.T_warmup = 50

        self.rejected_clients = set()


        print(f"--- Initializing NNM + Multi-Krum Server ---")
        print(f"    Krum f: {self.krum_f}")
        print(f"    Krum m: {self.krum_m}")
        print(f"    NNM k: {self.nnm_k}")

    # ------------------------------------------------------------
    # Flatten model parameters
    # ------------------------------------------------------------
    def _flatten(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([p.view(-1) for p in params.values()])

    # ------------------------------------------------------------
    # Reconstruct model
    # ------------------------------------------------------------
    def _vector_to_state(self, vec, reference_state):
        new_state = {}
        pointer = 0

        for k, v in reference_state.items():
            numel = v.numel()
            new_state[k] = vec[pointer:pointer + numel].view(v.shape)
            pointer += numel

        return new_state

    # ------------------------------------------------------------
    # NNM with contribution tracking
    # ------------------------------------------------------------
    def _nnm(self, updates: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, List[int]]]:
        """
        Returns:
            mixed_updates: (N, D)
            neighbor_map: dict[i] = list of neighbors contributing to i
        """
        n = updates.shape[0]
        mixed = torch.zeros_like(updates)
        neighbor_map = {}

        for i in range(n):
            dists = torch.norm(updates[i] - updates, dim=1)

            k = min(self.nnm_k + 1, n)
            neighbors = torch.topk(dists, k=k, largest=False).indices[1:]

            neighbor_map[i] = neighbors.tolist()

            mixed[i] = (
                updates[i] +
                updates[neighbors].sum(dim=0)
            ) / (len(neighbors) + 1)

        return mixed, neighbor_map

    # ------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------
    def aggregate(self, current_round=0) -> Dict[str, torch.Tensor]:

        if not self.received_updates:
            return

        client_ids = list(self.received_updates.keys())
        n = len(client_ids)

        # --- Krum feasibility ---
        if n < self.krum_f + 3:
            print(f"Warning: Not enough clients ({n}) for Krum. Falling back to FedAvg.")
            super().aggregate()
            return

        # --------------------------------------------------
        # 1. Flatten
        # --------------------------------------------------
        flat_updates = []
        param_refs = []

        for cid in client_ids:
            params = self.received_updates[cid]['params']
            param_refs.append(params)
            flat_updates.append(self._flatten(params))

        flat_updates = torch.stack(flat_updates)

        # --------------------------------------------------
        # 2. NNM
        # --------------------------------------------------
        print(f"Applying NNM (k={self.nnm_k}) on {n} updates...")
        mixed_updates, neighbor_map = self._nnm(flat_updates)

        # --------------------------------------------------
        # 3. Pairwise distances
        # --------------------------------------------------
        dists = torch.cdist(mixed_updates, mixed_updates, p=2)

        k_neighbors = max(1, n - self.krum_f - 2)
        scores = torch.zeros(n)

        for i in range(n):
            d_sorted, _ = torch.sort(dists[i])
            scores[i] = torch.sum(d_sorted[1:k_neighbors + 1])

        # --------------------------------------------------
        # 4. Multi-Krum selection
        # --------------------------------------------------
        m_to_select = min(n, max(1, self.krum_m))

        _, top_indices = torch.topk(
            scores,
            k=m_to_select,
            largest=False
        )

        # --------------------------------------------------
        # 5. Compute contributing clients (FIX)
        # --------------------------------------------------
        contributing_clients = set()

        for i in top_indices.tolist():
            contributing_clients.add(client_ids[i])  # self

            for j in neighbor_map[i]:
                contributing_clients.add(client_ids[j])

        rejected_clients = set(client_ids) - contributing_clients
        self.rejected_clients = rejected_clients

        print(f"Selected centers: {[client_ids[i] for i in top_indices.tolist()]}")
        print(f"Effective contributing clients: {sorted(contributing_clients)}")
        print(f"Rejected clients (true exclusion): {sorted(rejected_clients)}")

        # --------------------------------------------------
        # 6. Aggregate selected mixed updates
        # --------------------------------------------------
        selected_updates = mixed_updates[top_indices]
        final_update = torch.mean(selected_updates, dim=0)

        # --------------------------------------------------
        # 7. Reconstruct model
        # --------------------------------------------------
        reference = param_refs[0]
        new_params = self._vector_to_state(final_update, reference)

        # --------------------------------------------------
        # 8. Update metrics (CORRECTED)
        # --------------------------------------------------
        self.update_defense_metrics(
            client_ids_received=set(client_ids),
            rejected_client_ids=rejected_clients
        )

        # --------------------------------------------------
        # Update reputation 
        # --------------------------------------------------
        self.update_reputation(
          client_ids=client_ids,
          scores=scores
        )

        # --------------------------------------------------
        # 9. Update global model
        # --------------------------------------------------
        self.set_params({k: v.to(self.device) for k, v in new_params.items()})

        self.received_updates = {}

    

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


