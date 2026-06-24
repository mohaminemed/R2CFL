import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
import copy

# Flame requires hdbscan for clustering
try:
    import hdbscan
except ImportError:
    print("Please install hdbscan: pip install hdbscan")
    hdbscan = None

from ..fl.server import FedAvgAggregator
from .metrics_logger import DefenseMetrics

class FlameServer(DefenseMetrics, FedAvgAggregator):
    """
    Implements the FLAME defense mechanism, adapted to be compatible
    with the new FedAvgAggregator base class.
    
    It filters clients based on clustering and then performs robust
    aggregation with clipping and adaptive noise.
    """
    def __init__(self, *args, **kwargs):
        
        # Call the parent constructor
        super().__init__(*args, **kwargs)

        if hdbscan is None:
            raise ImportError("hdbscan is not installed. Please install it (pip install hdbscan) to use FlameServer.")

        config = kwargs.get('defense_cfg', None)
        if config is None and len(args) >= 4:
            # Safely check the position where config was passed in the runner call
            config = args[3]

        # FLAME pecific params
        self.config = config if config is not None else {}
        self.lamda = self.config.get('flame_lamda', 0.001) # Noise std dev factor
        self.eta = self.config.get('flame_eta', 1.0)       # Global learning rate
        
        print(f"Initialized FlameServer with lamda={self.lamda}, eta={self.eta}")

    def aggregate(self, current_round=0) -> Dict[str, torch.Tensor]:
        """
        Overrides the FedAvg aggregate method.
        
        1. Detects anomalies using `detect_anomalies`.
        2. Filters out malicious clients.
        3. Performs robust aggregation (clipping, noise) on benign clients.
        4. Updates the global model.
        5. Clears the received updates buffer.
        """
        if not self.received_updates:
            print("Warning: No updates to aggregate.")
            # Return current model params
            return self.get_params()

        client_ids_received_list = list(self.received_updates.keys())
        client_ids_received_set = set(client_ids_received_list)
        
        # 1. Detect anomalies
        benign_client_ids, malicious_client_ids, client_distances = self.detect_anomalies()
        
        print(f"Flame detected {len(malicious_client_ids)} anomalous clients.")
        if malicious_client_ids:
            print(f"Filtering out clients: {malicious_client_ids}")

        rejected_client_ids = set(malicious_client_ids)
        
        self.update_defense_metrics(
            client_ids_received=client_ids_received_set,
            rejected_client_ids=rejected_client_ids
        )

        # 2. Handle filtering
        if not benign_client_ids:
            print("Warning: Flame filtered out all clients. Global model will not be updated.")
            self.received_updates = {} # Clear buffer
            return self.get_params()

        # 3. Robust Aggregation with Clipping and Noise 
        
        # Calculate clipping norm (median distance of benign clients)
        benign_distances = [client_distances[cid] for cid in benign_client_ids]
        clip_norm = torch.median(torch.tensor(benign_distances)).item() if benign_distances else 1.0

        weight_accumulator = {
            name: torch.zeros_like(param).to(self.device) 
            for name, param in self.model.state_dict().items()
        }
        
        global_params_cpu = self.get_params() 

        for client_id in benign_client_ids:
            local_params = self.received_updates[client_id]['params']
            
            weight = 1.0 / len(benign_client_ids) 

            for name, param_cpu in local_params.items():
                if name.endswith('num_batches_tracked'): 
                    continue
                
                diff = param_cpu.to(self.device) - global_params_cpu[name].to(self.device)
                
                # Apply clipping
                client_dist = client_distances[client_id]
                if client_dist > clip_norm:
                    diff.mul_(clip_norm / client_dist) 
                
                weight_accumulator[name].add_(diff * weight)

        # 4. Update global model parameters (in-place on device)
        final_state_dict = self.model.state_dict()
        std_dev = self.lamda * clip_norm 

        for name, param in final_state_dict.items():
            if name in weight_accumulator:
                param.add_(weight_accumulator[name] * self.eta)

                if 'weight' in name or 'bias' in name:
                    noise = torch.normal(0, std_dev, param.shape, device=self.device)
                    param.add_(noise)
        
        # 5. Clear buffer and return
        self.received_updates = {} 
        
        return self.get_params() # Return new model state_dict (on CPU)

    def detect_anomalies(self) -> Tuple[List[int], List[int], Dict[int, float]]:
        """
        Detects anomalies using cosine similarity clustering on last layer weights.
        
        Reads from `self.received_updates`.
        
        Returns:
            - List[int]: benign_client_ids
            - List[int]: malicious_client_ids
            - Dict[int, float]: {client_id: euclidean_distance} for all clients
        """
        num_clients = len(self.received_updates)
        
        client_ids_list = list(self.received_updates.keys())
        index_to_id = {i: cid for i, cid in enumerate(client_ids_list)}

        if num_clients < 2:
            return client_ids_list, [], {cid: 0.0 for cid in client_ids_list}

        global_params_cpu = self.get_params()
        
        first_client_data = next(iter(self.received_updates.values()))
        last_layer_names = self._get_last_layers(first_client_data['params'])

        all_client_weights_for_clustering = []
        client_id_to_distance: Dict[int, float] = {}

        for client_id in client_ids_list:
            local_params = self.received_updates[client_id]['params']
            
            flat_update_diff = []
            last_layer_weights = []

            for name, param in local_params.items():
                if 'weight' in name or 'bias' in name:
                    diff = param.to(self.device) - global_params_cpu[name].to(self.device)
                    flat_update_diff.append(diff.flatten())
                
                if name in last_layer_names:
                    last_layer_weights.append(param.cpu().flatten())

            euclidean_dist = torch.linalg.norm(torch.cat(flat_update_diff)).item()
            client_id_to_distance[client_id] = euclidean_dist
            
            all_client_weights_for_clustering.append(
                torch.cat(last_layer_weights).numpy().astype(np.float64)
            )
        
        client_weights_array = np.array(all_client_weights_for_clustering, dtype=np.float64)

        clusterer = hdbscan.HDBSCAN(
            metric="cosine", 
            algorithm="generic",
            min_cluster_size=max(2, num_clients // 2 + 1),
            allow_single_cluster=True
        )
        
        labels = clusterer.fit_predict(client_weights_array)

        benign_indices = []
        if np.all(labels == -1) or len(np.unique(labels)) == 1:
            benign_indices = list(range(num_clients))
        else:
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(unique_labels) > 0:
                largest_cluster_label = unique_labels[np.argmax(counts)]
                benign_indices = [i for i, label in enumerate(labels) if label == largest_cluster_label]
            else:
                benign_indices = list(range(num_clients))

        benign_client_ids = [index_to_id[i] for i in benign_indices]
        all_client_ids_set = set(client_ids_list)
        malicious_client_ids = list(all_client_ids_set - set(benign_client_ids))

        return benign_client_ids, malicious_client_ids, client_id_to_distance

    def _get_last_layers(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        """Get names of last two layers with parameters."""
        layer_names = list(state_dict.keys())
        # Filter for layers that have parameters (weights or biases)
        param_layers = [name for name in layer_names if 'weight' in name or 'bias' in name]
        # Return the last two (e.g., fc.weight, fc.bias)
        return param_layers[-2:]