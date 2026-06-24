import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
import numpy as np
import copy

try:
    import hdbscan
    from scipy.spatial.distance import pdist, squareform
except ImportError:
    print("Please install hdbscan and scipy: pip install hdbscan scipy")
    hdbscan = None

from ..fl.server import FedAvgAggregator
from .utils import NoiseDataset
from .const import NUM_CLASSES, IMG_SIZE
from .metrics_logger import DefenseMetrics

class DeepSightServer(DefenseMetrics, FedAvgAggregator):
    """
    Implements the DeepSight defense mechanism, adapted to be compatible
    with the new FedAvgAggregator base class.
    """
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        if hdbscan is None:
            raise ImportError("hdbscan is not installed. Please install it (pip install hdbscan scipy) to use DeepSightServer.")
        
        config = kwargs.get('defense_cfg', None)
        if config is None and len(args) >= 4:
             # Safely check the position where config was passed in the runner call
            config = args[3]

        self.config = config if config is not None else {}
        self.num_samples = self.config.get('deepsight_num_samples', 256)
        self.num_seeds = self.config.get('deepsight_num_seeds', 3)
        self.deepsight_batch_size = self.config.get('deepsight_batch_size', 64)
        self.deepsight_tau = self.config.get('deepsight_tau', 0.5)
        
        print("Initialized DeepSightServer.")

    def aggregate(self, current_round=0) -> Dict[str, torch.Tensor]:
        """
        Overrides the FedAvg aggregate method.
        
        1. Detects anomalies using `detect_anomalies`.
        2. Filters out malicious clients.
        3. Performs robust aggregation (clipping) on benign clients.
        4. Updates the global model.
        5. Clears the received updates buffer.
        """
        if not self.received_updates:
            print("Warning: No updates to aggregate.")
            return self.get_params()
        
        all_client_ids = set(self.received_updates.keys())
        anomalous_client_ids = []
        client_distances = {}

        try:
            # 1. Detect anomalies
            anomalous_client_ids, client_distances = self.detect_anomalies()
            
            print(f"DeepSight detected {len(anomalous_client_ids)} anomalous clients.")
            if anomalous_client_ids:
                print(f"Filtering out clients: {anomalous_client_ids}")

            # 2. Handle filtering
            malicious_client_ids_set = set(anomalous_client_ids)
            benign_client_ids = list(all_client_ids - malicious_client_ids_set)
            
            self.update_defense_metrics(
                client_ids_received=all_client_ids,
                rejected_client_ids=malicious_client_ids_set
            )
            
            if not benign_client_ids:
                print("Warning: DeepSight filtered out all clients. Global model will not be updated.")
                return self.get_params() 

            # 3. Robust Aggregation with Clipping
            benign_distances = [client_distances[cid] for cid in benign_client_ids]
            clip_norm = torch.median(torch.tensor(benign_distances)).item()

            global_params_cpu = self.get_params()
            aggregated_delta = {name: torch.zeros_like(param) for name, param in global_params_cpu.items()}
            total_benign_samples = sum(self.received_updates[cid]['length'] for cid in benign_client_ids)
            
            if total_benign_samples == 0:
                print("Warning: Benign clients reported 0 total samples. Model will not be updated.")
                return self.get_params()

            for cid in benign_client_ids:
                local_params = self.received_updates[cid]['params'] # On CPU
                num_samples = self.received_updates[cid]['length']
                weight = num_samples / total_benign_samples
                
                delta = {name: local_params[name] - global_params_cpu[name] for name in local_params}
                
                # Apply clipping to the delta
                client_dist = client_distances[cid]
                if client_dist > clip_norm:
                    scaling_factor = clip_norm / (client_dist + 1e-10) 
                    for name in delta:
                        if not name.endswith('num_batches_tracked'):
                            delta[name].mul_(scaling_factor)

                # Accumulate weighted, clipped delta
                for name, param_delta in delta.items():
                    if name in aggregated_delta:
                        aggregated_delta[name].add_(param_delta, alpha=weight)

            # 4. Apply the final aggregated delta to the global model 
            new_global_state = self.model.state_dict()
            for name, param in new_global_state.items():
                if name in aggregated_delta:
                    new_global_state[name].add_(aggregated_delta[name].to(self.device))
            
            self.set_params(new_global_state) 

            return self.get_params()
        
        except Exception as e:
            print(f"!!! Error during DeepSight aggregation: {e}. Falling back to non-filtering FedAvg. !!!")
            self.update_defense_metrics(
                client_ids_received=all_client_ids,
                rejected_client_ids=set()
            )
            return super().aggregate()

        finally:
            # 5. Clear buffers
            self.received_updates = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("DeepSight aggregation round finished, buffer and cache cleared.")
        
        

    def detect_anomalies(self) -> Tuple[List[int], Dict[int, float]]:
        """
        Orchestrates the DeepSight detection process.
        
        Reads from `self.received_updates` and maps results back to client IDs.
        
        Returns:
            - List[int]: anomalous_client_ids
            - Dict[int, float]: {client_id: euclidean_distance} for all clients
        """
        num_clients = len(self.received_updates)
        if num_clients < 2: 
            return [], {cid: 0.0 for cid in self.received_updates.keys()}

        client_ids_list = list(self.received_updates.keys())
        index_to_id = {i: cid for i, cid in enumerate(client_ids_list)}
        
        local_model_updates_list = [self.received_updates[cid]['params'] for cid in client_ids_list]

        param_names = [name for name, _ in self.model.named_parameters()]
        last_layer_weight_name = param_names[-2]
        last_layer_bias_name = param_names[-1]
        num_classes = self.model.state_dict()[last_layer_weight_name].shape[0]

        neups, TEs, euclidean_distances_list = self._calculate_neups(
            local_model_updates_list, num_classes, last_layer_weight_name, last_layer_bias_name
        )
        
        classification_boundary = np.median(TEs) if TEs else 0
        te_labels = [te <= classification_boundary * 0.5 for te in TEs]

        ddifs_per_seed = self._calculate_ddifs(local_model_updates_list)
        dist_cosine = self._calculate_cosine_distances(local_model_updates_list, last_layer_bias_name)

        neup_clusters = hdbscan.HDBSCAN().fit_predict(neups)
        neup_dists = self._dists_from_clust(neup_clusters, num_clients)

        cosine_clusters = hdbscan.HDBSCAN(metric='precomputed').fit_predict(dist_cosine)
        cosine_dists = self._dists_from_clust(cosine_clusters, num_clients)
        
        ddif_dists_list = []
        for i in range(self.num_seeds):
            ddif_clusters = hdbscan.HDBSCAN().fit_predict(ddifs_per_seed[i])
            ddif_dists_list.append(self._dists_from_clust(ddif_clusters, num_clients))

        merged_ddif_dists = np.average(ddif_dists_list, axis=0)
        merged_distances = np.mean([merged_ddif_dists, neup_dists, cosine_dists], axis=0)
        final_clusters = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=2, allow_single_cluster=True).fit_predict(merged_distances)

        benign_clients_set, malicious_clients_set = set(), set()
        unique_clusters = [l for l in np.unique(final_clusters) if l != -1]
        
        for cluster_label in unique_clusters:
            member_indices = np.where(final_clusters == cluster_label)[0]
            num_benign = sum(1 for i in member_indices if not te_labels[i])
            if len(member_indices) > 0 and (num_benign / len(member_indices)) >= self.deepsight_tau:
                benign_clients_set.update(member_indices)
            else:
                malicious_clients_set.update(member_indices)
        
        outlier_indices = np.where(final_clusters == -1)[0]
        for i in outlier_indices:
            if not te_labels[i]: benign_clients_set.add(i)
            else: malicious_clients_set.add(i)

        anomalous_client_ids = [index_to_id[i] for i in malicious_clients_set]
        
        client_distances_dict = {
            index_to_id[i]: dist for i, dist in enumerate(euclidean_distances_list)
        }

        return anomalous_client_ids, client_distances_dict
    
    
    def _dists_from_clust(self, clusters: np.ndarray, N: int) -> np.ndarray:
        """Helper to create pairwise distance matrix from cluster labels."""
        pairwise_dists = np.ones((N, N))
        for i in range(N):
            for j in range(i, N):
                if clusters[i] == clusters[j] and clusters[i] != -1:
                    pairwise_dists[i, j] = pairwise_dists[j, i] = 0
        return pairwise_dists
    
    def _calculate_neups(self, 
                         local_model_updates: List[Dict[str, torch.Tensor]], 
                         num_classes: int, 
                         weight_name: str, 
                         bias_name: str
                         ) -> Tuple[np.ndarray, List[float], List[float]]:
        """
        Calculates NEUPs, TEs, and Euclidean distances.
        Takes a list of model parameter dicts.
        Returns list-based results.
        """
        NEUPs, TEs, euclidean_distances = [], [], []
        global_params_cpu = self.get_params() # On CPU

        for local_update in local_model_updates: # local_update is on CPU
            # Move params to device for calculation
            device = self.device 
            
            flat_update = []
            for name, param in local_update.items():
                if 'weight' in name or 'bias' in name:
                    diff = param.cpu() - global_params_cpu[name]
                    flat_update.append(diff.flatten())
            
            if not flat_update:
                euclidean_distances.append(0.0)
            else:
                euclidean_distances.append(torch.linalg.norm(torch.cat(flat_update)).item())

            # Send relevant params to device for diff calculation
            global_weight = global_params_cpu[weight_name].to(device)
            global_bias = global_params_cpu[bias_name].to(device)
            local_weight = local_update[weight_name].to(device)
            local_bias = local_update[bias_name].to(device)

            diff_weight = torch.sum(torch.abs(local_weight - global_weight), dim=1)
            diff_bias = torch.abs(local_bias - global_bias)

            UPs_squared = (diff_bias + diff_weight) ** 2
            NEUP = UPs_squared / (torch.sum(UPs_squared) + 1e-10)
            NEUP_np = NEUP.cpu().numpy()
            NEUPs.append(NEUP_np)

            max_NEUP = np.max(NEUP_np)
            threshold = (1 / num_classes) * max_NEUP if num_classes > 0 else 0
            TEs.append(np.sum(NEUP_np >= threshold))
            
        return np.array(NEUPs), TEs, euclidean_distances

    def _calculate_ddifs(self, 
                         local_model_updates: List[Dict[str, torch.Tensor]]
                         ) -> np.ndarray:
        """
        Calculates DDifs.
        Takes a list of model parameter dicts.
        Returns list-based results.
        """
        dataset_name = self.config.get('dataset', '').upper()
        if not dataset_name or dataset_name not in NUM_CLASSES:
             # Try to infer from num_classes
             ds_map = {v: k for k, v in NUM_CLASSES.items()}
             num_classes_model = len(self.model.state_dict()[list(self.model.state_dict().keys())[-1]])
             if num_classes_model in ds_map:
                 dataset_name = ds_map[num_classes_model]
                 print(f"Warning: 'dataset' not in config. Inferred '{dataset_name}' from model output size.")
             else:
                 raise ValueError("Dataset name must be provided in config for DeepSight")

        num_classes = NUM_CLASSES[dataset_name]; img_height, img_width, num_channels = IMG_SIZE[dataset_name]

        self.model.eval() # Global model on device
        local_model = copy.deepcopy(self.model).to(self.device) # Temp model on device
        DDifs = []
        
        for seed in range(self.num_seeds):
            torch.manual_seed(seed)
            dataset = NoiseDataset((num_channels, img_height, img_width), self.num_samples)
            loader = torch.utils.data.DataLoader(dataset, self.deepsight_batch_size, shuffle=False)
            seed_ddifs = []
            
            for local_update in local_model_updates: 
                local_model.load_state_dict({k: v.to(self.device) for k, v in local_update.items()})
                local_model.eval()
                
                DDif = torch.zeros(num_classes, device=self.device)
                for inputs in loader:
                    inputs = inputs.to(self.device)
                    with torch.no_grad():
                        output_local = local_model(inputs)
                        output_global = self.model(inputs)
                    ratio = torch.div(output_local, output_global + 1e-30)
                    DDif.add_(ratio.sum(dim=0))
                
                DDif /= self.num_samples
                seed_ddifs.append(DDif.cpu().numpy())
            
            DDifs.append(seed_ddifs)
            
        return np.array(DDifs)

    def _calculate_cosine_distances(self, 
                                    local_model_updates: List[Dict[str, torch.Tensor]], 
                                    bias_name: str
                                    ) -> np.ndarray:
        """
        Calculates cosine distances of last layer bias.
        Takes a list of model parameter dicts.
        Returns list-based results.
        """
        N = len(local_model_updates)
        distances = np.zeros((N, N))
        global_bias = self.get_params()[bias_name] 
        
        bias_diffs = [(update[bias_name].cpu() - global_bias).flatten() for update in local_model_updates]

        for i in range(N):
            for j in range(i + 1, N):
                # Ensure diffs are 1D tensors
                diff_i = bias_diffs[i].view(1, -1)
                diff_j = bias_diffs[j].view(1, -1)
                
                similarity = F.cosine_similarity(diff_i, diff_j, dim=1, eps=1e-10)
                dist = 1.0 - similarity.item()
                distances[i, j] = distances[j, i] = dist
        return distances