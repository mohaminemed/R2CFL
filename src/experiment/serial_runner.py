import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import copy
import os # Added import
import time # Added import for timing aggregation

from .loggings import MetricsLogger
from .utils import get_data_and_model, get_client_instance, get_server_instance, select_clients 
from ..fl.client import BaseClient
from ..fl.server import FedAvgAggregator, BaseServer
from ..datasets.adapter import DatasetAdapter




class FederatedExperiment:
    def __init__(self, config: Dict):
        """Initializes the experiment from a configuration dictionary."""
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        self.cached_backdoor_loader = None
        
        self.seed = config.get("seed", 42) 
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.adapter: Optional[DatasetAdapter] = None 
        self.clients: List[BaseClient] = []
        self.server: Optional[BaseServer] = None 
        
        self.attack_enabled = False
        self.attack_start_round = float('inf')
        self.attack_end_round = float('inf')
        self.malicious_ids = set()

        self.prev_global_params_cpu: Optional[Dict[str, torch.Tensor]] = None
        self.prev_global_grad_cpu: Optional[Dict[str, torch.Tensor]] = None

        self._setup()

        headers = [
            'round', 'main_accuracy', 'main_loss', 'attack_success_rate',
            'is_attack_active', 'bottleneck_distance', 'bottleneck_threshold', 'agg_time_sec'
        ]
        self.logger = MetricsLogger(
            output_dir=self.config.get("output_dir", "results"), 
            experiment_name=self.config['experiment_name'],
            headers=headers
        )

    def _setup(self):
        """
        Sets up all components dynamically using factory functions.
        """
        # 1. Get dataset adapter and an initial model instance from the factory
        self.adapter, initial_model = get_data_and_model(self.config['data_params'])
        self.adapter.setup()

        load_pretrained = self.config['data_params'].get('load_pretrained', False)
        load_model_path = self.config['data_params'].get('load_model_path', '')
        
        if load_pretrained:
            try:
                print(f"--- Loading pre-trained model from: {load_model_path} ---")
                # Load the state dict, making sure it's on the correct device
                state_dict = torch.load(load_model_path, map_location=self.device, weights_only=True)
                initial_model.load_state_dict(state_dict)
                print("--- Model loaded successfully. ---")
            except FileNotFoundError:
                print(f"Warning: load_model_path '{load_model_path}' not found. Starting from scratch.")
            except Exception as e:
                print(f"Warning: Error loading model '{load_model_path}': {e}. Starting from scratch.")   

        # 2. Get the server instance (now includes various defenses)
        
        self.server: BaseServer = get_server_instance( # Use BaseServer type hint
            self.config,
            model=copy.deepcopy(initial_model), # Server gets its own copy
            test_loader=self.adapter.get_test_loader(),
            device=self.device
        )

        # 3. Get client instances (Benign, attacks, etc.) from the factory
        fl_params = self.config['fl_params']
        # --- Make data loading parameters explicit ---
        train_loaders = self.adapter.get_client_loaders(
            num_clients=fl_params['num_clients'],
            batch_size=self.config['training_params'].get('batch_size', 64), # Get batch_size from training_params
            strategy=self.config['data_params'].get('strategy', 'iid'), # Get strategy from data_params
            seed=self.seed, # Pass the experiment seed
            alpha=self.config['data_params'].get('alpha', 0.5) # Pass alpha if strategy is dirichlet
        )
        
        self.clients: List[BaseClient] = []
        for i in range(fl_params['num_clients']):
             # Handle cases where a client might not get data (e.g., extreme non-IID)
             if i not in train_loaders:
                  print(f"Warning: No data loader created for client {i}. Skipping client creation.")
                  continue
                  
             # The factory decides which client class to instantiate based on the config
             client = get_client_instance(
                 self.config,
                 client_id=i,
                 train_loader=train_loaders[i], # Pass the specific loader
                 model=copy.deepcopy(initial_model), # Give each client its own deep copy
                 device=self.device
             )
             self.clients.append(client)

        # --- 4. Extract Attack Parameters for Metric Tracking (New) ---
        attack_cfg = self.config.get('attack_params', {})
        self.attack_enabled = attack_cfg.get('enabled', False)

        if self.attack_enabled:
            # Use float('inf') as default end if not specified, matching the helper
            self.attack_start_round = attack_cfg.get('attack_start_round', 1) 
            self.attack_end_round = attack_cfg.get('attack_end_round', float('inf')) 
            self.malicious_ids = set(attack_cfg.get('malicious_client_ids', []))
        else:
            self.malicious_ids = set()
            self.attack_start_round = float('inf')
            self.attack_end_round = float('inf')

    def run(self):
        """
        Executes the FL rounds, tracks global updates, and evaluates backdoor success.
        """
        print(f"--- Starting Experiment: {self.config['experiment_name']} (Serial Runner) ---") # Clarify runner type
        fl_cfg = self.config['fl_params']
        train_cfg = self.config['training_params']
        
        self.prev_global_params_cpu = {k: v.cpu().clone() for k, v in self.server.get_params().items()}
        self.prev_global_grad_cpu = None
        self.agg_log = []
        for round_idx in range(fl_cfg['num_rounds']):
            current_round_num = round_idx + 1 
            print(f"\n--- Round {current_round_num}/{fl_cfg['num_rounds']} ---")

            # --- TPR/FPR Metric Setup: Set Ground Truth on Server (New) ---
            malicious_ids_this_round = []
            
            # Check if the attack is enabled and if the current round falls within the attack window
            if self.attack_enabled and (self.attack_start_round <= current_round_num <= self.attack_end_round):
                # The set of malicious IDs is passed as the ground truth
                malicious_ids_this_round = list(self.malicious_ids)
            
            # Pass the ground truth to the defense server if it has the set_malicious_ids method (from Mixin)
            if hasattr(self.server, 'set_malicious_ids'):
                self.server.set_malicious_ids(malicious_ids_this_round)
            # -------------------------------------------------------------
            
            selected_clients = select_clients( 
                client_list=self.clients,
                num_selected_clients=fl_cfg['clients_per_round'],
                selection_strat=fl_cfg.get('selection_strategy', 'random')
            )
            
            current_global_params = self.server.get_params() 
            
            for client in selected_clients:
                # Give client the latest global model state
                client.set_params(current_global_params) 
                
                update = client.local_train(
                    round_idx=round_idx,
                    epochs=train_cfg.get('local_epochs', 1),
                    prev_global_grad= self.prev_global_grad_cpu,
                    prev_global_params=self.prev_global_params_cpu 
                )

                if update and 'weights' in update and 'num_samples' in update:
                    # Server expects weights on its device
                    weights_on_device = {k: v.to(self.device) for k,v in update['weights'].items()}
                    self.server.receive_update(
                        client_id=client.get_id(), 
                        params=weights_on_device, 
                        length=update['num_samples']
                    )
                else:
                    print(f"Warning: Client {client.get_id()} did not return a valid update.")

            
            # --- Server Aggregation ---
            params_before_agg_cpu = {k: v.cpu().clone() for k, v in self.server.get_params().items()}

            # AGG time start
            agg_start_time = time.time()

            
            self.server.aggregate(current_round=current_round_num)

            # AGG time end
            agg_end_time = time.time()
            agg_duration = agg_end_time - agg_start_time
            print(f"Aggregation Time: {agg_duration:.2f} seconds")
        

            # Store params *after* aggregation for the next round's TDFed
            params_after_agg_cpu = {k: v.cpu().clone() for k, v in self.server.get_params().items()}
            self.prev_global_params_cpu = params_before_agg_cpu # Update for TDFed

            # --- Calculate and store GRADIENT for the next round's Neurotoxin ---
            self.prev_global_grad_cpu = {
                k: params_after_agg_cpu[k] - params_before_agg_cpu.get(k, torch.tensor(0.0)) # Use .get for safety
                for k in params_after_agg_cpu
            }
            
            # --- Evaluation ---
            main_metrics = self.server.evaluate() # Evaluate the newly aggregated model
            main_acc = main_metrics['metrics'].get('main_accuracy', main_metrics['metrics'].get('accuracy', -1.0)) # Handle potential key variations
            main_loss = main_metrics['metrics'].get('loss', -1.0)
            print(f"Global Model Accuracy: {main_acc:.4f}")
            
            # --- ASR Evaluation & Logging (Using helper function) ---
            self._evaluate_and_log_round(round_idx, main_acc, main_loss, agg_duration)
            


        self.logger.close()    
        
        
        # --- NEW: Call server.close() to trigger final metric reporting ---
        if hasattr(self.server, 'close'):
             self.server.close()
        # ----------------------------------------------------------------------

        print("\n--- Experiment Finished ---")
        # Ensure output directory exists for saving model
        output_dir = self.config.get("output_dir", "results")
        os.makedirs(output_dir, exist_ok=True)
        self.server.save_model(f"{output_dir}/{self.config['experiment_name']}_final_model.pth")

    # --- Keep Helper Function for ASR/Logging ---
    def _evaluate_and_log_round(self, round_idx, main_acc, main_loss, agg_duration):
        """Helper function for ASR evaluation and logging."""
        
        attack_cfg = self.config.get('attack_params', {})
        attack_enabled = attack_cfg.get('enabled', False)
        is_attack_active = False
        asr = 0.0
        current_round_num = round_idx + 1 # Use 1-based index consistent with checks
        
        if attack_enabled:
            # Note: Using the attributes set in _setup now, which match the original config structure
            attack_start_round = self.attack_start_round # or attack_cfg.get('attack_start_round', 1) 
            end = self.attack_end_round # or attack_cfg.get('attack_end_round', float('inf'))
            
            # Check if current round is within the attack window (inclusive end)
            if attack_start_round <= current_round_num <= end:
                is_attack_active = True            

            # Only start ASR check once attack could have started
            if current_round_num >= attack_start_round:
                malicious_client_ids = attack_cfg.get('malicious_client_ids', [])
                if not malicious_client_ids:
                    print("Warning: Attack enabled but no 'malicious_client_ids' specified.")
                else:
                    mal_client = next((c for c in self.clients if c.get_id() == malicious_client_ids[0]), None)
                    
                    trigger_obj = None
                    if mal_client:
                         # Try accessing trigger directly or via attack_config
                         trigger_obj = getattr(mal_client, 'trigger', None) or mal_client.attack_config.get('trigger', None)

                    if trigger_obj:
                        # Check if trigger is static (might not have 'is_static')
                        trigger_is_static = getattr(trigger_obj, 'is_static', True) 
                        
                        # Determine if loader needs update (dynamic trigger in active round)
                        update_loader = (self.cached_backdoor_loader is None) or (not trigger_is_static and is_attack_active)
                        
                        if update_loader:
                            print(f"--- {'Creating' if self.cached_backdoor_loader is None else 'Updating'} backdoor test loader (Trigger type: {'Static' if trigger_is_static else 'Dynamic'}) ---")
                            # Ensure generator is on the right device if it exists (for IBA)
                            if hasattr(trigger_obj, 'generator') and hasattr(trigger_obj.generator, 'to'):
                                trigger_obj.generator.to(self.device) # Ensure generator is on eval device

                            self.cached_backdoor_loader = self.adapter.get_backdoor_test_loader(
                                trigger_fn=trigger_obj.apply,
                                target_label=attack_cfg['target_label']
                            )

                        if self.cached_backdoor_loader:
                            self.server.model.eval() 
                            asr_metrics = self.server.evaluate(valloader=self.cached_backdoor_loader)
                            asr = asr_metrics['metrics'].get('main_accuracy', asr_metrics['metrics'].get('accuracy', -1.0))
                            print(f"Attack Success Rate (ASR): {asr:.4f}")
                            
                    else:
                        print(f"Warning: Could not find trigger object on malicious client {malicious_client_ids[0]}. Cannot evaluate ASR.")
        # --- Logging ---           
        log_data = {
            'round': current_round_num, 
            'main_accuracy': main_acc,
            'main_loss': main_loss,
            'attack_success_rate': asr, 
            'is_attack_active': int(is_attack_active),
            'agg_time_sec': agg_duration 
            
        }
        self.logger.log_round(log_data)