import torch
import numpy as np
import copy
from typing import Dict
import multiprocessing as mp

from .serial_runner import FederatedExperiment
from ..attacks.neurotoxin_client import NeurotoxinClient
from ..attacks.triggers.a3fl import A3FLTrigger
from ..attacks.triggers.patch_trigger import PatchTrigger
from ..defenses.rep_nnm import REPNNMKrumServer # Import the reputation-based server for dynamic aggregation

# This function must be at the top level of the module so it can be "pickled"
def run_client_task(client, global_params, epochs, round_idx, prev_global_grad):
    """
    A wrapper function to run a single client's training task.
    """
    try:
        # Set the client's model to the current global model
        client.set_params(copy.deepcopy(global_params))
        
        # Execute the local training
        if isinstance(client, NeurotoxinClient):
            update = client.local_train(
                epochs=epochs,
                round_idx=round_idx,
                prev_global_grad=prev_global_grad
            )
        else:
            update = client.local_train(
                epochs=epochs,
                round_idx=round_idx
            )
    
        # Return the results
        return update
    except Exception as e:
        print(f"Error in client {client.get_id()}: {e}")
        return None


class ParallelFederatedExperiment(FederatedExperiment):
    """
    An experiment runner that parallelizes client training using multiprocessing.
    """
    def __init__(self, config: Dict):
        super().__init__(config)
        # Get the number of parallel workers from the config
        self.num_workers = self.config.get('num_workers', 10)
        print(f"--- Initialized Parallel Runner with {self.num_workers} workers ---")

    def run(self):
        """
        Overrides the run method to execute client training in parallel.
        """
        print(f"--- Starting Experiment: {self.config['experiment_name']} ---")
        fl_cfg = self.config['fl_params']
        train_cfg = self.config['training_params']
        prev_global_grad = None

        # Set the multiprocessing start method to 'spawn' for CUDA safety
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass # Start method can only be set once

        for round_idx in range(fl_cfg['num_rounds']):
            print(f"\n--- Round {round_idx + 1}/{fl_cfg['num_rounds']} ---")
            current_round_num = round_idx + 1 
            
            selected_clients = np.random.choice(self.clients, fl_cfg['clients_per_round'], replace=False)
            global_params = self.server.get_params()
            
            # --- Parallel Client Training ---
            # 1. Prepare the arguments for each client task
            tasks = []
            for client in selected_clients:
                task_args = (
                    client,
                    global_params,
                    train_cfg['local_epochs'],
                    round_idx,
                    prev_global_grad
                )
                tasks.append(task_args)
    

            # 2. Create a process pool and run the tasks in parallel
            with mp.Pool(processes=self.num_workers) as pool:
                # starmap is used to pass multiple arguments to the worker function
                client_updates = pool.starmap(run_client_task, tasks)

            # 3. Process the results from the parallel execution
            print(f"Received updates from {len(client_updates)} clients.")
            for update in client_updates:
                client_id = update['client_id']
                #print(f"Processing update from client {client_id} with update id={id(update)}")
                if update and 'weights' in update and 'num_samples' in update:
                    weights_on_device = {k: v.to(self.device) for k,v in update['weights'].items()}
                    self.server.receive_update(
                        client_id=client_id, 
                        params=weights_on_device, 
                        length=update['num_samples']
                    )
            
            # --- Server Aggregation & Update Tracking ---
            params_before_agg = copy.deepcopy(self.server.get_params())
            if isinstance(self.server, REPNNMKrumServer): # If using the reputation-based server, pass the current round for dynamic reputation updates
                self.server.aggregate(current_round=current_round_num)
            else:    
                self.server.aggregate()
            params_after_agg = self.server.get_params()

            prev_global_grad = {k: params_after_agg[k] - params_before_agg[k] for k in params_before_agg}

            # --- Evaluation ---
            main_metrics = self.server.evaluate()
            main_acc = main_metrics['metrics'].get('main_accuracy', main_metrics['metrics'].get('accuracy', -1.0))
            main_loss = main_metrics['metrics'].get('loss', -1.0)
            print(f"Global Model Accuracy: {main_acc:.4f}")

            # Determine if the attack is configured and currently active
            attack_cfg = self.config.get('attack_params', {})
            attack_enabled = attack_cfg.get('enabled', False)
            is_attack_active = False
            attack_start_round = 0
            if attack_enabled:
                attack_start_round = attack_cfg.get('attack_start_round', 0)
                end = attack_cfg.get('attack_end_round', float('inf'))
                if attack_start_round <= round_idx <= end:
                    is_attack_active = True            

            # --- ASR Evaluation Logic ---
            asr = 0.0             
            # Only start computing ASR after the attack begins
            if attack_enabled and round_idx >= attack_start_round:
                # Get the trigger object (it exists even if the attack isn't active this round)
                malicious_client_id = attack_cfg['malicious_client_ids'][0]
                trigger_obj = self.clients[malicious_client_id].trigger
                
                # Determine if we need to update the loader
                update_loader = (self.cached_backdoor_loader is None) or (not trigger_obj.is_static and is_attack_active)
                
                # Update the loader if needed
                if update_loader:
                    print(f"--- {'Creating' if self.cached_backdoor_loader is None else 'Updating'} backdoor test loader (Trigger type: {'Static' if trigger_obj.is_static else 'Dynamic'}) ---")
                    self.cached_backdoor_loader = self.adapter.get_backdoor_test_loader(
                        trigger_fn=trigger_obj.apply,
                        target_label=attack_cfg['target_label']
                    )

                # Evaluate using the (potentially updated) cached loader
                if self.cached_backdoor_loader:
                    asr_metrics = self.server.evaluate(valloader=self.cached_backdoor_loader)
                    asr = asr_metrics['metrics'].get('main_accuracy', asr_metrics['metrics'].get('accuracy', -1.0))
                    print(f"Attack Success Rate (ASR): {asr:.4f}")

            # --- Logging ---

            log_data = {
                'round': round_idx,
                'main_accuracy': main_acc,
                'main_loss': main_loss,
                'attack_success_rate': asr, 
                'is_attack_active': int(is_attack_active),
            }
            self.logger.log_round(log_data)

        self.logger.close()    
        print("\n--- Experiment Finished ---")
        self.server.save_model(f"{self.config['experiment_name']}_final_model.pth")