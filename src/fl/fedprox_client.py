import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import copy

from .client import BenignClient # Assuming BenignClient is in client.py

class FedProxClient(BenignClient):
    """
    Implements the FedProx client-side algorithm.

    Adds a proximal term to the local loss function to mitigate
    issues related to data heterogeneity (non-IID).
    Loss = Local_Loss + (mu / 2) * ||w - w_t||^2
    """
    def __init__(self, mu: float = 0.01, *args, **kwargs):
        """
        Initializes the FedProx client.

        Args:
            mu (float): The hyperparameter controlling the strength of the
                        proximal term. Defaults to 0.01.
            *args, **kwargs: Arguments passed to the parent BenignClient.
        """
        super().__init__(*args, **kwargs)
        self.mu = mu
        # Store for the proximal term calculation
        self.initial_params: Optional[Dict[str, torch.Tensor]] = None
        # print(f"FedProx Client {self.id} initialized with mu={self.mu}") # Debug print

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        """
        Loads parameters from the server and stores a copy
        of the initial parameters for the proximal term.
        """
        super().set_params(params) # Load params into self.model
        # Store a deep copy of the initial parameters (w_t) on the client's device
        self.initial_params = {
            name: param.clone().detach().to(self.device)
            for name, param in self.model.named_parameters()
        }


    def local_train(self, round_idx: int, epochs: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Performs local training using the FedProx objective function.

        Args:
            round_idx (int): The current federated learning round index.
            **kwargs: Additional arguments (e.g., prev_global_params for TDFed compatibility,
                      though not used by FedProx itself).

        Returns:
            Dict[str, Any]: Dictionary containing updated weights, num_samples, metrics etc.
        """
        if self.trainloader is None:
            print(f"Warning: Client {self.id} has no trainloader. Skipping training.")
            # Return current weights or handle as appropriate
            return {
                'client_id': self.get_id(),
                'num_samples': 0,
                'weights': self.get_params(), # Return current (global) weights
                'metrics': {'loss': float('nan'), 'accuracy': float('nan')},
                'round_idx': round_idx
            }

        if self.initial_params is None:
             raise RuntimeError(f"Client {self.id}: FedProxClient cannot train before set_params is called.")

        self.model.train() # Set model to training mode
        # Recreate optimizer to reset state, bound to current self.model parameters
        self._create_optimizer()

        train_loss, correct, total = 0.0, 0, 0
        proximal_term_total = 0.0 # To track the proximal term magnitude

        for _ in range(epochs):
            num_batches_epoch = 0
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)

                # 1. Calculate standard local loss (L_i(w))
                local_loss = self.loss_fn(output, target)

                # 2. Calculate FedProx proximal term: (mu / 2) * ||w - w_t||^2
                proximal_term = torch.tensor(0.0, device=self.device)
                # Iterate through current parameters (w) and initial parameters (w_t)
                for name, param_current in self.model.named_parameters():
                    if param_current.requires_grad: # Only include trainable params
                        # Ensure initial param exists and is on the correct device
                        param_initial = self.initial_params.get(name)
                        if param_initial is not None:
                             # Add squared L2 norm of the difference
                             proximal_term += torch.sum((param_current - param_initial).pow(2))
                        else:
                             print(f"Warning: Client {self.id}: Initial parameter '{name}' not found for prox term.")


                # 3. Combine losses
                total_loss = local_loss + (self.mu / 2.0) * proximal_term

                total_loss.backward()

                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=...)

                self.optimizer.step()

                # --- Accumulate Metrics ---
                train_loss += total_loss.item() # Log the total FedProx loss
                proximal_term_total += proximal_term.item() # Log the proximal term value
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                num_batches_epoch += 1

        if self.scheduler:
            self.scheduler.step()

        # --- Calculate Average Metrics ---
        total_batches_processed = num_batches_epoch * epochs
        if total_batches_processed > 0:
            avg_loss = train_loss / total_batches_processed
            avg_prox_term = proximal_term_total / total_batches_processed
        else:
            avg_loss = float('nan')
            avg_prox_term = float('nan')

        accuracy = correct / total if total > 0 else 0.0

        # --- Package results ---
        final_weights_cpu = self.get_params() # Gets CPU state dict
        return_metrics = {
             'loss': avg_loss, # Total FedProx loss
             'accuracy': accuracy,
             'proximal_term': avg_prox_term # Include prox term magnitude in metrics
        }
        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': final_weights_cpu,
            'metrics': return_metrics,
            'round_idx': round_idx
        }
