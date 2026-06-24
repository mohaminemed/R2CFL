import torch
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader

from ..fl.client import BenignClient
from ..datasets.backdoor import BackdoorDataset
from .triggers.patch_trigger import PatchTrigger


class NeurotoxinClient(BenignClient):
    """
    Implements the Neurotoxin model poisoning attack.

    This client acts as a stealthy backdoor attacker. When activated, it trains
    on a mixed dataset of clean and poisoned (triggered) samples. The key aspect
    of the attack is that it constrains the learning to the "least important"
    parameters of the model, minimizing the impact on the main task's accuracy
    to remain undetected.

    Importance is determined using the aggregated global model update from the
    previous round (`prev_global_grad`). Parameters that changed the most are
    considered important, and their gradients are zeroed out during local training.

    Attributes:
        attack_config (Dict): A dictionary containing all attack-specific
            hyperparameters. See the `__init__` method for expected keys.
    """
    def __init__(self, attack_config: Dict, *args, **kwargs):
        """
        Initializes the malicious Neurotoxin client.

        Args:
            attack_config (Dict): A dictionary of attack parameters. Expected keys:
                - 'trigger' (BaseTrigger, optional): An instantiated trigger object.
                - 'target_label' (int): The target class for the backdoor.
                - 'attack_start_round' (int): The round to start the attack.
                - 'attack_end_round' (int): The round to end the attack.
                - 'mask_k_percent' (float): The percentage of important parameters
                  to mask (e.g., 0.05 for top 5%).
                - 'poison_fraction' (float): The fraction of local data to poison.
                - 'seed' (int, optional): Seed for poisoning reproducibility.
            *args: Variable length argument list passed to the parent constructor.
            **kwargs: Arbitrary keyword arguments passed to the parent constructor.
        """
        super().__init__(*args, **kwargs)
        self.attack_config = attack_config

        # Extract attack parameters from the config dictionary
        self.trigger = attack_config.get('trigger', PatchTrigger())
        self.target_label = attack_config.get('target_label', 0)
        self.attack_start_round = attack_config.get('attack_start_round', 0)
        self.attack_end_round = attack_config.get('attack_end_round', float('inf'))
        self.mask_k_percent = attack_config.get('mask_k_percent', 0.05)
        self.poison_fraction = attack_config.get('poison_fraction', 0.25)
        self.malicious_epochs = attack_config.get('malicious_epochs', 5)
        
        # Create the poisoned dataset once for performance
        self.poisoned_dataset = BackdoorDataset(
            original_dataset=self.trainloader.dataset,
            trigger_fn=self.trigger.apply,
            target_label=self.target_label,
            poison_fraction=self.poison_fraction,
            seed=attack_config.get('seed', 42)
        )
        
    def local_train(self, round_idx: int, epochs: int = 1, prev_global_grad: Optional[Dict[str, torch.Tensor]] = None, **kwargs) -> Dict[str, Any]:
        """
        Performs a poisoned local training round using the Neurotoxin strategy.

        If the current round is within the attack window, this method:
        1. Builds an importance mask from the previous global update (`prev_global_grad`).
        2. Trains on a local poisoned dataset.
        3. Applies the mask to the gradients during training, constraining the
           update to only the "unimportant" parameters.
        4. Returns the malicious model update.
        
        If outside the window, it behaves like a benign client.

        Args:
            epochs (int): The number of local training epochs, specified by the server.
            round_idx (int): The current federated learning round index.
            prev_global_grad (Dict[str, torch.Tensor], optional): A dictionary
                containing the aggregated model update (delta) from the previous
                round. This is used for importance calculation. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the client's ID, number of
                samples, the malicious model weights, and tracked training metrics.
        """
        
        # If outside attack window, behave like a benign client
        if not (self.attack_start_round <= round_idx <= self.attack_end_round):
            print(f"\n--- Neurotoxin Client [{self.id}] behaving benignly for round {round_idx} ---")
            return super().local_train(epochs, round_idx)
        
        print(f"\n--- Neurotoxin Client [{self.id}] starting hybrid attack for round {round_idx} ---")

        # --- Build robust grad mask (top-k by normalized importance) ---
        grad_mask: Optional[Dict[str, torch.Tensor]] = None
        if prev_global_grad is None:
            print(f"Client [{self.id}]: No previous global gradient. Attacking without mask.")
        else:
            # (Detailed mask creation logic as you implemented it)
            model_param_keys = set(name for name, _ in self.model.named_parameters())
            importance_parts = []
            key_to_delta = {}
            eps = 1e-12

            for name, delta in prev_global_grad.items():
                if name not in model_param_keys:
                    continue
                d_cpu = delta.detach().cpu().to(torch.float32)
                param_cpu = self.model.state_dict()[name].detach().cpu().to(torch.float32)
                importance = (d_cpu.abs() / (param_cpu.abs() + eps)).flatten()
                importance_parts.append(importance)
                key_to_delta[name] = d_cpu

            if not importance_parts:
                print(f"Client [{self.id}]: No matching trainable keys in prev_global_grad. Attacking without mask.")
            else:
                all_importances = torch.cat(importance_parts)
                k = max(1, int(self.mask_k_percent * all_importances.numel()))
                threshold = torch.topk(all_importances, k, largest=True, sorted=True)[0][-1].item()
                
                grad_mask = {}
                for name, delta_cpu in key_to_delta.items():
                    param_cpu = self.model.state_dict()[name].detach().cpu().to(torch.float32)
                    importance_key = (delta_cpu.abs() / (param_cpu.abs() + eps))
                    grad_mask[name] = (importance_key < threshold)

        # --- Create Dataloader and perform local training ---
        poisoned_loader = DataLoader(self.poisoned_dataset, batch_size=self.trainloader.batch_size, shuffle=True)
        self.model.train()
        train_loss, correct, total = 0.0, 0, 0

        for _ in range(self.malicious_epochs):
            for data, target in poisoned_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                
                if grad_mask is not None:
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if param.grad is not None and name in grad_mask:
                                mask = grad_mask[name].to(param.grad.dtype).to(param.grad.device)
                                param.grad.mul_(mask)
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        if self.scheduler:
            self.scheduler.step()
        
        # --- Package and return results ---
        num_batches = len(poisoned_loader)
        avg_loss = train_loss / (num_batches * self.malicious_epochs) if num_batches > 0 else float('nan')
        accuracy = correct / total if total > 0 else 0.0
        metrics = {'loss': avg_loss, 'accuracy': accuracy}
       
        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': metrics,
            'round_idx': round_idx
        }