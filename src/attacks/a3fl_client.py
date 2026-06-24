from typing import Dict, Any
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

from ..fl.client import BenignClient
from ..datasets.backdoor import BackdoorDataset
from ..attacks.triggers.a3fl import A3FLTrigger

class A3FLClient(BenignClient):
    """
    Implements the Adversarial Adaptive FL backdoor Attack (A3FL).

    This client performs a two-stage attack in each round:
    1. It first optimizes a trigger pattern to be robust against a hardened
       version of the current global model.
    2. It then performs standard local training on its data, which has been
       poisoned with this newly optimized trigger.
    """
    def __init__(self, attack_config: Dict, *args, **kwargs):
        """
        Initializes the A3FL client.

        Args:
            attack_config (Dict): Attack parameters. Expected keys:
                - 'trigger' (A3FLTrigger): An instantiated A3FLTrigger object.
                - 'target_label' (int): The target class for the backdoor.
                - 'attack_start_round' (int): The round to start the attack.
                - 'poison_fraction' (float): Fraction of local data to poison.
                - 'trigger_sample_size' (int): Number of local data samples
                  to use for optimizing the trigger pattern.
        """
        super().__init__(*args, **kwargs)
        self.attack_config = attack_config
        self.trigger: A3FLTrigger = attack_config.get('trigger', A3FLTrigger())
        self.target_label = attack_config.get('target_label', 0)
        self.attack_start_round = attack_config.get('attack_start_round', 0)
        self.attack_end_round = attack_config.get('attack_end_round', float('inf'))
        self.poison_fraction = attack_config.get('poison_fraction', 0.25)
        self.seed = attack_config.get('seed', 42)
        self.trigger_sample_size = attack_config.get('trigger_sample_size', 512)
        self.malicious_epochs = attack_config.get('malicious_epochs', 5)
        
        
    def _build_trigger_dataloader(self) -> DataLoader:
        """Samples a small subset of local data for trigger optimization."""
        base_dataset = self.trainloader.dataset
        num_samples = len(base_dataset)
        
        # Ensure we don't sample more than available
        k = min(self.trigger_sample_size, num_samples)
        if k == 0: return None # Handle case with no data

        indices = np.random.choice(np.arange(num_samples), size=k, replace=False).tolist()
        sampled_ds = Subset(base_dataset, indices)
        
        # Ensure batch size is not larger than the sample size
        batch_size = min(getattr(self.trainloader, "batch_size", 32), k)
        return DataLoader(sampled_ds, batch_size=batch_size, shuffle=True)

    def local_train(self, round_idx: int, epochs: int = 1, **kwargs) -> Dict[str, Any]:
        """Performs the two-stage A3FL attack."""

        if not (self.attack_start_round <= round_idx <= self.attack_end_round):
            return super().local_train(epochs, round_idx)
        
        # --- Stage 1: Optimize the trigger for the current global model ---
        print(f"\n--- A3FL Client [{self.id}] optimizing trigger for round {round_idx} ---")
        trigger_dl = self._build_trigger_dataloader()
        if trigger_dl:
            self.trigger.train_trigger(
                classifier_model=self.model,
                dataloader=trigger_dl,
                target_class=self.target_label
            )

        # --- Stage 2: Train on data poisoned with the new trigger ---
        poisoned_dataset = BackdoorDataset(
            original_dataset=self.trainloader.dataset,
            trigger_fn=self.trigger.apply,
            target_label=self.target_label,
            poison_fraction=self.poison_fraction,
            seed=self.seed
        )
        poisoned_loader = DataLoader(poisoned_dataset, batch_size=self.trainloader.batch_size, shuffle=True)

        # Use the parent's training method on the poisoned loader
        original_loader = self.trainloader
        try:
            self.trainloader = poisoned_loader
            # Use the 'epochs' argument from the runner for this training stage
            result = super().local_train(self.malicious_epochs, round_idx)
        finally:
            # Always restore the original loader
            self.trainloader = original_loader

        return result