from torch.utils.data import Dataset
from typing import Callable
import numpy as np
import torch 


# class BackdoorDataset(Dataset):
#     """
#     A Dataset wrapper that applies a backdoor trigger to a subset of the data
#     and changes their labels to a target label.

#     Args:
#         original_dataset (Dataset): The clean dataset to be poisoned.
#         trigger_fn (Callable): A function that takes a data sample (e.g., an image tensor)
#             and returns the sample with the trigger applied.
#         target_label (int): The new label for the poisoned samples.
#         poison_fraction (float, optional): The fraction of the dataset to poison.
#             Defaults to 1.0 (all samples).
#         seed (int, optional): Random seed for selecting which samples to poison.
#             Defaults to 0.
#     """
#     def __init__(self,
#                  original_dataset: Dataset,
#                  trigger_fn: Callable,
#                  target_label: int,
#                  poison_fraction: float = 1.0,
#                  seed: int = 0,
#                  poison_exclude_target: bool = True): 
        
#         self.original_dataset = original_dataset
#         self.trigger_fn = trigger_fn
#         self.target_label = target_label
#         self.poison_fraction = poison_fraction
#         self.dataset_size = len(self.original_dataset)

#         all_indices = np.arange(self.dataset_size)
#         eligible_indices = all_indices

#         if poison_exclude_target:
#             try:
#                 targets = np.asarray(self.original_dataset.targets)
#             except AttributeError:
#                 print("Warning: BackdoorDataset falling back to slow label extraction.")
#                 targets = np.asarray([self.original_dataset[i][1] for i in all_indices])
            
#             eligible_indices = all_indices[targets != self.target_label]

#         num_eligible = len(eligible_indices)
#         num_poisoned = int(num_eligible * self.poison_fraction)

#         rng = np.random.RandomState(seed)
#         self.poisoned_indices = set(rng.choice(eligible_indices, num_poisoned, replace=False))

#     def __len__(self) -> int:
#         return self.dataset_size

#     def __getitem__(self, index: int) -> tuple:
#         # Get the original data and label once
#         data, label = self.original_dataset[index]

#         # Apply backdoor if the index is marked for poisoning
#         if index in self.poisoned_indices:
#             data = self.trigger_fn(data)
#             label = self.target_label

#         return data, label
    
class BackdoorDataset(Dataset):
    """
    A Dataset wrapper that applies a backdoor trigger to a subset of the data
    and changes their labels to a target label.

    Args:
        original_dataset (Dataset): The clean dataset to be poisoned.
        trigger_fn (Callable): A function that takes a data sample (e.g., an image tensor)
            and returns the sample with the trigger applied.
        target_label (int): The new label for the poisoned samples.
        poison_fraction (float, optional): The fraction of the dataset to poison.
            Defaults to 1.0 (all samples).
        seed (int, optional): Random seed for selecting which samples to poison.
            Defaults to 0.
    """
    def __init__(self,
                 original_dataset: Dataset,
                 trigger_fn: Callable,
                 target_label: int,
                 poison_fraction: float = 1.0,
                 seed: int = 0,
                 poison_exclude_target: bool = True): 
        
        self.original_dataset = original_dataset
        self.trigger_fn = trigger_fn
        self.target_label = target_label
        self.poison_fraction = poison_fraction
        
        dataset_size = len(self.original_dataset)
        all_indices = np.arange(dataset_size)
        eligible_indices = all_indices

        if poison_exclude_target:
            try:
                targets = np.asarray(self.original_dataset.targets)
            except AttributeError:
                print("Warning: BackdoorDataset falling back to slow label extraction for eligibility check.")
                targets = np.asarray([self.original_dataset[i][1] for i in all_indices])
            
            eligible_indices = all_indices[targets != self.target_label]

        num_eligible = len(eligible_indices)
        num_poisoned = int(num_eligible * self.poison_fraction)

        rng = np.random.RandomState(seed)
        self.poisoned_indices = set(rng.choice(eligible_indices, num_poisoned, replace=False))
        
        self.cached_data = []
        self.cached_labels = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for index in range(dataset_size):
            data, label = self.original_dataset[index]
            
            if index in self.poisoned_indices:
                if isinstance(data, torch.Tensor):
                    data = data.to(device)
                
                data = self.trigger_fn(data)
                label = self.target_label
                
                if isinstance(data, torch.Tensor):
                    data = data.cpu()

            self.cached_data.append(data)
            self.cached_labels.append(label)
        
        print("Caching complete.")

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index: int) -> tuple:
        return self.cached_data[index], self.cached_labels[index]

