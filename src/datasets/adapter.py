from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
import torch
from .backdoor import BackdoorDataset

class DatasetAdapter(ABC):
    """
    Abstract adapter for datasets in Federated Learning experiments.

    A single instance of an adapter manages both the training and testing
    datasets. It provides methods to create partitioned DataLoaders for clients
    (from the training set) and a centralized DataLoader for evaluation
    (from the test set).
    """

    def __init__(self, root: str = "data", download: bool = True, train_transform: Optional[Callable] = None, test_transform: Optional[Callable] = None):
        self.root = root
        self.download = download
        self.train_transform = train_transform
        self.test_transform = test_transform
        self._train_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None

    @abstractmethod
    def load_datasets(self) -> None:
        """
        Loads the training and testing datasets.
        This method must be implemented by subclasses and should populate
        `self._train_dataset` and `self._test_dataset`.
        """
        raise NotImplementedError

    def setup(self) -> None:
        """Loads both datasets if they haven't been loaded yet."""
        if self._train_dataset is None or self._test_dataset is None:
            self.load_datasets()

    @property
    def train_dataset(self) -> Dataset:
        """Provides access to the loaded training dataset."""
        if self._train_dataset is None:
            raise RuntimeError("Datasets have not been loaded. Please call setup() first.")
        return self._train_dataset

    @property
    def test_dataset(self) -> Dataset:
        """Provides access to the loaded test dataset."""
        if self._test_dataset is None:
            raise RuntimeError("Datasets have not been loaded. Please call setup() first.")
        return self._test_dataset

    def get_test_loader(self, batch_size: int = 256) -> DataLoader:
        """
        Returns a DataLoader for the entire centralized test set.
        """
        self.setup()
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def get_backdoor_test_loader(
        self,
        trigger_fn: Callable,
        target_label: int,
        batch_size: int = 256
    ) -> DataLoader:
        """
        Returns a DataLoader for the test set where ALL samples have a
        backdoor trigger applied and are relabeled to the target label.
        """
        self.setup()

        # Use the wrapper, hardcoding poison_fraction to 1.0
        backdoor_dataset = BackdoorDataset(
            original_dataset=self.test_dataset,
            trigger_fn=trigger_fn,
            target_label=target_label,
            poison_fraction=1.0  # All samples are poisoned
        )

        return DataLoader(backdoor_dataset, batch_size=batch_size, shuffle=False)
    
    def get_client_loaders(
        self,
        num_clients: int,
        batch_size: int = 64,
        strategy: str = "iid",
        seed: int = 0,
        **strategy_args
    ) -> Dict[int, DataLoader]:
        """
        Partitions the TRAINING dataset among clients and returns DataLoaders.
        """
        self.setup()
        labels = self._extract_labels(self.train_dataset)
        dataset_size = len(self.train_dataset)

        if strategy.lower() == "iid":
            partitions = self.partition_iid(dataset_size, num_clients, seed)
        elif strategy.lower() == "dirichlet":
            alpha = float(strategy_args.get("alpha", 0.5))
            partitions = self.partition_dirichlet(labels, num_clients, alpha, seed)
        else:
            raise ValueError(f"Unknown partitioning strategy: {strategy}")

        loaders: Dict[int, DataLoader] = {}
        for cid, indices in partitions.items():
            if not indices:
                print(f"Warning: Client {cid} received 0 samples.")
                continue
            subset = Subset(self.train_dataset, indices)
            loaders[cid] = DataLoader(subset, batch_size=batch_size, shuffle=True)
        return loaders

    # ... (partitioning helpers and _extract_labels remain the same) ...
    @staticmethod
    def partition_iid(dataset_size: int, num_clients: int, seed: int = 0) -> Dict[int, List[int]]:
        rng = np.random.RandomState(seed)
        indices = np.arange(dataset_size)
        rng.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        return {i: s.tolist() for i, s in enumerate(splits)}

    @staticmethod
    def partition_dirichlet(labels: np.ndarray, num_clients: int, alpha: float, seed: int) -> Dict[int, List[int]]:
        """Partitions a dataset non-IID based on a Dirichlet distribution over classes."""
        rng = np.random.RandomState(seed)
        num_classes = int(labels.max()) + 1
        label_distribution = rng.dirichlet([alpha] * num_classes, num_clients)
        class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
        for indices in class_indices:
            rng.shuffle(indices)
        client_partitions: List[List[int]] = [[] for _ in range(num_clients)]
        for c_idx, indices in enumerate(class_indices):
            num_samples_for_class = len(indices)
            if num_samples_for_class == 0: continue
            class_proportions = label_distribution[:, c_idx]
            proportions = (num_samples_for_class / class_proportions.sum()) * class_proportions
            counts = proportions.astype(int)
            shortfall = num_samples_for_class - counts.sum()
            if shortfall > 0:
                order = np.argsort(proportions)[::-1]
                for i in range(shortfall): counts[order[i % len(order)]] += 1
            start = 0
            for client_id, count in enumerate(counts):
                client_partitions[client_id].extend(indices[start:start + count])
                start += count
        for i in range(num_clients): rng.shuffle(client_partitions[i])
        return {i: client_partitions[i] for i in range(num_clients)}

    @staticmethod
    def _extract_labels(ds: Dataset) -> np.ndarray:
        for attr in ("targets", "labels"):
            if hasattr(ds, attr): return np.asarray(getattr(ds, attr))
        if hasattr(ds, "samples"): return np.asarray([s[1] for s in ds.samples])
        # print("Warning: Falling back to slow label extraction by iterating.")
        return np.asarray([int(ds[i][1]) for i in range(len(ds))])