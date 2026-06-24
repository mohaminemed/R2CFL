from torchvision.datasets import FashionMNIST
from torchvision import transforms
from .adapter import DatasetAdapter


class FashionMNISTDataset(DatasetAdapter):
    def __init__(self, root: str = "data", download: bool = True):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])

        super().__init__(root, download, train_transform, test_transform)

    def load_datasets(self):
        self._train_dataset = FashionMNIST(
            root=self.root,
            train=True,
            download=self.download,
            transform=self.train_transform
        )

        self._test_dataset = FashionMNIST(
            root=self.root,
            train=False,
            download=self.download,
            transform=self.test_transform
        )