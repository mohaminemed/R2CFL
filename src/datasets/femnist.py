from .adapter import DatasetAdapter
from torchvision.datasets import EMNIST
from torchvision import transforms



class FEMNISTDataset(DatasetAdapter):
    def __init__(self, root: str = "data", download: bool = True):
        train_transform = transforms.Compose([
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)) # Normalize for grayscale images
            ])
        test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        
        super().__init__(root, download, train_transform, test_transform)

    def load_datasets(self):
        self._train_dataset = EMNIST(
            root=self.root,
            split='byclass', # This split has 62 classes (digits, upper, lower)
            train=True,
            download=True,
            transform=self.train_transform
            )
        self._test_dataset = EMNIST(
            root=self.root,
            split='byclass',
            train=False,
            download=True,
            transform=self.test_transform
            )