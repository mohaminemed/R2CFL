from .adapter import DatasetAdapter
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10
from typing import Optional
from torch.utils.data import DataLoader

class CIFAR10Dataset(DatasetAdapter):
    def __init__(self, root: str = "data", download: bool = True):
        train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        
        super().__init__(root, download, train_transform, test_transform)
    
    def load_datasets(self) -> None:
        self._train_dataset = CIFAR10(
            root=self.root, 
            train=True, 
            transform=self.train_transform, 
            download=self.download
            )
        self._test_dataset = CIFAR10(
            root=self.root, 
            train=False, 
            transform=self.test_transform, 
            download=self.download
            )
