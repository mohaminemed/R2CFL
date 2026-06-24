from .adapter import DatasetAdapter
from torchvision.datasets import MNIST
from torchvision import transforms

class MNISTDataset(DatasetAdapter):
    def __init__(self, root: str = "data", download: bool = True):
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        super().__init__(root, download, train_transform, test_transform)

    def load_datasets(self):
        self._train_dataset = MNIST(
            root=self.root, 
            train=True, 
            transform=self.train_transform, 
            download=self.download
            )
        self._test_dataset = MNIST(
            root= self.root, 
            train=False, 
            transform=self.test_transform, 
            download=self.download
            )