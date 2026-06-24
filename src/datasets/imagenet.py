from torchvision.datasets import ImageFolder
from torchvision import transforms
from .adapter import DatasetAdapter


class ImageNetAdapter(DatasetAdapter):
    def __init__(self, root: str = "data", download: bool = True):
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        
        super().__init__(root, download, train_transform, test_transform)

    def load_datasets(self):
        self._train_dataset = ImageFolder(
            root=f"{self.root}/train",
            transform=self.train_transform
            )
        self._test_dataset = ImageFolder(
            root=f"{self.root}/val",
            transform=self.test_transform
            )
        
class TinyImageNetAdapter(DatasetAdapter):
    def __init__(self, root: str = "data", download: bool = True):
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
            ])
        test_transform = transforms.Compose([
                transforms.Resize(72),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
            ])
        
        super().__init__(root, download, train_transform, test_transform)

    def load_datasets(self):
        self._train_dataset = ImageFolder(
            root=f"{self.root}/train",
            transform=self.train_transform
            )
        self._test_dataset = ImageFolder(
            root=f"{self.root}/val",
            transform=self.test_transform
            )