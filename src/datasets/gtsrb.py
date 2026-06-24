from torchvision.datasets import GTSRB
from torchvision import transforms
from .adapter import DatasetAdapter


class GTSRBDataset(DatasetAdapter):
    def __init__(self, root: str = "data", download: bool = True):
        train_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomRotation(10),       # Rotate by up to 10 degrees
                transforms.ColorJitter(brightness=0.2, contrast=0.2), # Adjust brightness/contrast
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669])
            ])
        test_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669])
            ])
        
        super().__init__(root, download, train_transform, test_transform)

    def load_datasets(self):
        self._train_dataset = GTSRB(
            root= self.root, 
            split='train', 
            download = self.download, 
            transform = self.train_transform
            )
        self._test_dataset = GTSRB(
            root= self.root, 
            split= 'test', 
            download=self.download, 
            transform= self.test_transform
            )