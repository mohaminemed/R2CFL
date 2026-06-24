from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import copy as cp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class BaseClient(ABC):
    @abstractmethod
    def get_id(self) -> int:
        pass

    @abstractmethod
    def num_samples(self) -> int:
        pass

    @abstractmethod
    def set_params(self, state_dict: Dict[str, torch.Tensor]) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def local_evaluate(self) -> Dict[str, Any]:
        pass


    

class BenignClient(BaseClient):
    def __init__(
        self,
        id: int,
        trainloader: Optional[DataLoader],
        testloader: Optional[DataLoader],
        model: torch.nn.Module,
        lr: float,
        weight_decay: float,
        epochs: int = 1,
        device: Optional[torch.device] = None,
    ):
        self.id = id
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device if device is not None else torch.device("cpu")
        self.epochs_default = epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self._model = model.to(self.device)
        self.dataset_len = len(trainloader.dataset) if trainloader is not None else 0
        
        # Optimizer and scheduler will now be created on-demand
        self.optimizer = None
        self.scheduler = None
        self._create_optimizer() # Initial creation
        
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    def _create_optimizer(self) -> None:
        """Recreates the optimizer, binding it to the current model parameters."""
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.97)

    def get_id(self) -> int:
        return self.id

    def num_samples(self) -> int:
        return self.dataset_len

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        """Load parameters and recreate the optimizer to reset its state."""
        self.model.load_state_dict(params)
        self.model.to(self.device)
        self._create_optimizer()
        
    def get_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self._model.state_dict().items()}

    def local_train(self, epochs: int, round_idx: int, **kwargs) -> Dict[str, Any]:
        """Train locally and return metrics collected during training."""
        self.model.train()
        
        train_loss, correct, total = 0.0, 0, 0
        
        for _ in range(epochs or self.epochs_default):
            if self.trainloader is None: break
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # Accumulate metrics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        if self.scheduler:
            self.scheduler.step()

        num_batches = len(self.trainloader) if self.trainloader else 1
        avg_loss = train_loss / (num_batches * (epochs or self.epochs_default))
        accuracy = correct / total if total > 0 else 0.0
        
        metrics = {'loss': avg_loss, 'accuracy': accuracy}
        
        result = {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': metrics,
            'round_idx': round_idx
        }
        return result

    def local_evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on the local test set (or train set if no test set)."""
        self.model.eval()
        loss_sum, correct, total, iters = 0.0, 0, 0, 0
        # Use testloader if available, otherwise fallback to trainloader
        valloader = self.testloader or self.trainloader
        if valloader is None:
            return {'metrics': {'loss': float('nan'), 'accuracy': float('nan')}}

        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                
                correct += (preds == targets).sum().item()
                loss_sum += self.loss_fn(outputs, targets).item()
                total += targets.size(0)
                iters += 1
                
        loss_avg = (loss_sum / iters) if iters > 0 else float('nan')
        acc = (correct / total) if total > 0 else float('nan')
        
        return {'client_id': self.get_id(), 'num_samples': total, 'metrics': {'loss': loss_avg, 'accuracy': acc}}