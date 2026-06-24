from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import copy as cp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class BaseServer(ABC):
    @abstractmethod
    def set_params(self, state_dict: Dict[str, torch.Tensor]) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def aggregate(self) -> Dict[str, torch.Tensor]:
        pass

class FedAvgAggregator(BaseServer):
    def __init__(self, model: torch.nn.Module, testloader=None, device: Optional[torch.device]=None, *args, **kwargs):
        self.device = device if device is not None else torch.device("cpu")
        self.model = cp.deepcopy(model).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.testloader = testloader  
        self.received_updates: Dict[int, Dict[str, Any]] = {}

    def load_testdata(self, testloader):
        self.testloader = testloader

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        # load params onto server model (move to device)
        self.model.load_state_dict({k: v.to(self.device) for k, v in params.items()})

    def receive_update(self, client_id: int, params: Dict[str, torch.Tensor], length: int) -> None:
        # store params as CPU tensors to make aggregation stable
        params_cpu = {k: v.cpu().clone().float() for k, v in params.items()}
        self.received_updates[client_id] = {
            'params': params_cpu,
            'length': int(length)
        }

    def evaluate(self, valloader=None) -> Dict[str, object]:
        valloader = valloader or self.testloader
        self.model.eval()
        if valloader is None:
            return {'num_samples': 0, 'metrics': {'loss': float('nan'), 'main_accuracy': float('nan')}}
        loss_sum, correct, total, iters = 0.0, 0, 0, 0
        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                correct += (preds == targets).sum().item()
                loss_sum += self.loss_fn(outputs, targets).item()
                total += targets.size(0)
                iters += 1
        return {'num_samples': total, 'metrics': {'loss': (loss_sum / iters) if iters else float('nan'), 'main_accuracy': (correct / total) if total else float('nan')}}

    def aggregate(self) -> Dict[str, torch.Tensor]:
        assert len(self.received_updates) > 0, "No client updates received for aggregation."
        total_samples = sum(data['length'] for data in self.received_updates.values())

        averaged = {}
        # initialize zeros on cpu and accumulate weighted sums
        first_client_data = next(iter(self.received_updates.values()))
        first_params = first_client_data['params']

        for k in first_params.keys():
            acc = torch.zeros_like(first_params[k], dtype=torch.float32)
            # Iterate over the dictionary values
            for client_data in self.received_updates.values():
                client_params = client_data['params']
                client_len = client_data['length']
                weight = float(client_len) / float(total_samples)
                if k in client_params:
                   acc += client_params[k] * weight
                # else: print(f"Warning: Key {k} not in client update.") # Optional warning
            averaged[k] = acc  # cpu tensors

        # load averaged params to server model
        self.set_params({k: v.to(self.device) for k, v in averaged.items()})

        # reset buffers
        self.received_updates = {}

        return {k: v.cpu().clone() for k, v in averaged.items()}

    def save_model(self, path: str) -> None:
        # save state_dict for portability
        torch.save(self.model.state_dict(), path)
