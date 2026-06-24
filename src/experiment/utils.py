from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
import copy
from torch.utils.data import DataLoader

# --- Component Imports ---

# Datasets
from ..datasets.gtsrb import GTSRBDataset
from ..datasets.cifar10 import CIFAR10Dataset
from ..datasets.mnist import MNISTDataset
from ..datasets.femnist import FEMNISTDataset
from ..datasets.adapter import DatasetAdapter
from ..datasets.cifar100 import CIFAR100Dataset
from ..datasets.fashionmnist import FashionMNISTDataset

# Models
from ..models.gtsrb import GTSRB_CNN
from ..models.cifar import CifarNetGN
from ..models.mnist import MNISTNet 
from ..models.mnist import EMNIST_CNN 
from ..models.mnist import Fashion_CNN
from ..models.unet import UNet, FEMNISTAutoencoder


# Core FL Components
from ..fl.client import BaseClient, BenignClient
from ..fl.fedprox_client import FedProxClient
from ..fl.server import FedAvgAggregator

# Attack Components
from ..attacks.neurotoxin_client import NeurotoxinClient
from ..attacks.a3fl_client import A3FLClient
from ..attacks.triggers.a3fl import A3FLTrigger
from ..attacks.triggers.patch_trigger import PatchTrigger

from ..attacks.omp_client import OMPClient


# Defense Components
from ..defenses.krum import MKrumServer
from ..defenses.flame import FlameServer

from ..defenses.deepsight import DeepSightServer

from ..defenses.nnm import NNMKrumServer
from ..defenses.rep_nnm import REPNNMKrumServer
from ..defenses.autodfl import AutoDFLFilteringServer
from ..defenses.autobfl import AutoBFLFilteringServer
from ..defenses.ssmtd_hmm import SSMTDHMMFilteringServer


def get_data_and_model(data_config: Dict) -> Tuple[DatasetAdapter, torch.nn.Module]: 
    """Returns the appropriate dataset adapter and model instance."""
    dataset_name = data_config.get('dataset_name', 'gtsrb') 
    root = data_config.get('root', 'data')
    download = data_config.get('download', True) 

    if dataset_name.lower() == 'gtsrb':
        adapter = GTSRBDataset(root, download)
        model = GTSRB_CNN(num_classes=43)
    elif dataset_name.lower() == 'cifar10':
        adapter = CIFAR10Dataset(root, download)
        model = CifarNetGN(num_classes=10)
    elif dataset_name.lower() == 'cifar100':
        adapter = CIFAR100Dataset(root, download)
        model = CifarNetGN(num_classes=100)
    elif dataset_name.lower() == 'mnist':
        adapter = MNISTDataset(root, download)
        model = MNISTNet() 
    elif dataset_name.lower() == 'femnist':
        adapter = FEMNISTDataset(root, download)
        model = EMNIST_CNN(num_classes=62)
    elif dataset_name.lower() == 'fashionmnist':
        adapter = FashionMNISTDataset(root, download)
        model =  Fashion_CNN(num_classes=10)      
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    return adapter, model

def select_clients(client_list: List, num_selected_clients: int, selection_strat: str = "random") -> List: 
    """
    Selects a subset of clients for the training round.
    Note: 'selection_strat' is a placeholder for future strategies.
    """
    if not client_list: return []

    if selection_strat.lower() != "random":
        print(f"Warning: Selection strategy '{selection_strat}' not implemented. Defaulting to 'random'.")

    num_to_select = min(num_selected_clients, len(client_list))
    chosen_clients = np.random.choice(client_list, num_to_select, replace=False)
    return list(chosen_clients)


def get_server_instance(config: Dict, model, test_loader, device):
    """
    Factory function to create the server instance.
    Includes defense mechanisms.
    """
    defense_cfg = config.get('defense_params', {})
    defense_enabled = defense_cfg.get('enabled', False) 
    defense_name = defense_cfg.get('name', 'none').lower() if defense_enabled else 'none'
    dataset_name = config.get('data_params', {}).get('dataset_name', 'gtsrb').lower()

    logging_kwargs = {
        'output_dir': config.get("output_dir", "results"),
        'experiment_name': config.get('experiment_name', 'default_exp')
    }

    if defense_name == 'krum':
        print("Instantiating MKrum server.")
        return MKrumServer(model, test_loader, device, defense_cfg, **logging_kwargs)
    elif defense_name == 'flame':
        print("Instantiating Flame server.")
        return FlameServer(model, test_loader, device, defense_cfg, **logging_kwargs)
    elif defense_name == 'deepsight':
        print("Instantiating DeepSight server.")
        defense_cfg['dataset'] = dataset_name
        return DeepSightServer(model, test_loader, device, defense_cfg, **logging_kwargs)
    elif defense_name == 'nnm_krum':
        return NNMKrumServer(model=model, testloader=test_loader, device=device, config=config, defense_config=defense_cfg, **logging_kwargs)  
    elif defense_name == 'rep_nnm':      
        return REPNNMKrumServer(model=model, testloader=test_loader, device=device, config=config, defense_config=defense_cfg, **logging_kwargs)
    elif defense_name == 'autodfl':
        return AutoDFLFilteringServer(model=model, testloader=test_loader, device=device, config=config, defense_config=defense_cfg, **logging_kwargs)
    elif defense_name == 'autobfl':
        return AutoBFLFilteringServer(model=model, testloader=test_loader, device=device, config=config, defense_config=defense_cfg, **logging_kwargs)
    elif defense_name == 'ssmtd_hmm':
        return SSMTDHMMFilteringServer(model=model, testloader=test_loader, device=device, config=config, defense_config=defense_cfg, **logging_kwargs)
    else: # Default case: No defense or unknown defense name
        if defense_enabled and defense_name != 'none':
             print(f"Warning: Unknown defense '{defense_name}'. Falling back to standard FedAvg.")
        print("Instantiating standard FedAvg server.")
        return FedAvgAggregator(model=model, testloader=test_loader, device=device)


def get_client_instance(
    config: Dict,
    client_id: int,
    train_loader: Optional[DataLoader], 
    model, 
    device
) -> BaseClient: 
    """
    Factory function to create a client instance based on the config.
    Now optimized for potential parallel execution where loader might be None initially.
    """
    attack_cfg = config.get('attack_params', {})
    malicious_ids = set(attack_cfg.get('malicious_client_ids', []))
    training_params = config['training_params']

    # --- 1. Define all base arguments for any client ---
    base_client_args = {
        'id': client_id,
        'trainloader': train_loader, 
        'testloader': None,
        'model': model, 
        'lr': training_params.get('lr', 0.01),
        'weight_decay': training_params.get('weight_decay', 5e-4), 
        'epochs': training_params.get('local_epochs', 1), 
        'device': device
    }

    if attack_cfg.get('enabled') and client_id in malicious_ids:
        attack_name = attack_cfg.get('name')
        print(f"Instantiating malicious client {client_id} for attack: {attack_name}")

        # --- 2. Create the Trigger object explicitly ---
        trigger_obj = None
        if 'trigger' in attack_cfg:
            trigger_cfg = attack_cfg['trigger']
            trigger_name = trigger_cfg.get('name')
            data_cfg = config.get('data_params', {}) 

            img_size_map = {'mnist': (28, 28), 'femnist': (28, 28), 'fashionmnist': (28, 28), 'cifar10': (32, 32), 'gtsrb': (32, 32)}
            channels_map = {'mnist': 1, 'femnist': 1, 'fashionmnist': 1, 'cifar10': 3, 'gtsrb': 3}
            dataset_name_lower = data_cfg.get('dataset_name', 'mnist').lower() 
            image_size = tuple(trigger_cfg.get('image_size', img_size_map.get(dataset_name_lower, (32,32)))) 
            in_channels = trigger_cfg.get('in_channels', channels_map.get(dataset_name_lower, 3)) 
            if trigger_name == 'a3fl':
                trigger_obj = A3FLTrigger(
                    position=tuple(trigger_cfg.get('position', [image_size[0]-4, image_size[1]-4])), 
                    size=tuple(trigger_cfg.get('size', [3, 3])),
                    in_channels=in_channels,
                    image_size=image_size,
                    trigger_epochs=trigger_cfg.get('trigger_epochs', 5),
                    trigger_lr=trigger_cfg.get('trigger_lr', 0.01),
                    lambda_balance=trigger_cfg.get('lambda_balance', 0.1),
                    adv_epochs=trigger_cfg.get('adv_epochs', 10),
                    adv_lr=trigger_cfg.get('adv_lr', 0.01)
                )
            elif trigger_name == 'patch':
                position = tuple(trigger_cfg.get('position', [image_size[0]-4, image_size[1]-4]))
                size = tuple(trigger_cfg.get('size', [3, 3]))

                # Ensure color matches number of channels
                if in_channels == 1:
                   color = (1.0,)  # grayscale → single value
                else:
                   color = tuple(trigger_cfg.get('color', [1.0] * in_channels))
                trigger_obj = PatchTrigger(
                   position=position,
                   size=size,
                   color=color
                )
            else:
                 print(f"Warning: Unknown trigger name '{trigger_name}'. Trigger object will be None.")


        # Prepare config dict to pass to the malicious client constructor
        malicious_config = copy.copy(attack_cfg) 
        malicious_config['trigger'] = trigger_obj
        malicious_config['seed'] = config.get('seed', 42)

        # --- 3. Create Malicious Client ---
        if attack_name == 'neurotoxin':
            return NeurotoxinClient(attack_config=malicious_config, **base_client_args)
        elif attack_name == 'a3fl':
            return A3FLClient(attack_config=malicious_config, **base_client_args)
        elif attack_name == 'omp':
            return OMPClient(attack_config=malicious_config, **base_client_args)
        else:
            # Fallback or error for unknown attack
             print(f"Warning: Unknown attack name '{attack_name}' for malicious client {client_id}. Creating BenignClient instead.")
             return BenignClient(**base_client_args) # Fallback to benign if attack name unknown
    else:
        fedprox_mu = training_params.get('fedprox_mu', 0.0)
        if fedprox_mu > 0.0:
            print(f"Instantiating FedProx client {client_id} with mu={fedprox_mu}.")
            return FedProxClient(mu=fedprox_mu, **base_client_args)  
    return BenignClient(**base_client_args)

