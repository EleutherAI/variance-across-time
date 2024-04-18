import os
from typing import List
from itertools import product

import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from torch.utils.data import Dataset
from pandas import DataFrame
import torchvision.transforms as T
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from torchvision.datasets import CIFAR10

from tools.ensemble_modelling import Ensemble

CIFARC_CORRUPTIONS = [
    'brightness', 'frost', 'jpeg_compression', 'shot_noise', 'contrast', 'gaussian_blur', 
    'snow', 'defocus_blur', 'gaussian_noise', 'motion_blur', 'spatter', 'elastic_transform', 
    'glass_blur', 'pixelate', 'speckle_noise', 'fog', 'impulse_noise', 'saturate', 'zoom_blur'
]

DEFAULT_DATASET_TYPES = [
    'cifarc', 'train', 'test', 'cifar5m'
]


class CIFAR10_Dataset(Dataset):
    """CIFAR 10 class for Out of distribution dataset

    Args:
        data (ndarray): Array of shape (data_size, 32, 32, 3), image pixels
        labels (ndarray): Array of shape (data_size), target labels
        corruption (str): Name of the corruption
    """
    def __init__(self, data: ndarray, labels: ndarray, corruption: str):
        self.targets = labels
        self.data = torch.tensor(data)
        self.corruption = corruption
        
    def __getitem__(self, idx):
        # put channels first, as input
        return (self.data[idx].permute(-1, 0, 1)/255, self.targets[idx])

    def __len__(self):
        return len(self.targets)


def get_model_paths(
        models_path: str,
        warps: int = 128,
        models_per_warp: int = 32,
        step: int = 1) -> List[str]:
    """Gets all paths to pytorch checkpoint files
    """
    all_model_paths = []
    for warp, idx in product(range(warps), range(models_per_warp)):
        
        model_path = os.path.join(
            models_path, f'warp_{warp}', f'model_{idx}', f'step={step}.pt')
        
        if os.path.exists(model_path):
            all_model_paths.append(model_path)

    return all_model_paths


def get_datasets(dataset: list | str, dataset_path: str) -> DataFrame:
    """Yields appropriate dataset based on dataset distributions

    Args:
        args (Namespace): Input arguments
    
    Yields:
        (DataFrame) dataset from the given dataset distribution
    """
    if isinstance(dataset, list):
        for dist in dataset:
            yield from get_datasets(dist, dataset_path)
    
    elif dataset == 'cifarc':
        for corruption in CIFARC_CORRUPTIONS:
            data = np.load(os.path.join(dataset_path, f'{corruption}.npy'))
            labels = np.load(os.path.join(dataset_path, 'labels.npy'))
            yield CIFAR10_Dataset(data, labels, corruption)
    
    elif dataset == 'cifar5m':
        data = np.load(os.path.join(dataset_path, 'cifar5m_sample_images.npy'))
        labels = np.load(os.path.join(dataset_path, 'cifar5m_sample_labels.npy'))
        yield CIFAR10_Dataset(data, labels, "cifar5m")

    elif dataset == 'train':
        trf = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.RandAugment(),
            T.ToTensor(),
        ])
        train = CIFAR10(
            root='./cifar-train', train=True, download=True, transform=trf,
        )
        train.corruption = 'train'
        yield train
        
    elif dataset == 'test':
        test = CIFAR10(
            root='./cifar-test',
            train=False,
            download=True,
            transform=T.ToTensor(),
        )
        test.corruption = 'none'
        yield test
    else:
        raise NotImplementedError(f"{dataset} is not implemented.")


def get_logits(
        dataset: Dataset,
        batch_size: int,
        models_path: str,
        models_hidden_sizes: list[int] = [48, 96, 192, 384],
        warps: int = 128,
        models_per_warp: int = 64,
        models_per_gpu: int = 512,
        gpu_id: int = 6,
        step: int = 0) -> Tensor:
    """Calculates log probabilities of samples of dataset across all models in all warps

    Args:
        args (Namespace): Passed input arguments

    Returns:
        log probabilities: (torch.Tensor of shape (dataset_size, num_models, num_classes))
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    model_paths = get_model_paths(
        models_path,
        warps=warps,
        models_per_warp=models_per_warp,
        step=step
    )
    all_log_probs = []

    lig_trainer = Trainer(
        accelerator='gpu',
        devices=[gpu_id],
        precision='16-mixed'
    )

    for i in range(0, len(model_paths), models_per_gpu):
        print(f"Iterating over models {i}...{min(i+models_per_gpu, len(model_paths))}")
        curr_model_paths = model_paths[i:i+models_per_gpu]
        
        # Initializing with a random model seed, seed does not matter as we load them anyways
        models = Ensemble(100, len(curr_model_paths),
                          model_hidden_sizes=models_hidden_sizes)

        models.from_pretrained(curr_model_paths)

        log_probs = torch.cat(lig_trainer.predict(models, dataloader))
        all_log_probs.append(log_probs)
    
    all_log_probs = torch.cat(all_log_probs, dim=-2)
    return all_log_probs
