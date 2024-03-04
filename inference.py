"""
Modularized inference code. Runs a given data point across all models and calculates 
"""
import os
import torch
import random
import string
import pandas as pd
import numpy as np

from tools import Ensemble, CIFAR10_Dataset 
from argparse import ArgumentParser, Namespace


from typing import Any, Callable, Dict, List, TypeAlias, Tuple
from torch import Tensor
from pandas import DataFrame
from numpy import ndarray

from torch.utils.data import ConcatDataset, Dataset, DataLoader
from pytorch_lightning import Trainer   
from inference_stats import PIPELINE
import torchvision.transforms as T
from torchvision.datasets import CIFAR10


DEFAULT_MODELS_PATH = '/mnt/ssd-1/variance-across-time/cifar-ckpts'
DEFAULT_OOD_DATASET_PATH = '/mnt/ssd-1/sai/variance-across-time/own/'
DEFAULT_RES_SAVE_PATH = '/mnt/ssd-1/sai/variance-across-time/datasets/'
DEFAULT_OOD_DATASET_CORRUPTIONS = [
    'brightness', 'frost', 'jpeg_compression', 'shot_noise', 'contrast', 'gaussian_blur', 
    'snow', 'defocus_blur', 'gaussian_noise', 'motion_blur', 'spatter', 'elastic_transform', 
    'glass_blur', 'pixelate', 'speckle_noise', 'fog', 'impulse_noise', 'saturate', 'zoom_blur'
]
DEFAULT_DATASET_TYPES = [
    'out_of_distribution', 'train', 'test'
]

def get_model_paths(args: Namespace) -> List[str]:
    """Gets all paths to pytorch checkpoint files
    """
    all_model_paths = []
    for warp in range(args.warps):
        # TODO: warp 3 does not seem to exist
        if warp == 3: continue

        for idx in range(args.models_per_warp):
            model_path = os.path.join(args.models_path,f'warp_{warp}', f'model_{idx}', f'step={args.step}.pt')
            all_model_paths.append(model_path)
    
    return all_model_paths

def get_datasets(args) -> DataFrame:
    """Yields appropriate dataset based on dataset distributions

    Args:
        args (Namespace): Input arguments
    
    Yields:
        (DataFrame) dataset from the given dataset distribution
    """
    if isinstance(args.dataset_distribution, list):
        for dist in args.dataset_distribution:
            args.dataset_distribution = dist
            yield from get_datasets(args)
    
    if args.dataset_distribution == 'out_of_distribution':
        for corruption in args.ood_dataset_corruptions:
            data = np.load(os.path.join(args.ood_dataset_path, f'{corruption}_srs1000.npy'))
            labels = np.load(os.path.join(args.ood_dataset_path, 'labels_srs1000.npy'))
            yield CIFAR10_Dataset(data, labels, corruption)

    elif args.dataset_distribution == 'train':
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

    elif args.dataset_distribution == 'test':
        test = CIFAR10(
            root='./cifar-test', train=False, download=True, transform=T.ToTensor(),
        )
        test.corruption = 'none'
        yield test
    else:
        raise NotImplementedError(f"{args.dataset_distribution} is not implemented.")
    


def get_logits(args, dataset) -> Tensor:
    """Calculates log probabilities of samples of dataset across all models in all warps

    Args:
        args (Namespace): Passed input arguments

    Returns:
        log probabilities: (torch.Tensor of shape (dataset_size, num_models, num_classes))
    """
    dataloader = DataLoader(dataset, batch_size=args.dataset_batch_size, num_workers=4)

    model_paths = get_model_paths(args)
    all_log_probs = []

    lig_trainer = Trainer(
        accelerator = 'gpu',
        devices = [args.gpu_id],
        precision = '16-mixed'
    )

    for i in range(0, len(model_paths), args.models_per_gpu):
        print(f"Iterating over models {i}...{min(i+args.models_per_gpu, len(model_paths))}")
        curr_model_paths = model_paths[i:i+args.models_per_gpu]
        
        # Initializing with a random model seed, seed does not matter as we load them anyways
        models = Ensemble(100, len(curr_model_paths), "")
        models.from_pretrained(curr_model_paths)

        log_probs = torch.cat(lig_trainer.predict(models, dataloader))
        all_log_probs.append(log_probs)
    
    all_log_probs = torch.cat(all_log_probs, dim = -2)
    return all_log_probs

def run_pipeline_and_save(args):
    """Runs all inference metrics on datasets and saves the results

    Args:
        args (Namespace): Input arguments
    """
    all_results = []
    all_data = {
        'datasets': [],
        'labels': [],
        'corruption': []
    }
    for dataset in get_datasets(args):
        all_data['datasets'].append(dataset)
        all_data['labels'].extend(dataset.targets)
        all_data['corruption'].extend([dataset.corruption for _ in range(len(dataset))])
    
    concat_dataset = ConcatDataset(all_data['datasets'])
    all_data.pop('datasets')

    results = DataFrame.from_dict(all_data)
    logits = get_logits(args, concat_dataset)
    PIPELINE.transform(logits, results)   
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, f'{args.run_id}_inference_metrics.parquet')
    results.to_parquet(save_path)


if __name__ == '__main__':
    # Use Tensor Cores even for float32
    torch.set_float32_matmul_precision("high")
    parser = ArgumentParser()
    parser.add_argument('--step', type=int, default=16384)
    parser.add_argument('--models-per-warp', type=int, default=32)
    parser.add_argument('--warps', type=int, default=128)
    parser.add_argument('--models-path', type=str, default=DEFAULT_MODELS_PATH)
    parser.add_argument('--gpu-id', type=int, default=6)
    parser.add_argument('--models-per-gpu', type=int, default=512)
    parser.add_argument('--dataset-batch-size', type=int, default=64)
    parser.add_argument('--ood-dataset-path', type=str, default=DEFAULT_OOD_DATASET_PATH)
    parser.add_argument(
        '--ood-dataset-corruptions', type=list, 
        choices=DEFAULT_OOD_DATASET_CORRUPTIONS,
        default=DEFAULT_OOD_DATASET_CORRUPTIONS
    )
    parser.add_argument('--save_path', type=str, default=DEFAULT_RES_SAVE_PATH)
    parser.add_argument(
        '--dataset-distribution', type=str, 
        choices=DEFAULT_DATASET_TYPES, 
        default=DEFAULT_DATASET_TYPES
    )
    default_random_id = random.choices(string.ascii_lowercase, k=10)
    random.shuffle(default_random_id)
    parser.add_argument('--run-id',type=str,default=''.join(default_random_id))
    args = parser.parse_args()
    run_pipeline_and_save(args)
    
