"""
Modularized inference code. Runs a given data point across all models and calculates 
"""
import os
import torch
import pandas as pd
import numpy as np
from train import Ensemble
from argparse import ArgumentParser

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer   

DEFAULT_MODELS_PATH = '/mnt/ssd-1/variance-across-time/cifar-ckpts'
DEFAULT_DATASET_PATH = '/mnt/ssd-1/sai/variance-across-time/own/'
DEFAULT_RES_SAVE_PATH = '/mnt/ssd-1/sai/variance-across-time/datasets/'
DEFAULT_DATASET_CORRUPTIONS = [
    'brightness', 'frost', 'jpeg_compression', 'shot_noise', 'contrast', 'gaussian_blur', 
    'snow', 'defocus_blur', 'gaussian_noise', 'motion_blur', 'spatter', 'elastic_transform', 
    'glass_blur', 'pixelate', 'speckle_noise', 'fog', 'impulse_noise', 'saturate', 'zoom_blur'
]

def get_model_paths(args):
    all_model_paths = []
    for warp in range(args.warps):
        # TODO: warp 3 does not seem to exist
        if warp == 3: continue

        for idx in range(args.models_per_warp):
            model_path = os.path.join(args.models_path,f'warp_{warp}', f'model_{idx}', f'step={args.step}.pt')
            all_model_paths.append(model_path)
    
    return all_model_paths

class CIFAR10_Dataset(Dataset):
    def __init__(self, data, labels):
        self.labels = torch.tensor(labels, dtype=torch.int64)
        self.data = torch.tensor(data)
        

    def __getitem__(self, idx):
        # put channels first, as input
        return (self.data[idx].permute(-1, 0, 1)/255, self.labels[idx])

    def __len__(self):
        return len(self.labels)



def get_jenson_shannon_divergances(args, dataset_corruption):
    """Calculates log probabilities of samples of dataset across all models in all warps

    Args:
        args (argparse.Namespace): Passed input arguments

    Returns:
        (torch.Tensor of shape (dataset_size, num_models, num_classes))
    """
    model_paths = get_model_paths(args)
    all_log_probs = []
    data = np.load(os.path.join(args.dataset_path, f'{dataset_corruption}_srs10000.npy'))
    labels = np.load(os.path.join(args.dataset_path, 'labels_srs10000.npy'))
    dataloader = DataLoader(CIFAR10_Dataset(data, labels), batch_size=args.dataset_batch_size, num_workers=4)
    lig_trainer = Trainer(
        accelerator = 'gpu',
        devices = [args.gpu_id],
        precision = '16-mixed'
    )
    for i in range(0, len(model_paths), args.models_per_gpu):
        print(f"Iterating over models {i}...{min(i+args.models_per_gpu, len(model_paths))}")
        curr_model_paths = model_paths[i:i+args.models_per_gpu]
        models = Ensemble(100, len(curr_model_paths), "")
        models.from_pretrained(curr_model_paths)
        log_probs = torch.cat(lig_trainer.predict(models, dataloader))
        all_log_probs.append(log_probs)
    
    all_log_probs = torch.cat(all_log_probs, dim = -2)
    all_jsds = models.jenson_shannon_divergance(all_log_probs)

    
    return all_jsds.numpy(), labels




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
    parser.add_argument('--dataset-path', type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        '--dataset-corruptions', type=list, 
        choices=DEFAULT_DATASET_CORRUPTIONS,
        default=DEFAULT_DATASET_CORRUPTIONS
    )
    parser.add_argument('--save_path', type=str, default=DEFAULT_RES_SAVE_PATH)
    args = parser.parse_args()
    result = pd.DataFrame()
    for corruption in args.dataset_corruptions:
        all_jsds, labels = get_jenson_shannon_divergances(args, corruption)
        result[f'{corruption}_jenson_shannon_divergances'] = all_jsds
    
    result['labels'] = labels
    result['ids'] = range(len(labels))        
        
    os.makedirs(args.save_path, exist_ok=True)
    result.to_parquet(os.path.join(args.save_path, f'jsds.parquet'))
    
