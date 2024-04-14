"""
Modularized inference code. Runs a given data point across all models and calculates 
"""
import os
import torch
import random
import string

from tools import get_datasets, get_logits, DEFAULT_DATASET_TYPES
from argparse import ArgumentParser, BooleanOptionalAction

from pandas import DataFrame
from torch.utils.data import ConcatDataset
from inference_stats import PIPELINE

DEFAULT_MODELS_PATH = './cifar-ckpts'
DEFAULT_OOD_DATASET_PATH = './own/'
DEFAULT_RES_SAVE_PATH = './datasets/'


def run_pipeline_and_save(args):
    """Runs all inference metrics on datasets and saves the results

    Args:
        args (Namespace): Input arguments
    """
    all_data = {
        'datasets': [],
        'labels': [],
        'corruption': []
    }

    for dataset in get_datasets(args.dataset_distribution, args.ood_dataset_path):
        all_data['datasets'].append(dataset)
        all_data['labels'].extend(dataset.targets)
        all_data['corruption'].extend(
            [dataset.corruption for _ in range(len(dataset))])

    concat_dataset = ConcatDataset(all_data['datasets'])
    all_data.pop('datasets')

    results = DataFrame.from_dict(all_data)
    logits = get_logits(args, concat_dataset)

    # if logit tensor is requested, save to .pt
    if args.save_logits:
        filename = os.path.join(args.save_path, f"{args.run_id}_logits.pt")
        torch.save(logits, filename)

    PIPELINE.transform(logits, results, device=args.gpu_id)

    # save results to parquet file
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, f'{args.run_id}_inference_metrics.parquet')
    results.to_parquet(save_path)


if __name__ == '__main__':
    # Use Tensor Cores even for float32
    torch.set_float32_matmul_precision("high")
    parser = ArgumentParser()
    parser.add_argument('--save-logits', '-l', action=BooleanOptionalAction)
    parser.add_argument('--step', '-s', type=int, default=16384)
    parser.add_argument('--models-per-warp', '-m', type=int, default=32)
    parser.add_argument('--warps', '-w', type=int, default=128)
    parser.add_argument('--models-path', '-p', type=str, default=DEFAULT_MODELS_PATH)
    parser.add_argument('--gpu-id', '-g', type=int, default=6)
    parser.add_argument('--models-per-gpu', type=int, default=512)
    parser.add_argument('--dataset-batch-size', type=int, default=64)
    parser.add_argument('--ood-dataset-path', type=str, default=DEFAULT_OOD_DATASET_PATH)
    parser.add_argument('--save_path', type=str, default=DEFAULT_RES_SAVE_PATH)
    parser.add_argument(
        '--dataset-distribution', type=str,
        choices=DEFAULT_DATASET_TYPES,
        default=DEFAULT_DATASET_TYPES
    )
    default_random_id = random.choices(string.ascii_lowercase, k=10)
    random.shuffle(default_random_id)
    parser.add_argument('--run-id', type=str, default=''.join(default_random_id))
    
    # for handling different model configs
    parser.add_argument('--models-hidden-sizes',
                        '-z', nargs=4, default=[48, 96, 192, 384], type=int)
    
    args = parser.parse_args()
    run_pipeline_and_save(args)
    
