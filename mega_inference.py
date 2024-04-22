from argparse import ArgumentParser, BooleanOptionalAction
from uuid import uuid4
import os

import torch
import numpy as np
import pandas as pd

from tools import (
    get_datasets,
    get_logits,
    DEFAULT_DATASET_TYPES,
    logit_pca
)

STEP_PROGESSION = np.power(2, range(17))

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = ArgumentParser()
    parser.add_argument('--save-logits', '-l', action=BooleanOptionalAction)
    parser.add_argument(
        '--steps', '-s', nargs='+',
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536], type=int
    )
    parser.add_argument(
        '--datasets', '-d', nargs='+', type=str, choices=DEFAULT_DATASET_TYPES, required=True)
    
    parser.add_argument('--gpu-id', '-g', type=int, default=6)
    parser.add_argument('--models-path', '-p', type=str, required=True)
    parser.add_argument('--fallback-dataset-path', type=str)
    parser.add_argument('--models-hidden-sizes', '-z', nargs=4, default=[48, 96, 192, 384], type=int)
    parser.add_argument('--models-per-warp', '-m', type=int, default=32)
    parser.add_argument('--warps', '-w', type=int, default=128)
    parser.add_argument('--models-per-gpu', type=int, default=512)
    parser.add_argument('--save_path', type=str, default='./mega_inference_output/')
    
    parser.add_argument('--run-id', '-i', type=str, default=str(uuid4()))

    args = parser.parse_args()
    
    # make run output folder
    save_dir = os.path.join(args.save_path, args.run_id)
    os.makedirs(save_dir, exist_ok=True)

    # for each dataset, load dataset
    for dataset_i, dataset in enumerate(
        get_datasets(args.datasets, args.fallback_dataset_path)
    ):

        # variances for all steps in this run-dataset combo
        dataset_variances = pd.DataFrame([])
        
        # dataset name
        dataset_name = args.datasets[dataset_i]

        # for each step, run inference
        for step in args.steps:
            print(f"Starting inference for step {step}")
            
            # if matching logit file is found, skip computation
            logit_filename = os.path.join(save_dir, f"{dataset_name}_{step}_logits.pt")
            
            if os.path.exists(logit_filename):
                logits: torch.Tensor = torch.load(logit_filename)
                print("Loaded logit from file!")
            else:
                logits: torch.Tensor = get_logits(
                    dataset,
                    64,
                    args.models_path,
                    args.models_hidden_sizes,
                    args.warps,
                    args.models_per_warp,
                    args.models_per_gpu,
                    args.gpu_id,
                    step=step
                )

                # save logits
                if args.save_logits:
                    torch.save(logits, logit_filename)

            # run pca
            try:
                variances = logit_pca(logits.cuda(args.gpu_id)).cpu().numpy()
                new_column = pd.DataFrame({
                    f'step {step}': variances
                })
                dataset_variances = pd.concat([dataset_variances, new_column], axis=1)
            except torch.cuda.OutOfMemoryError as oome:
                print("CUDA OUT OF MEMORY")
                print(torch.cuda.memory_summary())
            
        # save variances and graphs
        variances_filepath = os.path.join(save_dir, f"{dataset_name}_variances.csv")
        dataset_variances.to_csv(variances_filepath)
