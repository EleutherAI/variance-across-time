from argparse import ArgumentParser
import os
import re
import uuid

import torch as t
import torchvision.transforms as T
import torch.multiprocessing as mp
from torchvision.datasets import CIFAR10
import pandas as pd

from resnet_manager import ConvNet

_checkpoint_name_fmt = "step=(\\d*).pt"
_warp_dir_name_fmt = "warp_\\d*"
_model_dir_name_fmt = "model_\\d*"

def pca_analysis(checkpoint: dict, device: str, dataset: t.Tensor) -> int:
    print(f"Loading {checkpoint['path']}")
    
    # cannot share cuda'd tensors in mp, must cuda them here
    dataset = dataset.to(device)
    
    model = ConvNet(checkpoint['path']).to(device)
    
    variances = model.get_logit_variance_ratios(dataset, batch_size=5000)
    
    checkpoint['top-1 variance'] = variances[0].sum().item()
    checkpoint['top-5 variance'] = variances[:5].sum().item()
    checkpoint['top-9 variance'] = variances[:9].sum().item()
    
    print(f"Finished {checkpoint['path']}")
    
    return checkpoint


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('warps_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('-w', '--workers', type=int, default=4)
    
    args = parser.parse_args()
    
    warps: str = args.warps_dir
    output_dir: str = args.output_dir
    device = args.device
    num_workers = args.workers
    
    t.set_num_threads(num_workers)

    train = CIFAR10(
        root='./cifar-train',
        train=True,
        download=True,
        transform=T.ToTensor(),
    )

    # nontrain = CIFAR10(
    #     root='./cifar-test',
    #     train=False,
    #     download=True,
    #     transform=T.ToTensor(),
    # )

    batch, height, width, channels = train.data.shape
    cifar10_train = t.reshape(
        t.tensor(train.data, dtype=t.float32), (batch, channels, height, width)
    )
    cifar10_train.share_memory_()
    
    checkpoints: list[dict[str, str]] = []
    
    pcas = pd.DataFrame()
    
    # TODO clean this monstrosity
    
    # go into warps dir, find all named warp folders
    for warp in os.listdir(warps):
        if re.search(_warp_dir_name_fmt, warp) is None:
            continue
        
        warp_folder = os.path.join(warps, warp)
        
        # go into each warp folder, find all named models
        for model in os.listdir(warp_folder):
            if re.search(_model_dir_name_fmt, model) is None:
                continue
            
            model_folder = os.path.join(warp_folder, model)
            
            # go into each model, find each checkpoint
            for checkpoint in os.listdir(model_folder):
                if re.search(_checkpoint_name_fmt, checkpoint) is None:
                    continue
                
                step_no = re.search(_checkpoint_name_fmt, checkpoint).groups()[0]
                
                # finally, add checkpoint to list
                checkpoints.append({
                    'warp': warp,
                    'model': model,
                    'checkpoint': checkpoint,
                    'step': step_no,
                    'path': os.path.join(model_folder, checkpoint)
                })


    multiprocessing_food = [
        (c, device, cifar10_train) for c in checkpoints
    ]
    
    mp.set_start_method('spawn', force=True)
    with mp.Pool(num_workers) as p:
        print(f"{len(multiprocessing_food)} checkpoints found.")
        results = p.starmap(pca_analysis, multiprocessing_food)
        pcas = pd.DataFrame(results)
        csv_filename = f"pcas_{uuid.uuid4()}.csv"
        csv_filepath = os.path.join(output_dir, csv_filename)
        pcas.to_csv(csv_filepath)
        print(f"Saved PCAs to {csv_filepath}")