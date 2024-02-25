from argparse import ArgumentParser
import os
import re
import multiprocessing

import torch as t
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

from resnet_manager import ConvWarp


def pca_analysis(warp_folder: str, device: str) -> int:
    print(f"Starting {warp_folder}...")
    warp = ConvWarp(warp_folder)

    for bucket, df in warp.pca_over_time(cifar10_train, device=device).items():
        df.to_csv(os.path.join(warp_folder, f"pca_top_{bucket}.csv"))
    
    print("done")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('warps_folder', type=str)
    parser.add_argument('output_folder', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
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
        t.tensor(train.data, dtype=t.float32), (batch, channels, height, width)).to(args.device)
    
    dirs = filter(lambda dir: re.match('warp_\\d*', dir) is not None, os.listdir(args.warps_folder))
    dirs = map(lambda dir: (os.path.join(args.warps_folder, dir), args.device), dirs)
    
    with multiprocessing.Pool() as p:
        results = p.starmap(pca_analysis, list(dirs))
        print(results)
