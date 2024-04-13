from argparse import ArgumentParser
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable, Generator
import os
import math
import tempfile
import torch

from PIL import Image
from pytorch_lightning.loggers import WandbLogger
from torch import nn, optim
from torch.func import functional_call, stack_module_state
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, random_split

from tools.ensemble_modelling import Ensemble
from torchvision.datasets import CIFAR10
import torch

import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as T


NUM_STEPS = 2 ** 16

class Cifar5m(Dataset):
    def __init__(self, root: str, offset: int = 0, end: int = -1, transform: Callable = lambda x: x):
        N = os.stat(f"{root}/X.bin").st_size // (32 * 32 * 3) 
        num_images = max(end, N) - offset

        self.transform = transform
        self.X = np.memmap(
            f"{root}/X.bin", dtype=np.uint8, mode='r', offset=offset * 32 * 32 * 3, shape=(num_images, 32, 32, 3)
        )
        self.Y = np.memmap(
            f"{root}/Y.bin", dtype=np.uint64, mode='r', offset=offset, shape=(num_images,)
        )

    def __getitem__(self, idx) -> tuple[Image.Image, int]:
        return self.transform(Image.fromarray(self.X[idx])), int(self.Y[idx])

    def __len__(self):
        return len(self.X)

    def __iter__(self) -> Generator[tuple[Image.Image, int], None, None]:
        yield from zip(map(self.transform, map(Image.fromarray, self.X)), map(int, self.Y))


if __name__ == '__main__':
    # Use Tensor Cores even for float32
    torch.set_float32_matmul_precision("high")

    parser = ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('--cifar5m', type=str, default="")
    parser.add_argument('--num-models', type=int, default=64)
    parser.add_argument('--rsync-dest', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    trf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.RandAugment(),
        T.ToTensor(),
    ])
    if args.cifar5m:
        train = Cifar5m(args.cifar5m, transform=trf, end=5_992_688)
        nontrain = Cifar5m(args.cifar5m, offset=5_992_688, transform=T.ToTensor())
    else:
        train = CIFAR10(
            root='./cifar-train', train=True, download=True, transform=trf,
        )
        nontrain = CIFAR10(
            root='./cifar-test', train=False, download=True, transform=T.ToTensor(),
        )
    # Use the fixed seed 0 to do the val-test split
    torch.manual_seed(0)
    val, test = random_split(nontrain, [0.1, 0.9])

    # Set the random seed used for initializing the models
    seed = args.seed * args.num_models
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    print(f"\033[92m--- Using seed {seed} ---\033[0m")

    dm = pl.LightningDataModule.from_datasets(
        train, val, test, batch_size=64, num_workers=4,
    )
    ensemble = Ensemble(
        seed, args.num_models, args.rsync_dest
    )

    trainer = pl.Trainer(
        #logger=WandbLogger(
        #    name=args.name,
        #    project="variance",
        #    entity="eleutherai",
        #    save_dir=f"warp_{args.seed}",
        #),
        max_steps=NUM_STEPS,
        # Mixed precision with (b)float16 activations
        precision="bf16-mixed",
    )
    trainer.fit(ensemble, dm)
    trainer.test(ensemble, dm)
