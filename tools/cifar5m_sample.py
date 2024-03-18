"""
Script that takes an archive of images and labels for CIFAR-5m and produces a
simple random sample.
"""

import argparse
import torch as t
import numpy as np
import os
from rich.table import Table
from rich.console import Console

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--seed", help="Set a random seed for reproducibility", type=int, default=None)
    parser.add_argument("--images", help="images .bin or .pt file", type=str)
    parser.add_argument("--labels", help="labels .bin or .pt file", type=str)
    parser.add_argument(
        "--num",
        "-n",
        help="size of sample (default 50k)",
        type=int,
        default=50_000
    )

    args = parser.parse_args()

    assert os.path.exists(args.images)
    assert os.path.exists(args.labels)

    # set seed if neede
    if args.seed is not None:
        t.manual_seed(args.seed)

    # load data

    labels = np.load(args.labels)
    print("Labels shape:", labels.shape)

    images = np.load(args.images)
    # images = einops.rearrange(labels, "(m h w c) -> m h w c", h=32, w=32, c=3)
    print("Images shape:", images.shape)

    num = args.num

    # randomly generate indices
    indices = t.randperm(len(labels))[:num]
    
    labels_subet = labels[indices]
    images_subset = images[indices]
    
    freqs = np.bincount(labels_subet)
    
    table = Table(title="Simple Random Sampling Label Frequencies")
    table.add_column("Class", style="green", no_wrap=True)
    table.add_column("Frequency", style="red", no_wrap=True)
    
    for index, count in enumerate(freqs):
        table.add_row(str(index), str(count))

    console = Console()
    console.print(table)

    # mask and save
    np.save("cifar5m_sample_labels", labels_subet)
    np.save("cifar5m_sample_images", images_subset)
