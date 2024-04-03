"""
run some analysis on the CIFAR-C dataset
"""
import os
from rich import print as rprint
from rich.table import Table, Column
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
# import matplotlib.pyplot as plt

# list .npy files in dataset
archive_dir = "./CIFAR-10-C"
archives = [file for file in os.listdir(archive_dir) if file.endswith('.npy') and file != "labels.npy"]


final_table = Table(
    Column(header="Name"),
    Column(header="Shape"),
    Column(header="Top-1 P. Component Variance"),
    Column(header="Top-5 P. Component Variance"),
    Column(header="Top-10 P. Component Variance"),
    Column(header="RGB Mean"),
    Column(header="RGB Std. Dev.")
)

pca_machine = PCA(n_components=10)

for archive in tqdm(archives):
    # load .npy archive
    npy_path = os.path.join(archive_dir, archive)
    loaded_archive = np.load(npy_path)
   
    rgb_mean = loaded_archive.mean((0, 1, 2))
    rgb_std = loaded_archive.std((0, 1, 2))

    n, height, width, channels = loaded_archive.shape
    flattened_archive = loaded_archive.reshape(n, height * width * channels)
    
    # pca stuff

    pca = pca_machine.fit(flattened_archive)
    variance_ratios = pca.explained_variance_ratio_
    cumulative_ratios = [
        f"{x:.4f}" for x in
        [
        variance_ratios[0],
        sum(variance_ratios[:4]),
        sum(variance_ratios)
        ]
    ]

    final_table.add_row(
        archive.rstrip('.npy'),
        str(loaded_archive.shape),
        *cumulative_ratios,
        np.array2string(rgb_mean, precision=3, separator=', '),
        np.array2string(rgb_std, precision=3, separator=', ')
    )


rprint(final_table)
