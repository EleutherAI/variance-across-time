from argparse import ArgumentParser

from tools.ensemble_modelling import Ensemble
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10

import pytorch_lightning as pl
import torchvision.transforms as T


NUM_STEPS = 2 ** 16

if __name__ == '__main__':
    # Use Tensor Cores even for float32
    torch.set_float32_matmul_precision("high")

    parser = ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('--num-models', type=int, default=32)
    parser.add_argument('--rsync-dest', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    trf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.RandAugment(),
        T.ToTensor(),
    ])
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
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)

    print(f"\033[92m--- Using seed {args.seed} ---\033[0m")

    dm = pl.LightningDataModule.from_datasets(
        train, val, test, batch_size=64, num_workers=4,
    )
    ensemble = Ensemble(
        args.seed, args.num_models, args.rsync_dest
    )

    trainer = pl.Trainer(
        logger=WandbLogger(
            name=args.name, project="variance-across-time", entity="eleutherai"
        ),
        max_steps=NUM_STEPS,
        # Mixed precision with (b)float16 activations
        precision="16-mixed",
    )
    trainer.fit(ensemble, dm)
    trainer.test(ensemble, dm)
