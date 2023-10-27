from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial
import math

from pytorch_lightning.loggers import WandbLogger
from torch import nn, optim
from torch.func import functional_call, stack_module_state
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from vit_pytorch import ViT

import pytorch_lightning as pl
import torch
import torchvision.transforms as T


def patched_resnet18(num_classes: int):
    from torchvision.models import resnet18

    model = resnet18(num_classes=num_classes)
    model.maxpool = torch.nn.Identity() # type: ignore[attr-type]

    return model


class Ensemble(pl.LightningModule):
    def __init__(self, num_models: int):
        super().__init__()
        self.save_hyperparameters()

        models = [
            patched_resnet18(num_classes=10).to(torch.bfloat16)
            for _ in range(num_models)
        ]
        self.params, self.bufs = stack_module_state(models) # type: ignore[arg-type]
        self.template = models[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        @partial(torch.vmap, in_dims=(0, 0, None), randomness="same")
        def fmodel(params, bufs, x):
            return functional_call(self.template, (params, bufs), x)

        return fmodel(self.params, self.bufs, x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        with torch.no_grad():
            denom = math.log(len(logits))
            lps = logits.log_softmax(dim=-1)
            mixture_lps = lps.logsumexp(dim=0).sub(denom)

            jsd = torch.sum(lps.exp() * (lps - mixture_lps), dim=-1).mean()
            self.log('train_jsd', jsd)

        y_broadcast = y.expand(len(logits), -1).flatten()
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y_broadcast)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        with torch.no_grad():
            denom = math.log(len(logits))
            lps = logits.log_softmax(dim=-1)
            mixture_lps = lps.logsumexp(dim=0).sub(denom)

            jsd = torch.sum(lps.exp() * (lps - mixture_lps), dim=-1).mean()
            self.log('val_jsd', jsd)

        y_broadcast = y.expand(len(logits), -1).flatten()
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y_broadcast)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        with torch.no_grad():
            denom = math.log(len(logits))
            lps = logits.log_softmax(dim=-1)
            mixture_lps = lps.logsumexp(dim=0).sub(denom)

            jsd = torch.sum(lps.exp() * (lps - mixture_lps), dim=-1).mean()
            self.log('test_jsd', jsd)

        y_broadcast = y.expand(len(logits), -1).flatten()
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y_broadcast)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        opt = optim.SGD(self.params.values(), lr=0.005, momentum=0.9, weight_decay=5e-4)
        schedule = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
        return [opt], [schedule]


@dataclass
class LogSpacedCheckpoint(pl.Callback):
    """Save checkpoints at log-spaced intervals"""

    dirpath: str

    base: float = 2.0
    next: int = 1

    def on_train_batch_end(self, trainer: pl.Trainer, *_):
        if trainer.global_step >= self.next:
            self.next = round(self.next * self.base)
            trainer.save_checkpoint(self.dirpath + f"/step={trainer.global_step}.ckpt")


if __name__ == '__main__':
    SEED = 0
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    torch.set_default_dtype(torch.bfloat16)

    # Use Tensor Cores even for float32
    torch.set_float32_matmul_precision("high")

    parser = ArgumentParser()
    parser.add_argument('num_models', type=int, default=128)
    args = parser.parse_args()

    with torch.device("cuda"):
        ensemble = Ensemble(args.num_models)

    trf = T.Compose([
        T.ToTensor(), T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),
    ])
    nontest = CIFAR10(
        root='./data', train=False, download=True, transform=trf,
    )
    train, val = random_split(nontest, [0.9, 0.1])
    test = CIFAR10(
        root='./data', train=False, download=True, transform=T.ToTensor(),
    )

    dm = pl.LightningDataModule.from_datasets(
        train, val, test, batch_size=128, num_workers=8
    )
    trainer = pl.Trainer(
        callbacks=[LogSpacedCheckpoint(dirpath='./checkpoints')],
        logger=WandbLogger(project='variance-across-time', entity='eleutherai'),
        max_epochs=200,
    )
    trainer.fit(ensemble, dm)
    trainer.test(ensemble, dm)
