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
# from vit import ViT

import pytorch_lightning as pl
import torch
import torchvision.transforms as T


def mangle_name(name: str) -> str:
    return name.replace('.', '-')


def unmangle_name(name: str) -> str:
    return name.replace('-', '.')


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
            ViT(
                image_size=32,
                patch_size=4,
                num_classes=10,
                dim=512,
                depth=6,
                heads=8,
                mlp_dim=512,
                dropout=0.1,
                emb_dropout=0.1,
            ).to(torch.bfloat16).cuda()
            # patched_resnet18(num_classes=10).to(torch.bfloat16)
            for _ in range(num_models)
        ]
        template = models[0]

        params, bufs = stack_module_state(models) # type: ignore[arg-type]
        self.params = nn.ParameterDict({
            mangle_name(k): v for k, v in params.items()
        })
        self.bufs = nn.ParameterDict({
            mangle_name(k): v for k, v in bufs.items()
        })

        @partial(torch.vmap, in_dims=(0, 0, None), randomness="same")
        def fmodel(params, bufs, x):
            return functional_call(template, (params, bufs), x)
        
        self.forward = partial(fmodel, params, bufs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

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
            self.log('val_variance', lps.exp().var(dim=0).mean())

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
        opt = optim.RAdam(self.params.values(), lr=3e-4, weight_decay=1e-4, foreach=False)
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
    parser.add_argument('name', type=str)
    parser.add_argument('--num_models', type=int, default=256)
    args = parser.parse_args()

    with torch.device("cuda"):
        ensemble = Ensemble(args.num_models)

    trf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
    ])
    nontest = CIFAR10(
        root='./cifar-train', train=False, download=True, transform=trf,
    )
    train, val = random_split(nontest, [0.9, 0.1])
    test = CIFAR10(
        root='./cifar-test', train=False, download=True, transform=T.ToTensor(),
    )

    dm = pl.LightningDataModule.from_datasets(
        train, val, test, batch_size=1, num_workers=8
    )
    trainer = pl.Trainer(
        accumulate_grad_batches=128,
        callbacks=[LogSpacedCheckpoint(dirpath='./checkpoints')],
        logger=WandbLogger(
            args.name, project='variance-across-time', entity='eleutherai'
        ),
        max_epochs=200,
    )
    trainer.fit(ensemble, dm)
    trainer.test(ensemble, dm)
