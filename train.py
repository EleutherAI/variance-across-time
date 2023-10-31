from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial
import math
import torchvision as tv

from pytorch_lightning.loggers import WandbLogger
from torch import nn, optim
from torch.func import functional_call, stack_module_state
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from vit_pytorch import ViT

import pytorch_lightning as pl
import torch
import torchvision.transforms as T


NUM_STEPS = 2 ** 16

def mangle_name(name: str) -> str:
    return name.replace('.', '-')


def unmangle_name(name: str) -> str:
    return name.replace('-', '.')


class Ensemble(pl.LightningModule):
    def __init__(self, num_models: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.save_hyperparameters()

        models = [self.build_net().to(dtype) for _ in range(num_models)]
        template = models[0]

        params, bufs = stack_module_state(models) # type: ignore[arg-type]
        for k, v in bufs.items():
            self.register_buffer(mangle_name(k), v)
        for k, v in params.items():
            self.register_parameter(
                mangle_name(k), nn.Parameter(v)
            )

        @partial(torch.vmap, in_dims=(0, 0, None), randomness="error")
        def fmodel(params, bufs, x):
            return functional_call(template, (params, bufs), x)
        
        self.forward = partial(fmodel, params, bufs)
        self._params = params.values()
    
    def build_net(self):
        return ViT(
            image_size=32,
            patch_size=4,
            num_classes=10,
            dim=384,
            depth=6,
            heads=8,
            mlp_dim=384,
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        with torch.no_grad():
            denom = math.log(len(logits))
            lps = logits.log_softmax(dim=-1)
            mixture_lps = lps.logsumexp(dim=0).sub(denom)

            jsd = torch.sum(lps.exp() * (lps - mixture_lps), dim=-1).mean()
            self.log('train_jsd', jsd, on_step=True)

        loss_fn = torch.vmap(torch.nn.functional.cross_entropy, (0, None), 0)
        losses = loss_fn(logits, y)
        self.log('train_loss', losses.mean(), on_step=True, prog_bar=True)

        return losses.sum()

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

        loss_fn = torch.vmap(torch.nn.functional.cross_entropy, (0, None), 0)
        losses = loss_fn(logits, y)
        self.log('val_loss', losses.mean())

        acc = logits.argmax(dim=-1).eq(y).float().mean()
        self.log('val_acc', acc, on_step=False, on_epoch=True)

        return losses.sum()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss_fn = torch.vmap(torch.nn.functional.cross_entropy, (0, None), 0)
        losses = loss_fn(logits, y)
        self.log('test_loss', losses.mean())

        return losses.sum()

    def configure_optimizers(self):
        opt = optim.RAdam(self._params, lr=0.0005)
        schedule = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_STEPS)
        return [opt], [{"scheduler": schedule, "interval": "step"}]


class RegNetEnsemble(Ensemble):
    def build_net(self):
        model = tv.models.regnet_y_400mf(num_classes=10)
        model.stem[0].stride = 1
        model.stem.insert(0, nn.Upsample(scale_factor=2))

        return model

    def configure_optimizers(self):
        opt = optim.SGD(self._params, lr=0.1, momentum=0.9, weight_decay=5e-5)
        schedule = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_STEPS)
        return [opt], [{"scheduler": schedule, "interval": "step"}]


@dataclass
class LogSpacedCheckpoint(pl.Callback):
    """Save checkpoints at log-spaced intervals"""

    dirpath: str

    base: float = 2.0
    next: int = 1
    """One-indexed step number of the next checkpoint to save"""

    def on_train_batch_end(self, trainer: pl.Trainer, *_):
        if trainer.global_step >= self.next:
            self.next = round(self.next * self.base)
            trainer.save_checkpoint(
                self.dirpath + f"/step={trainer.global_step}.ckpt", weights_only=True
            )


if __name__ == '__main__':
    SEED = 0
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)

    # Use Tensor Cores even for float32
    torch.set_float32_matmul_precision("high")

    parser = ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('--num_models', type=int, default=32)
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
    val, test = random_split(nontrain, [0.1, 0.9])

    dm = pl.LightningDataModule.from_datasets(
        train, val, test, batch_size=128, num_workers=8
    )

    # Split the total number of models requested into the number of GPUs
    with torch.device("cuda"):
        has_bf16 = torch.cuda.is_bf16_supported()
        ensemble = Ensemble(
            args.num_models, dtype=torch.bfloat16 if has_bf16 else torch.float16
        )

    trainer = pl.Trainer(
        callbacks=[LogSpacedCheckpoint(dirpath='./checkpoints')],
        logger=WandbLogger(
            args.name, project='variance-across-time', entity='eleutherai'
        ),
        max_steps=NUM_STEPS,
        # Mixed precision with (b)float16 activations
        precision="bf16-mixed" if has_bf16 else 16,
    )
    trainer.fit(ensemble, dm)
    trainer.test(ensemble, dm)
