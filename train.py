from argparse import ArgumentParser
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from pathlib import Path
import math
import tempfile

from pytorch_lightning.loggers import WandbLogger
from torch import nn, optim
from torch.func import functional_call, stack_module_state
from torch.optim.optimizer import Optimizer
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from transformers import (
    ConvNextV2Config, ConvNextV2ForImageClassification,
    get_cosine_schedule_with_warmup,
)

import pytorch_lightning as pl
import torch
import torchvision.transforms as T


NUM_STEPS = 2 ** 16

class Ensemble(pl.LightningModule):
    def __init__(
        self, seed: int, num_models: int, rsync_dest: str | None = None
    ):
        super().__init__()
        self.save_hyperparameters()

        dtype = torch.float32
        self.models = nn.ModuleList()

        for offset in range(num_models):
            torch.cuda.manual_seed_all(seed + offset)
            torch.manual_seed(seed + offset)

            self.models.append(self.build_net().to(dtype))

    def build_net(self) -> nn.Module:
        cfg = ConvNextV2Config(
            image_size=32,
            # Femto architecture
            depths=[2, 2, 6, 2],
            hidden_sizes=[48, 96, 192, 384],
            num_labels=10,
            # The default of 4 x 4 patches shrinks the image too aggressively for
            # low-resolution images like CIFAR-10
            patch_size=2,
        )
        model = ConvNextV2ForImageClassification(cfg)

        # HuggingFace initialization is terrible; use PyTorch init instead
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.reset_parameters()

        return model

    def configure_optimizers(self):
        params, bufs = stack_module_state(list(self.models))
        template = self.models[0]

        @partial(torch.vmap, in_dims=(0, 0, None), randomness="different")
        def fmodel(params, bufs, x):
            return functional_call(template, (params, bufs), x)

        # forward should call vmap
        self.forward = partial(fmodel, params, bufs)

        # Replace the parameters in each module with a view onto the stacked params.
        # This allows us to easily run and save each model individually, while still
        # using torch.vmap and training them all in a vectorized fashion.
        for i, model in enumerate(self.models):
            for name, buf in model.named_buffers():
                buf.data = bufs[name][i]
            for name, param in model.named_parameters():
                param.data = params[name][i]

        opt = optim.AdamW(params.values(), lr=0.002, weight_decay=0.05)
        schedule = get_cosine_schedule_with_warmup(opt, 1_000, NUM_STEPS)
        return [opt], [{"scheduler": schedule, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x).logits

        with torch.no_grad():
            denom = math.log(len(logits))
            lps = logits.log_softmax(dim=-1)
            mixture_lps = lps.logsumexp(dim=0).sub(denom)

            jsd = torch.sum(lps.exp() * (lps - mixture_lps), dim=-1).mean()
            self.log(f'{stage}_jsd', jsd, on_step=True)

        loss_fn = torch.vmap(torch.nn.functional.cross_entropy, (0, None), 0)
        losses = loss_fn(logits, y)
        self.log(f'{stage}_loss', losses.mean())

        self.log(
            f'{stage}_acc',
            logits.argmax(dim=-1).eq(y).float().mean(),
            on_epoch=not self.training,
            on_step=self.training,
            prog_bar=(stage == 'val'),
        )
        return losses.sum()
    
    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, 'test')

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        # Log spaced checkpoints
        step = self.global_step + 1

        # Only save checkpoints at powers of 2
        if not math.log2(step).is_integer():
            return

        self.print(f"Saving step {step} checkpoints")
        dest = self.hparams.get('rsync_dest')

        with tempfile.TemporaryDirectory() if dest else nullcontext() as log_dir:
            log_dir = log_dir or self.trainer.log_dir
            assert log_dir is not None

            for i, model in enumerate(self.models):
                # Weird thing we have to do to prevent PyTorch from saving the entire
                # stack of models in the checkpoint instead of just the one we want
                model = deepcopy(model)

                p = Path(f"{log_dir}/model_{i}/step={step}.pt")
                p.parent.mkdir(parents=True, exist_ok=True)

                torch.save(model.state_dict(), p)

            # Push to rsync
            if dest:
                self.print(f"Pushing to {dest}")

                import sysrsync
                sysrsync.run(
                    source=str(log_dir),
                    destination=dest,
                    options=["-a"],
                )

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
