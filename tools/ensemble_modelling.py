"""
Ensemble model, vmaps across multiple, ConvNext models for more speedup
"""

import torch
import math
import tempfile

from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from pathlib import Path
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch import nn, optim
from torch.func import functional_call, stack_module_state
from torch.optim.optimizer import Optimizer


from transformers import (
    ConvNextV2Config, ConvNextV2ForImageClassification,
    get_cosine_schedule_with_warmup,
)

NUM_STEPS = 2 ** 16

class Ensemble(pl.LightningModule):
    def __init__(
        self, seed: int, num_models: int, rsync_dest: str | None = None
    ):
        """ConvNextV2 model ensemble. forward calls and grads are parallelized 
        with vmap across multiple models.

        Args:
            seed (int): Seed for initialization
            num_models (int): Number of models to create an ensemble of
            rsync_dest (str): [Only during training] Path to sync checkpoints into
        """
        super().__init__()
        self.save_hyperparameters()

        dtype = torch.float32
        self.models = nn.ModuleList()
        self.is_model_vectorized = False

        for offset in range(num_models):
            torch.cuda.manual_seed_all(seed + offset)
            torch.manual_seed(seed + offset)

            self.models.append(self.build_net().to(dtype))

    def build_net(self) -> nn.Module:
        """Builds a single ConvNextV2 model

        Returns:
            (nn.Module): Instance of `ConvNextV2ForImageClassification
        """
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
    
    def from_pretrained(self, paths: list[str]):
        """
        loads models from the given list of checkpoint paths. 

        Args:
            paths (list[str]): List of saved checkpoint paths. 
                Length of list must be same as `num_models`
        """

        if len(paths) != len(self.models):
            raise ValueError(f"Number of model paths should be {len(self.models)}")
        
        for idx, path in enumerate(paths):
            self.models[idx].load_state_dict(torch.load(path))

    def vectorize_models(self):
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
        self.is_model_vectorized = True
        return params, bufs

    def configure_optimizers(self):
        # Vectorize models while building optimizers during training.
        if not self.is_model_vectorized:
            params, bufs = self.vectorize_models()

        opt = optim.AdamW(params.values(), lr=0.002, weight_decay=0.05)
        schedule = get_cosine_schedule_with_warmup(opt, 1_000, NUM_STEPS)
        return [opt], [{"scheduler": schedule, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train').sum()

    def predict_step(self, batch, batch_idx):
        # Vectorize models if they are not vectorized earlier
        if not self.is_model_vectorized:
            self.vectorize_models()
    
        log_probs = self.shared_step(batch, 'predict')
        return log_probs

    @staticmethod
    def jenson_shannon_divergence(lps):
        """Calculates Jenson Shannon Divergance (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)
        
        Args:
            lps (torch.Tensor): Log probabilities of shape (*, `num_models`, `num_classes`)
        """
        n = np.log(lps.shape[-2])
        mixture_lps = lps.logsumexp(dim=-2, keepdim=True).sub(n)
        return torch.sum(lps.exp() * (lps - mixture_lps), dim=-1).mean(dim=-1)
    
    def shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x).logits 

        if stage == 'predict':
            return logits.log_softmax(dim=-1).transpose(-2, -3)

        with torch.no_grad():
            lps = logits.log_softmax(dim=-1)
            jsd = self.jenson_shannon_divergence(lps)
            self.log(f'{stage}_jsd', jsd.mean(), on_step=True)

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
        return losses
    
    def validation_step(self, batch, batch_idx):
        # Vectorize model here if it is not vectorized already
        if not self.is_model_vectorized:
            params, bufs = self.vectorize_models()
        return self.shared_step(batch, 'val')

    def test_step(self, batch, batch_idx=0):
        # Vectorize model here if it is not vectorized already
        if not self.is_model_vectorized:
            params, bufs = self.vectorize_models()
        return self.shared_step(batch, 'test')

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
