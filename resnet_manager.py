import os
import re
from itertools import product

import torch as t
from transformers import ConvNextV2Config, ConvNextV2ForImageClassification
import pandas as pd
from torch.utils.data import DataLoader
# from pytorch_lightning import LightningDataModule, LightningModule


# a singular instance of a RestNet
class ConvNet(t.nn.Module):
    _cfg = ConvNextV2Config(
        image_size=32,
        # Femto architecture
        depths=[2, 2, 6, 2],
        hidden_sizes=[48, 96, 192, 384],
        num_labels=10,
        # The default of 4 x 4 patches shrinks the image too aggressively for
        # low-resolution images like CIFAR-10
        patch_size=2,
    )

    def __init__(self, model_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = ConvNextV2ForImageClassification.from_pretrained(
            model_path,
            config=self._cfg,
            local_files_only=True
        )
        
    def forward(self, x):
        return self.model(x)


class ConvNetTimeLapse(t.nn.Module):
    """A module that loads and handles all of the checkpoints
    across a single models training, allowing batch computation of 
    statistics.
    """
    _checkpoint_name_fmt = "step=(\\d*).pt"

    def __init__(self, model_folder_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.checkpoints = t.nn.ModuleDict()
        self.model_folder_path = model_folder_path

        # find all .pt files in folder
        checkpoints = os.listdir(model_folder_path)

        # keep only files that match naming convention
        checkpoints = filter(lambda x: re.match(self._checkpoint_name_fmt, x) is not None, checkpoints)

        # concat full path to checkpoint
        checkpoints = map(lambda x: os.path.join(model_folder_path, x), checkpoints)

        # load all models
        for checkpoint in checkpoints:
            # grab step no.
            step_no = re.search(self._checkpoint_name_fmt, checkpoint).groups()[0]

            # note that OrderedDict requires Str keys
            self.checkpoints[step_no] = ConvNet(checkpoint)

        # ensure the is at least one valid checkpoint
        if len(self.checkpoints) == 0:
            raise FileExistsError(f"No valid checkpoints found in {model_folder_path}, valid checkpoint is named 'step=[step_no].pt'")

        # print(f"Successfully loaded {len(self.checkpoints)} checkpoints from {model_folder_path}")

    def forward(self, x) -> dict[str, t.Tensor]:
        return {
            step: self.checkpoints[step](x) for step in self.checkpoints
        }

    def pca_logits(
        self,
        x: t.Tensor,
        components: int = 10
    ) -> dict[str, t.Tensor]:
        """
        Using x images (batch, channels, height, width), computes variance
        ratios for the principal components of the logits.

        Args:
            x (Tensor): images
            components (int): number of components to report variance ratios
                for. Will be overriden if x.shape[0] < components. Defaults to 10

        Returns:
            dict[str, t.Tensor]: dictionary of step: Principal component variances
        """
        
        # assuming the dataset we're passing (CIFAR10) is large, batch the network forward
        batch_x = DataLoader(x, batch_size=256, shuffle=False)
        logits_acc: dict[str, t.Tensor] = {}
        
        for batch in batch_x:
            for step, output in self(batch).items():
                if step in logits_acc:
                    logits_acc[step] = t.concat([logits_acc[step], output.logits])
                else:
                    logits_acc[step] = output.logits

        for step in logits_acc:
            logits = logits_acc[step]
            centered = logits - logits.mean(dim=0)
            
            e_vals, _ = t.linalg.eigh(t.matmul(centered.T, centered))
            
            logits_acc[step] = t.flip(e_vals, dims=(0, ))

        return logits_acc

    def jsd_logits(self, x: t.Tensor) -> dict[str, t.Tensor]:
        pass


class ConvWarp:
    """ Loading/Handling models in one entire Warp
    """
    def __init__(self, warp_path: str) -> None:
        self.warp_path = warp_path
        self.models: t.nn.ModuleDict[str, ConvNetTimeLapse] = t.nn.ModuleDict()
        
        # check for folders
        try:
            dirs = os.listdir(warp_path)
            self.models = t.nn.ModuleDict({
                dir: ConvNetTimeLapse(os.path.join(warp_path, dir))
                for dir in dirs
            })
        except FileExistsError:
            raise ValueError(f"Warp is malformed; ensure the provided folder ({warp_path}) contains valid models.")
    
    def pca_over_time(
        self,
        x: t.Tensor,
        component_buckets: list[int] = [1, 5, 9],
        device: str = "cpu"
    ) -> dict[int, pd.DataFrame]:
        pca_data = {
            k: pd.DataFrame() for k in component_buckets
        }

        self.models = self.models.to(device)
        self.models.eval()

        with t.no_grad():
            for model_name, model in self.models.items():
                # grab dict of [step, pcas]
                step_pcas = model.pca_logits(x, components=max(component_buckets))

                # clean so we have a map of step: component-variance buckets
                # for each model also convert steps to ints now
                step_variance_buckets = {
                    k: {} for k in component_buckets
                }
                for (step, bucket) in product(step_pcas, component_buckets):
                    # print(step_pcas[step])
                    step_variance_buckets[bucket][int(step)] = t.sum(step_pcas[step][:bucket]).item()

                # go through each component-variance bucket, append model name,
                # and add to larger dataframes
                for bucket in component_buckets:
                    step_variance_buckets[bucket]["model"] = model_name
                    pca_data[bucket] = pd.concat([pca_data[bucket], pd.DataFrame(step_variance_buckets[bucket], index=[0])])

            # reorder columns
            for k in pca_data:
                ordered_cols = sorted(pca_data[k].columns, key=lambda x: -1 if isinstance(x, str) else x)
                pca_data[k] = pd.DataFrame(pca_data[k][ordered_cols])
        
        # load to cpu when done
        self.models = self.models.to('cpu')
                
        return pca_data
