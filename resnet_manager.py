import os
import re
from itertools import product

import torch as t
# import torchvision
from transformers import ConvNextV2Config, ConvNextV2ForImageClassification
from sklearn.decomposition import PCA
import pandas as pd

device = 'cuda' if t.cuda.is_available() else 'cpu'

# import pytorch_lightning as pl

# TODO add doc comments!

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

        print(f"Successfully loaded {len(self.checkpoints)} checkpoints from {model_folder_path}")
    
    def forward(self, x) -> dict[str, t.Tensor]:
        return {
            step: self.checkpoints[step](x) for step in self.checkpoints
        }
    
    def pca_logits(self, x: t.Tensor, components: int = 10) -> dict[str, t.Tensor]:
        """
        Using x images (batch, channels, height, width), computes logits for each image.
        Takes logits and creates vectors of the probabilities for each class (10, batch).
        Run PCA on the  10 x batch sized matrix, returning an array (10, ) of the variance that
        the top-10 components contribute.
        
        Args:
            x (Tensor): images
            components (int): number of components to report variance ratios for. Will be overriden if x.shape[0] < components. Defaults to 10

        Returns:
            dict[str, t.Tensor]: dictionary of step: Principal component variance
        """
        pcas: dict[str, t.Tensor] = {}
        pca = PCA(n_components=min([components, x.shape[0]]))
        
        for step, checkpoint in self.checkpoints.items():
            logits = checkpoint(x).logits
            pca_out = pca.fit(logits)
            pcas[step] = pca_out.explained_variance_ratio_
        
        return pcas

    
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
                dir: ConvNetTimeLapse(os.path.join(warp_path, dir)) for dir in dirs
            })
        except FileExistsError:
            raise ValueError(f"Warp is malformed; ensure the provided folder ({warp_path}) contains valid models.")
    
    def pca_over_time(
        self,
        x: t.Tensor,
        component_buckets: list[int] = [1, 5, 9]
    ) -> dict[int, pd.DataFrame]:
        pca_data = {
            k: pd.DataFrame() for k in component_buckets
        }
        with t.no_grad():
            for model_name, model in warp.models.items():
                # grab dict of [step, pcas]
                step_pcas = model.pca_logits(x, components=max(component_buckets))
                
                # clean so we just have component variance buckets for each step of the model
                # also convert steps to ints now
                step_variance_buckets = {
                    k: {} for k in component_buckets
                }
                for (step, bucket) in product(step_pcas, component_buckets):
                    step_variance_buckets[bucket][int(step)] = sum(step_pcas[step][:bucket])
                
                for bucket in component_buckets:
                    step_variance_buckets[bucket]["model"] = model_name
                    pca_data[bucket] = pd.concat([pca_data[bucket], pd.DataFrame(step_variance_buckets[bucket], index=[0])])
            
            for k in pca_data:
                ordered_cols = sorted(pca_data[k].columns, key=lambda x: -1 if isinstance(x, str) else x)
                pca_data[k] = pd.DataFrame(pca_data[k][ordered_cols])
                
        return pca_data

warp = ConvWarp('../warp_0')

rando = t.rand((32, 3, 32, 32), requires_grad=False)
for bucket, df in warp.pca_over_time(rando).items():
    print(df.head())
