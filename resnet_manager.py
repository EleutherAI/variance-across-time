import os
import re

import torch as t
# import torchvision
from transformers import ConvNextV2Config, ConvNextV2ForImageClassification

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
    
    def pca_logits(self, x: t.Tensor) -> dict[str, t.Tensor]:
        """
        Using x images (batch, channels, height, width), computes logits for each image.
        Takes logits and creates vectors of the probabilities for each class (10, batch).
        Run PCA on the  10 x batch sized matrix, returning an array (10, ) of the variance that
        the top-10 components contribute.
        
        Args:
            x (Tensor): images

        Returns:
            dict[str, t.Tensor]: dictionary of step: Principal component variance
        """
        pass
    
    def jsd_logits(self, x: t.Tensor) -> dict[str, t.Tensor]:
        pass


class ConvWarp:
    def __init__(self, warp_path: str) -> None:
        self.warp_path = warp_path
        self.models = t.nn.ModuleDict()
        
        # check for folders
        try:
            dirs = os.listdir(warp_path)
            for dir in dirs:
                path = os.path.join(warp_path, dir)
                self.models[dir] = ConvNetTimeLapse(path)
        except FileExistsError:
            raise ValueError(f"Warp is malformed; ensure the provided folder ({warp_path}) contains valid models.")
        

if __name__ == "__main__":
    # time_lapse = ConvNetTimeLapse('../warp_0/model_0')

    # rando = t.rand((5, 3, 32, 32), requires_grad=False)
    # with t.no_grad():
        # print(time_lapse(rando))
    warp = ConvWarp('../warp_0')

