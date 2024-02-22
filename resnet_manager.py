import os
import re

import torch as t
import torchvision
from transformers import (ConvNextV2Config,
                          ConvNextV2ForImageClassification, AutoConfig)

import pytorch_lightning as pl


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
        self.model(x)

class ConvNetTimeLapse(t.nn.Module):
    _checkpoint_name_fmt = "step=(\\d*).pt"
    
    def __init__(self, model_folder_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.models = t.nn.ModuleDict()
        
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
            self.models[step_no] = ConvNet(checkpoint)
        
        print(f"Successfully loaded {len(self.models)} checkpoints from {model_folder_path}")
            
    
    def forward(self, x):
        pass

if __name__ == "__main__":
    time_lapse = ConvNetTimeLapse('../warp_0/model_0')
    # (t.ones(50, 32, 32, 3))
    # print(time_lapse.models['1'](t.ones(32, 32, 3)))

