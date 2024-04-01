"""
Script for initializing and saving models of different hidden_state sizes.

For now, we are testing whether the third hidden state/depth is possibly limiting the variance
of an untrained model.
"""
from argparse import ArgumentParser
from ensemble_modelling import Ensemble
from pathlib import Path
import torch
from copy import deepcopy

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--num', '-n', type=int, default=4096)
    parser.add_argument('--out', '-o', type=str, default='./width_testing_models')
    parser.add_argument('--batch', '-b', type=int, default=64)

    args = parser.parse_args()
    
    """
    Remember that the chosen, regular config was:
    depths=[2, 2, 6, 2]
    hidden_sizes=[48, 96, 192, 384]
    
    We will augment the third value in `hidden_sizes`,
    increasing it by 24, starting from (192 + 24 =) 216 until 384
    """
    for set_no, size in enumerate(range(216, 384 + 1, 24)):
        print(f"Generating hidden layer {size}")
        
        remaining = args.num
        while remaining > 0:
            batch = min(remaining, args.batch)
            remaining -= batch
        
            # 1. Initialize Ensemble Models
            
            ensemble = Ensemble(
                (size + remaining) * 10_000,  # to avoid seed collisions
                batch,
                model_depths=[2, 2, 6, 2],  # TODO add option for changing
                model_hidden_sizes=[48, 96, size, 384]
            )
            
            # 2. save models
            # Note that I'm mimicking the warp layout to remain compatible with
            # the layout expected in `inference.py`
            
            for i, model in enumerate(ensemble.models):  # TODO seperate out this code in ensemble
                model = deepcopy(model)

                out_path = Path(args.out, f"hidden_layer_size_{size}",
                                f"warp_0/model_{i + remaining}/step=1.pt")
                out_path.parent.mkdir(parents=True, exist_ok=True)

                torch.save(model.state_dict(), out_path)
