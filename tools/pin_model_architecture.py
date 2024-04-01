"""
Script for initializing and saving models of different hidden_state sizes.

For now, we are testing whether the third hidden state/depth is possibly limiting the variance
of an untrained model. 
"""
from argparse import ArgumentParser
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--num', '-n', type=int, default=4096)
    parser.add_argument('--out', '-o', type=str, default='./width_testing_models')
    
    args = parser.parse_args()
    
    """
    Remember that the chosen, regular config was:
    depths=[2, 2, 6, 2] 
    hidden_sizes=[48, 96, 192, 384]
    
    We will augment the third value in `hidden_sizes`,
    increasing it by 24, starting from (192 + 24 =) 216 until 384
    """
    