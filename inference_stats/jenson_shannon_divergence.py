from .base import PIPELINE
from torch import Tensor
from pandas import DataFrame
import numpy as np
import torch

def jenson_shannon_divergence(lps):
    """Calculates Jenson Shannon Divergance (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)
    
    Args:
        lps (torch.Tensor): Log probabilities of shape (*, `num_models`, `num_classes`)
    """
    n = np.log(lps.shape[-2])
    mixture_lps = lps.logsumexp(dim=-2, keepdim=True).sub(n)
    return torch.sum(lps.exp() * (lps - mixture_lps), dim=-1).mean(dim=-1)

@PIPELINE.register_filter()
def calculate_jenson_shannon_divergence(logits: Tensor, results: DataFrame) -> DataFrame:
    """Calculates and stores jenson shannon divergences in a Dataframe
    """
    results['jenson_shannon_divergence'] = jenson_shannon_divergence(logits).cpu().numpy()
    return results