from .base import PIPELINE
from torch import Tensor
from pandas import DataFrame, Series
import einops
import torch as t

# TODO replace with t.cov, depending on memory consumption
def covar(x: t.Tensor) -> t.Tensor:
    centered = (x - x.mean(dim=0))
    return (centered.T @ centered) / (x.shape[0] - 1)

@PIPELINE.register_filter()
def class_covariance_eigvals(logits: Tensor, results: DataFrame) -> DataFrame:
    with t.no_grad():
        
        for c in range(10):
            concated_logits = einops.rearrange(logits[:, :, c::10], "i m l -> m (i l)")
            
            e_vals: t.Tensor = t.linalg.eigvalsh(covar(concated_logits))
            
            results[f'eig_vals_{c}'] = Series(e_vals.cpu().numpy())
    
    return results

@PIPELINE.register_filter()
def covariance_eigvals(logits: Tensor, results: DataFrame) -> DataFrame:
    """ Calculates the eigvals of the centered covariance matrix of logits.

    Args:
        logits (Tensor): _description_
        results (DataFrame): _description_

    Returns:
        DataFrame: _description_
    """
    
    with t.no_grad():
        concated_logits = einops.rearrange(logits, "i m l -> m (i l)")

        e_vals: t.Tensor = t.linalg.eigvalsh(covar(concated_logits))

        results['eig_vals'] = Series(e_vals.cpu().numpy())
    
    return results