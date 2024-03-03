from .base import PIPELINE
from torch import Tensor
from pandas import DataFrame, Series
import torch as t
    

@PIPELINE.register_filter()
def covariance_eigvals(logits: Tensor, results: DataFrame) -> DataFrame:
    """ Calculates the eigvals of the centered covariance matrix of logits.
    
    This is kind of scuffed since results is one image per row. But, the column eig_vals
    contains the eig_vals of the covar matrix in ascending order.

    Args:
        logits (Tensor): _description_
        results (DataFrame): _description_

    Returns:
        DataFrame: _description_
    """
    with t.no_grad():
        num_images, num_models, d_logits = logits.shape
        
        concated_logits = logits.reshape(num_images, num_models * d_logits)
        
        centered = concated_logits - concated_logits.mean(dim=0)
        
        e_vals = t.linalg.eigvalsh(t.matmul(centered.T, centered))

        results['eig_vals'] = Series(e_vals)
    
    return results