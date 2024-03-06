from .base import PIPELINE
from torch import Tensor
from pandas import DataFrame, Series
import torch as t
import einops


def svd_pca(x: t.Tensor) -> t.Tensor:
    x -= x.mean(0)
    s = t.linalg.svdvals(x)
    return t.square(s) / (x.size(0) - 1)


@PIPELINE.register_filter()
@t.no_grad()
def class_covariance_eigvals(logits: Tensor, results: DataFrame) -> DataFrame:
    for c in range(10):
        sliced_logits = logits[:, :, c].T

        e_vals: t.Tensor = svd_pca(sliced_logits)

        results[f'eig_vals_{c}'] = Series(e_vals.cpu().numpy())

    return results


@PIPELINE.register_filter()
@t.no_grad()
def covariance_eigvals(logits: Tensor, results: DataFrame) -> DataFrame:
    """ Calculates the eigvals of the centered covariance matrix of logits.

    Args:
        logits (Tensor): _description_
        results (DataFrame): _description_

    Returns:
        DataFrame: _description_
    """
    model_stacked_logits = einops.rearrange(logits, "i m l -> m (i l)")

    e_vals: t.Tensor = svd_pca(model_stacked_logits)

    results['eig_vals'] = Series(e_vals.cpu().numpy())

    return results
