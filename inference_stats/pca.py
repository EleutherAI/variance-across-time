from .base import PIPELINE
from torch import Tensor
from pandas import DataFrame, Series
import torch as t
import einops


@t.no_grad()
def svd_pca(x: t.Tensor) -> t.Tensor:
    x -= x.mean(0)
    s = t.linalg.svdvals(x)
    return t.square(s) / (x.size(0) - 1)


@PIPELINE.register_filter()
@t.no_grad()
def class_p_component_variance(logits: Tensor, results: DataFrame) -> DataFrame:
    for c in range(logits.size(2)):
        sliced_logits = logits[:, :, c].T

        variances: t.Tensor = svd_pca(sliced_logits)

        results[f'eig_vals_{c}'] = Series(variances.cpu().numpy())

    return results


@PIPELINE.register_filter()
@t.no_grad()
def p_component_variance(logits: Tensor, results: DataFrame) -> DataFrame:
    """ Calculates the eigvals of the centered covariance matrix of logits.

    Args:
        logits (Tensor): _description_
        results (DataFrame): _description_

    Returns:
        DataFrame: _description_
    """
    model_stacked_logits = einops.rearrange(logits, "i m l -> m (i l)")

    variances: t.Tensor = svd_pca(model_stacked_logits)

    results['eig_vals'] = Series(variances.cpu().numpy())

    return results
