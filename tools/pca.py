import torch as t


@t.no_grad()
def svd_pca(input: t.Tensor) -> t.Tensor:
    """Takes a 2-dim tensor, returns variances of the principal components (PCs).
    PCs are calculated by squaring the singular values of the tensor,
    then dividing by the number of rows, minus 1. 

    Args:
        x (t.Tensor): The input Tensor

    Returns:
        t.Tensor: a 1d tensor of the principal components in descending order of magnitude
    """
    input -= input.mean(0)  # center
    s = t.linalg.svdvals(input)  # get singular values
    return t.square(s) / (input.size(0) - 1)  # return PCs
