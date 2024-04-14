import torch as t
import einops


@t.no_grad()
def logit_pca(logits: t.Tensor) -> t.Tensor:
    if logits.dim() != 3:
        raise "Passed Tensor is not dim 3"
    
    stacked_logits = einops.rearrange(logits, 'i m l -> m (i l)')
    return svd_pca(stacked_logits)


@t.no_grad()
def logit_class_pca(logits: t.Tensor) -> dict[int, t.Tensor]:
    if logits.dim() != 3:
        raise "Passed Tensor is not dim 3"
    
    class_variances = {}
    for cls_no in range(10):
        class_logits = logits[:, :, cls_no].T
        
        class_variances[cls_no] = svd_pca(class_logits)
    
    return class_variances
    

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
