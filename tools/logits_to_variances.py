""" TODO Delete and replace
Script for taking logits of a dataset over all model checkpoints and computing 
the variances for all logits, and logits for each individual classifier label. 
"""

from argparse import ArgumentParser
import torch as t
import einops
from pandas import DataFrame, Series
from pca import svd_pca


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu-id', '-g', type=int, default=6)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--logits', '-l', type=str, required=True)
    args = parser.parse_args()
    
    metrics = DataFrame()
    device = f"cuda:{args.gpu_id}"

    assert args.out.endswith('.csv')
    logits = t.load(args.logits, map_location=device)
    stacked_logits = einops.rearrange(logits, 'i m l -> m (i l)')

    for cls_no in range(10):
        class_logits = logits[:, :, cls_no].T
        metrics[f'class_{cls_no}'] = Series(svd_pca(class_logits).cpu().numpy())

    metrics['all'] = Series(svd_pca(stacked_logits).cpu().numpy())

    # save to csv
    metrics = metrics.to_csv(args.out)