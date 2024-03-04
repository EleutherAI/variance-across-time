from .base import PIPELINE

_has_registered_all_metrics = False

if not _has_registered_all_metrics:
    from .jenson_shannon_divergence import calculate_jenson_shannon_divergence
    _has_registered_all_metrics = True