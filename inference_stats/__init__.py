from .base import PIPELINE

_has_registered_all_metrics = False

if not _has_registered_all_metrics:
    from .jenson_shannon_divergence import calculate_jenson_shannon_divergence
    from .pca import p_component_variance
    from .pca import class_p_component_variance
    _has_registered_all_metrics = True
