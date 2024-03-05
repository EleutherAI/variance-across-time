from typing import Any, Callable, Dict, List, TypeAlias
InferFunc: TypeAlias = Callable[..., Any]
from pandas import DataFrame
from torch import Tensor

class InferencePipeline:
    def __init__(self):
        """
        Pipeline for applying stats to model logits 
        """
        self.stat_funcs: List[InferFunc] = []

    def register_filter(self) -> InferFunc:
        """
        Decorator function, registering a inference function to the pipeline
        """
        def decorator(metric_func: InferFunc):
            def wrapper(*args, **kwargs):
                return metric_func(*args, **kwargs)

            print("Registring Metric: ", metric_func.__name__)
            self.stat_funcs.append(metric_func)
            return wrapper
        
        return decorator
    
    def transform(self, logits: Tensor, results: DataFrame, device: str | None = None):
        """Saves all stats on logits on the resultant dataframe. 
        
        Saving format is determined by individual functions

        Args:
            logits (torch.Tensor): Model logits of shape (batch_size, num_models, num_classes)
            results (DataFrame): Dataframe to save further statistics onto
        """
        if device is not None:
            logits = logits.to(device)
        
        for stat_func in self.stat_funcs:
            stat_func(logits, results)

PIPELINE = InferencePipeline() 