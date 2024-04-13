import torch
from numpy import ndarray
from torch.utils.data import Dataset


class CIFAR10_Dataset(Dataset):
    """CIFAR 10 class for Out of distribution dataset

    Args:
        data (ndarray): Array of shape (data_size, 32, 32, 3), image pixels
        labels (ndarray): Array of shape (data_size), target labels
        corruption (str): Name of the corruption
    """
    def __init__(self, data: ndarray, labels: ndarray, corruption: str):
        self.targets = labels
        self.data = torch.tensor(data)
        self.corruption = corruption
        
    def __getitem__(self, idx):
        # put channels first, as input
        return (self.data[idx].permute(-1, 0, 1)/255, self.targets[idx])

    def __len__(self):
        return len(self.targets)


def get_model_paths(
        models_path: str,
        warps: int = 128,
        models_per_warp: int = 32,
        step: int = 1) -> List[str]:
    """Gets all paths to pytorch checkpoint files
    """
    all_model_paths = []
    for warp, idx in product(range(warps), range(models_per_warp)):
        
        model_path = os.path.join(
            models_path, f'warp_{warp}', f'model_{idx}', f'step={step}.pt')
        
        if os.path.exists(model_path):
            all_model_paths.append(model_path)

    return all_model_paths