import torch as t
from transformers import ConvNextV2Config, ConvNextV2ForImageClassification
from torch.utils.data import DataLoader

class ConvNet(t.nn.Module):
    """A singular RestNet, loaded from a checkpoint
    """
    
    # see train.py; Pytorch needs same config for weight loading
    _cfg = ConvNextV2Config(
        image_size=32,
        depths=[2, 2, 6, 2],
        hidden_sizes=[48, 96, 192, 384],
        num_labels=10,
        patch_size=2,
    )

    def __init__(self, model_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = ConvNextV2ForImageClassification.from_pretrained(
            model_path,
            config=self._cfg,
            local_files_only=True
        )

    def forward(self, x: t.Tensor):
        return self.model(x)

    def get_logit_variance_ratios(self, x: t.Tensor, batch_size: int | None = None) -> t.Tensor:
        self.eval()
        with t.no_grad():
            logits: t.Tensor = None
            
            if batch_size is not None:
                dl = DataLoader(x, batch_size=batch_size, shuffle=False)
                for batch in dl:
                    batch_out = self(batch).logits
                    logits = batch_out if logits is None else t.concat([logits, batch_out])
            else:
                logits = self(x).logits
            
            centered = logits - logits.mean(dim=0)
            

            e_vals, _ = t.linalg.eigh(t.matmul(centered.T, centered))
            e_vals_total = t.sum(e_vals)

            return t.flip(e_vals / e_vals_total, dims=(0, ))