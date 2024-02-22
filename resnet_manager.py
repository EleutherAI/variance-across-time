import torch as t
import torchvision
from torchsummary import summary
from transformers import ConvNextV2Config, ConvNextV2ForImageClassification

import pytorch_lightning as pl
from transformers.models.vision_encoder_decoder.configuration_vision_encoder_decoder import OrderedDict

# a singular instance of a RestNet
class ConvNet():
    def __init__(self, model_path: str):
        self.model = t.load(model_path)


if __name__ == "__main__":
    net = ConvNet("../test_model/step=1.pt")
