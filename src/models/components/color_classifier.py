import torch
from torch import nn

from . import backbones as B
from .utils import *

NAMED_MODEL = get_named_function(B)


class ColorClassifier(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        backbone: str,
        output_size: int = 10,
    ) -> None:
        """Initialize a `ColorClassifier` module.

        :param input_size: The number of input features.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()
        self.backbone = NAMED_MODEL[backbone](pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.dim, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        x = self.backbone(x)
        return x


if __name__ == "__main__":
    _ = SimpleDenseNet()
