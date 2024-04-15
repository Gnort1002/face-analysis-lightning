import torch
from torch import nn


class SkintoneClassifier(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        num_of_features: int = 2048,
        output_size: int = 4,
    ) -> None:
        """Initialize a `ColorClassifier` module.

        :param input_size: The number of input features.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()
        self.output_size = output_size
        if num_of_features > 1024:
            self.classifier = self.fc1 = nn.Sequential(
                nn.Linear(num_of_features, 1024),
                nn.ReLU(inplace=True),

                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),

                nn.Linear(512, 256),
                nn.ReLU(inplace=True),

                nn.Linear(256, 128),
                nn.ReLU(inplace=True),

                nn.Linear(128, output_size))
        elif num_of_features > 512:
            self.classifier = self.fc1 = nn.Sequential(
                nn.Linear(num_of_features, 512),
                nn.ReLU(inplace=True),

                nn.Linear(512, 256),
                nn.ReLU(inplace=True),

                nn.Linear(256, 128),
                nn.ReLU(inplace=True),

                nn.Linear(128, output_size))
        elif num_of_features > 256:
            self.classifier = self.fc1 = nn.Sequential(
                nn.Linear(num_of_features, 256),
                nn.ReLU(inplace=True),

                nn.Linear(256, 128),
                nn.ReLU(inplace=True),

                nn.Linear(128, output_size))
        else:
            self.classifier = self.fc1 = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),

                nn.Linear(128, output_size))
        self.softmax = nn.Softmax()

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        # use the pretrained model
        x = self.classifier(representation)
        # x = self.softmax(x)
        return x


if __name__ == "__main__":
    _ = SkintoneClassifier()
