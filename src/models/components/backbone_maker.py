import torch
from torch import nn

from timm import create_model
from .backbones import InceptionResnetV1, create_vgg_face_dag


class BackboneMaker(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        name: str,
        pretrained: bool = False,
        checkpoint_path: str = None,
    ) -> None:
        """Initialize a `backbone` module.

        :name: The name of the backbone to create by timm.
        :pretrained: Whether to use pretrained weights.
        """
        super().__init__()
        self.name = name
        self.pretrained = pretrained
        self.checkpoint_path = checkpoint_path
        # use the pretrained model
        if self.name == "inception_resnet_v1":
            self.backbone = InceptionResnetV1(
                pretrained=self.pretrained,
                checkpoint_path=self.checkpoint_path)
        elif self.name == "vgg":
            self.backbone = create_vgg_face_dag(
                weights_path=self.checkpoint_path
            )
        else:
            self.backbone = create_model(
                self.name,
                pretrained=self.pretrained,
                checkpoint_path=self.checkpoint_path,
                num_classes=10,
                global_pool="avg",
            )
            self.backbone.reset_classifier(0)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        
        representations = self.backbone(x)
        return representations


if __name__ == "__main__":
    _ = BackboneMaker()
