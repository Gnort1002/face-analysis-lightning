import torch
from torch import nn

import torchvision.models as models
from .age_classifier import AgeClassifier
from .gender_classifier import GenderClassifier
from .race_classifier import RaceClassifier
from .skintone_classifier import SkintoneClassifier
from .emotion_classifier import EmotionClassifier
from .masked_classifier import MaskedClassifier
from .backbones import create_vgg_face_dag

class FaceAttrsClassifier(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        backbone: str,
        race_output_size: int = 3,
        gender_output_size: int = 2,
        age_output_size: int = 6,
        skintone_output_size: int = 4,
        emotion_output_size: int = 7,
        masked_output_size: int = 2
    ) -> None:
        """Initialize a `ColorClassifier` module.

        :param input_size: The number of input features.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()
        if backbone == "resnet50":
            self.backbone = models.resnet50(weights="DEFAULT")
        elif backbone == "resnet101":
            self.backbone = models.resnet101(weights="DEFAULT")
        elif backbone == "vgg":
            self.backbone = create_vgg_face_dag()
        else:
            raise KeyError
        num_filters = self.backbone.fc.in_features if "resnet" in backbone else 2622

        # Freeze the backbone
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        # for param in self.backbone.fc.parameters():
        #     param.requires_grad = True
        layers = list(self.backbone.children())[:-1]
        if "resnet" in backbone:
            self.feature_extractor = nn.Sequential(*layers)
        else:
            self.feature_extractor = self.backbone
        
        # use the pretrained model
        self.race_classifier = RaceClassifier(
            num_of_features=num_filters,
            output_size=race_output_size
        )
        self.gender_classifier = GenderClassifier(
            num_of_features=num_filters,
            output_size=gender_output_size
        )
        self.age_classifier = AgeClassifier(
            num_of_features=num_filters,
            output_size=age_output_size
        )
        self.skintone_classifier = SkintoneClassifier(
            num_of_features=num_filters,
            output_size=skintone_output_size
        )
        self.emotion_classifier = EmotionClassifier(
            num_of_features=num_filters,
            output_size=emotion_output_size
        )
        self.masked_classifier = MaskedClassifier(
            num_of_features=num_filters,
            output_size=masked_output_size
        )
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        # self.feature_extractor.eval()
        # with torch.no_grad():
        representations = self.feature_extractor(x)
        # x = self.classifier(representations)
        race_pred = self.race_classifier(representations)
        gender_pred = self.gender_classifier(representations)
        age_pred = self.age_classifier(representations)
        skintone_pred = self.skintone_classifier(representations)
        emotion_pred = self.emotion_classifier(representations)
        maksed_pred = self.masked_classifier(representations)
        
        race_pred = self.softmax(race_pred)
        gender_pred = self.softmax(gender_pred)
        age_pred = self.sigmoid(age_pred)
        skintone_pred = self.softmax(skintone_pred)
        emotion_pred = self.softmax(emotion_pred)
        maksed_pred = self.softmax(maksed_pred)
        
        pred = {"race": race_pred,
                "gender": gender_pred,
                "age": age_pred,
                "skintone": skintone_pred,
                "emotion": emotion_pred,
                "masked": maksed_pred}
        return pred


if __name__ == "__main__":
    _ = FaceAttrsClassifier()
