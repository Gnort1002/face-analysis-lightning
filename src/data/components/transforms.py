import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
import numpy as np


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def get_img_trans(phase,
                  image_size=256,
                  crop_size=224,
                  mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225),
                  backbone_name="resnet50"):
    normalize = A.Normalize(mean=mean, std=std)
    if backbone_name == "inception_resnet_v1":
        if phase == "train":
            return A.Compose(
                [
                    # np.float32,
                    A.Resize(160, 160),
                    A.HorizontalFlip(),
                    ToTensor(),
                ]
            )
        elif phase in ["test", "val"]:
            return A.Compose(
                [
                    A.Resize(160, 160),
                    ToTensor(),
                ]
            )
        elif phase in ["predict"]:
            return A.Compose(
                [
                    A.Resize(160, 160),
                    ToTensor(),
                ]
            )
        else:
            raise KeyError
    elif backbone_name in ["resnet50", "vgg", "resnet18"]:
        if phase == "train":
            return A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.RandomCrop(crop_size, crop_size),
                    A.HorizontalFlip(),
                    normalize,
                    ToTensor(),
                ]
            )
        elif phase in ["test", "val"]:
            return A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.CenterCrop(crop_size, crop_size),
                    normalize,
                    ToTensor(),
                ]
            )
        elif phase in ["predict"]:
            return A.Compose(
                [
                    A.Resize(image_size, image_size),
                    normalize,
                    ToTensor(),
                ]
            )
        else:
            raise KeyError
    else:
        raise KeyError