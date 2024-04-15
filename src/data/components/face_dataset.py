import os.path as osp
import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset
from .transforms import get_img_trans, fixed_image_standardization


class FaceDataset(Dataset):
    """Fashion Color dataset."""

    def __init__(self,
                 root_dir,
                 image_list,
                 mean,
                 std,
                 image_size,
                 crop_size,
                 backbone_name,
                 mode="train",
                 transform=None,
                 predict_mode=False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(image_list, delimiter=" ", header=None)
        self.data = np.array(self.data)
        self.root_dir = root_dir
        self.backbone_name = backbone_name
        if transform:
            self.transform = get_img_trans(mode,
                                           image_size=image_size,
                                           crop_size=crop_size,
                                           mean=mean,
                                           std=std,
                                           backbone_name=backbone_name)
        self.age_classes = 6
        self.predict_mode = predict_mode
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.data[idx][0]
        img_path = osp.join(self.root_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        label = self.data[idx][1:]
        if self.transform:
            image = self.transform(image=image)["image"]
            if self.backbone_name == "inception_resnet_v1":
                image = fixed_image_standardization(image)
        if self.predict_mode:
            return image, img_name
        target = {"race": label[0],
                  "gender": label[1],
                  "age": label[2],
                  "skintone": label[3],
                  "emotion": label[4],
                  "masked": label[5]}
        label = self._transform_ages_to_one_hot_ordinal(target,
                                                        self.age_classes
                                                        )
        return image, label

    def _transform_ages_to_one_hot_ordinal(self, target, age_classes):
        age = target["age"]
        new_age = np.zeros(shape=age_classes)
        new_age[:age] = 1
        target["age"] = new_age
        return target
