import os.path as osp
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset
from .transforms import get_img_trans


class FashionDataset(Dataset):
    """Fashion Color dataset."""

    def __init__(self, root_dir, metadata_file, mode="train", transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data = pd.read_csv(osp.join(root_dir, metadata_file))
        self.data = data[data["phase"] == mode].reset_index(drop=True)
        self.root_dir = root_dir
        if transform:
            self.transform = get_img_trans(mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = osp.join(self.root_dir, "images", self.data.iloc[idx, 0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.data.iloc[idx, 3]

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
