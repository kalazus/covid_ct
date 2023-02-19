from typing import List

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    r"""Basic dataset for segmentation task

    Args
    ----------
    df : pd.DataFrame
        Dataframe with 'image' and 'mask' columns
    classes : List[int]
        List of classes in segmentation image.
    transformation : albumentations.BasicTransform
        Albumentation transformation for 'image' and 'mask'.
    """

    def __init__(
        self,
        df: "pd.DataFrame",
        classes: List[int],
        transformation: "A.BasicTransform" = None,
    ) -> None:
        super().__init__()

        self.images = df["image"].values
        self.masks = df["mask"].values
        self.classes = classes

        self.transformation = transformation

    def __len__(self):
        return len(self.masks)

    def read_image(self, file: str):
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image, 2).astype(np.float32)
        return image

    def read_mask(self, file: str):
        mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        class_masks = [mask == item for item in self.classes]
        mask = np.stack(class_masks, 2, dtype=np.float32)
        return mask

    def __getitem__(self, index):
        image: np.ndarray = self.read_image(self.images[index])
        mask: np.ndarray = self.read_mask(self.masks[index])

        if self.transformation is not None:
            items = self.transformation(image=image, mask=mask)
            image, mask = items["image"], items["mask"]

        image = np.moveaxis(image, 2, 0)
        mask = np.moveaxis(mask, 2, 0)

        return image, mask
