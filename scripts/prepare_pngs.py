import os

import cv2
import numpy as np

from ctcovseg import nii_dir_to_png


def scans_to_pngs(data: np.ndarray):
    image: np.ndarray
    images = []

    for image in (data[..., i] for i in range(data.shape[-1])):
        image = cv2.resize(image, shape)
        image -= image.min()
        image /= image.max()
        image *= 255
        image = image.astype(np.uint8)
        images.append(image)

    return images


def masks_to_png(data: np.ndarray):
    image: np.ndarray
    images = []

    for image in (data[..., i] for i in range(data.shape[-1])):
        image = cv2.resize(image, shape, interpolation=cv2.INTER_NEAREST)
        image = image.astype(np.uint8)
        images.append(image)

    return images


if __name__ == "__main__":
    shape = (256, 256)

    nii_dir_to_png(
        "./input/covid19-ct-scans/ct_scans", "./input/pngs/ct_scans", scans_to_pngs
    )
    for mask_dir in ["lung_and_infection_mask"]: # "infection_mask", "lung_mask", 
        nii_dir_to_png(
            os.path.join("./input/covid19-ct-scans", mask_dir),
            os.path.join("./input/pngs", mask_dir),
            masks_to_png,
        )
