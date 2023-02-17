import os
from glob import glob
from typing import Callable, List, Union

import cv2
import numpy as np

from utils import nii_to_np


def nii_to_png(
    filename: Union[str, "os.PathLike"],
    new_filename: Union[str, "os.PathLike"],
    preprocessing: Callable[["np.ndarray"], List[np.ndarray]],
) -> None:
    r"""Converts nii file to group of png files"""
    nii_array = nii_to_np(filename)
    to_png_list = preprocessing(nii_array)
    os.makedirs(os.path.dirname(new_filename), exist_ok=True)

    for i, item in enumerate(to_png_list):
        cv2.imwrite(f"{new_filename}_{i:03}.png", item)


def nii_dir_to_png(
    root_nii: Union[str, "os.PathLike"],
    root_png: Union[str, "os.PathLike"],
    preprocessing: Callable[["np.ndarray"], List[np.ndarray]],
) -> None:
    r"""Converts directory with nii files to group of png files. Subdirs and filenames are preserved, with slice number added to filename.

    Parameters
    ----------
    root_nii : str or PathLike
        Specification of root directory with nii files
    root_png : str or PathLike
        Specification of root directory for pngs
    preprocessing : Callable
        Function to procees 3d nii array, returning list of 2d arrays to be written.
    """
    olddir = os.getcwd()
    os.chdir(root_nii)
    files = glob("**/*.nii", recursive=True)
    os.chdir(olddir)

    for file in files:
        nii_to_png(
            os.path.join(root_nii, file),
            os.path.join(root_png, os.path.splitext(file)[0]),
            preprocessing,
        )


if __name__ == "__main__":
    shape = (256, 256)

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

    nii_dir_to_png(
        "./data/covid19-ct-scans/ct_scans", "./data/pngs/ct_scans", scans_to_pngs
    )
    for mask_dir in ["infection_mask", "lung_mask", "lung_and_infection_mask"]:
        nii_dir_to_png(
            os.path.join("./data/covid19-ct-scans", mask_dir),
            os.path.join("./data/pngs", mask_dir),
            masks_to_png,
        )
