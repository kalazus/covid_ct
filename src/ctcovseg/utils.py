import os
import random
from typing import List, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch


def nii_to_np(filepath: Union[str, "os.PathLike"]) -> "np.ndarray":
    r"""Load file given flename and returns its raw data

    Parameters
    ----------
    filename : str or PathLike
        Specification of file to load

    Returns
    -------
    img : ndarray
        Data floating point data array of loaded file
    """
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.rot90(array)  # for better visualization
    return array


def seed_all(seed: int):
    r"""Seeds randomisation for torch, numpy and randoms

    Parameters
    ----------
    seed : unsigned int
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    r"""Seeds current worker"""
    seed = torch.utils.data.get_worker_info().seed % (2 ^ 32)
    seed_all(seed)


def plot_segmentations(img: "np.ndarray", masks: List["np.ndarray"]):
    r"""Plot image and countour of masks in a row

    Parameters
    ----------
    img : ndarray
        Main 2d image data.
    masks : sequence of ndarray
        Arrays with same shape as img.
    """
    full_len = len(masks) + 1
    fig, axes = plt.subplots(1, full_len, figsize=(10, full_len * 10))
    axes[0].imshow(img, cmap="bone")

    for i, mask in enumerate(masks, 1):
        axes[i].imshow(img, cmap="bone")
        axes[i].contour(mask, alpha=0.5)

    plt.tight_layout()
    plt.show()
