import os
import random
from typing import Union

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
