import os
from typing import Union

import nibabel as nib
import numpy as np


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
