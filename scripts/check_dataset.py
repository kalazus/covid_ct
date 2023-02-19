from glob import glob

import pandas as pd
from torch.utils.data import DataLoader

from ctcovseg import SegmentationDataset

if __name__ == "__main__":
    images = glob("input/pngs/ct_scans/*.png")
    masks = glob("input/pngs/lung_and_infection_mask/*.png")
    df = pd.DataFrame(dict(image=images, mask=masks))

    dataset = SegmentationDataset(df, [0, 1, 2])
    item = dataset[0]
    for el in item:
        print(el.shape, el.dtype)

    dl = DataLoader(dataset, 8, False)
    item = next(iter(dl))
    for el in item:
        print(el.shape, el.dtype)
