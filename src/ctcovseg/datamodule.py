import os
from glob import glob

import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from .config import Config
from .dataset import SegmentationDataset
from .utils import seed_worker


class PngSegmentationDataModule(pl.LightningDataModule):
    r"""Datamodule for segmentation task with files where pngs should grouped by name before last `_`.

    Parameters
    ----------
    config : Config
    image_data_root : str or PathLike
        Root dir for image pngs.
    mask_data_root : str or PathLike
        Root dir for mask pngs.
    """

    def __init__(
        self,
        config: Config,
        image_data_root: str,
        mask_data_root: str,
    ) -> None:
        super().__init__()

        self.config = config
        self.image_data_root = image_data_root
        self.mask_data_root = mask_data_root

    def setup(self, stage: str = None) -> None:
        config = self.config

        image_files = glob(os.path.join(self.image_data_root, "*.png"))
        mask_files = glob(os.path.join(self.mask_data_root, "*.png"))

        df = pd.DataFrame(dict(image=image_files, mask=mask_files))
        df = df.iloc[:100].copy()

        df["patient"] = df["image"].str.rsplit("_", n=1).str[0]

        splitter = GroupShuffleSplit(
            n_splits=1, train_size=config.train_size, random_state=config.seed
        )
        train_idx, valid_idx = next(splitter.split(df, groups=df["patient"]))
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]

        self.train_ds = SegmentationDataset(
            train_df, classes=config.classes, transformation=config.train_aug
        )
        self.valid_ds = SegmentationDataset(
            valid_df, classes=config.classes, transformation=config.valid_aug
        )

    def _dataloader(self, dataset, train=False):
        return DataLoader(
            dataset,
            self.config.batch_size,
            shuffle=train,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker if train else None,
        )

    def train_dataloader(self):
        return self._dataloader(self.train_ds, True)

    def val_dataloader(self):
        return self._dataloader(self.valid_ds)

    def test_dataloader(self):
        return self._dataloader(self.valid_ds)

    def predict_dataloader(self):
        return self._dataloader(self.valid_ds)
