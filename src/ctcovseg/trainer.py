import os
from glob import glob
from typing import Union

import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from .config import Config
from .dataset import SegmentationDataset
from .train_model import SegmentationModel
from .utils import seed_all, seed_worker


def train(config: Config, data_root: Union[str, "os.PathLike"]):
    r"""Train configuration

    Parameters
    ----------
    config : Config
    data_root : str or os.PathLike
        Path to directory with ct_scans and lung_and_infection_mask subdirectories with png files for datasets.

    Returns
    -------
    metrics : dict
        Dictionary of metrics name and value.
    """
    image_files = glob(os.path.join(data_root, "ct_scans", "*.png"))
    mask_files = glob(os.path.join(data_root, "lung_and_infection_mask", "*.png"))

    df = pd.DataFrame(dict(image=image_files, mask=mask_files))
    df = df.iloc[:100].copy()

    df["patient"] = df["image"].str.rsplit("_", n=1).str[0]

    splitter = GroupShuffleSplit(
        n_splits=1, train_size=config.train_size, random_state=config.seed
    )
    train_idx, valid_idx = next(splitter.split(df, groups=df["patient"]))

    train_df = df.loc[train_idx]
    valid_df = df.loc[valid_idx]

    seed_all(42)

    train_ds = SegmentationDataset(
        train_df, classes=config.classes, transformation=config.train_aug
    )
    valid_ds = SegmentationDataset(
        valid_df, classes=config.classes, transformation=config.valid_aug
    )

    train_dl = DataLoader(
        train_ds,
        config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    valid_dl = DataLoader(
        valid_ds, config.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    model = SegmentationModel(config, len(train_dl) * config.epochs)

    trainer_kwargs = dict(
        max_epochs=config.epochs,
        accelerator="auto",
        num_sanity_val_steps=0,
        enable_model_summary=False,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                dirpath="models",
                filename=f"{config.experiment_name}",
                auto_insert_metric_name=False,
                save_weights_only=True,
            ),
        ],
    )
    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(model, train_dl, valid_dl)

    metrics = trainer.test(model, valid_dl, ckpt_path="best", verbose=False)[0]

    return metrics
