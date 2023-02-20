import os
from glob import glob
from typing import Union

import pytorch_lightning as pl

from .config import Config
from .datamodule import PngSegmentationDataModule
from .train_model import SegmentationModel
from .utils import seed_all


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

    images_root = os.path.join(data_root, "ct_scans")
    masks_root = os.path.join(data_root, "lung_and_infection_mask")
    data_module = PngSegmentationDataModule(config, images_root, masks_root)

    seed_all(42)
    data_module.setup()
    train_dl = data_module.train_dataloader()
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

    trainer.fit(model, datamodule=data_module)

    metrics = trainer.test(
        model, datamodule=data_module, ckpt_path="best", verbose=False
    )[0]

    return metrics
