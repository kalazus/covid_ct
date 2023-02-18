import pytorch_lightning as pl
import torch
import torchmetrics

from config import Config


class SegmentationModel(pl.LightningModule):
    r"""Pytorch Lightning model for semantic segmentation

    Parameters
    ----------
    config : Config
        Config class with loss, classes, lr arguments
    train_size : int
        Number of steps for scheduler
    """

    def __init__(self, config: Config, train_size: int) -> None:
        super().__init__()

        self.config = config
        self.train_size = train_size

        self.model = config.get_model()

        # Loss function
        self.loss = config.loss

        # Metric
        self.f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=len(config.classes)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        preds = self(imgs)
        loss = self.loss(preds, targets)

        self.log("train_loss", loss)

        return loss
    
    def _evaluation_step(self, batch, batch_idx, prefix):
        imgs, targets = batch

        preds = self(imgs)
        loss = self.loss(preds, targets)

        self.f1(preds, targets)
        self.log(f"{prefix}f1", self.f1)
        self.log(f"{prefix}loss", loss)

    def validation_step(self, batch, batch_idx):
        self._evaluation_step(batch, batch_idx, 'val_')

    def test_step(self, batch, batch_idx):
        self._evaluation_step(batch, batch_idx, 'test_')

    def predict_step(self, batch, batch_idx):
        imgs = batch[0]
        return self(imgs)

    def get_scheduler(self, optimizer: "torch.optim.lr_scheduler.Optimizer"):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in optimizer.param_groups],
            total_steps=self.train_size,
            div_factor=3,
            final_div_factor=2,  # lr_min = max_lr / (div_factor * final_div_factor)
            pct_start=0.25,
        )

        return {"scheduler": scheduler, "interval": "step"}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)

        return [optimizer], [self.get_scheduler(optimizer)]
