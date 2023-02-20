import albumentations as A
import cv2
import segmentation_models_pytorch as smp
from torch import nn


class Config:
    experiment_name = "start"

    classes = [0, 1, 2]
    loss = nn.BCEWithLogitsLoss()
    lr = 1e-4
    epochs = 1

    input_size = (256, 256)
    train_size = 0.8
    batch_size = 16
    seed = 42

    depth = 4

    model_params = dict(
        type=smp.Unet,
        encoder_name="vgg16_bn",
        in_channels=1,
        classes=len(classes),
        encoder_depth=depth,
        decoder_channels=tuple(32 * (2**i) for i in range(depth)),
        activation=None,
    )

    valid_augmentations = [
        A.Normalize(mean=0, std=1, max_pixel_value=255),
    ]

    train_augmentations = [
        A.ShiftScaleRotate(
            border_mode=cv2.BORDER_CONSTANT,
            rotate_limit=0.0,
            shift_limit=0.0,
            scale_limit=0.15,
        ),
        A.HorizontalFlip(),
        A.Perspective(p=0.1),
        A.RandomBrightnessContrast(p=0.15),
    ] + valid_augmentations

    @property
    def valid_aug(self):
        return A.Compose(self.valid_augmentations)

    @property
    def train_aug(self):
        return A.Compose(self.train_augmentations)

    def get_model(self):
        params = self.model_params.copy()
        model_class = params.pop("type")

        model = model_class(**params)

        return model
