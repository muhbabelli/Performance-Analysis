import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from easydict import EasyDict as edict


class SemanticSegmentationDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if "cityscapes" in self.config.DATA.NAME:
            from datasets.cityscapes import Cityscapes

            transformations = self.get_transformations()

            if "ood" in self.config.DATA.NAME:
                raise NotImplementedError("OOD is not implemented in this version")
            else:
                self.train_dataset = Cityscapes(
                    hparams=self.config.DATA,
                    transform=transformations.cityscapes_train,
                    split="train",
                )

                self.valid_dataset = Cityscapes(
                    hparams=self.config.DATA,
                    transform=transformations.cityscapes_val,
                    split="val",
                )
        else:
            raise ValueError(f"Undefined Dataset: {self.config.DATA.NAME}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.SOLVER.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.config.SOLVER.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.SOLVER.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def get_transformations(self):
        min_height, min_width = 512, 1024
        if self.config.MODEL.BACKBONE.NAME == "DINOv2":
            min_height, min_width = 518, 1036

        min_height_val, min_width_val = 1024, 2048
        if self.config.MODEL.BACKBONE.NAME == "DINOv2":
            min_height_val, min_width_val = 1036, 2058

        transformations = edict(
            cityscapes_train=A.Compose(
                [
                    A.RandomScale(
                        scale_limit=[0.5 - 1, 2.0 - 1], p=1.0
                    ),  # subtracted 1 because albumentations uses scale factor
                    A.RandomCrop(height=512, width=1024, p=1.0),
                    A.PadIfNeeded(
                        min_height=min_height,
                        min_width=min_width,
                        p=1.0,
                        mask_value=self.config.MODEL.IGNORE_INDEX,
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            ),
            cityscapes_val=A.Compose(
                [
                    A.PadIfNeeded(
                        min_height=min_height_val,
                        min_width=min_width_val,
                        p=1.0,
                        mask_value=self.config.MODEL.IGNORE_INDEX,
                    ),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            ),
            road_anomaly=A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            ),
        )

        return transformations
