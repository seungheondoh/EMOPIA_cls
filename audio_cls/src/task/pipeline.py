import pickle
from typing import Callable, Optional
import os
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from ..data import PEmo_Dataset


class PEmoPipeline(LightningDataModule):
    def __init__(self, config, fix_config) -> None:
        super(PEmoPipeline, self).__init__()
        self.config = config
        self.fix_config = fix_config
        self.dataset_builder = PEmo_Dataset

    def get_fl(self):
        if self.split == "TRAIN":
            self.fl = pd.read_csv("../dataset/split/train.csv", index_col=0)
        elif self.split == "VALID":
            self.fl = pd.read_csv("../dataset/split/val.csv", index_col=0)
        elif self.split == "TEST":
            self.fl = pd.read_csv("../dataset/split/test.csv", index_col=0)
        else:
            print("Split should be one of [TRAIN, VALID, TEST]")


    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = PEmoPipeline.get_dataset(
                self.dataset_builder,
                self.fix_config.wav.feature_path,
                self.fix_config.task.labels,
                "TRAIN",
                self.fix_config.task.cls_type,
                self.fix_config.wav.input_length, 
                self.fix_config.wav.sr
            )

            self.val_dataset = PEmoPipeline.get_dataset(
                self.dataset_builder,
                self.fix_config.wav.feature_path,
                self.fix_config.task.labels,
                "VALID",
                self.fix_config.task.cls_type,
                self.fix_config.wav.input_length, 
                self.fix_config.wav.sr
            )

        if stage == "test" or stage is None:
            self.test_dataset = PEmoPipeline.get_dataset(
                self.dataset_builder,
                self.fix_config.wav.feature_path,
                self.fix_config.task.labels,
                "TEST",
                self.fix_config.task.cls_type,
                self.fix_config.wav.input_length, 
                self.fix_config.wav.sr
            )

    def train_dataloader(self) -> DataLoader:
        return PEmoPipeline.get_dataloader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.fix_config.hparams.num_workers,
            drop_last=False,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return PEmoPipeline.get_dataloader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.fix_config.hparams.num_workers,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return PEmoPipeline.get_dataloader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.fix_config.hparams.num_workers,
            drop_last=False,
            shuffle=False,
        )

    @classmethod
    def get_dataset(cls, dataset_builder: Callable, feature_path, labels, split, cls_type, input_length, sr) -> Dataset:
        dataset = dataset_builder(feature_path, labels, split, cls_type, input_length, sr)
        return dataset

    @classmethod
    def get_dataloader(cls, dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, drop_last: bool, **kwargs) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, **kwargs)