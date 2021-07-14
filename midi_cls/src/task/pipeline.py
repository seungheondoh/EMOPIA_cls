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
                self.fix_config.midi.feature_path,
                self.fix_config.task.labels,
                "TRAIN",
                self.fix_config.task.cls_type,
                self.fix_config.midi.pad_idx
            )

            self.val_dataset = PEmoPipeline.get_dataset(
                self.dataset_builder,
                self.fix_config.midi.feature_path,
                self.fix_config.task.labels,
                "VALID",
                self.fix_config.task.cls_type,
                self.fix_config.midi.pad_idx
            )

        if stage == "test" or stage is None:
            self.test_dataset = PEmoPipeline.get_dataset(
                self.dataset_builder,
                self.fix_config.midi.feature_path,
                self.fix_config.task.labels,
                "TEST",
                self.fix_config.task.cls_type,
                self.fix_config.midi.pad_idx
            )

    def train_dataloader(self) -> DataLoader:
        return PEmoPipeline.get_dataloader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.fix_config.hparams.num_workers,
            drop_last=False,
            shuffle=True,
            collate_fn = self.train_dataset.batch_padding
        )

    def val_dataloader(self) -> DataLoader:
        return PEmoPipeline.get_dataloader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.fix_config.hparams.num_workers,
            drop_last=False,
            shuffle=False,
            collate_fn= self.val_dataset.batch_padding
        )

    def test_dataloader(self) -> DataLoader:
        return PEmoPipeline.get_dataloader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.fix_config.hparams.num_workers,
            drop_last=False,
            shuffle=False,
            collate_fn= self.test_dataset.batch_padding
        )

    @classmethod
    def get_dataset(cls, dataset_builder: Callable, feature_path, labels, split, cls_type, pad_idx) -> Dataset:
        dataset = dataset_builder(feature_path, labels, split, cls_type, pad_idx)
        return dataset

    @classmethod
    def get_dataloader(cls, dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, drop_last: bool, collate_fn, **kwargs) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fn, **kwargs)