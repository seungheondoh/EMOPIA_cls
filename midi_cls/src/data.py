import os
import json
import numpy as np
import pandas as pd
import torch

import random
import pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class PEmo_Dataset(Dataset):
    def __init__(self, feature_path, labels, split, cls_type, pad_idx):
        self.pt_dir = feature_path
        self.labels = labels
        self.split = split
        self.get_fl()
        self.cls_type = cls_type
        self.pad_idx = pad_idx

    def get_fl(self):
        if self.split == "TRAIN":
            self.fl = pd.read_csv("../dataset/split/train.csv", index_col=0)
        elif self.split == "VALID":
            self.fl = pd.read_csv("../dataset/split/val.csv", index_col=0)
        elif self.split == "TEST":
            self.fl = pd.read_csv("../dataset/split/test.csv", index_col=0)
        else:
            print("Split should be one of [TRAIN, VALID, TEST]")

    def __getitem__(self, index):
        audio_fname = self.fl.iloc[index].name
        label = self.fl.iloc[index]['label']
        if self.cls_type == "AV":
            labels = self.labels.index(label)
        elif self.cls_type == "A":
            if label in ['Q1','Q2']:
                labels = self.labels.index('HA')
            elif label in ['Q3','Q4']:
                labels = self.labels.index('LA')
        elif self.cls_type == "V":
            if label in ['Q1','Q4']:
                labels = self.labels.index('HV')
            elif label in ['Q2','Q3']:
                labels = self.labels.index('LV')
        processed_midi = torch.load(os.path.join(self.pt_dir, audio_fname + ".pt"))
        return processed_midi, labels, audio_fname

    def __len__(self):
        return len(self.fl)
    
    def batch_padding(self, data):
        texts, labels, audio_fname = list(zip(*data))
        max_len = max([len(s) for s in texts])
        texts = [s+[self.pad_idx]*(max_len-len(s)) if len(s) < max_len else s for s in texts]
        return torch.LongTensor(texts), torch.LongTensor(labels), audio_fname