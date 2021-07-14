import os
import json
import numpy as np
import pandas as pd
import torch

import random
import pickle
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa

class PEmo_Dataset(Dataset):
    def __init__(self, feature_path, labels, split, cls_type, input_length, sr):
        self.pt_dir = feature_path
        self.labels = labels
        self.split = split
        self.cls_type = cls_type
        self.input_length = input_length
        self.sr = sr
        self.sample_length = self.sr* self.input_length
        self.get_fl()

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
        audio_pt = torch.load(os.path.join(self.pt_dir, audio_fname + ".pt"))

        if self.split == "TEST":
            frame = (audio_pt.shape[1] - self.sample_length) // self.sample_length
            audio_sample = torch.zeros(frame, 1, self.sample_length)
            for i in range(frame):
                audio_sample[i] = torch.Tensor(audio_pt[:,i*self.sample_length:(i+1)*self.sample_length])
            return audio_sample, labels, audio_fname
        else:
            random_idx = int(torch.floor(torch.rand(1) * (audio_pt.shape[1] - self.sample_length)))
            audio_sample = audio_pt[:,random_idx : random_idx + self.sample_length]
            return audio_sample, labels, audio_fname

    def __len__(self):
        return len(self.fl)
