import torch.nn as nn
import numpy as np

def accuracy(source, target):
    source = source.max(1)[1].long()
    target = target.long()
    correct = (source == target).sum()
    mean_correct = correct/ source.shape[0]
    return mean_correct