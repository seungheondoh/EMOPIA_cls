import torch.nn as nn
import numpy as np

def accuracy(source, target):
    source = source.max(1)[1].long().detach().cpu().numpy()
    target = target.long().detach().cpu().numpy()
    correct = (source == target).sum().item()
    return correct/float(source.shape[0])