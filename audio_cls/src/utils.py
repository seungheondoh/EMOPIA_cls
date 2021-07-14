import torch
import torchaudio
from pydub import AudioSegment
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
            
def torch_sox_effect_load(mp3_path, resample_rate):
    effects = [
        ['rate', str(resample_rate)]
    ]
    waveform, source_sr = torchaudio.load(mp3_path)
    waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, source_sr, effects, channels_first=True)
    return waveform

def _sox_effect_resample(waveform, source_sr, target_sr):
    effects = [
        ['rate', str(target_sr)]
    ]
    waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, source_sr, effects, channels_first=True)
    return waveform


def _pydub_resampling(audio_path, target_sr):
    if audio_path[-3:] in ('m4a', 'aac'):
        song = AudioSegment.from_file(audio_path, 'm4a').set_frame_rate(target_sr).set_channels(1)._data
    else:
        song = AudioSegment.from_file(audio_path, audio_path[-3:]).set_frame_rate(target_sr).set_channels(1)._data
    decoded = np.frombuffer(song, dtype=np.int16) / 32768
    audio_tensor = torch.Tensor(decoded).unsqueeze(0)
    return audio_tensor


def _torchaudio_resampling(mp3_path, target_sr):
    waveform, source_sr = torchaudio.load(mp3_path)
    fn_resample = torchaudio.transforms.Resample(source_sr, target_sr, resampling_method= 'sinc_interpolation')
    waveform = fn_resample(waveform)
    audio_tensor = waveform.mean(0, True)
    return audio_tensor

def _librosa_resampling(mp3_path, target_sr):
    y, sr = librosa.load(mp3_path)
    y_16k = librosa.resample(y, sr, target_sr, res_type='kaiser_fast')
    return y_16k

def batch_first(batch):
    waveforms = []
    labels = []
    for (audio_tensor, tag_binary) in batch:
        print(audio_tensor.shape)
        waveform = audio_tensor.permute(1,0)
        print(waveform)
        break
    #     waveforms.append(waveform)
    #     labels.append(labels)
    # return (waveforms, labels)