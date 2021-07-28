import os
from pathlib import Path
import torchaudio
import pandas as pd
import torch
from tqdm import tqdm
import pickle
from midi_cls.midi_helper.remi.midi2event import analyzer, corpus, event
from midi_cls.midi_helper.magenta.processor import encode_midi

def torch_sox_effect_load(mp3_path, resample_rate):
    effects = [
        ['rate', str(resample_rate)]
    ]
    waveform, source_sr = torchaudio.load(mp3_path)
    if source_sr != 22050:
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, source_sr, effects, channels_first=True)
    return waveform

def audio_resample():
    audio_path = "../dataset/PEmoDataset/audios/seg"
    save_path = "./dataset/resample22050"
    for fn in tqdm(total):
        pt_path = Path(save_path, fn + ".pt")
        resample = torch_sox_effect_load(Path(audio_path, fn + ".mp3"), 22050).mean(0, True)
        if not os.path.exists(os.path.dirname(pt_path)):
            os.makedirs(os.path.dirname(pt_path))
        torch.save(resample, pt_path)

def remi_extractor(midi_path, event_to_int):
    midi_obj = analyzer(midi_path)
    song_data = corpus(midi_obj)
    event_sequence = event(song_data)
    quantize_midi = [event_to_int[str(i['name'])+"_"+str(i['value'])] for i in event_sequence]
    return quantize_midi

def magenta_extractor(midi_path):
    return encode_midi(midi_path)

def midi_feature_extract():
    # load remi dictionary
    path_data_root = "./midi_cls/midi_helper/remi/"
    path_dictionary = os.path.join(path_data_root, 'dictionary.pkl')
    midi_dictionary = pickle.load(open(path_dictionary, "rb"))
    event_to_int = midi_dictionary[0]

    midi_path = "../dataset/PEmoDataset/midis"
    remi_path = "./dataset/remi_midi"
    magenta_path = "./dataset/magenta_midi"

    for midi in os.listdir(midi_path):
        remi_fn = os.path.join(remi_path, midi).replace(".mid",".pt")
        try:
            remi_midi = remi_extractor(os.path.join(midi_path, midi), event_to_int)
        except:
            print("remi error: ", midi)
        if not os.path.exists(os.path.dirname(remi_fn)):
            os.makedirs(os.path.dirname(remi_fn))
        torch.save(remi_midi, remi_fn)

        magenta_fn = os.path.join(magenta_path, midi).replace(".mid",".pt")
        try:
            magenta_midi = magenta_extractor(os.path.join(midi_path, midi))
        except:
            print("magenta error: ", midi)
        if not os.path.exists(os.path.dirname(magenta_fn)):
            os.makedirs(os.path.dirname(magenta_fn))
        torch.save(magenta_midi, magenta_fn)

if __name__ == "__main__": 
    midi_feature_extract()
    # audio_domain_resample()
