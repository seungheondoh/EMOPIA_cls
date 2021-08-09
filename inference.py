import os
import json
import pickle
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf
from audio_cls.src.model.net import ShortChunkCNN_Res
from midi_cls.src.model.net import SAN
from midi_cls.midi_helper.remi.midi2event import analyzer, corpus, event
from midi_cls.midi_helper.magenta.processor import encode_midi

path_data_root = "./midi_cls/midi_helper/remi/"
path_dictionary = os.path.join(path_data_root, 'dictionary.pkl')
midi_dictionary = pickle.load(open(path_dictionary, "rb"))
event_to_int = midi_dictionary[0]

def torch_sox_effect_load(mp3_path, resample_rate):
    effects = [
        ['rate', str(resample_rate)]
    ]
    waveform, source_sr = torchaudio.load(mp3_path)
    if source_sr != 22050:
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, source_sr, effects, channels_first=True)
    return waveform

def remi_extractor(midi_path, event_to_int):
    midi_obj = analyzer(midi_path)
    song_data = corpus(midi_obj)
    event_sequence = event(song_data)
    
    quantize_midi = []
    for i in event_sequence:
        try:
            quantize_midi.append(event_to_int[str(i['name'])+"_"+str(i['value'])])
        except KeyError:
            
            if 'Velocity' in str(i['name']):
                quantize_midi.append(event_to_int[str(i['name'])+"_"+str(i['value']-2)])
            else:
                #skip the unknown event
                continue
    
    return quantize_midi

def magenta_extractor(midi_path):
    return encode_midi(midi_path)

def predict(args) -> None:
    device = args.cuda if args.cuda and torch.cuda.is_available() else 'cpu'
    if args.cuda:
        print('GPU name: ', torch.cuda.get_device_name(device=args.cuda))
    config_path = Path("best_weight", args.types, args.task, "hparams.yaml")
    checkpoint_path = Path("best_weight", args.types, args.task, "best.ckpt")
    config = OmegaConf.load(config_path)
    label_list = list(config.task.labels)
    if args.types == "wav":
        model = ShortChunkCNN_Res(
                sample_rate = config.wav.sr,
                n_fft = config.hparams.n_fft,
                f_min = config.hparams.f_min,
                f_max = config.hparams.f_max,
                n_mels = config.hparams.n_mels,
                n_channels = config.hparams.n_channels,
                n_class = config.task.n_class
        )
        state_dict = torch.load(checkpoint_path, map_location=torch.device(args.cuda))
        new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
        new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
        model.load_state_dict(new_state_dict)
        model.eval()
    else:
        model = SAN( 
            num_of_dim= config.task.num_of_dim, 
            vocab_size= config.midi.pad_idx+1, 
            lstm_hidden_dim= config.hparams.lstm_hidden_dim, 
            embedding_size= config.hparams.embedding_size, 
            r= config.hparams.r)
        state_dict = torch.load(checkpoint_path, map_location=torch.device(args.cuda))
        new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
        new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
        model.load_state_dict(new_state_dict)
        model.eval()
    model = model.to(args.cuda)

    if args.types == "midi_like":
        quantize_midi = magenta_extractor(args.file_path)
        model_input = torch.LongTensor(quantize_midi).unsqueeze(0)
        prediction = model(model_input.to(args.cuda))
    elif args.types == "remi":
        quantize_midi = remi_extractor(args.file_path, event_to_int)
        model_input = torch.LongTensor(quantize_midi).unsqueeze(0)
        prediction = model(model_input.to(args.cuda))
    elif args.types == "wav":
        model_input = torch_sox_effect_load(args.file_path, 22050).mean(0, True)
        sample_length = config.wav.sr * config.wav.input_length
        frame = (model_input.shape[1] - sample_length) // sample_length
        audio_sample = torch.zeros(frame, 1, sample_length)
        for i in range(frame):
            audio_sample[i] = torch.Tensor(model_input[:,i*sample_length:(i+1)*sample_length])
        prediction = model(audio_sample.to(args.cuda))
        prediction = prediction.mean(0,False)
    
    pred_label = label_list[prediction.squeeze(0).max(0)[1].detach().cpu().numpy()]
    pred_value = prediction.squeeze(0).detach().cpu().numpy()
    print("========")
    print(args.file_path, " is emotion", pred_label)
    print("Inference values: ", pred_value)

    return pred_label, pred_value

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--types", default="midi_like", type=str, choices=["midi_like", "remi", "wav"])
    parser.add_argument("--task", default="ar_va", type=str, choices=["ar_va", "arousal", "valence"])
    parser.add_argument("--file_path", default="./dataset/sample_data/Sakamoto_MerryChristmasMr_Lawrence.mid", type=str)
    parser.add_argument('--cuda', default='cuda:0', type=str)
    args = parser.parse_args()
    _, _ = predict(args)
