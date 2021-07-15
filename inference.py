import os
import json
import pickle
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
from midi_cls.src.model.net import SAN
from midi_cls.midi_helper.remi.midi2event import analyzer, corpus, event
from midi_cls.midi_helper.magenta.processor import encode_midi

path_data_root = "./midi_cls/midi_helper/remi/"
path_dictionary = os.path.join(path_data_root, 'dictionary.pkl')
midi_dictionary = pickle.load(open(path_dictionary, "rb"))
event_to_int = midi_dictionary[0]

def remi_extractor(midi_path, event_to_int):
    midi_obj = analyzer(midi_path)
    song_data = corpus(midi_obj)
    event_sequence = event(song_data)
    quantize_midi = [event_to_int[str(i['name'])+"_"+str(i['value'])] for i in event_sequence]
    return quantize_midi

def magenta_extractor(midi_path):
    return encode_midi(midi_path)

def main(args) -> None:
    device = args.cuda if args.cuda and torch.cuda.is_available() else 'cpu'
    if args.cuda:
        print('GPU name: ', torch.cuda.get_device_name(device=args.cuda))

    if args.task == "ar_va":
        labels= ["Q1","Q2","Q3","Q4"]
        checkpoint_path = "./best_weight/midi/ar_va/best.ckpt"
        model = SAN( num_of_dim= 4, vocab_size= 389, lstm_hidden_dim= 128, embedding_size= 300, r=16)
        state_dict = torch.load(checkpoint_path, map_location=torch.device(args.cuda))
        new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
        new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
        model.load_state_dict(new_state_dict)
        model.eval()
        quantize_midi = magenta_extractor(args.midi_path)

    elif args.task == "arousal":
        labels= ["HA","LA"]
        checkpoint_path = "./best_weight/midi/arousal/best.ckpt"
        model = SAN( num_of_dim= 2, vocab_size= 389, lstm_hidden_dim= 128, embedding_size= 300, r=16)
        state_dict = torch.load(checkpoint_path, map_location=torch.device(args.cuda))
        new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
        new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
        model.load_state_dict(new_state_dict)
        model.eval()
        quantize_midi = magenta_extractor(args.midi_path)

    elif args.task == "valence":
        labels= ["HV","LV"]
        checkpoint_path = "./best_weight/midi/valence/best.ckpt"
        model = SAN( num_of_dim= 2, vocab_size= 389, lstm_hidden_dim= 128, embedding_size= 300, r=8)
        state_dict = torch.load(checkpoint_path, map_location=torch.device(args.cuda))
        new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
        new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
        model.load_state_dict(new_state_dict)
        model.eval()
        quantize_midi = magenta_extractor(args.midi_path)
        
    model = model.to(args.cuda)
    torch_midi = torch.LongTensor(quantize_midi).unsqueeze(0)
    prediction = model(torch_midi.to(args.cuda))
    print("========")
    print(args.midi_path, " is emotion", labels[prediction.squeeze(0).max(0)[1].detach().cpu().numpy()])
    print("Inference values: ",prediction)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--types", default="midi", type=str, choices=["midi, audio"])
    parser.add_argument("--task", default="ar_va", type=str, choices=["ar_va", "arousal", "valence"])
    parser.add_argument("--midi_path", default="./dataset/sample_data/example_generative.mid", type=str)
    parser.add_argument('--cuda', default='cuda:0', type=str)
    args = parser.parse_args()
    main(args)
