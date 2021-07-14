# EMOPIA_cls

This is the official repository of **EMOPIA: A Multi-Modal Pop Piano Dataset For Emotion Recognition and Emotion-based Music Generation**. The paper has been accepted by International Society for Music Information Retrieval Conference 2021. 

- [Demo Page](https://annahung31.github.io/EMOPIA/)
- [Dataset at Zenodo (Coming soon)]()

# Conditional Generation
For the generation models and codes, please refer to [this repo](https://github.com/annahung31/EMOPIA).

# Emotion Classification

## Environment

1. Install python and PyTorch:
    - python==3.8.5
    - torch==1.8.0 (Please install it according to your [CUDA version](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4).)
    
2. Other requirements:
    - pip install -r requirements.txt

3. MIDI processor
    - [MIDI-like(magenta)](https://github.com/jason9693/midi-neural-processor)
    - [REMI](https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/Dataset.md)

## Usage

### Inference
1. MIDI domain inference

        python inference.py --types midi --task ar_va --midi_path {your_midi} --cuda {cuda}

2. Audio domain inference

        python inference.py --types audio --task ar_va --midi_path {your_midi} --cuda {cuda}

### Training from scratch
1. Download the data files from [HERE]().
    
2. Preprocessing

        python preprocessing.py
    a. audio
        - resampling to 22050
    b. midi
        - magenta feature extraction
        - remi feature extraction

3. training options:  

    a. MIDI domain classification

        cd midi_cls
        python train_test.py --midi magenta --task ar_va --embedding_size 300 --r 16 --lstm_hidden_dim 128
        python train_test.py --midi magenta --task arousal --embedding_size 300 --r 16 --lstm_hidden_dim 128
        python train_test.py --midi magenta --task valence --embedding_size 300 --r 16 --lstm_hidden_dim 128


    b. Wav domain clasfficiation

        cd audio_cls
        python train_test.py --wav sr22k --task ar_va --n_channels 128 --n_fft 1024 --n_mels 128
        python train_test.py --wav sr22k --task arousal --n_channels 128 --n_fft 1024 --n_mels 128
        python train_test.py --wav sr22k --task valence --n_channels 128 --n_fft 1024 --n_mels 128


## Results


## Authors

The paper is a co-working project with [Anna](https://github.com/annahung31), [Joann](https://github.com/joann8512) and Nabin. This repository is mentained by me.


## License
The EMOPIA dataset is released under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). It is provided primarily for research purposes and is prohibited to be used for commercial purposes. When sharing your result based on EMOPIA, any act that defames the original music owner is strictly prohibited.


## Cite the dataset

```
@inproceedings{{EMOPIA},
         author = {Hung, Hsiao-Tzu and Ching, Joann and Doh, Seungheon and Kim, Nabin and Nam, Juhan and Yang, Yi-Hsuan},
         title = {{MOPIA}: A Multi-Modal Pop Piano Dataset For Emotion Recognition and Emotion-based Music Generation},
         booktitle = {Proc. Int. Society for Music Information Retrieval Conf.},
         year = {2021}
}
```

### Reference
- https://github.com/YatingMusic/compound-word-transformer
- https://github.com/jason9693/midi-neural-processor