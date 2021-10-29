# EMOPIA_cls

This is the official repository of **EMOPIA: A Multi-Modal Pop Piano Dataset For Emotion Recognition and Emotion-based Music Generation**. The paper has been accepted by International Society for Music Information Retrieval Conference 2021. This repository is the **Emotion Recognition** part (Audio and MIDI domain).

- [Demo Page](https://annahung31.github.io/EMOPIA/)
- [Dataset at Zenodo](https://zenodo.org/record/5090631#.YQEZZ1Mzaw5)
- [Conditional Generation Repo](https://github.com/annahung31/EMOPIA).


**News!**

`2021-10-29` update [matlab feature](https://drive.google.com/file/d/1lG3KMYhRZsBr3ILqcES2aVY21xhvCr7i/view?usp=sharing) (key, tempo, note density)

`2021-07-21` update [dataset](https://zenodo.org/record/5090631#.YQEZZ1Mzaw5)

`2021-07-20` Upload all pretrained [weight](https://drive.google.com/file/d/1L_NOVKCElwcYUEAKp1-FZj_G6Hcq2g2c/view?usp=sharing)

<figure><img src="figs/results.png">

you can check ML performance in [notebook](https://github.com/Dohppak/EMOPIA_cls/blob/main/notebook/1.ML%20Classifier.ipynb)

## Environment

1. Install python and PyTorch:
    - python==3.8.5
    - torch==1.8.0 (Please install it according to your [CUDA version](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4).)
    
2. Other requirements:
    - pip install -r requirements.txt

3. git clone MIDI processor (already done)
    - [MIDI-like(magenta)](https://github.com/jason9693/midi-neural-processor)
    - [REMI](https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/Dataset.md)
    - If you want to bulid new REMI corpus, vocab from other dataset, plz check official repo of compund-word-transfomer and `EMOPIA_cls/midi_cls/midi_helper/remi/src`

## Usage

### Inference
download model weight in [Here](https://drive.google.com/file/d/1L_NOVKCElwcYUEAKp1-FZj_G6Hcq2g2c/view?usp=sharing), unzip in project dir.

1. MIDI domain inference

        python inference.py --types {midi_like or remi} --task ar_va --file_path {your_midi} --cuda {cuda}
        python inference.py --types {midi_like or remi} --task arousal --file_path {your_midi} --cuda {cuda}
        python inference.py --types {midi_like or remi} --task valence --file_path {your_midi} --cuda {cuda}

2. Audio domain inference

        python inference.py --types wav --task ar_va --file_path {your_mp3} --cuda {cuda}
        python inference.py --types wav --task arousal --file_path {your_mp3} --cuda {cuda}
        python inference.py --types wav --task valence --file_path {your_mp3} --cuda {cuda}

### Inference results

        python inference.py --types wav --task ar_va --file_path ./dataset/sample_data/Sakamoto_MerryChristmasMr_Lawrence.mp3

        ./dataset/sample_data/Sakamoto_MerryChristmasMr_Lawrence.mp3  is emotion Q3
        Inference values:  [0.33273646 0.17223473 0.63210356 0.07314324]

        python inference.py --types midi_like --task ar_va --file_path ./dataset/sample_data/Sakamoto_MerryChristmasMr_Lawrence.mid

        ./dataset/sample_data/Sakamoto_MerryChristmasMr_Lawrence.mid  is emotion Q3
        Inference values:  [-1.3685153 -1.3001229  2.2495744 -0.873877 ]

### Training from scratch
1. Download the data files from [HERE](https://zenodo.org/record/5090631#.YQEZZ1Mzaw5).
    
2. Preprocessing

    a. audio: resampling to 22050

    b. midi: magenta feature extraction, remi feature extraction

        python preprocessing.py

3. training options:  

    a. MIDI domain classification

        cd midi_cls
        python train_test.py --midi {midi_like or remi} --task ar_va
        python train_test.py --midi {midi_like or remi} --task arousal
        python train_test.py --midi {midi_like or remi} --task valence


    b. Wav domain clasfficiation

        cd audio_cls
        python train_test.py --wav sr22k --task ar_va
        python train_test.py --wav sr22k --task arousal
        python train_test.py --wav sr22k --task valence

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
