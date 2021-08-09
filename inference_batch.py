
import os
import glob
import pandas as pd
import ipdb
from inference import predict
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from argparse import ArgumentParser, Namespace
from sklearn.metrics import accuracy_score


def evaluate(args):
    config_path = Path("best_weight", args.types, args.task, "hparams.yaml")
    config = OmegaConf.load(config_path)
    n_cls = config.task.num_of_dim

    if args.task == "ar_va":    
        
        mapping = {
            1: 1,
            2: 2,
            3: 3,
            4: 4
        }

        
    elif args.task == "arousal":
        mapping = {
                1: 1,
                2: 1,
                3: 2,
                4: 2
            }

    elif args.task == "valence":

        mapping = {
                1: 1,
                2: 2,
                3: 2,
                4: 1
            }

    files = glob.glob(os.path.join(args.folder_path, '*.mid'))
    assert len(files) != 0
    print('evaluating {} clips'.format(len(files)))


    y_true = []
    y_preds = []
    class_1_score = []
    class_2_score = []
    class_3_score = []
    class_4_score = []
    cls_score = [
                class_1_score,
                class_2_score,
                class_3_score,
                class_4_score
                ]
    filenames = []

    for a_file in files:
            
        target = int(a_file.split('/')[-1][4])
        args.file_path = a_file
        pred_label, pred_value= predict(args)
        for i in range(n_cls):
            cls_score[i].append(pred_value[i])
        

        print('target: ', mapping[target])
        y_true.append(mapping[target])
        y_preds.append(pred_value.argmax() + 1 )
        filenames.append(a_file)



    acc = accuracy_score(y_true, y_preds)
    print(args.folder_path, ':', acc)


    if n_cls == 2:
        pred_df = pd.DataFrame({'filename': filenames, 'target': y_true, 'class_1_score': cls_score[0], 'class_2_score': cls_score[1]})
    elif n_cls == 4:
        pred_df = pd.DataFrame({'filename': filenames, 'target': y_true, 
                                'class_1_score': cls_score[0], 
                                'class_2_score': cls_score[1],
                                'class_3_score': cls_score[2],
                                'class_4_score': cls_score[3]
                                })


    if args.output_file:    
        pred_df = pred_df.append({'filename': 'acc_score', 'target': acc}, ignore_index=True)
        pred_df.to_csv(args.output_file)
        print('predict result save to ',  args.output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--types", default="midi_like", type=str, choices=["midi_like", "remi", "wav"])
    parser.add_argument("--task", default="ar_va", type=str, choices=["ar_va", "arousal", "valence"])
    parser.add_argument("--file_path", type=str)
    parser.add_argument('--cuda', default='cuda:0', type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--folder_path", type=str, required=True)
    args = parser.parse_args()
    evaluate(args)