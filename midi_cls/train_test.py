from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import os
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.model.net import SAN
from src.task.pipeline import PEmoPipeline
from src.task.runner import Runner
import wandb

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config(args: Namespace) -> DictConfig:
    parent_config_dir = Path("conf/")
    child_config_dir = parent_config_dir / args.dataset
    task_config_dir = child_config_dir / "task"
    midi_config_dir = child_config_dir / "midi"

    config = OmegaConf.create()
    task_config = OmegaConf.load(task_config_dir / f"{args.task}.yaml")
    midi_config = OmegaConf.load(midi_config_dir / f"{args.midi}.yaml")
    config.update(task=task_config, midi=midi_config, hparams=vars(args))
    return config


def get_tensorboard_logger(args: Namespace) -> TensorBoardLogger:
    logger = TensorBoardLogger(
        save_dir=f"exp/{args.dataset}", name=args.task, version=f"{args.midi}/"
    )
    return logger

def get_wandb_logger(model):
    logger = WandbLogger()
    logger.watch(model)
    return logger 


def get_checkpoint_callback(args, save_path) -> ModelCheckpoint:
    prefix = save_path
    suffix = "Best-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=prefix,
        filename=suffix,
        save_top_k=1,
        save_last= True,
        monitor="val_acc",
        mode='max',
        save_weights_only=True,
        verbose=True,
    )
    return checkpoint_callback

def get_best_performance(runner, save_path):
    for fnames in os.listdir(save_path):
        if "Best" in fnames:
            checkpoint_path = Path(save_path,fnames)
    state_dict = torch.load(checkpoint_path)
    runner.load_state_dict(state_dict.get("state_dict"))
    return runner

def get_early_stop_callback(args: Namespace) -> EarlyStopping:
    early_stop_callback = EarlyStopping(
        monitor="val_acc", min_delta=0.00, patience=5, verbose=True, mode="max"
    )
    return early_stop_callback


def main(args) -> None:
    if args.reproduce:
        seed_everything(42)

    save_path = f"exp/{args.dataset}/{args.task}/{args.midi}/batch:{args.batch_size}-h:{args.lstm_hidden_dim}-emd:{args.embedding_size}-wd:{args.weight_decay}-attn:{args.r}-T_0:{args.T_0}-lr:{args.lr}/"
    fix_config = get_config(args)
    pipeline = PEmoPipeline(config = args, fix_config=fix_config)
    model = SAN(
                r = args.r,
                num_of_dim=fix_config.task.num_of_dim, 
                vocab_size=fix_config.midi.pad_idx + 1, 
                embedding_size= args.embedding_size
                )
    runner = Runner(model, args, eval_type="last")

    # logger = get_wandb_logger(model)
    checkpoint_callback = get_checkpoint_callback(args, save_path)
    early_stop_callback = get_early_stop_callback(args)

    trainer = Trainer(
                        max_epochs= args.max_epochs,
                        gpus= [args.gpus],
                        distributed_backend= args.distributed_backend,
                        benchmark= args.benchmark,
                        deterministic= args.deterministic,
                        # logger=logger,
                        callbacks=[
                            # early_stop_callback,
                            checkpoint_callback
                        ]
                      )

    trainer.fit(runner, datamodule=pipeline)
    trainer.test(runner, datamodule=pipeline)
    results = {"last": runner.test_results}

    best_runner = Runner(model, args, eval_type="best")
    best_runner = get_best_performance(best_runner, save_path)
    trainer.test(best_runner, datamodule=pipeline)
    results.update({"best": best_runner.test_results})

    with open(Path(save_path, "results_last.json"), mode="w") as io:
        json.dump(results, io, indent=4)

    OmegaConf.save(config=fix_config, f= Path(save_path, "hparams.yaml"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="EMOPIA", type=str)
    parser.add_argument("--midi", default="magenta", type=str)
    parser.add_argument("--task", default="ar_va", type=str)
    # model
    parser.add_argument("--r", default=14, type=int)
    parser.add_argument("--lstm_hidden_dim", default=128, type=float)
    parser.add_argument("--embedding_size", default=300, type=float)
    # pipeline
    parser.add_argument("--batch_size", default=8, type=float)
    parser.add_argument("--num_workers", default=8, type=float)
    # runner
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--T_0", default=45, type=int)
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--distributed_backend", default="dp", type=str)
    parser.add_argument("--deterministic", default=True, type=str2bool)
    parser.add_argument("--benchmark", default=False, type=str2bool)
    parser.add_argument("--reproduce", default=True, type=str2bool)

    args = parser.parse_args()
    main(args)
 