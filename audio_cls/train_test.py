
from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import os
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.model.net import ShortChunkCNN_Res
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
    wav_config_dir = child_config_dir / "wav"
    monitor_config_dir = child_config_dir / "monitor"
    config = OmegaConf.create()
    task_config = OmegaConf.load(task_config_dir / f"{args.task}.yaml")
    wav_config = OmegaConf.load(wav_config_dir / f"{args.wav}.yaml")
    monitor_config = OmegaConf.load(monitor_config_dir / f"{args.monitor}.yaml")
    config.update(task=task_config, wav=wav_config, monitor=monitor_config, hparams=vars(args))
    return config


def get_tensorboard_logger(args: Namespace) -> TensorBoardLogger:
    logger = TensorBoardLogger(
        save_dir=f"exp/{args.dataset}", name=args.task, version=args.wav
    )
    return logger

def get_wandb_logger(model):
    logger = WandbLogger()
    logger.watch(model)
    return logger 


def get_checkpoint_callback(fix_config, save_path) -> ModelCheckpoint:
    prefix = save_path
    suffix = "Best-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=prefix,
        filename=suffix,
        save_top_k=1,
        save_last= True,
        monitor=fix_config.monitor.metric,
        mode=fix_config.monitor.mode,
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

def get_early_stop_callback(config, fix_config) -> EarlyStopping:
    early_stop_callback = EarlyStopping(
        monitor=fix_config.monitor.metric, min_delta=0.00, patience=config.patience, verbose=True, mode=fix_config.monitor.mode
    )
    return early_stop_callback


def main(args) -> None:
    if args.reproduce:
        seed_everything(42)

    wandb.init(config=args)
    config = wandb.config

    save_path = f"exp/{args.dataset}/{args.task}/{args.wav}/batch:{args.batch_size}-monitor:{args.monitor}-mels:{args.n_mels}-T_0:{args.T_0}-lr:{args.lr}/"

    fix_config = get_config(args)
    pipeline = PEmoPipeline(config=config, fix_config=fix_config)
    model = ShortChunkCNN_Res(
                sample_rate = fix_config.wav.sr,
                n_fft = config.n_fft,
                f_min = config.f_min,
                f_max = config.f_max,
                n_mels = config.n_mels,
                n_channels = config.n_channels,
                n_class = fix_config.task.n_class
            )
    runner = Runner(model, config, eval_type="last")

    logger = get_wandb_logger(model)
    checkpoint_callback = get_checkpoint_callback(fix_config, save_path)
    early_stop_callback = get_early_stop_callback(config, fix_config)

    trainer = Trainer(
                        max_epochs= config.max_epochs,
                        gpus= [fix_config.hparams.gpus],
                        distributed_backend= fix_config.hparams.distributed_backend,
                        benchmark= fix_config.hparams.benchmark,
                        deterministic= fix_config.hparams.deterministic,
                        logger=logger,
                        callbacks=[
                            early_stop_callback,
                            checkpoint_callback
                        ]
                      )

    trainer.fit(runner, datamodule=pipeline)
    trainer.test(runner, datamodule=pipeline)
    results = {"last": runner.test_results}

    best_runner = Runner(model, config, eval_type="best")
    best_runner = get_best_performance(best_runner, save_path)
    trainer.test(best_runner, datamodule=pipeline)
    results.update({"best": best_runner.test_results})

    with open(Path(save_path, "results_last.json"), mode="w") as io:
        json.dump(results, io, indent=4)
    for update_key in fix_config.hparams.keys():
        if update_key in config.keys():
            fix_config.hparams[update_key] = config[update_key] 
    OmegaConf.save(config=fix_config, f= Path(save_path, "hparams.yaml"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="EMOPIA", type=str)
    parser.add_argument("--task", default="ar_va", type=str)
    parser.add_argument("--wav", default="sr22k", type=str)
    parser.add_argument("--monitor", default="acc", type=str)
    parser.add_argument("--patience", default=5, type=int)
    # model
    parser.add_argument("--n_channels", default=128, type=int)
    parser.add_argument("--n_fft", default=1024, type=int)
    parser.add_argument("--n_mels", default=128, type=int)
    parser.add_argument("--f_min", default=0, type=int)
    parser.add_argument("--f_max", default=11025, type=int)
    # pipeline
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    # runner
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=0.0001, type=float)
    parser.add_argument("--T_0", default=16, type=int)
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--gpus", default=0, type=int)
    parser.add_argument("--distributed_backend", default="dp", type=str)
    parser.add_argument("--deterministic", default=True, type=str2bool)
    parser.add_argument("--benchmark", default=False, type=str2bool)
    parser.add_argument("--reproduce", default=True, type=str2bool)

    args = parser.parse_args()
    main(args)
 