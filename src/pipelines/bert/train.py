"""Полезные ссылки


Дока по tokenizers: https://huggingface.co/docs/transformers/internal/tokenization_utils
Тренировка BERT:
    https://skimai.com/fine-tuning-bert-for-sentiment-analysis/,https://github.com/maastrichtlawtech/bsard.git
Какая-то странная статья про finetuning:
    https://towardsdatascience.com/how-to-make-the-most-out-of-bert-finetuning-d7c9f2ca806c
"""
import random
from pathlib import Path

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.pipelines.bert.data.datamodule import BERTDatamodule
from src.pipelines.bert.model import BertLightningModel
from pytorch_lightning.loggers import WandbLogger  # noqa
import os
from string import ascii_lowercase, digits
import wandb

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["HYDRA_FULL_ERROR"] = "1"

repo_path = pyrootutils.find_root()


def instantiate_callbacks(experiment_dir: Path, model_name: str):
    # bar = RichProgressBar()

    dir_path = experiment_dir / "checkpoints"
    dir_path.mkdir(exist_ok=True, parents=True)
    model_checkpoint = ModelCheckpoint(
        dirpath=dir_path,
        filename=f"{model_name}-val-loss_" + "{val/loss_epoch:.04f}-val-acc_{val/acc_epoch:.4f}",
        monitor="val/acc_epoch",
        mode="max",
        auto_insert_metric_name=False,
        verbose=True,
    )

    return [model_checkpoint]


def instantiate_logger(experiment_dir: Path):
    wandb_dir = experiment_dir / "wandb"
    wandb_dir.mkdir(exist_ok=True, parents=True)

    logger = WandbLogger(
        name=experiment_dir.name,
        save_dir=wandb_dir,
        project="thesis",
        entity="homomorfism",
    )
    include = ["Makefile", ".yaml", ".py", ".md", ".toml", ".lock", ".ipynb", ".png"]
    if os.environ.get("WANDB_MODE") != "offline":
        wandb.run.log_code(root=str(repo_path), include_fn=lambda item: any(x in item for x in include))

    return logger


def generate_string(length: int):
    letters = digits + ascii_lowercase
    return "".join([random.choice(letters) for _ in range(length)])


@hydra.main(config_path="config", config_name="finetune_bert.yaml", version_base="1.1")
def train(config: DictConfig):
    experiment_name = config.experiment_name + "-" + generate_string(length=5)
    experiment_dir = repo_path / ".logs" / experiment_name
    experiment_dir.mkdir(exist_ok=True, parents=True)

    pl.seed_everything(config.seed)

    model = BertLightningModel(config=config)
    datamodule = BERTDatamodule(config=config, tokenizer=model.tokenizer)

    logger = instantiate_logger(experiment_dir)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=instantiate_callbacks(experiment_dir, experiment_name),
        **config.train,
    )

    trainer.fit(model, datamodule)

    ckpt_path = experiment_dir / "checkpoints"
    wandb.save(str(ckpt_path / "*"))

    print(f"Results are saved in: {ckpt_path.resolve()}")
    print("Training is finished!")

    wandb.finish()


if __name__ == "__main__":
    if torch.cuda.is_available():
        # this is for 3090
        torch.set_float32_matmul_precision(precision="medium")

    train()
