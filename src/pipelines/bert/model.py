import pprint

import pyrootutils
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from transformers import BertForSequenceClassification
from transformers import AutoTokenizer
from torch.optim import AdamW
from torchmetrics import Accuracy
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

repo_path = pyrootutils.find_root(search_from=__file__, indicator="pyproject.toml")


class BertLightningModel(pl.LightningModule):
    model: BertForSequenceClassification

    def __init__(self, config: DictConfig):
        super(BertLightningModel, self).__init__()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        self.model = BertForSequenceClassification.from_pretrained(self.config.model.name, num_labels=2)
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")

    def configure_optimizers(self):
        # TODO - add cosine scheduler

        # implementation of decaying, ref: transformers/trainer.py#L1044
        decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_params = [
            {
                "params": [p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                "weight_decay": self.config.model.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimiser = AdamW(optimizer_params, lr=self.config.model.lr, eps=1e-6)

        return [optimiser]

    def training_step(self, batch: dict, batch_idx: int):
        outputs = self.model.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
        )

        predictions = torch.argmax(outputs.logits, dim=1)
        self.train_acc.update(predictions, batch["labels"])

        loss = outputs.loss
        self.log("train/loss", loss, batch_size=len(predictions))

        return {"loss": loss}

    def training_epoch_end(self, outputs: list[dict]) -> None:
        mean_loss = torch.stack([x["loss"] for x in outputs]).mean()
        train_acc = self.train_acc.compute()

        self.log_dict({"train/loss_epoch": mean_loss, "train/acc_epoch": train_acc})

    def validation_step(self, batch: dict, batch_idx: int):
        outputs = self.model.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
        )

        predictions = torch.argmax(outputs.logits, dim=1)
        self.val_acc.update(predictions, batch["labels"])

        return {"loss": outputs.loss}

    def validation_epoch_end(self, outputs: list[dict]) -> None:
        mean_loss = torch.stack([x["loss"] for x in outputs]).mean()
        val_acc = self.val_acc.compute()

        result_dict = {"val/loss_epoch": mean_loss.item(), "val/acc_epoch": val_acc.item()}

        print()
        print("-" * 20)
        print(f"Validation results, epoch: {self.current_epoch}")
        pprint.pprint(result_dict)
        print("-" * 20)

        self.log_dict(result_dict)
