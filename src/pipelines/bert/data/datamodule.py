import pytorch_lightning as pl
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.pipelines.bert.data.dataset import CustomBERTDataset
from transformers import BertTokenizerFast
from src.data_reader import read_json
from transformers import DataCollatorWithPadding
import pyrootutils

repo_path = pyrootutils.find_root()


class BERTDatamodule(pl.LightningDataModule):
    def __init__(self, config: DictConfig, tokenizer: BertTokenizerFast):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        paths = self.config.data.collie
        queries = read_json(repo_path / paths.queries)
        articles = read_json(repo_path / paths.articles)
        train_queries, val_queries = train_test_split(queries, test_size=self.config.val_size, random_state=42)

        self.train_dataset = CustomBERTDataset(
            tokenizer=self.tokenizer,
            articles_list=articles,
            query_articles_list=train_queries,
            config=self.config,
            with_negative_samples=True,
        )

        self.val_dataset = CustomBERTDataset(
            tokenizer=self.tokenizer,
            articles_list=articles,
            query_articles_list=val_queries,
            config=self.config,
            with_negative_samples=True,
        )

        self.collate_fn = DataCollatorWithPadding(
            tokenizer=tokenizer, padding="longest", max_length=config.model.max_length, return_tensors="pt"
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            sampler=WeightedRandomSampler(
                self.train_dataset.sampling_weights,
                num_samples=len(self.train_dataset),
            ),
        )

    def val_dataloader(self):
        # without weight sampling
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
