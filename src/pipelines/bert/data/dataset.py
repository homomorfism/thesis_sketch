import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.data_reader import read_json
import pyrootutils
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


def generate_pairs_between_queries_and_articles(
    articles_list: list[dict], query_articles_list: list[dict], with_negative_samples: bool
) -> list[dict]:
    """Создает все-возможные пары query-article с проставленным relevancy (1=Да/1=Нет)."""
    article_name_to_id = {}
    for item in articles_list:
        article_name_to_id[item["article_name"]] = item["article_content"]

    generated_pairs = []
    for item in query_articles_list:
        query_content = item["query"]
        relevant_articles = set(item["articles"])
        for article_name in relevant_articles:
            generated_pairs.append(
                {
                    "query": query_content,
                    "article": article_name_to_id[article_name],
                    "is_relevant": 1,
                }
            )

        if with_negative_samples:
            for article_name in set(article_name_to_id).difference(relevant_articles):
                generated_pairs.append(
                    {"query": query_content, "article": article_name_to_id[article_name], "is_relevant": 0}
                )

    return generated_pairs


class CustomBERTDataset(Dataset):
    def __init__(
        self,
        tokenizer: BertTokenizerFast,
        articles_list: list[dict],
        query_articles_list: list[dict],
        config: DictConfig,
        with_negative_samples: bool,
    ):
        self.tokenizer = tokenizer
        self.max_length = config.model.max_length

        # self.query_to_ids = {query_item['query']: ii for ii, query_item in enumerate(query_articles_list)}
        # self.article_to_ids = {article_item['article_content']: ii for ii, }

        self.generated_pairs = generate_pairs_between_queries_and_articles(
            articles_list=articles_list,
            query_articles_list=query_articles_list,
            with_negative_samples=with_negative_samples,
        )

        # needed for calculating sampling weight
        # positive : negative - 1:5
        # positive - x, negative - y
        # p(positive) = alpha * x / (x + y) = 1 / 6
        # alpha = 1 / 6 * (x + y) / x
        n_positives = len(list(filter(lambda x: x["is_relevant"] == 1, self.generated_pairs)))
        n_negatives = len(list(filter(lambda x: x["is_relevant"] == 0, self.generated_pairs)))
        sampling_ratio = config.sampling_negative_ratio
        probs = 1 / (1 + sampling_ratio) * (n_positives + n_negatives) / n_positives

        self.sampling_weights = list(map(lambda x: probs if x["is_relevant"] else 1, self.generated_pairs))

    def __getitem__(self, item):
        sample = self.generated_pairs[item]

        query, article, is_relevant = sample["query"], sample["article"], sample["is_relevant"]

        result = self.tokenizer(
            text=query, text_pair=article, padding=True, truncation=True, max_length=self.max_length
        )

        return {
            "input_ids": result["input_ids"],
            "token_type_ids": result["token_type_ids"],
            "attention_mask": result["attention_mask"],
            "label": torch.as_tensor([is_relevant], dtype=torch.long),
            "query": query,
            "article": article,
        }

    def __len__(self):
        return len(self.generated_pairs)


@hydra.main(config_path="../config", config_name="finetune_bert.yaml", version_base="1.1")
def main(config: DictConfig):
    repo_path = pyrootutils.find_root(search_from=__file__, indicator="pyproject.toml")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
    )

    queries = read_json(repo_path / config.data.collie.queries.split("../../")[1])
    articles = read_json(repo_path / config.data.collie.articles.split("../../")[1])

    dataset = CustomBERTDataset(
        tokenizer=tokenizer,
        query_articles_list=queries,
        articles_list=articles,
        config=config,
        with_negative_samples=True,
    )

    print(dataset[1])

    print(tokenizer.decode(dataset[1]["input_ids"]))


if __name__ == "__main__":
    main()
