# Тут будут описаны проведенные эксперименты (с кодом их запуска) и полученные результаты.

Как сконвертировать json в markdown-table: [link](https://kdelmonte.github.io/json-to-markdown-table/).

## BM25

Как запустить:

```bash
make connect
make train_bm25
```

Результаты:

| dataset_name | f_score | ndcg_at_2 | ndcg_at_3 | ndcg_at_5 | ndcg_at_10 | recall_at_2 | recall_at_3 | recall_at_5 | recall_at_10 |
|--------------|---------|-----------|-----------|-----------|------------|-------------|-------------|-------------|--------------|
| collie       | 0.4792  | 0.5328    | 0.5477    | 0.5626    | 0.5842     | 0.5456      | 0.5823      | 0.6162      | 0.6779       |
| vietnam      | 0.0940  | 0.1061    | 0.1265    | 0.1489    | 0.1741     | 0.1111      | 0.1540      | 0.2062      | 0.2788       |

## Tf-Idf


## BERT


Command to run training: 
```bash
poetry run python src/pipelines/bert/train.py --config-name finetune_bert.yaml
```


## ColBERT

##       

##       