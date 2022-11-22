import json
from pathlib import Path

import pandas as pd

pd.set_option('display.max_columns', None)


def extract_query(query_df: pd.DataFrame):
    data_json = []
    for _, row in query_df.iterrows():
        query = row.query

        article_list = []
        for document in row.documents:
            if 'articles' not in document.keys():
                continue

            articles = [{
                'document_id': document['name'],
                'article_name': item['name']
            } for item in document['articles']]
            article_list.extend(articles)

        data_json.append({
            'query': query,
            'articles': article_list
        })

    return data_json


def extract_article_content(data: list[str]):
    data_json = []

    for line in data:
        row = json.loads(line)
        document_id = row['so_hieu']

        for article_data in row['cac_dieu']:
            article_name = article_data['ten_dieu']
            article_content = article_data['noi_dung']

            data_json.append({
                'document_id': document_id,
                'article_name': article_name,
                'article_content': article_content
            })

    return data_json


def main():
    query_path = Path("../../../data/raw/VietnamQA/ground_truth.jsonl")
    query_df = pd.read_json(query_path, lines=True)

    print(f"Processing {query_path} file.")
    query_processed = extract_query(query_df)
    print(f"    Found {len(query_processed)} queries")

    query_processed_output_path = Path("../../../data/preprocessed/Vietnam/query_article.json")
    with open(query_processed_output_path, 'w') as f:
        json.dump(query_processed, f, ensure_ascii=False)

    articles_path = Path("../../../data/raw/VietnamQA/original_2020_12_22.jsonl")
    with open(articles_path) as f:
        articles_data = f.readlines()

    print(f"Processing {articles_path.name} file.")
    articles_preprocessed = extract_article_content(articles_data)

    print(f"    Found: {len(articles_preprocessed)} articles.")

    articles_preprocessed_output_path = Path("../../../data/preprocessed/Vietnam/articles.json")
    with open(articles_preprocessed_output_path, 'w') as f:
        json.dump(articles_preprocessed, f, ensure_ascii=False)

    print("Finished processing Vietnam dataset")


if __name__ == '__main__':
    main()
