import json
from pathlib import Path


def convert_vietnam_to_collie_format_dataset(query_articles: list[dict], articles: list[dict]):
    new_query_articles = []
    for query_item in query_articles:
        query = query_item['query']
        new_articles_names = []

        for document in query_item['articles']:
            new_articles_names.append(document['document_id'] + "_" + document['article_name'])

        new_query_articles.append({
            'query': query,
            'articles': new_articles_names
        })

    new_articles = []
    for article_item in articles:
        new_articles.append({
            'article_name': article_item['document_id'] + "_" + article_item['article_name'],
            'article_content': article_item['article_content']
        })

    return new_query_articles, new_articles


def main():
    query_article_preprocess_path = Path("../../../data/preprocessed/VietnamTranslated/query_article.json")
    with open(query_article_preprocess_path) as f:
        query_article_preprocess_data = json.load(f)

    article_preprocess_path = Path("../../../data/preprocessed/VietnamTranslated/articles.json")
    with open(article_preprocess_path) as f:
        article_preprocess_data = json.load(f)

    new_query_articles, new_articles = convert_vietnam_to_collie_format_dataset(
        query_article_preprocess_data,
        article_preprocess_data
    )

    query_article_saving_path = Path('../../../data/cleaned/Vietnam/query_article.json')
    query_article_saving_path.parent.mkdir(parents=True, exist_ok=True)
    with open(query_article_saving_path, 'w') as f:
        json.dump(new_query_articles, f, indent=4)

    article_saving_path = Path('../../../data/cleaned/Vietnam/articles.json')
    article_saving_path.parent.mkdir(parents=True, exist_ok=True)
    with open(article_saving_path, 'w') as f:
        json.dump(new_articles, f, indent=4)


if __name__ == '__main__':
    main()
