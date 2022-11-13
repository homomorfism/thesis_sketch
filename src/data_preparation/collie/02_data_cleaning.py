import json
import re
from pathlib import Path


def collie_clean(query_article: list[dict], articles: list[dict]):
    # removing articles which not in queries

    # used_articles = []
    # for item in query_article:
    #     used_articles.extend(item['articles'])
    #
    # used_articles = set(used_articles)
    #
    # cleaned_articles = []
    # for article_item in articles:
    #     if article_item['article_name'] in used_articles:
    #         cleaned_articles.append(article_item)
    #
    # articles = cleaned_articles

    # remove (1), (i), \n, " from articles
    for article_item in articles:
        text = article_item['article_content']
        text = re.sub(r"\(\w+\)", "", text)
        text = re.sub(r"\n", "", text)
        text = re.sub(r"\"", "", text)
        article_item['article_content'] = text

    for query_item in query_article:
        text = query_item['query']
        text = re.sub("\"", "", text)
        query_item['query'] = text

    # removing queries with empty article list
    new_query_article = []
    for query_item in query_article:
        if query_item['articles']:
            new_query_article.append(query_item)

    query_article = new_query_article

    return query_article, articles


def main():
    query_article_path = Path("../../../data/preprocessed/COLLIE/query_article.json")
    with open(query_article_path) as f:
        query_article = json.load(f)

    articles_path = Path("../../../data/preprocessed/COLLIE/articles.json")

    with open(articles_path) as f:
        articles = json.load(f)

    query_article, articles = collie_clean(query_article, articles)

    saving_dir = Path("../../../data/cleaned/COLLIE/")
    saving_dir.mkdir(exist_ok=True, parents=True)

    with open(saving_dir / "query_article.json", 'w') as f:
        json.dump(query_article, f)

    with open(saving_dir / 'articles.json', 'w') as f:
        json.dump(articles, f)

    print(f"Generated {len(query_article)} queries")
    print(f"Generated {len(articles)} articles")

    print("Finished, saved files to ...")
    print((saving_dir / "query_article.json").resolve())
    print((saving_dir / 'articles.json').resolve())


if __name__ == '__main__':
    main()
