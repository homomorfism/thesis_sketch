import json
from collections import Counter
from pathlib import Path


def convert_vietnam_to_collie_format_dataset(query_articles: list[dict], articles: list[dict]):
    new_query_articles = []
    for query_item in query_articles:
        query = query_item["query"]
        new_articles_names = []

        for document in query_item["articles"]:
            new_articles_names.append(document["document_id"] + "_" + document["article_name"])

        new_query_articles.append({"query": query, "articles": new_articles_names})

    new_articles = []
    for article_item in articles:
        new_articles.append(
            {
                "article_name": article_item["document_id"] + "_" + article_item["article_name"],
                "article_content": article_item["article_content"],
            }
        )

    return new_query_articles, new_articles


def replace_space_to_underscore_in_names(query_articles: list[dict], articles: list[dict]):
    new_query_articles = []

    for query_item in query_articles:
        new_query_articles.append(
            {"query": query_item["query"], "articles": list(map(lambda x: x.replace(" ", "_"), query_item["articles"]))}
        )

    new_articles = []
    for article_item in articles:
        new_articles.append(
            {
                "article_name": article_item["article_name"].replace(" ", "_"),
                "article_content": article_item["article_content"],
            }
        )

    return new_query_articles, new_articles


def drop_non_unique_articles(articles: list[dict]):
    counter = Counter([item["article_name"] for item in articles])

    new_articles = []
    for article in articles:
        article_name = article["article_name"]
        if counter[article_name] == 1:
            new_articles.append(article)

    return new_articles


def remove_article_names_not_in_queries(query_data: list[dict], article_data: list[dict]):
    article_names = {item["article_name"] for item in article_data}

    new_query_data = []
    for item in query_data:
        new_articles = []
        for article_name in item["articles"]:
            if article_name in article_names:
                new_articles.append(article_name)
            else:
                print(f"Not found: {article_name}")

        new_query_data.append({"query": item["query"], "articles": new_articles})

    return new_query_data, article_data


def remove_queries_with_no_answers(query_articles):
    new_query_article = []
    for query_item in query_articles:
        if query_item["articles"]:
            new_query_article.append(query_item)

    return new_query_article


def main():
    query_article_preprocess_path = Path("../../../data/preprocessed/VietnamTranslated/query_article.json")
    with open(query_article_preprocess_path) as f:
        query_article_preprocess_data = json.load(f)

    article_preprocess_path = Path("../../../data/preprocessed/VietnamTranslated/articles.json")
    with open(article_preprocess_path) as f:
        article_preprocess_data = json.load(f)

    new_query_articles, new_articles = convert_vietnam_to_collie_format_dataset(
        query_article_preprocess_data, article_preprocess_data
    )

    new_query_articles, new_articles = replace_space_to_underscore_in_names(new_query_articles, new_articles)

    new_articles = drop_non_unique_articles(new_articles)

    new_query_articles, new_articles = remove_article_names_not_in_queries(new_query_articles, new_articles)

    new_query_articles = remove_queries_with_no_answers(new_query_articles)

    query_article_saving_path = Path("../../../data/cleaned/Vietnam/query_article.json")
    query_article_saving_path.parent.mkdir(parents=True, exist_ok=True)
    with open(query_article_saving_path, "w") as f:
        json.dump(new_query_articles, f, indent=4)

    article_saving_path = Path("../../../data/cleaned/Vietnam/articles.json")
    article_saving_path.parent.mkdir(parents=True, exist_ok=True)
    with open(article_saving_path, "w") as f:
        json.dump(new_articles, f, indent=4)

    print("Finished!")


if __name__ == "__main__":
    main()
