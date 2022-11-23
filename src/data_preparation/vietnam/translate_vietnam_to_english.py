# pip install --upgrade google-cloud-translate
import json
import logging
import os

from google.cloud import translate
from tqdm import tqdm

logging.basicConfig(filename='translate.log', level=logging.DEBUG)


def set_tokens_for_api():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = \
        "/home/shamil/thesis_sketch/configs/sixth-impulse-365412-c93f2c3c0b95.json"

    project_id = "sixth-impulse-365412"
    parent = f"projects/{project_id}"

    os.environ['PROJECT_ID'] = parent


def translate_sentence(src_sentence: str, client: translate.TranslationServiceClient):
    response = client.translate_text(
        contents=[src_sentence],
        source_language_code='vi',
        target_language_code='en',
        parent=os.environ['PROJECT_ID']
    )

    return response.translations[0].translated_text


def translate_queries(
        queries: list[dict],
        client: translate.TranslationServiceClient) -> list[dict]:
    translated = []
    for item in tqdm(queries, desc='Translating queries...'):
        query = item['query']
        articles = item['articles']

        for element in articles:
            element['article_name'] = element['article_name'].lower().replace("điều", 'chapter')

        query_translated = translate_sentence(query, client)

        translated.append({
            'query': query_translated,
            'articles': articles
        })

    return translated


def translate_articles(
        articles: list[dict],
        client: translate.TranslationServiceClient
) -> list[dict]:
    translated = []
    for item in tqdm(articles, desc='Translating articles...'):
        try:
            item['article_content'] = translate_sentence(item['article_content'], client)
        except Exception as e:
            logging.warning(f"Can not translate {item}, e={e}, setting it to '' ")
        item['article_name'] = item['article_name'].lower().replace("điều", 'chapter')

        translated.append(item)

    return translated


def main():
    set_tokens_for_api()
    client = translate.TranslationServiceClient()

    query_path = "../../data/preprocessed/Vietnam/query_article.json"
    query_saving_path = "../../data/preprocessed/VietnamTranslated/query_article.json"

    with open(query_path) as f:
        query_json = json.load(f)

    query_translated_json = translate_queries(query_json, client)

    with open(query_saving_path, 'w') as f:
        json.dump(query_translated_json, f, ensure_ascii=False)

    print("Finished translating queries!")

    article_path = "../../data/preprocessed/Vietnam/articles.json"
    article_saving_path = "../../data/preprocessed/VietnamTranslated/articles.json"

    with open(article_path) as f:
        article_json = json.load(f)

    article_translated_json = translate_articles(article_json, client)

    with open(article_saving_path, 'w') as f:
        json.dump(article_translated_json, f, ensure_ascii=True)

    print("Finished translating articles!")


if __name__ == '__main__':
    main()
