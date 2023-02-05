import json
import re
from pathlib import Path

from bs4 import BeautifulSoup


def extract_query_pairs(data: BeautifulSoup):
    query_articles_json = []

    for tag in data.find_all("pair"):
        article = tag.t1.text

        article_names = re.findall(r"Article \d+-?\d*\n", article)
        article_names = list(map(lambda x: x.strip(), article_names))

        query = tag.t2.text.replace("\n", "")

        query_articles_json.append({"query": query, "articles": article_names})

    return query_articles_json


def extract_article_content(data: list[str]):
    preprocessed_data = []

    for line in data:
        line = line.strip()

        if "  " in line:
            preprocessed_data.extend(line.split("  "))
            continue

        if line.startswith(("Section", "Chapter", "Civil Code", "Part")):
            continue

        # Removing meta-headers to laws like Standards for Construction
        line = re.sub(r"^\([^\divx].*\)", "", line)
        if line:
            preprocessed_data.append(line)

    data_json = []

    curr_article_name, curr_article_content = None, []

    for ii, line in enumerate(preprocessed_data):
        if line.startswith("Article"):
            if ii != 0:
                data_json.append(
                    {"article_name": curr_article_name, "article_content": "\n".join(curr_article_content)}
                )
                curr_article_name, curr_article_content = None, []

            curr_article_name = line

        else:
            curr_article_content.append(line)

    data_json.append({"article_name": curr_article_name, "article_content": "\n".join(curr_article_content)})

    return data_json


def main():
    query_xml_raw_path = Path("../../../data/raw/COLLIE/task3/COLIEE2022statute_data-English/train")

    print(f"Processing {query_xml_raw_path.__str__()} directory")
    query_data = []
    for xml_path in query_xml_raw_path.glob("**/*.xml"):
        data = BeautifulSoup(xml_path.read_bytes(), features="xml")

        result_data = extract_query_pairs(data)
        query_data.extend(result_data)

    print(f"    Found {len(query_data)} queries.")

    query_xml_preprocessed_path = Path("../../../data/preprocessed/COLLIE/query_article.json")
    with open(query_xml_preprocessed_path, "w") as f:
        json.dump(query_data, f)

    law_path = Path("../../../data/raw/COLLIE/task3/" "COLIEE2022statute_data-English/text/civil_code_en-1to724-2.txt")

    print(f"Processing {law_path.__str__()} file")
    with open(law_path, encoding="utf-8-sig") as f:
        law_data = f.readlines()

    law_data = extract_article_content(law_data)
    print(f"    Found {len(law_data)} laws.")

    law_preprocessed_path = Path("../../../data/preprocessed/COLLIE/articles.json")
    with open(law_preprocessed_path, "w") as f:
        json.dump(law_data, f)

    print("Finished processing COLLIE dataset!")

    print("Saved files")
    print(query_xml_preprocessed_path.resolve())
    print(law_preprocessed_path.resolve())


if __name__ == "__main__":
    main()
