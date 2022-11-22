import re
import subprocess
from pathlib import Path

import scrapy
from scrapy import Spider
from tqdm import tqdm

question_xpath = "/html/body/div[2]/div[3]/div[2]/div/div[2]/div[2]/ul/li/div[1]"
answer_xpath = "/html/body/div[2]/div[3]/div[2]/div/div[2]/div[2]/ul/li/div[2]//" \
               "*[normalize-space(text())]/text()"

re_pattern = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
spaces = re.compile('\xa0')


def clean_text(text):
    text = re.sub(re_pattern, '', text)
    text = re.sub(spaces, " ", text)
    text = re.sub("\n|\t|\r", " ", text)

    return text


class YaPravoSpider(Spider):
    name = 'yapravo'
    start_urls = ['https://yapravo.ru/chasto_zadavaemye_voprosy/']
    allowed_domains = ['yapravo.ru', 'https://yapravo.ru/']

    custom_settings = {
        'CONCURRENT_REQUESTS': 2,
        'DNS_TIMEOUT': 300,
        "HTTPCACHE_ENABLED": True,
        'AUTOTHROTTLE_ENABLED': True,
    }

    bar = tqdm(total=3556, desc="Parsing yapravo.ru")

    def parse(self, response, **kwargs):
        for ii, item in enumerate(response.xpath("//a[text()='Показать ответ']/@href").getall()):
            _, _, url = item.split("/", maxsplit=2)
            url = self.start_urls[0] + url

            yield scrapy.Request(url=url, callback=self.parse_page)

    def parse_page(self, response, **kwargs):  # noqa
        question = response.xpath(question_xpath).get() or ""
        question = clean_text(question)

        answer = response.xpath(answer_xpath).getall()
        answer = " ".join(answer)
        answer = clean_text(answer)

        self.bar.update()

        yield {
            'url': response.url,
            'query': question,
            'answer': answer
        }


if __name__ == '__main__':
    output_path = Path("../raw/scraped_data_ru/yapravo.csv")
    output_path.unlink(missing_ok=True)

    subprocess.run(['scrapy', 'runspider', 'parse_yapravo.py',
                    '-o', str(output_path), '-s', 'LOG_FILE=parse_yapravo.log'])
    print("Finished parsing yapravo.ru/chasto_zadavaemye_voprosy !")
