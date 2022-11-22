import subprocess
from pathlib import Path

import scrapy
from bs4 import BeautifulSoup


class MiralsSpider(scrapy.Spider):
    name = 'mirals'
    start_urls = ['https://mirals.ru/voprosy-yuristam']
    allowed_domains = ['mirals.ru', 'https://mirals.ru/']

    def parse(self, response, **kwargs):
        for item in response.xpath("//ul[@class='page-subpages']/li/a"):
            uri = item.xpath("@href").get()
            uri = f"https://mirals.ru{uri}"

            yield scrapy.Request(uri, callback=self.parse_item)

    def parse_item(self, response, **kwargs):
        question = response.xpath("/html/body/div[2]/div[1]/div[2]/h1/text()").get()
        question = question.replace("\xa0", " ")

        answers = []
        for item in response.xpath('//p[@style="text-align: justify;"]'):
            soap = BeautifulSoup(item.get())
            answers.append(soap.get_text())

        answer_str = "\n".join(answers)
        answer_str = answer_str.replace("\xa0", " ")
        answer_str = answer_str.replace('Отвечает юрист Юридической компании "Миралс":', "")

        return {
            'query': question,
            'answer': answer_str
        }


if __name__ == '__main__':
    output_path = Path("../raw/scraped_data_ru/mirals.csv")
    output_path.unlink(missing_ok=True)
    subprocess.run(["scrapy", "runspider", "parse_mirals.py", "-o", str(output_path), '--nolog'])

    print("Finished scraping mirals.ru !")
