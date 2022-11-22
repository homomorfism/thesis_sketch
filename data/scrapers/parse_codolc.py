import subprocess
from pathlib import Path

from scrapy import Spider


class CodolcSpider(Spider):
    name = 'codolc'
    start_urls = ['https://codolc.com/faq/']
    allowed_domains = ['codolc.com']

    def __init__(self):
        super(CodolcSpider, self).__init__()

        cat_ids = [
            '19', '25', '4', '23', '22', '6', '21',
            '7', '16', '8', '24', '9', '20', '10', '11',
            '17', '18', '12', '13', '14', '15'
        ]
        url = self.start_urls[0]
        self.start_urls = list(map(lambda cat: url + '?CRFaq[catID]=' + cat, cat_ids))

    def parse(self, response, **kwargs):
        questions = response.xpath("//a[@class='accordion-toggle']/h4/span/text()").getall()
        answers = response.xpath("//div[@class='accordion-inner']")

        for question, answer in zip(questions, answers):
            answer_data = answer.xpath(".//*[normalize-space(text())]/text()").getall()
            answer_data = list(filter(
                lambda line: 'text-indent' not in line,
                answer_data
            ))  # cleaning css tags
            answer_data = "\n".join(answer_data)
            answer_data = answer_data.replace('\xa0', ' ').replace("\n", ' ')
            question = question.replace('\xa0', ' ').replace("\n", ' ')

            yield {
                'query': question,
                'answer': answer_data
            }


if __name__ == '__main__':
    output_path = Path("../raw/scraped_data_ru/codolc.csv")
    output_path.unlink(missing_ok=True)
    subprocess.run(["scrapy", "runspider", "parse_codolc.py", "-o", str(output_path), '--nolog'])

    print("Finished scraping codolc.com/faq !")
