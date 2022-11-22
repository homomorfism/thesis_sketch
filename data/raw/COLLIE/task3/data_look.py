import re
from pathlib import Path

import seaborn as sns
import xmltodict
from matplotlib import pyplot as plt

data = []
count = 0
train_path = Path("COLIEE2022statute_data-English/train")
for file_path in train_path.glob("**/*.xml"):
    print(file_path)

    # count += file_path.read_text().count("t2")

    train_data = xmltodict.parse(file_path.read_text())['dataset']['pair']
    # count += len(train_data)
    # for item in train_data:
    #     print(item)
    #     import sys; sys.exit(1)
    #     data.append(item)

print(count)


def count_number_rows(t1: str):
    return len(re.findall("Article \\d+-*\\d*[\\n\\s]", t1))


n_laws = []
for item in data:
    n = count_number_rows(item['t1'])
    if n == 0:
        print(item)

    n_laws.append(n)


sns.displot(n_laws)
plt.savefig("displot.png")
plt.show()
