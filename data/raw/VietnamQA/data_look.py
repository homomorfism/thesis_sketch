import pprint

import pandas as pd

# pd.set_option('display.max_columns', None)
# pd.set_option('max_colwidth', None)

# ground_truth = pd.read_json("ground_truth.jsonl", lines=True)
# print(ground_truth.sample(4))


original = pd.read_json("original_2020_12_22.jsonl", lines=True)
# sample = original[original.so_hieu == '28/1999/NĐ-CP']
# print(original.head(1))
# original = original.iloc[10:11]
# print(json.dumps(original.iloc[0].to_json(), indent=2))
# print(original.columns)
# print(original['cac_dieu'])

#
sample = original.iloc[60]
data = sample.to_dict()

pp = pprint.PrettyPrinter(indent=2)

pp.pprint(data)
