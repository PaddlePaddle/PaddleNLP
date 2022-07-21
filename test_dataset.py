import paddlenlp
from datasets import load_dataset


data_set = load_dataset("cote", "mfw")
# print(type(train_set))
print(data_set)

for idx, example in enumerate(data_set["train"]):
    if idx < 10:
        print(example)