import paddlenlp
from datasets import load_dataset


train_set, dev_set, test_set = load_dataset("chnsenticorp", split=["train", "validation", "test"])

print(len(train_set))
print(len(dev_set))
print(len(test_set))

for idx, example in enumerate(test_set):
    if idx < 10:
        print(example)

