import os
import json
from functools import partial
from paddlenlp.datasets import load_dataset
from paddlenlp.prompt import InputExample


def load_local_dataset_qic(data_path, splits, label_list):
    """
    Read datasets from files.
    
    Args:
        data_path (str):
            Path to the dataset directory, including label.txt, train.txt, 
            dev.txt (and data.txt).
        splits (list):
            Which file(s) to load, such as ['train', 'dev', 'test'].
        task_type(str):
            It determines the formation of data.
            Support `multi-class`, `multi-label` and `hierachical`.
    """

    def _reader(data_file, label_list):
        with open(data_file, "r", encoding="utf-8") as fp:
            for idx, line in enumerate(fp):
                data = line.strip().split("\t")
                if len(data) == 1:
                    yield InputExample(text_a=data[0])
                else:
                    text, label = data
                    yield InputExample(text_a=text, labels=label_list[label])

    assert isinstance(splits, list) and len(splits) > 0

    split_map = {"train": "train.txt", "dev": "dev.txt", "test": "data.txt"}

    dataset = []
    for split in splits:
        data_file = os.path.join(data_path, split_map[split])
        dataset.append(
            load_dataset(_reader,
                         data_file=data_file,
                         label_list=label_list,
                         lazy=False))
    return dataset


def load_local_dataset(data_path, splits, label_list):

    def _reader(data_file, label_list):
        with open(data_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line.strip())
                yield InputExample(text_a=data["sentence"],
                                   labels=label_list[data["label_desc"]])

    assert isinstance(splits, list) and len(splits) > 0

    split_map = {
        "train": "train_0.json",
        "dev": "dev_0.json",
        "test": "test_public.json"
    }

    dataset = []
    for split in splits:
        data_file = os.path.join(data_path, split_map[split])
        dataset.append(
            load_dataset(_reader,
                         data_file=data_file,
                         label_list=label_list,
                         lazy=False))
    return dataset
