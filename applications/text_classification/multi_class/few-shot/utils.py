import os
from functools import partial
from paddlenlp.datasets import load_dataset
from paddlenlp.prompt import InputExample


def load_local_dataset(data_path, splits, task_type="multi-label"):
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

    def text_reader(data_file):
        with open(data_file, "r", encoding="utf-8") as fp:
            for idx, line in enumerate(fp):
                yield InputExample(uid=idx, text_a=line.strip())

    def multi_class_reader(data_file):
        with open(data_file, "r", encoding="utf-8") as fp:
            for idx, line in enumerate(fp):
                text, label = line.strip().split("\t")
                yield InputExample(uid=idx,
                                   text_a=text,
                                   text_b=None,
                                   labels=label)

    def multi_label_reader(data_file):
        with open(data_file, "r", encoding="utf-8") as fp:
            for idx, line in enumerate(fp):
                text, label = line.strip().split("\t")
                label = label.strip().split(",")
                yield InputExample(uid=idx,
                                   text_a=text,
                                   text_b=None,
                                   labels=label)

    def hierachical_reader(data_file, label_list):
        with open(data_file, "r", encoding="utf-8") as fp:
            for line in fp:
                data = line.strip().split("\t")
                depth = len(data) - 1
                labels = [x.strip().split(",") for x in data[1:]]
                shape = [len(layer) for layer in layers]
                offsets = [0] * len(shape)
                has_next = True
                labels = []
                while has_next:
                    l = ''
                    for i, off in enumerate(offsets):
                        if l == '':
                            l = layers[i][off]
                        else:
                            l += '##{}'.format(layers[i][off])
                        if l in label_list and label_list[l] not in labels:
                            labels.append(label_list[l])
                    for i in range(len(shape) - 1, -1, -1):
                        if offsets[i] + 1 >= shape[i]:
                            offsets[i] = 0
                            if i == 0:
                                has_next = False
                        else:
                            offsets[i] += 1
                            break
                yield InputExample(uid=idx,
                                   text_a=data[0],
                                   text_b=None,
                                   labels=labels)

    assert isinstance(splits, list) and len(splits) > 0

    label_file = os.path.join(data_path, "label.txt")
    with open(label_file, "r", encoding="utf-8") as fp:
        label_list = [x.strip() for x in fp.readlines()]

    reader_map = {
        "multi-class": multi_class_reader,
        "multi-label": multi_label_reader,
        "hierachical": partial(hierachical_reader, label_list=label_list)
    }

    try:
        reader = reader_map[task_type]
    except KeyError:
        raise ValueError(f"Unspported task type {task_type}.")

    split_map = {"train": "train.txt", "dev": "dev.txt", "test": "data.txt"}

    dataset = []
    for split in splits:
        data_file = os.path.join(data_path, split_map[split])
        if split == "test":
            dataset.append(
                load_dataset(text_reader, data_file=data_file, lazy=False))
            continue
        dataset.append(load_dataset(reader, data_file=data_file, lazy=False))

    return dataset
