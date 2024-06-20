# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from functools import partial

import paddle

from paddlenlp.dataaug import WordDelete, WordInsert, WordSubstitute, WordSwap
from paddlenlp.datasets import MapDataset, load_dataset


def extend_with_pseudo_data(data_ds, pseudo_path, labels_to_ids):
    """
    Extend train dataset with pseudo labeled examples if exists.
    """
    if pseudo_path is None:
        return data_ds
    with open(pseudo_path, "r", encoding="utf-8") as fp:
        pseudo_data = [json.loads(x.strip()) for x in fp]
    data_ds = MapDataset([x for x in data_ds] + pseudo_data)
    return data_ds


def extend_with_data_augment(data_ds, aug_type, num_aug=10, percent=0.1, aug_base="mlm", example_keys=None):
    """
    Extend train dataset with augmentation.
    """
    if example_keys is None:
        return data_ds
    if aug_type is None or aug_type == "None":
        return data_ds
    if aug_type == "delete":
        aug = WordDelete(create_n=num_aug, aug_percent=percent)
    elif aug_type == "substitute":
        aug = WordSubstitute(aug_base, create_n=num_aug, aug_percent=percent)
    elif aug_type == "insert":
        aug = WordInsert(aug_base, create_n=num_aug, aug_percent=percent)
    elif aug_type == "swap":
        aug = WordSwap(create_n=num_aug, aug_percent=percent)
    else:
        raise ValueError("Unsupported data augment strategy `{}`".format(aug_type))

    aug_data = []
    for example in data_ds:
        for key in example_keys:
            text_aug = aug.augment(example[key])
            for text in text_aug:
                new_example = example.copy()
                example[key] = text
                aug_data.append(new_example)

    data_ds = MapDataset([x for x in data_ds] + aug_data)
    return data_ds


def convert_chid(data_ds):
    """
    Insert idioms into positions of `#idiom#` so that the task is converted
    to binary classification.
    """
    split_data_ds = []
    for example in data_ds:
        fragments = example["content"].split("#idiom#")
        label = example.get("answer", None)
        for index, cand in enumerate(example["candidates"]):
            new_example = {"content_pre": fragments[0], "content_post": fragments[1], "idiom": cand}
            if label is not None:
                new_example["label"] = str(int(index == label))
            split_data_ds.append(new_example)
    return MapDataset(split_data_ds)


def convert_csl(data_ds):
    """
    Concatanate keywords and it can be replaced by keyword `options` in develop versioin.
    """
    concat_data_ds = []
    for example in data_ds:
        example["keyword"] = "，".join(example["keyword"])
        concat_data_ds.append(example)
    return MapDataset(concat_data_ds)


def convert_cluewsc(data_ds):
    """
    Mark the pronoun and entity with special tokens.
    """
    marked_data_ds = []
    for example in data_ds:
        target, text = example["target"], list(example["text"])
        pronoun, p_index = target["span2_text"], target["span2_index"]
        entity, e_index = target["span1_text"], target["span1_index"]
        label = example.get("label", None)
        if p_index > e_index:
            text.insert(p_index, "_")
            text.insert(p_index + len(pronoun) + 1, "_")
            text.insert(e_index, "[")
            text.insert(e_index + len(entity) + 1, "]")
        else:
            text.insert(e_index, "[")
            text.insert(e_index + len(entity) + 1, "]")
            text.insert(p_index, "_")
            text.insert(p_index + len(pronoun) + 1, "_")
        new_example = {"text": "".join(text), "pronoun": pronoun, "entity": entity}
        if label is not None:
            new_example["label"] = label
        marked_data_ds.append(new_example)
    return MapDataset(marked_data_ds)


def convert_labels_to_ids(example, orig_key, labels_to_ids):
    """
    Convert the keyword in datasets to `labels`.
    """
    if orig_key in example:
        example["label_ids"] = labels_to_ids[example.pop(orig_key)]
    return example


def convert_ids_to_words(example, token_ids):
    """
    Convert label id to the first word in mapping from labels to words,
    the length of which should coincide with that of `mask` in prompt.
    """
    if "label_ids" in example:
        labels = paddle.index_select(token_ids, paddle.to_tensor(example.pop("label_ids")), axis=0).squeeze(0)
        example["labels"] = labels
    return example


def load_fewclue_dataset(args, verbalizer, example_keys=None):
    """
    Load fewclue datasets and convert them to the standard format of PET.
    """
    split_id = args.split_id
    splits = [f"train_{split_id}", f"dev_{split_id}", "test_public", "test"]
    if args.task_name == "cluewsc":
        train_ds, dev_ds, public_test_ds, test_ds = load_dataset("fewclue", name=args.task_name, splits=splits)
        unlabeled_ds = None
    else:
        splits.append("unlabeled")
        train_ds, dev_ds, public_test_ds, test_ds, unlabeled_ds = load_dataset(
            "fewclue", name=args.task_name, splits=splits
        )
    data_ds = [train_ds, dev_ds, public_test_ds, test_ds, unlabeled_ds]

    # Preprocess data for mask prediction task.
    if args.task_name == "chid":
        for index, sub_data_ds in enumerate(data_ds):
            data_ds[index] = convert_chid(sub_data_ds)
    elif args.task_name == "cluewsc":
        for index, sub_data_ds in enumerate(data_ds[:-1]):
            data_ds[index] = convert_cluewsc(sub_data_ds)
    elif args.task_name == "csl":
        for index, sub_data_ds in enumerate(data_ds):
            data_ds[index] = convert_csl(sub_data_ds)
    orig_key = "label"
    if args.task_name == "tnews":
        orig_key = "label_desc"
    elif args.task_name == "iflytek":
        orig_key = "label_des"
    convert_label = partial(convert_labels_to_ids, orig_key=orig_key, labels_to_ids=verbalizer.labels_to_ids)
    for index, sub_data_ds in enumerate(data_ds):
        if sub_data_ds is not None:
            data_ds[index] = sub_data_ds.map(convert_label)

    # Extend train dataset with data augmentation and pseudo-label data.
    data_ds[0] = extend_with_data_augment(
        data_ds[0], args.augment_type, args.num_augment, args.word_augment_percent, args.augment_method, example_keys
    )
    data_ds[0] = extend_with_pseudo_data(data_ds[0], args.pseudo_data_path, verbalizer.labels_to_ids)

    dev_labels = [x["label_ids"] for x in data_ds[1]]
    test_labels = [x["label_ids"] for x in data_ds[2]]

    convert_fn = partial(convert_ids_to_words, token_ids=verbalizer.token_ids[:, 0, :])
    data_ds[:3] = [x.map(convert_fn) for x in data_ds[:3]]

    return data_ds, (dev_labels, test_labels)
