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

import numpy as np

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


def convert_efl(data_ds, label_words, orig_key, is_train=False, num_neg=5):
    efl_data_ds = []
    label_list = sorted(label_words.keys())
    for example in data_ds:
        label = label_words[example[orig_key]] if orig_key in example else None
        sub_list = label_list
        if is_train and label is not None and len(label_list) > num_neg:
            rand_index = np.random.permutation(len(label_list))
            sub_list = [example[orig_key]] + [label_list[i] for i in rand_index[:num_neg]]
        for key in sub_list:
            new_example = example.copy()
            cand = label_words[key]
            new_example["candidate_label"] = cand
            if label is not None:
                new_example["labels"] = int(cand == label)
            efl_data_ds.append(new_example)
    return MapDataset(efl_data_ds)


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


def load_fewclue_dataset(args, verbalizer):
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

    # Preprocess data for EFL.
    if args.task_name == "chid":
        for index, sub_data_ds in enumerate(data_ds):
            data_ds[index] = convert_chid(sub_data_ds)
    elif args.task_name == "cluewsc":
        for index, sub_data_ds in enumerate(data_ds[:-1]):
            data_ds[index] = convert_cluewsc(sub_data_ds)

    orig_key = "label"
    if args.task_name == "tnews":
        orig_key = "label_desc"
    elif args.task_name == "iflytek":
        orig_key = "label_des"
    for index, sub_data_ds in enumerate(data_ds):
        is_train = index == 0
        if sub_data_ds is not None:
            data_ds[index] = convert_efl(sub_data_ds, args.label_words, orig_key, is_train)

    # Extend train dataset with pseudo-label data.
    data_ds[0] = extend_with_pseudo_data(data_ds[0], args.pseudo_data_path, verbalizer.labels_to_ids)

    return data_ds
