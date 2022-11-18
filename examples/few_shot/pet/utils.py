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
import types
from functools import partial

import numpy as np

import paddle
from paddle import Tensor
from paddlenlp.datasets import load_dataset
from paddlenlp.prompt import ManualVerbalizer


def load_prompt_arguments(args):
    """
    Load prompt and label words according to prompt index.
    """
    with open(args.prompt_path, "r", encoding="utf-8") as fp:
        configs = json.load(fp)
        assert len(configs["verbalizer"]) == len(configs["template"])
        assert configs["verbalizer"][0] is not None
        verbalizer = [configs["verbalizer"][0]]
        last_verb_index = 0
        for index, verb in enumerate(configs["verbalizer"][1:]):
            if verb is None or len(verb) == 0:
                verbalizer.append(configs["verbalizer"][last_verb_index])
            else:
                verbalizer.append(verb)
                last_verb_index = index + 1
        args.prompt = configs["template"][args.prompt_index]["text"]
        label_words = configs["verbalizer"][args.prompt_index]
        if isinstance(label_words, list):
            label_words = {k: k for k in label_words}
        args.label_words = label_words
        return args


def maskedlm_wrapper(verbalizer):

    def process_outputs(self,
                        outputs: Tensor,
                        masked_positions: Tensor = None,
                        **kwargs):
        if masked_positions is None:
            return outputs
        batch_size, _, num_pred = outputs.shape
        outputs = outputs.reshape([-1, num_pred])
        outputs = paddle.gather(outputs, masked_positions)
        outputs = outputs.reshape([batch_size, -1, num_pred])
        return outputs

    verbalizer.eval_process_outputs = verbalizer.process_outputs
    verbalizer.process_outputs = types.MethodType(process_outputs, verbalizer)
    return verbalizer


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


def extend_with_data_augment(data_ds,
                             aug_type,
                             num_aug=10,
                             percent=0.1,
                             aug_base="mlm"):
    """
    Extend train dataset with augmentation.
    """
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
    elif aug_type == "generate":
        aug = SentenceSynonym(create_n, generate_n=create_n + 3)
    else:
        raise ValueError(
            "Unsupported data augment strategy `{}`".format(aug_type))

    aug_data = []
    for example in data_ds:
        text_a_aug = aug.augment(example["text_a"])
        for text in text_a_aug:
            new_example = example.copy()
            example["text_a"] = text
            aug_data.append(new_example)

        if "text_b" in example and example["text_b"] is not None:
            text_b_aug = aug.augment(example["text_b"])
            for text in text_b_aug:
                new_example = example.copy()
                example["text_b"] = text
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
            text = fragments[0] + "（" + cand + "）" + fragments[1]
            new_example = {
                "content_pre": fragments[0],
                "content_post": fragments[1],
                "idiom": cand
            }
            if label is not None:
                new_example["labels"] = int(index == label)
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
        new_example = {
            "text": "".join(text),
            "pronoun": pronoun,
            "entity": entity
        }
        if label is not None:
            new_example["labels"] = label
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
        example["labels"] = paddle.index_select(token_ids,
                                                paddle.to_tensor(
                                                    example["label_ids"]),
                                                axis=0)
    return example


def load_fewclue_dataset(args, verbalizer):
    """
    Load fewclue datasets and convert them to the standard format of PET.
    """
    split_id = args.split_id
    splits = [f"train_{split_id}", f"dev_{split_id}", f"test_public", "test"]
    if args.task_name == "cluewsc":
        train_ds, dev_ds, public_test_ds, test_ds = load_dataset(
            "fewclue", name=args.task_name, splits=splits)
        unlabeled_ds = None
    else:
        splits.append("unlabeled")
        train_ds, dev_ds, public_test_ds, test_ds, unlabeled_ds = load_dataset(
            "fewclue", name=args.task_name, splits=splits)
    data_ds = [train_ds, dev_ds, public_test_ds, test_ds, unlabeled_ds]

    # Preprocess data for mask prediction task.
    if args.task_name == "chid":
        for index, sub_data_ds in enumerate(data_ds):
            data_ds[index] = convert_chid(sub_data_ds)
    elif args.task_name == "cluewsc":
        for index, sub_data_ds in enumerate(data_ds):
            data_ds[index] = convert_cluewsc(sub_data_ds)
    else:
        if args.task_name in ("tnews", "iflytek"):
            orig_key = "label_des"
        else:
            orig_key = "label"
        convert_label = partial(convert_labels_to_ids,
                                orig_key=orig_key,
                                labels_to_ids=verbalizer.labels_to_ids)
        for index, sub_data_ds in enumerate(data_ds):
            if sub_data_ds is not None:
                data_ds[index] = sub_data_ds.map(convert_label)

    # Extend train dataset with data augmentation and pseudo-label data.
    data_ds[0] = extend_with_data_augment(data_ds[0], args.augment_type,
                                          args.num_augment,
                                          args.word_augment_percent,
                                          args.augment_method)
    data_ds[0] = extend_with_pseudo_data(data_ds[0], args.pseudo_data_path,
                                         verbalizer.labels_to_ids)

    dev_labels = [x["label_ids"] for x in data_ds[1]]
    test_labels = [x["label_ids"] for x in data_ds[2]]

    convert_fn = partial(convert_ids_to_words,
                         token_ids=verbalizer.token_ids[:, 0, :])
    data_ds[:3] = [x.map(convert_fn) for x in data_ds[:3]]

    return data_ds, (dev_labels, test_labels)


def combine_data_label_and_save(data_path, data_ds, label_preds, labels):
    """
    Combine unsupervised data and corresponding predicted labels and 
    save one example per line.
    """
    predictions = paddle.to_tensor(label_preds.predictions)
    predictions = paddle.nn.functional.softmax(predictions, axis=1).numpy()
    label_preds = np.argmax(predictions, axis=1)
    label_probs = np.max(predictions, axis=1)
    with open(data_path, "w") as fp:
        for index, example in enumerate(data_ds):
            example["labels"] = labels[label_preds[index]]
            example["prob"] = label_probs[index]
            fp.write(json.dumps(example, ensure_ascii=False) + "\n")
