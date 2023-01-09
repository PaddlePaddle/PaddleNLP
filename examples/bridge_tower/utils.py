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

import bisect
import warnings

import numpy as np
import paddle
from paddle.io import Dataset
from src.datasets import (
    CocoCaptionKarpathyDataset,
    ConceptualCaptionDataset,
    SBUCaptionDataset,
    SNLIDataset,
    VisualGenomeCaptionDataset,
    VQAv2Dataset,
)

from paddlenlp.transformers import AutoTokenizer


def get_dataset(_config):

    train_transform_keys = (
        ["default_train"] if len(_config["train_transform_keys"]) == 0 else _config["train_transform_keys"]
    )

    val_transform_keys = ["default_val"] if len(_config["val_transform_keys"]) == 0 else _config["val_transform_keys"]

    tokenizer_name = _config["tokenizer"]
    # tokenizer = get_pretrained_tokenizer(tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    debug_num = _config["debug_num"]
    # _config["data_root"]='/root/paddlejob/workspace/env_run/output/dataset/fine-tune'

    if _config["group_name"] == "vqa":
        train_dataset = VQAv2Dataset(
            _config["data_root"],
            train_transform_keys,
            split="train",
            image_size=_config["image_size"],
            max_text_len=_config["max_text_len"],
            draw_false_image=_config["draw_false_image"],
            draw_false_text=_config["draw_false_text"],
            image_only=_config["image_only"],
            tokenizer=tokenizer,
            debug_num=debug_num,
        )

        eval_dataset = VQAv2Dataset(
            _config["data_root"],
            val_transform_keys,
            split="val",
            image_size=_config["image_size"],
            max_text_len=_config["max_text_len"],
            draw_false_image=_config["draw_false_image"],
            draw_false_text=_config["draw_false_text"],
            image_only=_config["image_only"],
            tokenizer=tokenizer,
            debug_num=debug_num,
        )
    if _config["group_name"] == "snli":
        train_dataset = SNLIDataset(
            _config["data_root"],
            train_transform_keys,
            split="train",
            image_size=_config["image_size"],
            max_text_len=_config["max_text_len"],
            draw_false_image=_config["draw_false_image"],
            draw_false_text=_config["draw_false_text"],
            image_only=_config["image_only"],
            tokenizer=tokenizer,
            debug_num=debug_num,
        )

        eval_dataset = SNLIDataset(
            _config["data_root"],
            val_transform_keys,
            split="val",
            image_size=_config["image_size"],
            max_text_len=_config["max_text_len"],
            draw_false_image=_config["draw_false_image"],
            draw_false_text=_config["draw_false_text"],
            image_only=_config["image_only"],
            tokenizer=tokenizer,
            debug_num=debug_num,
        )

    return tokenizer, train_dataset, eval_dataset


def get_pretrained_dataset(_config):
    train_transform_keys = (
        ["default_train"] if len(_config["train_transform_keys"]) == 0 else _config["train_transform_keys"]
    )

    val_transform_keys = ["default_val"] if len(_config["val_transform_keys"]) == 0 else _config["val_transform_keys"]

    tokenizer_name = _config["tokenizer"]
    # tokenizer = get_pretrained_tokenizer(tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    debug_num = _config["debug_num"]

    if _config["group_name"] == "mlm_itm":
        list_train_datasets = []
        list_valid_datasets = []
        for dataset_name in _config["datasets"]:
            eval_dataset = None
            if dataset_name == "coco":
                train_dataset = CocoCaptionKarpathyDataset(
                    _config["data_root"],
                    train_transform_keys,
                    split="train",
                    image_size=_config["image_size"],
                    max_text_len=_config["max_text_len"],
                    draw_false_image=_config["draw_false_image"],
                    draw_false_text=_config["draw_false_text"],
                    image_only=_config["image_only"],
                    tokenizer=tokenizer,
                    debug_num=debug_num,
                )
                eval_dataset = CocoCaptionKarpathyDataset(
                    _config["data_root"],
                    val_transform_keys,
                    split="val",
                    image_size=_config["image_size"],
                    max_text_len=_config["max_text_len"],
                    draw_false_image=_config["draw_false_image"],
                    draw_false_text=_config["draw_false_text"],
                    image_only=_config["image_only"],
                    tokenizer=tokenizer,
                    debug_num=debug_num,
                )
            elif dataset_name == "gcc":
                train_dataset = ConceptualCaptionDataset(
                    _config["data_root"],
                    train_transform_keys,
                    split="train",
                    image_size=_config["image_size"],
                    max_text_len=_config["max_text_len"],
                    draw_false_image=_config["draw_false_image"],
                    draw_false_text=_config["draw_false_text"],
                    image_only=_config["image_only"],
                    tokenizer=tokenizer,
                    debug_num=debug_num,
                )
            elif dataset_name == "vg":
                train_dataset = VisualGenomeCaptionDataset(
                    _config["data_root"],
                    train_transform_keys,
                    split="train",
                    image_size=_config["image_size"],
                    max_text_len=_config["max_text_len"],
                    draw_false_image=_config["draw_false_image"],
                    draw_false_text=_config["draw_false_text"],
                    image_only=_config["image_only"],
                    tokenizer=tokenizer,
                    debug_num=debug_num,
                )

            elif dataset_name == "sbu":
                train_dataset = SBUCaptionDataset(
                    _config["data_root"],
                    train_transform_keys,
                    split="train",
                    image_size=_config["image_size"],
                    max_text_len=_config["max_text_len"],
                    draw_false_image=_config["draw_false_image"],
                    draw_false_text=_config["draw_false_text"],
                    image_only=_config["image_only"],
                    tokenizer=tokenizer,
                    debug_num=debug_num,
                )
            list_train_datasets.append(train_dataset)
            if eval_dataset is not None:
                list_valid_datasets.append(eval_dataset)

    train_dataset = ConcatDataset(list_train_datasets)
    eval_dataset = ConcatDataset(list_valid_datasets)
    return tokenizer, train_dataset, eval_dataset


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn(
            "cummulative_sizes attribute is renamed to " "cumulative_sizes", DeprecationWarning, stacklevel=2
        )
        return self.cumulative_sizes


def collate_fn(batch, mlm_collator):
    batch_size = len(batch)
    keys = set([key for b in batch for key in b.keys()])
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
    img_sizes = list()

    for img_key in img_keys:
        img = dict_batch[img_key]
        img_sizes += [ii.shape for i in img if i is not None for ii in i]

    for size in img_sizes:
        assert len(size) == 3, f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

    if len(img_keys) != 0:
        max_height = max([i[1] for i in img_sizes])
        max_width = max([i[2] for i in img_sizes])

    for img_key in img_keys:
        img = dict_batch[img_key]
        view_size = len(img[0])

        new_images = [paddle.zeros([batch_size, 3, max_height, max_width]) for _ in range(view_size)]

        for bi in range(batch_size):
            orig_batch = img[bi]
            for vi in range(view_size):
                if orig_batch is None:
                    new_images[vi][bi] = None
                else:
                    orig = img[bi][vi]
                    new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

        dict_batch[img_key] = new_images

    txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

    if len(txt_keys) != 0:
        # texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        # draw_text_len = len(encodings)
        flatten_encodings = [e for encoding in encodings for e in encoding]
        flatten_mlms = mlm_collator(flatten_encodings)

        for i, txt_key in enumerate(txt_keys):
            texts, encodings = (
                [d[0] for d in dict_batch[txt_key]],
                [d[1] for d in dict_batch[txt_key]],
            )

            mlm_ids, mlm_labels = (
                flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
            )

            input_ids = paddle.zeros_like(mlm_ids)
            attention_mask = paddle.zeros_like(mlm_ids)
            for _i, encoding in enumerate(encodings):
                _input_ids, _attention_mask = (
                    paddle.to_tensor(encoding["input_ids"]),
                    paddle.to_tensor(encoding["attention_mask"]),
                )
                input_ids[_i, : len(_input_ids)] = _input_ids
                attention_mask[_i, : len(_attention_mask)] = _attention_mask

            dict_batch[txt_key] = texts
            dict_batch[f"{txt_key}_ids"] = input_ids
            dict_batch[f"{txt_key}_labels"] = paddle.full_like(input_ids, -100)
            dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
            dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
            dict_batch[f"{txt_key}_masks"] = attention_mask
            if "labels" in dict_batch:
                dict_batch["labels"] = np.array(dict_batch["labels"])

    return dict_batch
