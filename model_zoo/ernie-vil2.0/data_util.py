# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import base64
import json
import logging
import os
import pickle
from io import BytesIO
from math import ceil

import lmdb
import numpy as np
import paddle
from paddle.io import Dataset
from paddle.vision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from PIL import Image


def _convert_to_rgb(image):
    return image.convert("RGB")


def _preprocess_text(text):
    # Adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", '"').replace("”", '"')
    return text


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, split="val", max_txt_length=64, use_augment=False, resolution=224, tokenizer=None):
        self.lmdb_path = lmdb_path

        # Assert LMDB directories exist
        assert os.path.isdir(lmdb_path), "The LMDB directory {} of {} split does not exist!".format(lmdb_path, split)
        lmdb_pairs = os.path.join(lmdb_path, "pairs")
        assert os.path.isdir(lmdb_pairs), "The LMDB directory {} of {} image-text pairs does not exist!".format(
            lmdb_pairs, split
        )
        lmdb_imgs = os.path.join(lmdb_path, "imgs")
        assert os.path.isdir(lmdb_imgs), "The LMDB directory {} of {} image base64 strings does not exist!".format(
            lmdb_imgs, split
        )

        # Open LMDB files
        self.env_pairs = lmdb.open(lmdb_pairs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_pairs = self.env_pairs.begin(buffers=True)
        self.env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_imgs = self.env_imgs.begin(buffers=True)

        # Fetch number of pairs and images
        self.number_samples = int(self.txn_pairs.get(key=b"num_samples").tobytes().decode("utf-8"))
        self.number_images = int(self.txn_imgs.get(key=b"num_images").tobytes().decode("utf-8"))
        logging.info(
            "{} LMDB file contains {} images and {} pairs.".format(split, self.number_images, self.number_samples)
        )

        super(LMDBDataset, self).__init__()

        # The self.dataset_len will be edited to a larger value by calling pad_dataset()
        self.dataset_len = self.number_samples
        self.global_batch_size = 1  # Will be modified to the exact global_batch_size after calling pad_dataset()

        self.split = split
        self.max_txt_length = max_txt_length

        self.use_augment = use_augment
        self.transform = self._build_transform(resolution)
        self.tokenizer = tokenizer

    def _build_transform(self, resolution):
        if self.split == "train" and self.use_augment:
            transform = Compose(
                [
                    RandomResizedCrop(resolution, scale=(0.9, 1.0), interpolation="bicubic"),
                    RandomHorizontalFlip(0.5),
                    _convert_to_rgb,
                    ToTensor(),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            transform = Compose(
                [
                    Resize((resolution, resolution), interpolation="bicubic"),
                    _convert_to_rgb,
                    ToTensor(),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        return transform

    def __del__(self):
        if hasattr(self, "env_pairs"):
            self.env_pairs.close()
        if hasattr(self, "env_imgs"):
            self.env_imgs.close()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        sample_index = index % self.number_samples

        pair = pickle.loads(self.txn_pairs.get("{}".format(sample_index).encode("utf-8")).tobytes())
        image_id, text_id, raw_text = pair

        image_b64 = self.txn_imgs.get("{}".format(image_id).encode("utf-8")).tobytes()
        image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))  # already resized
        image = self.transform(image)
        texts = self.tokenizer(
            [_preprocess_text(raw_text)], max_seq_len=self.max_txt_length, truncation=True, padding="max_length"
        )
        text = texts["input_ids"][0]

        eos_index = text.index(self.tokenizer.vocab["[SEP]"])
        eos_index = np.array(eos_index)
        return {"pixel_values": image, "input_ids": text, "index": eos_index}


def pad_dataset(dataset, global_batch_size):
    # Edit dataset.__len__() of the dataset
    dataset.dataset_len = ceil(dataset.dataset_len / global_batch_size) * global_batch_size
    dataset.global_batch_size = global_batch_size


def get_eval_txt_dataset(args, max_txt_length=24, tokenizer=None):
    input_filename = args.text_data
    dataset = EvalTxtDataset(input_filename, max_txt_length=max_txt_length, tokenizer=tokenizer)
    return dataset


def get_eval_img_dataset(args):
    lmdb_imgs = args.image_data
    dataset = EvalImgDataset(lmdb_imgs, resolution=224)
    return dataset


def get_train_eval_dataset(args, epoch_id=0, max_txt_length=64, tokenizer=None):
    train_dataset = LMDBDataset(
        args.train_data,
        split="train",
        max_txt_length=max_txt_length,
        tokenizer=tokenizer,
        use_augment=True,
        resolution=224,
    )
    eval_dataset = LMDBDataset(
        args.val_data,
        split="val",
        max_txt_length=max_txt_length,
        tokenizer=tokenizer,
        use_augment=False,
        resolution=224,
    )
    return train_dataset, eval_dataset


def create_dataloader(dataset, mode="train", batch_size=1, num_workers=1, batchify_fn=None, trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == "train" else False
    if mode == "train":
        batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=batchify_fn, return_list=True
    )


class EvalTxtDataset(Dataset):
    def __init__(self, jsonl_filename, max_txt_length=24, tokenizer=None):
        assert os.path.exists(jsonl_filename), "The annotation datafile {} not exists!".format(jsonl_filename)

        logging.debug(f"Loading jsonl data from {jsonl_filename}.")
        self.texts = []
        with open(jsonl_filename, "r", encoding="utf-8") as fin:
            for line in fin:
                obj = json.loads(line.strip())
                text_id = obj["text_id"]
                text = obj["text"]
                self.texts.append((text_id, text))
        logging.debug(f"Finished loading jsonl data from {jsonl_filename}.")

        self.max_txt_length = max_txt_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_id, text = self.texts[idx]
        texts = self.tokenizer([_preprocess_text(str(text))], max_len=self.max_txt_length, padding="max_length")
        text = texts["input_ids"][0]
        return {"text_id": text_id, "input_ids": text}


class EvalImgDataset(Dataset):
    def __init__(self, lmdb_imgs, resolution=224):
        assert os.path.isdir(lmdb_imgs), "The image LMDB directory {} not exists!".format(lmdb_imgs)

        logging.debug(f"Loading image LMDB from {lmdb_imgs}.")

        self.env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_imgs = self.env_imgs.begin(buffers=True)
        self.cursor_imgs = self.txn_imgs.cursor()
        self.iter_imgs = iter(self.cursor_imgs)
        self.number_images = int(self.txn_imgs.get(key=b"num_images").tobytes().decode("utf-8"))
        logging.info("The specified LMDB directory contains {} images.".format(self.number_images))

        self.transform = self._build_transform(resolution)

    def _build_transform(self, resolution):
        normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return Compose(
            [
                Resize((resolution, resolution), interpolation="bicubic"),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ]
        )

    def __len__(self):
        return self.number_images

    def __getitem__(self, idx):
        img_id, image_b64 = next(self.iter_imgs)
        if img_id == b"num_images":
            img_id, image_b64 = next(self.iter_imgs)

        img_id = img_id.tobytes()
        image_b64 = image_b64.tobytes()

        img_id = int(img_id.decode(encoding="utf8", errors="ignore"))
        image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))  # already resized
        image = self.transform(image)

        return img_id, image
