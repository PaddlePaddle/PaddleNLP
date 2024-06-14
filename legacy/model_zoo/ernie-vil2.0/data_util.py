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
import json
import logging
import os
from io import BytesIO
from math import ceil

import numpy as np
import paddle
import pyarrow as pa
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


class ArrowDataset(Dataset):
    def __init__(
        self,
        arrow_path,
        split="val",
        max_txt_length=64,
        use_augment=False,
        resolution=224,
        tokenizer=None,
        text_column_name="caption",
    ):
        self.arrow_path = arrow_path
        # Assert Arrow directories exist
        print(os.path.join(arrow_path, split + ".arrow"))
        assert os.path.exists(
            os.path.join(arrow_path, split + ".arrow")
        ), "The arrow directory {} of {} split does not exist!".format(arrow_path, split)
        arrow_split_path = os.path.join(arrow_path, split + ".arrow")
        self.df = pa.ipc.open_file(arrow_split_path).read_pandas()
        # Fetch number of pairs and images
        self.number_samples = len(self.df)
        self.number_images = self.number_samples
        logging.info(
            "{} Arrow file contains {} images and {} pairs.".format(split, self.number_images, self.number_samples)
        )

        super(ArrowDataset, self).__init__()

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

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        sample_index = index % self.number_samples
        data_raw = self.df.iloc[sample_index, :]
        txt_raw = data_raw["caption"]
        image_raw = data_raw["image"]
        image_bytes = BytesIO(image_raw)
        image_bytes.seek(0)
        image = Image.open(image_bytes)
        image = self.transform(image)
        texts = self.tokenizer(
            [_preprocess_text(txt_raw)], max_length=self.max_txt_length, truncation=True, padding="max_length"
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
    arrow_imgs = args.image_data
    dataset = EvalImgDataset(arrow_imgs, resolution=224)
    return dataset


def get_train_eval_dataset(args, epoch_id=0, max_txt_length=64, tokenizer=None):
    train_dataset = ArrowDataset(
        args.train_data,
        split="train",
        max_txt_length=max_txt_length,
        tokenizer=tokenizer,
        use_augment=True,
        resolution=224,
    )
    eval_dataset = ArrowDataset(
        args.val_data,
        split="valid",
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
    def __init__(self, filename, max_txt_length=24, tokenizer=None):
        assert os.path.exists(filename), "The annotation datafile {} not exists!".format(filename)
        jsonl_filename = filename
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
    def __init__(self, arrow_imgs_filename, resolution=224):
        assert os.path.isfile(arrow_imgs_filename), "The image arrow filename {} not exists!".format(
            arrow_imgs_filename
        )

        logging.debug(f"Loading image arrow from {arrow_imgs_filename}.")
        self.img_df = pa.ipc.open_file(arrow_imgs_filename).read_pandas()
        self.number_images = len(self.img_df)
        self.transform = self._build_transform(resolution)
        logging.info("The specified arrow directory contains {} images.".format(self.number_images))

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
        img_raw, img_id = self.img_df.iloc[idx]["image"], self.img_df.iloc[idx]["image_id"]
        image_bytes = BytesIO(img_raw)
        image_bytes.seek(0)
        image = Image.open(image_bytes)
        image = self.transform(image)
        if img_id.isnumeric():
            img_id = int(img_id)

        return img_id, image
