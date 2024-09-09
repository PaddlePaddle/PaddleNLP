# -*- coding: utf-8 -*-

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

"""
This script serializes images and image-text pair annotations into arrow files,
which supports more convenient dataset loading and random access to samples during training
compared with TSV and Jsonl data files.
"""
import argparse
import base64
import json
import logging
import os
import random
from collections import defaultdict
from glob import glob

import jsonlines
import pandas as pd
import pyarrow as pa
from tqdm import tqdm


def id2rest(photo_id, iid2photo, iid2captions):
    captions = iid2captions[photo_id]
    photo_data = iid2photo[photo_id]
    photo_data = photo_data.encode("utf-8")
    photo_data = base64.urlsafe_b64decode(photo_data)
    return [photo_data, captions, photo_id]


def path2rest(path, iid2captions, iid2photo):
    name = path.split("/")[-1]
    with open(path, "rb") as fp:
        binary = fp.read()
    text_index = iid2photo[name.split("/")[-1]]
    bs_item = []
    for index in text_index:
        bs_item.append([binary, iid2captions[index], name.split("/")[-1]])
    return bs_item


def valid_img(path, iid2photo):
    name = path.split("/")[-1]
    with open(path, "rb") as fp:
        binary = fp.read()
    return [binary, name.split("/")[-1]]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="the directory which stores the image tsvfiles and the text jsonl annotations",
)
parser.add_argument(
    "--splits",
    type=str,
    required=True,
    help="specify the dataset splits which this script processes, concatenated by comma \ (e.g. train,valid,test)",
)
parser.add_argument(
    "--data_out_dir",
    type=str,
    default=None,
    help="specify the directory which stores the output arrow files. If set to None, the arrow_dir will be set to args.data_dir/arrow",
)
parser.add_argument("--t2i_type", type=str, default="jsonl", help="the type of text2photo filename")
parser.add_argument(
    "--image_dir",
    type=str,
    required=True,
    help="the directory which stores the images['png','jpg','JPG']",
)
args = parser.parse_args()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %p",
    level=10,
)
# yapf: enable
if __name__ == "__main__":
    assert os.path.isdir(args.data_dir), "The data_dir does not exist! Please check the input args..."
    specified_splits = list(set(args.splits.strip().split(",")))  # train test dev
    specified_type = args.t2i_type
    print("Dataset splits to be processed: {}".format(", ".join(specified_splits)))
    for split in specified_splits:
        datasplit_path = os.path.join(args.data_dir, split)
        iid2captions = dict()
        iid2photo = defaultdict(list)
        image_dir = args.image_dir
        assert os.path.isdir(image_dir), "The image_dir does not exist! Please check the input args..."
        assert specified_type == "jsonl", "the type of file is not jsonl"
        txt_path = datasplit_path + "_texts.jsonl"
        assert os.path.exists(txt_path) is True
        with open(txt_path, "r", encoding="utf-8") as fin_pairs:
            for index, line in tqdm(enumerate(fin_pairs)):
                line = line.strip()
                obj = json.loads(line)
                for field in ("text_id", "text", "image_ids"):
                    assert (
                        field in obj
                    ), "Field {} does not exist in line {}. \
                            Please check the integrity of the text annotation Jsonl file."
                image_ids = obj["image_ids"]
                if type(image_ids[0]) == str and image_ids[0].isnumeric() is True:
                    iid2captions[index] = obj["text"]
                    iid2photo[int(image_ids[0])].append(index)
                else:
                    iid2captions[index] = obj["text"]
                    iid2photo[image_ids[0]].append(index)
        paths = (
            list(glob(f"{args.image_dir}/*/*jpg"))
            + list(glob(f"{args.image_dir}/*/*.png"))
            + list(glob(f"{args.image_dir}/*/*.JPG"))
        )
        random.shuffle(paths)
        # 有效图片路径
        if type(list(iid2photo.keys())[0]) == int:
            caption_paths = [path for path in paths if int(path.split("/")[-1][:-4]) in iid2photo]
        elif type(list(iid2photo.keys())[0]) == str and "." not in list(iid2photo.keys())[0]:
            caption_paths = [path for path in paths if path.split("/")[-1][:-4] in iid2photo]
        else:
            caption_paths = [path for path in paths if path.split("/")[-1] in iid2photo]
        invalid_photo = [path for path in iid2photo if path not in [i.split("/")[-1] for i in caption_paths]]
        bs = []
        for path in tqdm(caption_paths):
            bs += path2rest(path, iid2captions, iid2photo)
        dataframe = pd.DataFrame(bs, columns=["image", "caption", "image_id"])
        table = pa.Table.from_pandas(dataframe)

        if args.data_out_dir is None:
            data_out_dir = os.path.join(args.data_dir, "arrow")
        else:
            data_out_dir = args.data_out_dir
        os.makedirs(data_out_dir, exist_ok=True)
        with pa.OSFile(f"{data_out_dir}/{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        if split in ["valid", "test"]:
            if len(invalid_photo) > 0:
                logging.info("The jsonl file contains invalid images")
                data_valid = []
                with open(txt_path, "r", encoding="utf-8") as fin_pairs:
                    for line in fin_pairs:
                        line = line.strip()
                        obj = json.loads(line)
                        obj["image_ids"] = [i for i in obj["image_ids"] if i not in invalid_photo]
                        if len(obj["image_ids"]) == 0:
                            continue
                        data_valid.append(obj)
                with jsonlines.open(datasplit_path + "_updata_texts.jsonl", mode="w") as writer:
                    for row in data_valid:
                        writer.write({"text_id": row["text_id"], "text": row["text"], "image_ids": row["image_ids"]})
            bs_img = [valid_img(path, iid2captions) for path in tqdm(caption_paths)]
            dataframe_img = pd.DataFrame(bs_img, columns=["image", "image_id"])
            table_img = pa.Table.from_pandas(dataframe_img)
            with pa.OSFile(f"{data_out_dir}/{split}_img.arrow", "wb") as sink:
                with pa.RecordBatchFileWriter(sink, table_img.schema) as writer:
                    writer.write_table(table_img)
