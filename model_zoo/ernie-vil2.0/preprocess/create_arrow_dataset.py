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
import os
import random
from collections import defaultdict
from glob import glob

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
    text_index = iid2photo[int(name.split("/")[-1][:-4])]
    bs_item = []
    for index in text_index:
        bs_item.append([binary, iid2captions[index], int(name.split("/")[-1][:-4])])
    return bs_item


def valid_img(path, iid2photo):
    name = path.split("/")[-1]
    with open(path, "rb") as fp:
        binary = fp.read()
    return [binary, int(name.split("/")[-1][:-4])]


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
        if specified_type == "jsonl" or specified_type == "json":
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
        elif specified_splits == "csv":
            txt_path = datasplit_path + "." + specified_type
            assert os.path.exists(txt_path) is True
            fin_data = pd.read_csv(txt_path, header=None)
            for index in range(len(fin_data)):
                raw_txt, photo_path = fin_data.iloc[index, 0], fin_data.iloc[index, 1]
                iid2captions[index] = raw_txt
                iid2photo[photo_path.split("/")[-1]].append(index)
        paths = (
            list(glob(f"{args.image_dir}/*jpg"))
            + list(glob(f"{args.image_dir}/*.png"))
            + list(glob(f"{args.image_dir}/*.JPG"))
            + list(glob(f"{args.image_dir}/*.gif"))
        )
        random.shuffle(paths)
        if type(list(iid2photo.keys())[0]) == int:
            caption_paths = [path for path in paths if int(path.split("/")[-1][:-4]) in iid2photo]  # 有效图片路径
        elif type(list(iid2photo.keys())[0]) == str and "." not in list(iid2photo.keys())[0]:
            caption_paths = [path for path in paths if path.split("/")[-1][:-4] in iid2photo]  # 有效图片路径
        else:
            caption_paths = [path for path in paths if path.split("/")[-1] in iid2photo]  # 有效图片路径
        bs = []
        for path in tqdm(caption_paths):
            bs += path2rest(path, iid2captions, iid2photo)
        dataframe = pd.DataFrame(bs, columns=["image", "caption", "image_id"])
        table = pa.Table.from_pandas(dataframe)

        if args.data_out_dir == "":
            data_out_dir = os.path.join(args.data_dir, "arrow")
        else:
            data_out_dir = args.data_out_dir
        os.makedirs(data_out_dir, exist_ok=True)
        with pa.OSFile(f"{data_out_dir}/{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        if split in ["valid", "test"]:
            bs_img = [valid_img(path, iid2captions) for path in tqdm(caption_paths)]
            dataframe_img = pd.DataFrame(bs_img, columns=["image", "image_id"])
            table_img = pa.Table.from_pandas(dataframe_img)
            with pa.OSFile(f"{data_out_dir}/{split}_img.arrow", "wb") as sink:
                with pa.RecordBatchFileWriter(sink, table_img.schema) as writer:
                    writer.write_table(table_img)
