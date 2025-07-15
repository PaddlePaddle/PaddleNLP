# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os

import jsonlines
import pandas as pd
import pyarrow as pa
from tqdm import tqdm


def get_all_image_paths(image_dir):
    valid_exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    paths = []
    for root, _, files in os.walk(image_dir):
        for fname in files:
            if os.path.splitext(fname)[1] in valid_exts:
                paths.append(os.path.join(root, fname))
    return paths


def build_image_map(image_paths):
    return {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}


def build_arrow_for_split(split_name, jsonl_path, image_map, output_dir):
    print(f"Processing split: {split_name}")
    all_entries = 0
    kept_entries = 0
    data = []
    data_img = []
    missing_image_ids = set()

    with jsonlines.open(jsonl_path, "r") as reader:
        for obj in tqdm(reader):
            all_entries += 1
            image_ids = [str(i) for i in obj.get("image_ids", [])]
            valid_image_ids = [i for i in image_ids if i in image_map]

            if not valid_image_ids:
                missing_image_ids.update(image_ids)
                continue

            for img_id in valid_image_ids:
                img_path = image_map[img_id]
                with open(img_path, "rb") as img_f:
                    img_bytes = img_f.read()
                data.append([img_bytes, obj["text"], img_id])
                data_img.append([img_bytes, img_id])
                kept_entries += 1

    # 保存图文对 arrow 文件
    df = pd.DataFrame(data, columns=["image", "caption", "image_id"])
    table = pa.Table.from_pandas(df)
    with pa.OSFile(os.path.join(output_dir, f"{split_name}.arrow"), "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

    # 保存图像-only arrow 文件（无caption）
    df_img = pd.DataFrame(data_img, columns=["image", "image_id"])
    table_img = pa.Table.from_pandas(df_img)
    with pa.OSFile(os.path.join(output_dir, f"{split_name}_img.arrow"), "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table_img.schema) as writer:
            writer.write_table(table_img)

    print(f"{split_name}: {kept_entries}/{all_entries} entries kept.")
    if missing_image_ids:
        miss_path = os.path.join(output_dir, f"missing_{split_name}.txt")
        with open(miss_path, "w", encoding="utf-8") as f:
            for mid in sorted(missing_image_ids):
                f.write(mid + "\n")
        print(f"⚠️ Missing image IDs written to: {miss_path}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    image_paths = get_all_image_paths(args.image_dir)
    image_map = build_image_map(image_paths)

    splits = args.splits.split(",")
    for split in splits:
        jsonl_path = os.path.join(args.data_dir, f"{split}_texts.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"File not found: {jsonl_path}")
            continue
        build_arrow_for_split(split, jsonl_path, image_map, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing *_texts.jsonl files")
    parser.add_argument("--image_dir", required=True, help="Directory containing image files")
    parser.add_argument("--output_dir", default=None, help="Directory to save output .arrow files")
    parser.add_argument("--splits", default="train,valid,test", help="Comma-separated list of dataset splits")

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "arrow")
    main(args)
