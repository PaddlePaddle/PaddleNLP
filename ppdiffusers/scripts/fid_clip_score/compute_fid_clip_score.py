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

import argparse
import json
import math
import os
import pathlib

import paddle
import pandas as pd
from fid_score import IMAGE_EXTENSIONS, calculate_fid_given_paths
from paddle.utils.download import get_path_from_url
from PIL import Image
from tqdm.auto import tqdm

from paddlenlp.transformers import CLIPModel, CLIPProcessor
from ppdiffusers.utils import DOWNLOAD_SERVER, PPDIFFUSERS_CACHE

base_url = DOWNLOAD_SERVER + "/CompVis/data/"
cache_path = os.path.join(PPDIFFUSERS_CACHE, "data")


def save_json(data, file_path="statistic_results.json"):
    with open(str(file_path), "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False)


def batchify(data, batch_size=16):
    one_batch = []
    for example in data:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            yield one_batch
            one_batch = []
    if one_batch:
        yield one_batch


@paddle.no_grad()
def compute_clip_score(model, processor, texts, images_path, batch_size=64):
    all_text_embeds = []
    all_image_embeds = []
    for text, image_path in tqdm(
        zip(batchify(texts, batch_size), batchify(images_path, batch_size)), total=math.ceil(len(texts) / batch_size)
    ):
        assert len(text) == len(image_path)
        batch_inputs = processor(
            text=text,
            images=[Image.open(image) for image in image_path],
            return_tensors="pd",
            max_length=processor.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
        )
        text_embeds = model.get_text_features(input_ids=batch_inputs["input_ids"])
        image_embeds = model.get_image_features(pixel_values=batch_inputs["pixel_values"])
        all_text_embeds.append(text_embeds)
        all_image_embeds.append(image_embeds)

    all_text_embeds = paddle.concat(all_text_embeds)
    all_image_embeds = paddle.concat(all_image_embeds)
    all_text_embeds = all_text_embeds / all_text_embeds.norm(axis=-1, keepdim=True)
    all_image_embeds = all_image_embeds / all_image_embeds.norm(axis=-1, keepdim=True)
    clip_score = (all_image_embeds * all_text_embeds).sum(-1) * model.logit_scale.exp()
    return clip_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default=None, nargs="+", type=str, help="image_path")
    parser.add_argument(
        "--text_file_name",
        default="coco30k",
        choices=["coco1k", "coco10k", "coco30k"],
        type=str,
        help="text file.",
    )
    parser.add_argument(
        "--clip_model_name_or_path", default="openai/clip-vit-base-patch32", type=str, help="clip_model_name_or_path"
    )
    parser.add_argument("--fid_batch_size", default=32, type=int, help="fid_batch_size")
    parser.add_argument("--clip_batch_size", default=64, type=int, help="clip_batch_size")
    parser.add_argument("--resolution", default=256, type=int, help="resolution of images")
    parser.add_argument("--device", default="gpu", type=str, help="device")
    parser.add_argument(
        "--only_fid",
        action="store_true",
        help=("Only eval fid. "),
    )
    args = parser.parse_args()

    paddle.set_device(args.device)
    all_path = args.image_path
    text_file_name = args.text_file_name
    # dont change
    image_num = text_file_name.replace("coco", "")
    if image_num == "30k":
        os.environ["FLAG_IMAGE_NUM"] = "30000"
    elif image_num == "10k":
        os.environ["FLAG_IMAGE_NUM"] = "10000"
    else:
        os.environ["FLAG_IMAGE_NUM"] = "1000"
    dataset_name = f"coco_{args.resolution}_{image_num}.npz"
    fid_target_file = get_path_from_url(base_url + dataset_name, cache_path) + ".npz"

    text_file = get_path_from_url(base_url + text_file_name + ".tsv", cache_path)
    df = pd.read_csv(text_file, sep="\t")
    texts = df["caption_en"].tolist()
    if not args.only_fid:
        model = CLIPModel.from_pretrained(args.clip_model_name_or_path)
        model.eval()
        processor = CLIPProcessor.from_pretrained(args.clip_model_name_or_path)
        # pad_token_id must be set to zero!
        processor.tokenizer.pad_token_id = 0

    results = {"file": [], "fid": []}
    for path in all_path:
        results["file"].append(path)
        # fid score
        fid_value = calculate_fid_given_paths(
            [fid_target_file, path],
            batch_size=args.fid_batch_size,
            dims=2048,
            num_workers=4,
        )
        results["fid"].append(fid_value)

        if not args.only_fid:
            # clip score
            images_path = sorted(
                [image_path for ext in IMAGE_EXTENSIONS for image_path in pathlib.Path(path).glob("*.{}".format(ext))]
            )
            clip_score = compute_clip_score(model, processor, texts, images_path, args.clip_batch_size)
            if "clip_score" not in results:
                results["clip_score"] = []
            _clip_score = clip_score.mean().item()
            results["clip_score"].append()
            if image_num == "30k":
                print(f"=====> clip_score 1k: {clip_score[:1000].mean().item()}")
                print(f"=====> clip_score 10k: {clip_score[:10000].mean().item()}")
            print(f"fid: {fid_value}, clip_score: {_clip_score}")
        else:
            print(f"fid: {fid_value}")
    # save json file results
    save_json(results)
    print(results)
