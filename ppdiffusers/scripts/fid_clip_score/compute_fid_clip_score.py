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
import pickle
import tempfile

import matplotlib.pyplot as plt
import paddle
from fid_score import (
    IMAGE_EXTENSIONS,
    InceptionV3,
    calculate_frechet_distance,
    compute_statistics_of_path,
)
from paddle.utils.download import get_path_from_url
from PIL import Image
from tqdm.auto import tqdm

from paddlenlp.transformers import CLIPModel, CLIPProcessor
from ppdiffusers.utils import DOWNLOAD_SERVER, PPDIFFUSERS_CACHE

base_url = DOWNLOAD_SERVER + "/CompVis/data/"
cache_path = os.path.join(PPDIFFUSERS_CACHE, "data")


def save_pickle(data, file_path):
    with open(str(file_path), "wb") as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    with open(str(input_file), "rb") as f:
        data = pickle.load(f)
    return data


def save_json(data, file_path="statistic_results.json"):
    with open(str(file_path), "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False)


def calculate_ms_given_path(path, batch_size, dims, save_path, num_workers=1):
    if not os.path.exists(path):
        raise RuntimeError("Invalid path: %s" % path)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    m, s = compute_statistics_of_path(path, model, batch_size, dims, num_workers)
    save_pickle(dict(m=m, s=s), save_path)


def calculate_fid_given_ms_file(ms1, dataset_name):
    ms1 = load_pickle(ms1)
    ms2 = get_path_from_url(base_url + dataset_name, cache_path)
    ms2 = load_pickle(ms2)

    m1, s1 = ms1["m"], ms1["s"]
    m2, s2 = ms2["m"], ms2["s"]
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


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
    return clip_score.mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default=None, nargs="+", type=str, help="image_path")
    parser.add_argument(
        "--text_file_name",
        default="mscoco.en.1k",
        choices=["mscoco.en.1k", "mscoco.en.30k"],
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
    args = parser.parse_args()

    paddle.set_device(args.device)
    all_path = args.image_path
    text_file_name = args.text_file_name
    # dont change
    dataset_name = f"mscoco_en_val2014_{args.resolution}.pkl"
    model = CLIPModel.from_pretrained(args.clip_model_name_or_path)
    model.eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model_name_or_path)
    # pad_token_id must be set to zero!
    processor.tokenizer.pad_token_id = 0

    text_file = get_path_from_url(base_url + text_file_name, cache_path)
    with open(text_file, "r") as f:
        texts = [p.strip() for p in f.readlines()]

    results = {"file": [], "fid": [], "clip_score": []}
    for path in all_path:
        results["file"].append(path)
        # fid score
        with tempfile.TemporaryDirectory() as tmpdirname:
            predict_ms_file = os.path.join(tmpdirname, "tmp.pkl")
            calculate_ms_given_path(
                path, batch_size=args.fid_batch_size, dims=2048, save_path=predict_ms_file, num_workers=4
            )
            fid_value = calculate_fid_given_ms_file(predict_ms_file, dataset_name)
        results["fid"].append(fid_value)

        # clip score
        images_path = sorted(
            [image_path for ext in IMAGE_EXTENSIONS for image_path in pathlib.Path(path).glob("*.{}".format(ext))]
        )
        clip_score = compute_clip_score(model, processor, texts, images_path, args.clip_batch_size)
        results["clip_score"].append(clip_score)
        print(f"fid: {fid_value}, clip_score: {clip_score}")

    # save json file results
    save_json(results)

    # plot Pareto Curves
    step = -1
    plt.plot(results["clip_score"], results["fid"], label=f"pd-{step}", linewidth=3, marker="o")
    plt.xlabel("CLIP Score")
    plt.ylabel(f"FID@{text_file_name}")
    plt.title("Pareto Curves")
    plt.legend()
    plt.savefig("pareto_curves.png")
    plt.show()
