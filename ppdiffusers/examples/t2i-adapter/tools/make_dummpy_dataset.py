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

import argparse
import json
import os

from tqdm import tqdm

from ppdiffusers.utils import load_image

dataset_base_name_one_type_one_url_base = ""
dataset_base_name_one_type_two_url_base = ""
dataset_base_name_two_type_one_url_base = ""
dataset_base_name_two_type_two_url_base = ""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_base_name",
    type=str,
    default="artv4_openpose_test13",
    help="The dataset basename.",
)
parser.add_argument(
    "--ids_list_path",
    type=str,
    default="artv4_openpose_test13_ids.txt",
    help="The ids list path.",
)
parser.add_argument(
    "--ids_list_path",
    type=str,
    default="artv4_openpose_test13_ids.txt",
    help="The ids list path.",
)
parser.add_argument(
    "--source_prompt_list_one_path",
    type=str,
    default="prompts_artv4_openpose_test1_en_prompts.txt",
    help="The first source prompt list path.",
)
parser.add_argument(
    "--source_prompt_list_two_path",
    type=str,
    default="prompts_artv4_openpose_test2_en_prompts.txt",
    help="The second source prompt list path.",
)
parser.add_argument(
    "--source_prompt_list_three_path",
    type=str,
    default="prompts_artv4_openpose_test3_en_prompts.txt",
    help="The third source prompt list path.",
)
parser.add_argument(
    "--dataset_prompt_json_name",
    type=str,
    default="prompt.json",
    help="The dataset prompt json name.",
)
args = parser.parse_args()


def get_images_form_urls(ids_list, dir_path, dataset_base_name, type=None, is_resize=False):
    for i, id in enumerate(tqdm(ids_list)):
        if dataset_base_name == "artv4_openpose_test13":
            if type == "原图":
                img_url = dataset_base_name_one_type_one_url_base + f"{id}/{id}_final00_control.png"
            elif type == "Openpose控制图":
                img_url = dataset_base_name_one_type_two_url_base + f"{id}/{id}_final00_control_openpose.png"
        if dataset_base_name == "artv4_openpose_test2":
            if type == "原图":
                img_url = dataset_base_name_two_type_one_url_base + f"{id}/{id}_final00_control.png"
            elif type == "Openpose控制图":
                img_url = dataset_base_name_two_type_one_url_base + f"{id}/{id}_final00_control_openpose.png"
        in_image = load_image(img_url)
        if is_resize:
            in_image = in_image.resize((512, 512))
        os.makedirs(dir_path, exist_ok=True)
        name = str(i) + "_" + id + ".png"
        in_image.save(os.path.join(dir_path, name))


def get_prompt_json_file(ids_list, prompt_lists, dataset_base_name):
    with open(os.path.join(dataset_base_name, args.dataset_prompt_json_name), "w") as wf:
        for i, id in enumerate(ids_list):
            which_prompt_list = int(id.split("_")[1][-1]) - 1
            which_prompt = int(id.split("_")[-1])
            name = str(i) + "_" + id + ".png"

            data = {
                "source": "source/" + name,
                "target": "target/" + name,
                "prompt": prompt_lists[which_prompt_list][which_prompt].strip(),
            }
            json_str = json.dumps(data)
            wf.write(json_str + "\n")


if __name__ == "__main__":
    dataset_base_name = args.dataset_base_name
    ids_list = [line.strip() for line in open(args.ids_list_path, "r", encoding="utf8").readlines()]

    source_prompt_lists = [
        [line.strip() for line in open(args.source_prompt_list_one_path, "r", encoding="utf8").readlines()],
        [line.strip() for line in open(args.source_prompt_list_two_path, "r", encoding="utf8").readlines()],
        [line.strip() for line in open(args.source_prompt_list_three_path, "r", encoding="utf8").readlines()],
    ]

    source_dir = os.path.join(dataset_base_name, "source")
    target_dir = os.path.join(dataset_base_name, "target")
    get_images_form_urls(ids_list, source_dir, dataset_base_name, type="Openpose控制图", is_resize=False)
    get_images_form_urls(ids_list, target_dir, dataset_base_name, type="原图", is_resize=False)
    get_prompt_json_file(ids_list, source_prompt_lists, dataset_base_name)
