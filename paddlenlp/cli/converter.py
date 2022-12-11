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
import os
import shutil
from typing import Type

from paddlenlp.utils.converter import Converter, load_all_converters
from paddlenlp.utils.log import logger


def convert_from_local_dir(pretrained_dir: str, output: str):
    """convert weight from local dir

    Args:
        pretrained_dir (str): the pretrained dir
        output (str): the output dir
    """

    # 1. checking the related files
    files = os.listdir(pretrained_dir)
    assert "pytorch_model.bin" in files, f"`pytorch_model.bin` file must exist in dir<{pretrained_dir}>"
    assert "config.json" in files, f"`config.json` file must exist in dir<{pretrained_dir}>"

    # 2. get model architecture from config.json
    config_file = os.path.join(pretrained_dir, "config.json")
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    architectures = config.pop("architectures", []) or config.pop("init_class", None)
    if not architectures:
        raise ValueError("can not find the model weight architectures with field: <architectures> and <init_class>")

    if isinstance(architectures, str):
        architectures = [architectures]

    if len(architectures) > 1:
        raise ValueError("only support one model architecture")

    architecture = architectures[0]

    # 3. retrieve Model Converter
    target_converter_classes = [
        converter_class for converter_class in load_all_converters() if architecture in converter_class.architectures
    ]
    if not target_converter_classes:
        logger.error(f"can not find target Converter based on architecture<{architecture}>")
    if len(target_converter_classes) > 1:
        logger.warning(
            f"{len(target_converter_classes)} found, we will adopt the first one as the target converter ..."
        )

    target_converter_class: Type[Converter] = target_converter_classes[0]

    # 4. do converting
    converter = target_converter_class()
    converter.convert(pretrained_dir, output_dir=output)


def convert_from_local_file(weight_file_path: str, output: str):
    """convert from the local weitht file

    TODO(wj-Mcat): no model info for weight file, this method is dangerous.

    Args:
        weight_file_path (str): the path of the weight file
        output (str): the output dir
    """
    # 1. check the name of weight file
    if not os.path.isdir(weight_file_path):
        weight_file_dir, filename = os.path.split(weight_file_path)
        if filename != "pytorch_model.bin":
            shutil.copy(weight_file_path, os.path.join(weight_file_dir, "pytorch_model.bin"))

        weight_file_path = weight_file_dir
    convert_from_local_dir(weight_file_path, output)


def convert_from_online_model(model_name: str, cache_dir: str, output_dir):
    """convert the model which is not maintained in paddlenlp community, eg: vblagoje/bert-english-uncased-finetuned-pos

    TODO(wj-Mcat): this feature will be done in next version

    Args:
        model_name (str): the name of model
        cache_dir (str): the cache_dir to save pytorch model
        output_dir (_type_): the output dir
    """
    # 1. checke the community models from paddle community server

    # 2. download config file from huggingface website

    # 3. download the pytorch model file from huggingface server

    # 4. convert the pytorch model file

    # 5. [Optional] forward the paddle/pytorch model and compare the logits
