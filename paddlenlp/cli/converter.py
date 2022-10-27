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
from paddlenlp.utils.converter import Converter
from paddlenlp.transformers.bert.converter import BertConverter


def convert_from_local_dir(pretrained_dir: str, output: str):
    """convert weight from local dir

    Args:
        pretrained_dir (str): the pretrained dir
        output (str): the output dir
    """
    converter = BertConverter()
    converter.convert(input_dir=pretrained_dir, output_dir=output)


def convert_from_local_file(weight_file_path: str, output: str):
    """convert from the local weitht file

    Args:
        weight_file_path (str): the path of the weight file
        output (str): the output dir
    """
    # 1. check the name of weight file
    if not os.path.isdir(weight_file_path):
        weight_file_dir, filename = os.path.split(weight_file_path)
        if filename != "pytorch_model.bin":
            shutil.copy(weight_file_path,
                        os.path.join(weight_file_dir, 'pytorch_model.bin'))

        weight_file_path = weight_file_dir
    convert_from_local_dir(weight_file_path, output)


def convert_from_online_model(model_name: str, cache_dir: str, output_dir):
    """convert the model which is not maintained in paddlenlp community, eg: vblagoje/bert-english-uncased-finetuned-pos

    Args:
        model_name (str): the name of model
        cache_dir (str): the cache_dir to save pytorch model
        output_dir (_type_): the output dir
    """
    # 1. auto save
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(cache_dir)

    # 2. resolve the converter
    config_file = os.path.join(cache_dir, 'config.json')
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    architectures = config['architectures']

    converter = BertConverter()
    converter.convert(input_dir=cache_dir, output_dir=output_dir)
