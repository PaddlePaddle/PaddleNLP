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

import os
import json
from typing import Type, List, Tuple
from typer import Typer
import shutil
import importlib, inspect
from paddlenlp.transformers import AutoModel, AutoTokenizer, PretrainedModel, PretrainedTokenizer
from paddlenlp.utils.log import logger
from paddlenlp.utils.env import MODEL_HOME
from paddlenlp.utils.downloader import is_url

from tabulate import tabulate


def load_all_models() -> List[Tuple[str, str]]:
    """load all model_name infos

    Returns:
        List[Tuple[str, str]]: [model_type, model_name]
    """
    module = importlib.import_module("paddlenlp.transformers")
    model_names = set()
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        obj = getattr(module, attr_name)
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, PretrainedModel):
            continue

        obj: Type[PretrainedModel] = obj
        if not obj.__name__.endswith("PretrainedModel"):
            continue
        configurations = obj.pretrained_init_configuration
        for model_name in configurations.keys():
            model_names.add((obj.base_model_prefix, model_name))
    return model_names


app = Typer()


@app.command()
def download(model_name: str,
             cache_dir: str = "./models",
             force_download: bool = False):
    """download the paddlenlp models with command, you can specific `model_name`

    Args:\n
        model_name (str): pretarined model name, you can checkout all of model from source code.
        cache_dir (str, optional): the cache_dir. Defaults to "./models".
    """
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(os.getcwd(), cache_dir)

    if is_url(model_name):
        logger.error("<MODEL_NAME> can not be url")
        return

    cache_dir = os.path.join(cache_dir, model_name)
    if force_download:
        shutil.rmtree(cache_dir, ignore_errors=True)

    model: PretrainedModel = AutoModel.from_pretrained(model_name)
    model.save_pretrained(cache_dir)

    tokenizer: PretrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(cache_dir)

    logger.info(f"successfully saved model into <{cache_dir}>")


@app.command()
def search(query: str):
    """search the model with query, eg: paddlenlp search bert

    Args:
        query (str): the str fragment of bert-name
    """
    model_names = load_all_models()

    tables = []
    for model_type, model_name in model_names:
        if query in model_name:
            tables.append([model_type, model_name])
    print(
        tabulate(tables, headers=['model type', 'model name'], tablefmt="grid"))


@app.command()
def convert(model_type: str,
            config_or_model_name: str,
            pytorch_checkpoint_path: str = 'pytorch',
            dump_output: str = "model_state.pdparams"):
    # convert pytorch weight file to paddle weight file

    # Args:
    #     model_type (str): the name of target paddle model name, which can be: bert, bert-base-uncased
    #     torch_checkpoint_path (str, optional): the path of target pytorch weight file . Defaults to 'pytorch'.

    # 1. resolve pytorch weight file path
    if os.path.isdir(pytorch_checkpoint_path):
        pytorch_checkpoint_path = os.path.join(pytorch_checkpoint_path,
                                               "pytorch_model.bin")
        if not os.path.isfile(pytorch_checkpoint_path):
            raise FileNotFoundError(
                "pytorch checkpoint file {} not found".format(
                    pytorch_checkpoint_path))
    elif not os.path.exists(pytorch_checkpoint_path):
        raise FileNotFoundError("pytorch checkpoint file {} not found".format(
            pytorch_checkpoint_path))

    def resolve_configuration(model_class: Type[PretrainedModel]) -> dict:
        if config_or_model_name in model_class.pretrained_init_configuration:
            return model_class.pretrained_init_configuration[
                config_or_model_name]
        assert os.path.isfile(
            config_or_model_name
        ), f'can"t not find the configuration file by <{config_or_model_name}>'
        with open(config_or_model_name, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config

    # 2. convert different model weight file with
    if model_type == 'bert':
        from paddlenlp.transformers.bert.modeling import convert_pytorch_weights, BertModel
        config = resolve_configuration(BertModel)
        model = BertModel(**config)
        convert_pytorch_weights(model,
                                pytorch_checkpoint_path=pytorch_checkpoint_path)
    elif model_type == 'albert':
        from paddlenlp.transformers.albert.modeling import AlbertModel
        config = resolve_configuration(AlbertModel)
        model = AlbertModel(**config)
        # call `convert` method


def main():
    """the PaddleNLPCLI entry"""
    app()
