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

from genericpath import isdir
import os
import json
from typing import Type, List, Tuple, Optional
import typer
from typer import Typer
import shutil
import importlib, inspect
from paddlenlp import __version__
from paddlenlp.transformers import AutoModel, AutoTokenizer, PretrainedModel, PretrainedTokenizer
from paddlenlp.utils.log import logger
from paddlenlp.utils.downloader import is_url
from paddlenlp.cli.converter import convert_from_local_file, convert_from_online_model
from paddlenlp.cli.utils.tabulate import tabulate
from paddlenlp.cli.download import load_community_models


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
            model_names.add(("official", obj.base_model_prefix, model_name))
    logger.info(f"find {len(model_names)} official models ...")

    # load & extend community models
    community_model_names = load_community_models()
    for model_name in community_model_names:
        model_names.add(model_name)

    logger.info(f"finding {len(model_names)} models ...")
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
def search(query):
    """search the model with query, eg: paddlenlp search bert

    Args:
        query (Optional[str]): the str fragment of bert-name
    """
    logger.info("start to search models ...")
    model_names = load_all_models()

    tables = []
    if query:
        for model_category, model_type, model_name in model_names:
            if query in model_name:
                tables.append([model_category, model_type, model_name])
    tabulate(tables,
             headers=["model source", 'model type', 'model name'],
             highlight_word=query)

    logger.info(f"the retrieved number of models results is {len(tables)} ...")


@app.command()
def convert(input: Optional[str] = None, output: Optional[str] = None):
    if os.path.isdir(input):
        convert_from_local_file()
    convert_from_online_model()


def main():
    """the PaddleNLPCLI entry"""
    app()


if __name__ == "__main__":
    main()
