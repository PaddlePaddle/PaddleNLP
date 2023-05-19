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
from pathlib import Path
from typing import List, Tuple, Type

from uvicorn.config import LOGGING_CONFIG

from paddlenlp.utils.import_utils import is_package_available

# check whether the package is avaliable and give friendly description.
if not is_package_available("typer"):
    raise ModuleNotFoundError(
        "paddlenlp-cli tools is not installed correctly, you can use the following command"
        " to install paddlenlp cli tool: >>> pip install paddlenlp[cli]"
    )

import importlib
import inspect
import shutil

import typer

from paddlenlp.cli.download import load_community_models
from paddlenlp.cli.install import install_package_from_bos
from paddlenlp.cli.server import start_backend
from paddlenlp.cli.utils.tabulate import print_example_code, tabulate
from paddlenlp.transformers import (
    AutoModel,
    AutoTokenizer,
    PretrainedModel,
    PretrainedTokenizer,
)
from paddlenlp.transformers.utils import find_transformer_model_type
from paddlenlp.utils.downloader import is_url
from paddlenlp.utils.log import logger


def load_all_models(include_community: bool = False) -> List[Tuple[str, str]]:
    """load all model_name infos

    Returns:
        List[Tuple[str, str]]: [model_type, model_name]
    """
    # 1. load official models
    module = importlib.import_module("paddlenlp.transformers")
    model_names = []
    model_names_dict = {}
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
        model_type = find_transformer_model_type(obj)
        for model_name in configurations.keys():
            # get model type with refactoring
            model_names.append((model_type, model_name))
            model_names_dict[model_name] = True

    logger.info(f"find {len(model_names)} official models ...")

    # 2. load & extend community models
    if include_community:
        community_model_names = load_community_models()
        for model_name in community_model_names:
            # there are some same model-names between codebase and community models
            if model_name in model_names_dict:
                continue

            model_names.append(model_name)
    # 3. sort result
    model_names.sort(key=lambda item: item[0] + item[1])
    return model_names


app = typer.Typer()


@app.command()
def download(
    model_name: str,
    cache_dir: str = typer.Option(
        "./pretrained_models", "--cache-dir", "-c", help="cache_dir for download pretrained model"
    ),
    force_download: bool = typer.Option(False, "--force-download", "-f", help="force download pretrained model"),
):
    """download the paddlenlp models with command, you can specific `model_name`

    >>> paddlenlp download bert \n
    >>> paddlenlp download -c ./my-models -f bert \n

    Args:\n
        model_name (str): pretarined model name, you can checkout all of model from source code. \n
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
def search(
    query=typer.Argument(..., help="the query of searching model"),
    include_community: bool = typer.Option(
        False, "--include-community", "-i", help="whether searching community models"
    ),
):
    """search the model with query, eg: paddlenlp search bert

    >>> paddlenlp search bert \n
    >>> paddlenlp search -i bert \n

    Args: \n
        query (Optional[str]): the str fragment of bert-name \n
        include_community (Optional[bool]): whether searching community models
    """
    logger.info("start to search models ...")
    model_names = load_all_models(include_community)

    tables = []
    for model_type, model_name in model_names:
        # TODO(wj-Mcat): ignore the model_category info
        if not query or query in model_name:
            tables.append([model_type, model_name])
    tabulate(tables, headers=["model type", "model name"], highlight_word=query)
    print_example_code()

    logger.info(f"the retrieved number of models results is {len(tables)} ...")


@app.command(help="Start the PaddleNLP SimpleServer.")
def server(
    app: str,
    host: str = typer.Option("127.0.0.1", "--host", help="Bind socket to this host."),
    port: int = typer.Option("8000", "--port", help="Bind socket to this port."),
    app_dir: str = typer.Option(None, "--app_dir", help="The application directory path."),
    workers: int = typer.Option(
        None,
        "--workers",
        help="Number of worker processes. Defaults to the $WEB_CONCURRENCY environment"
        " variable if available, or 1. Not valid with --reload.",
    ),
    log_level: int = typer.Option(None, "--log_level", help="Log level. [default: info]"),
    limit_concurrency: int = typer.Option(
        None, "--limit-concurrency", help="Maximum number of concurrent connections or tasks to allow, before issuing"
    ),
    limit_max_requests: int = typer.Option(
        None, "--limit-max-requests", help="Maximum number of requests to service before terminating the process."
    ),
    timeout_keep_alive: int = typer.Option(
        15, "--timeout-keep-alive", help="Close Keep-Alive connections if no new data is received within this timeout."
    ),
    reload: bool = typer.Option(False, "--reload", help="Reload the server when the app_dir is changed."),
):
    """The main function for the staring the SimpleServer"""
    logger.info("starting to PaddleNLP SimpleServer...")
    if app_dir is None:
        app_dir = str(Path(os.getcwd()))
    # Flags of uvicorn
    backend_kwargs = {
        "host": host,
        "port": port,
        "log_config": LOGGING_CONFIG,
        "log_level": log_level,
        "workers": workers,
        "limit_concurrency": limit_concurrency,
        "limit_max_requests": limit_max_requests,
        "timeout_keep_alive": timeout_keep_alive,
        "app_dir": app_dir,
        "reload": reload,
    }
    start_backend(app, **backend_kwargs)


@app.command(
    help="install the target version of paddlenlp, eg: paddlenlp install / paddlenlp install paddlepaddle==latest"
)
def install(
    package: str = typer.Argument(default="paddlenlp==latest", help="install the target version of paddlenlp")
):
    """The main function for the staring the SimpleServer"""
    package = package.replace(" ", "").strip()

    if not package:
        raise ValueError("please assign the package name")

    # 1. parse the version of paddlenlp
    splits = [item for item in package.split("==")]
    if len(splits) == 0 or len(splits) > 2:
        raise ValueError(
            "please set the valid package: <package-name>==<version>, eg: paddlenlp==latest, paddlenlp==3099, "
            f"but received: {package}"
        )

    tag = "latest"
    package_name = splits[0]

    # TODO(wj-Mcat): will support `pipelines`, `ppdiffusers` later.
    assert package_name in ["paddlenlp"], "we only support paddlenlp"

    if len(splits) == 2:
        tag = splits[1]

    # 2. download & install package from bos server
    install_package_from_bos(package_name=package_name, tag=tag)


def main():
    """the PaddleNLPCLI entry"""
    app()


if __name__ == "__main__":
    main()
