# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
from __future__ import annotations

import inspect
import io
import json
import os
from collections import defaultdict
from typing import Dict, List, Type

from huggingface_hub import hf_hub_download

from paddlenlp import __version__
from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.utils.downloader import (
    COMMUNITY_MODEL_PREFIX,
    get_path_from_url_with_filelock,
    url_file_exists,
)
from paddlenlp.utils.import_utils import import_module
from paddlenlp.utils.log import logger

from ..utils import resolve_cache_dir

__all__ = [
    "AutoConfig",
]


def get_configurations() -> Dict[str, List[Type[PretrainedConfig]]]:
    """load the configurations of PretrainedConfig mapping: {<model-name>: [<class-name>, <class-name>, ...], }

    Returns:
        dict[str, str]: the mapping of model-name to model-classes
    """
    # 1. search the subdir<model-name> to find model-names
    transformers_dir = os.path.dirname(os.path.dirname(__file__))
    exclude_models = ["auto"]

    mappings = defaultdict(list)
    for model_name in os.listdir(transformers_dir):
        if model_name in exclude_models:
            continue

        model_dir = os.path.join(transformers_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        # 2. find the `configuration.py` file as the identifier of PretrainedConfig class
        configuration_path = os.path.join(model_dir, "configuration.py")
        if not os.path.exists(configuration_path):
            continue

        configuration_module = import_module(f"paddlenlp.transformers.{model_name}.configuration")
        for key in dir(configuration_module):
            value = getattr(configuration_module, key)
            if inspect.isclass(value) and issubclass(value, PretrainedConfig):
                mappings[model_name].append(value)

    return mappings


class AutoConfig(PretrainedConfig):
    """
    AutoConfig is a generic config class that will be instantiated as one of the
    base PretrainedConfig classes when created with the AutoConfig.from_pretrained() classmethod.
    """

    MAPPING_NAMES: Dict[str, List[Type[PretrainedConfig]]] = get_configurations()

    # cache the builtin pretrained-model-name to Model Class
    name2class = None
    config_file = "config.json"

    # TODO(wj-Mcat): the supporting should be removed after v2.6
    legacy_config_file = "config.json"

    @classmethod
    def _get_config_class_from_config(
        cls, pretrained_model_name_or_path: str, config_file_path: str
    ) -> PretrainedConfig:
        with io.open(config_file_path, encoding="utf-8") as f:
            config = json.load(f)

        # add support for legacy config
        if "init_class" in config:
            architectures = [config.pop("init_class")]
        else:
            architectures = config.pop("architectures", None)
            if architectures is None:
                return cls

        model_name = architectures[0]
        model_class = import_module(f"paddlenlp.transformers.{model_name}")

        assert inspect.isclass(model_class) and issubclass(
            model_class, PretrainedModel
        ), f"<{model_class}> should be a PretarinedModel class, but <{type(model_class)}>"

        return cls if model_class.config_class is None else model_class.config_class

    @classmethod
    def from_file(cls, config_file: str, **kwargs) -> AutoConfig:
        """construct configuration with AutoConfig class to enable normal loading

        Args:
            config_file (str): the path of config file

        Returns:
            AutoConfig: the instance of AutoConfig
        """
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        config.update(kwargs)
        return cls(**config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, from_hf_hub=False, **kwargs):
        """
        Creates an instance of `AutoConfig`. Related resources are loaded by
        specifying name of a built-in pretrained model, or a community-contributed
        pretrained model, or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model or dir path
                to load from. The string can be:

                - Name of built-in pretrained model
                - Name of a community-contributed pretrained model.
                - Local directory path which contains processor related resources
                  and processor config file ("processor_config.json").
            *args (tuple): position arguments for model `__init__`. If provided,
                use these as position argument values for processor initialization.
            **kwargs (dict): keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for processor
                initialization.

        Returns:
            PretrainedConfig: An instance of `PretrainedConfig`.


        Example:
            .. code-block::
            from paddlenlp.transformers import AutoConfig
            config = AutoConfig.from_pretrained("bert-base-uncased")
            config.save_pretrained('./bert-base-uncased')
        """
        subfolder = kwargs.pop("subfolder", None)
        cache_dir = resolve_cache_dir(
            pretrained_model_name_or_path, from_hf_hub=from_hf_hub, cache_dir=kwargs.pop("cache_dir", None)
        )

        if not cls.name2class:
            cls.name2class = {}
            for model_classes in cls.MAPPING_NAMES.values():
                for model_class in model_classes:
                    cls.name2class.update(
                        {model_name: model_class for model_name in model_class.pretrained_init_configuration.keys()}
                    )

        # From built-in pretrained models
        if pretrained_model_name_or_path in cls.name2class:
            return cls.name2class[pretrained_model_name_or_path].from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )

        # From local dir path
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, cls.config_file)
            if not os.path.exists(config_file):
                # try to load legacy config file
                legacy_config_file = os.path.join(pretrained_model_name_or_path, cls.legacy_config_file)
                if not os.path.exists(legacy_config_file):
                    raise ValueError(
                        f"config file<{cls.config_file}> or legacy config file<{cls.legacy_config_file}> not found"
                    )

                logger.warning(f"loading legacy config file<{cls.legacy_config_file}> ...")
                config_file = legacy_config_file

            config_class = cls._get_config_class_from_config(pretrained_model_name_or_path, config_file)
            logger.info("We are using %s to load '%s'." % (config_class, pretrained_model_name_or_path))
            if config_class is cls:
                return cls.from_file(config_file)
            return config_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        elif from_hf_hub:
            file = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename=cls.config_file,
                cache_dir=cache_dir,
                subfolder=subfolder,
                library_name="PaddleNLP",
                library_version=__version__,
            )
            # from local dir path
            return cls.from_pretrained(os.path.dirname(file))

        # Assuming from community-contributed pretrained models
        else:
            # add support for legacy config file ...
            community_config_path = "/".join([COMMUNITY_MODEL_PREFIX, pretrained_model_name_or_path, cls.config_file])
            if not url_file_exists(community_config_path):
                legacy_community_config_path = "/".join(
                    [COMMUNITY_MODEL_PREFIX, pretrained_model_name_or_path, cls.legacy_config_file]
                )
                if not url_file_exists(legacy_community_config_path):
                    raise RuntimeError(
                        f"Can't load Config for '{pretrained_model_name_or_path}'.\n"
                        f"Please make sure that '{pretrained_model_name_or_path}' is:\n"
                        "- a correct model-identifier of built-in pretrained models,\n"
                        "- or a correct model-identifier of community-contributed pretrained models,\n"
                        "- or the correct path to a directory containing relevant config files.\n"
                    )
                logger.warning(f"loading legacy config file<{cls.legacy_config_file}> ...")
                community_config_path = legacy_community_config_path

            resolved_config_file = get_path_from_url_with_filelock(community_config_path, cache_dir)

            config_class = cls._get_config_class_from_config(pretrained_model_name_or_path, resolved_config_file)
            logger.info("We are using %s to load '%s'." % (config_class, pretrained_model_name_or_path))
            if config_class is cls:
                return cls.from_file(resolved_config_file, **kwargs)

            return config_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
