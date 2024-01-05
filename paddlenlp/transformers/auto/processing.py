# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import importlib
import io
import json
import os
from collections import OrderedDict

from huggingface_hub import hf_hub_download

from ... import __version__
from ...utils.downloader import COMMUNITY_MODEL_PREFIX, get_path_from_url_with_filelock
from ...utils.import_utils import import_module
from ...utils.log import logger
from ..aistudio_utils import aistudio_download
from ..utils import resolve_cache_dir

__all__ = [
    "AutoProcessor",
]

PROCESSOR_MAPPING_NAMES = OrderedDict(
    [
        ("ChineseCLIPProcessor", "chineseclip"),
        ("CLIPProcessor", "clip"),
        ("ErnieViLProcessor", "ernie_vil"),
        ("CLIPSegProcessor", "clipseg"),
        ("SpeechT5Processor", "speecht5"),
        ("ClapProcessor", "clap"),
    ]
)


def get_configurations():
    MAPPING_NAMES = OrderedDict()
    for key, class_name in PROCESSOR_MAPPING_NAMES.items():
        import_class = importlib.import_module(f"paddlenlp.transformers.{class_name}.processing")
        processor_name = getattr(import_class, key)
        name = tuple(processor_name.pretrained_init_configuration.keys())
        if MAPPING_NAMES.get(name, None) is None:
            MAPPING_NAMES[name] = []
        MAPPING_NAMES[name].append(processor_name)
    return MAPPING_NAMES


class AutoProcessor:
    """
    AutoClass can help you automatically retrieve the relevant model given the provided
    pretrained weights/vocabulary.
    Autoprocessor is a generic processor class that will be instantiated as one of the
    base processor classes when created with the Autoprocessor.from_pretrained() classmethod.
    """

    MAPPING_NAMES = get_configurations()
    _processor_mapping = MAPPING_NAMES
    _name_mapping = PROCESSOR_MAPPING_NAMES
    processor_config_file = "preprocessor_config.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path).`"
        )

    @classmethod
    def _get_processor_class_from_config(cls, pretrained_model_name_or_path, config_file_path):
        with io.open(config_file_path, encoding="utf-8") as f:
            init_kwargs = json.load(f)
        # class name corresponds to this configuration
        init_class = init_kwargs.pop("init_class", None)
        if init_class is None:
            init_class = init_kwargs.pop("processor_class", None)
            if init_class is None:
                init_class = init_kwargs.pop("image_processor_type", None)
                # replace old name to new name
                if init_class is not None and init_class.endswith("ImageProcessor"):
                    init_class = init_class.replace("ImageProcessor", "Processor")
            if init_class is None:
                init_class = init_kwargs.pop("feature_extractor_type", None)
                # replace old name to new name
                if init_class is not None and init_class.endswith("FeatureExtractor"):
                    init_class = init_class.replace("FeatureExtractor", "Processor")

        if init_class:
            try:
                class_name = cls._name_mapping[init_class]
                import_class = import_module(f"paddlenlp.transformers.{class_name}.processing")
                processor_class = getattr(import_class, init_class)
                return processor_class
            except Exception:
                init_class = None

        # If no `init_class`, we use pattern recognition to recognize the processor class.
        if init_class is None:
            logger.info("We use pattern recognition to recognize the processor class.")
            for key, pattern in cls._name_mapping.items():
                if pattern in pretrained_model_name_or_path.lower():
                    init_class = key
                    class_name = cls._name_mapping[init_class]
                    import_class = import_module(f"paddlenlp.transformers.{class_name}.processor")
                    processor_class = getattr(import_class, init_class)
                    break
            return processor_class

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Creates an instance of `Autoprocessor`. Related resources are loaded by
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
            Pretrainedprocessor: An instance of `Pretrainedprocessor`.


        Example:
            .. code-block::
            from paddlenlp.transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            processor.save_pretrained('clip_processor')
        """
        cache_dir = kwargs.get("cache_dir", None)
        subfolder = kwargs.get("subfolder", "")
        if subfolder is None:
            subfolder = ""
        from_aistudio = kwargs.get("from_aistudio", False)
        from_hf_hub = kwargs.get("from_hf_hub", False)
        cache_dir = resolve_cache_dir(from_hf_hub, from_aistudio, cache_dir)
        kwargs["subfolder"] = subfolder
        kwargs["cache_dir"] = cache_dir

        all_processor_names = []
        for names, processor_class in cls._processor_mapping.items():
            for name in names:
                all_processor_names.append(name)

        # From local dir path
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, subfolder, cls.processor_config_file)
            if os.path.exists(config_file):
                processor_class = cls._get_processor_class_from_config(pretrained_model_name_or_path, config_file)
                logger.info("We are using %s to load '%s'." % (processor_class, pretrained_model_name_or_path))
                return processor_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        # From built-in pretrained models
        elif pretrained_model_name_or_path in all_processor_names:
            for names, processor_classes in cls._processor_mapping.items():
                for pattern in names:
                    if pattern == pretrained_model_name_or_path:
                        actual_processor_class = processor_classes[0]
                        logger.info(
                            "We are using %s to load '%s'." % (actual_processor_class, pretrained_model_name_or_path)
                        )
                        return actual_processor_class.from_pretrained(
                            pretrained_model_name_or_path, *model_args, **kwargs
                        )

        # From AI Studio or HF Hub
        elif from_aistudio or from_hf_hub:
            if from_aistudio:
                config_file = aistudio_download(
                    repo_id=pretrained_model_name_or_path,
                    filename=cls.processor_config_file,
                    cache_dir=cache_dir,
                    subfolder=subfolder,
                )
            else:
                config_file = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename=cls.processor_config_file,
                    subfolder=subfolder,
                    cache_dir=cache_dir,
                    library_name="PaddleNLP",
                    library_version=__version__,
                )
            if os.path.exists(config_file):
                processor_class = cls._get_processor_class_from_config(
                    pretrained_model_name_or_path,
                    config_file,
                )
                logger.info(f"We are using {processor_class} to load '{pretrained_model_name_or_path}'.")
                return processor_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        # Assuming from community-contributed pretrained models
        else:
            url_list = [COMMUNITY_MODEL_PREFIX, pretrained_model_name_or_path, cls.processor_config_file]
            cache_dir = os.path.join(cache_dir, pretrained_model_name_or_path, subfolder)
            if subfolder != "":
                url_list.insert(2, subfolder)
            community_config_path = "/".join(url_list)

            try:
                resolved_vocab_file = get_path_from_url_with_filelock(community_config_path, cache_dir)
            except RuntimeError as err:
                logger.error(err)
                raise RuntimeError(
                    f"Can't load processor for '{pretrained_model_name_or_path}'.\n"
                    f"Please make sure that '{pretrained_model_name_or_path}' is:\n"
                    "- a correct model-identifier of built-in pretrained models,\n"
                    "- or a correct model-identifier of community-contributed pretrained models,\n"
                    "- or the correct path to a directory containing relevant processor files.\n"
                )

            if os.path.exists(resolved_vocab_file):
                processor_class = cls._get_processor_class_from_config(
                    pretrained_model_name_or_path, resolved_vocab_file
                )
                logger.info("We are using %s to load '%s'." % (processor_class, pretrained_model_name_or_path))
                return processor_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
