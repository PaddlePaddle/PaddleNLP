# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

from ...utils import is_tokenizers_available
from ...utils.download import resolve_file_path
from ...utils.import_utils import import_module
from ...utils.log import logger
from ..configuration_utils import PretrainedConfig
from ..tokenizer_utils_base import TOKENIZER_CONFIG_FILE
from ..tokenizer_utils_fast import PretrainedTokenizerFast
from .configuration import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    config_class_to_model_type,
    model_type_to_module_name,
)
from .factory import _LazyAutoMapping

__all__ = [
    "AutoTokenizer",
]

if TYPE_CHECKING:
    TOKENIZER_MAPPING_NAMES: OrderedDict[str, Tuple[Optional[str], Optional[str]]] = OrderedDict()
else:
    TOKENIZER_MAPPING_NAMES = OrderedDict(
        [
            ("albert", "AlbertTokenizer"),
            ("albert_chinese", "AlbertChineseTokenizer"),
            ("albert_english", "AlbertEnglishTokenizer"),
            ("bart", "BartTokenizer"),
            ("bert", "BertTokenizer"),
            ("blenderbot", "BlenderbotTokenizer"),
            ("bloom", "BloomTokenizer"),
            ("clip", "CLIPTokenizer"),
            ("codegen", "CodeGenTokenizer"),
            ("convbert", "ConvBertTokenizer"),
            ("ctrl", "CTRLTokenizer"),
            ("distilbert", "DistilBertTokenizer"),
            ("electra", "ElectraTokenizer"),
            ("ernie", "ErnieTokenizer"),
            ("ernie_m", "ErnieMTokenizer"),
            ("fnet", "FNetTokenizer"),
            ("funnel", "FunnelTokenizer"),
            ("gemma", "GemmaTokenizer"),
            ("jamba", "JambaTokenizer"),
            ("layoutlm", "LayoutLMTokenizer"),
            ("layoutlmv2", "LayoutLMv2Tokenizer"),
            ("layoutxlm", "LayoutXLMTokenizer"),
            (
                "llama",
                (
                    "LlamaTokenizer",
                    "LlamaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("luke", "LukeTokenizer"),
            ("mamba", "MambaTokenizer"),
            ("mbart", "MBartTokenizer"),
            ("mbart50", "MBart50Tokenizer"),
            ("mobilebert", "MobileBertTokenizer"),
            ("mpnet", "MPNetTokenizer"),
            ("nezha", "NeZhaTokenizer"),
            ("pegasus", "PegasusChineseTokenizer"),
            ("prophetnet", "ProphetNetTokenizer"),
            ("reformer", "ReformerTokenizer"),
            ("rembert", "RemBertTokenizer"),
            ("roberta", "RobertaBPETokenizer"),
            ("roformer", "RoFormerTokenizer"),
            ("speecht5", "SpeechT5Tokenizer"),
            ("squeezebert", "SqueezeBertTokenizer"),
            ("t5", "T5Tokenizer"),
            ("xlm", "XLMTokenizer"),
            ("xlnet", "XLNetTokenizer"),
            ("bert_japanese", "BertJapaneseTokenizer"),
            ("bigbird", "BigBirdTokenizer"),
            ("blenderbot_small", "BlenderbotSmallTokenizer"),
            ("chatglm", "ChatGLMTokenizer"),
            ("chatglm_v2", "ChatGLMv2Tokenizer"),
            ("chinesebert", "ChineseBertTokenizer"),
            ("dallebart", "DalleBartTokenizer"),
            ("ernie_ctm", "ErnieCtmTokenizer"),
            ("ernie_doc", "ErnieDocBPETokenizer"),
            ("ernie_gram", "ErnieGramTokenizer"),
            ("ernie_layout", "ErnieLayoutTokenizer"),
            ("ernie_code", "ErnieCodeTokenizer"),
            ("megatronbert", "MegatronBertTokenizer"),
            ("nystromformer", "NystromformerTokenizer"),
            ("ppminilm", "PPMiniLMTokenizer"),
            ("roformerv2", "RoFormerv2Tokenizer"),
            ("skep", "SkepTokenizer"),
            ("tinybert", "TinyBertTokenizer"),
            ("unified_transformer", "UnifiedTransformerTokenizer"),
            ("unimo", "UNIMOTokenizer"),
            ("gpt", "GPTChineseTokenizer"),
            ("gau_alpha", "GAUAlphaTokenizer"),
            ("artist", "ArtistTokenizer"),
            ("chineseclip", "ChineseCLIPTokenizer"),
            ("ernie_vil", "ErnieViLTokenizer"),
            ("glm", "GLMGPT2Tokenizer"),
            ("qwen", "QWenTokenizer"),
            ("yuan", "YuanTokenizer"),
        ]
    )

TOKENIZER_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TOKENIZER_MAPPING_NAMES)

CONFIG_TO_TYPE = {v: k for k, v in CONFIG_MAPPING_NAMES.items()}


def get_configurations():
    MAPPING_NAMES = OrderedDict()
    for key, class_name in TOKENIZER_MAPPING_NAMES.items():
        import_class = importlib.import_module(f"paddlenlp.transformers.{class_name}.tokenizer")
        tokenizer_name = getattr(import_class, key)
        name = tuple(tokenizer_name.pretrained_init_configuration.keys())
        MAPPING_NAMES[name] = tokenizer_name
    return MAPPING_NAMES


def tokenizer_class_from_name(class_name: str):
    if class_name == "PretrainedTokenizerFast":
        return PretrainedTokenizerFast

    for module_name, tokenizers in TOKENIZER_MAPPING_NAMES.items():
        if class_name in tokenizers:
            module_name = model_type_to_module_name(module_name)
            print(f"module_name: {module_name}")
            try:
                module = importlib.import_module(f".{module_name}", "paddlenlp.transformers")
                return getattr(module, class_name)
            except AttributeError:
                try:
                    print(f"module: {module}")
                    module = importlib.import_module(f".{module_name}.tokenizer", "paddlenlp.transformers")

                    return getattr(module, class_name)
                except AttributeError:
                    raise ValueError(f"Tokenizer class {class_name} is not currently imported.")

    for config, tokenizers in TOKENIZER_MAPPING._extra_content.items():
        for tokenizer in tokenizers:
            if getattr(tokenizer, "__name__", None) == class_name:
                return tokenizer

    # We did not fine the class, but maybe it's because a dep is missing. In that case, the class will be in the main
    # init and we return the proper dummy to get an appropriate error message.
    main_module = importlib.import_module("paddlenlp")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


def get_tokenizer_config(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
    **kwargs,
):
    """
    Loads the tokenizer configuration from a pretrained model tokenizer configuration.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download:
            Deprecated and ignored. All downloads are now resumed by default when possible.
            Will be removed in v5 of Transformers.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the tokenizer config is located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Dict`: The configuration of the tokenizer.

    Examples:

    ```python
    # Download configuration from huggingface.co and cache.
    tokenizer_config = get_tokenizer_config("google-bert/bert-base-uncased")
    # This model does not have a tokenizer config so the result will be an empty dict.
    tokenizer_config = get_tokenizer_config("FacebookAI/xlm-roberta-base")

    # Save a pretrained tokenizer locally and you can reload its config
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    tokenizer.save_pretrained("tokenizer-test")
    tokenizer_config = get_tokenizer_config("tokenizer-test")
    ```"""

    resolved_config_file = resolve_file_path(
        pretrained_model_name_or_path,
        TOKENIZER_CONFIG_FILE,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        subfolder=subfolder,
    )
    if resolved_config_file is None:
        logger.info("Could not locate the tokenizer configuration file, will try to use the model config instead.")
        return {}
    with open(resolved_config_file, encoding="utf-8") as reader:
        result = json.load(reader)

    return result


class AutoTokenizer:
    """
    AutoClass can help you automatically retrieve the relevant model given the provided
    pretrained weights/vocabulary.
    AutoTokenizer is a generic tokenizer class that will be instantiated as one of the
    base tokenizer classes when created with the AutoTokenizer.from_pretrained() classmethod.
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoTokenizer is designed to be instantiated "
            "using the `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def _get_tokenizer_class_from_config(cls, pretrained_model_name_or_path, config_file_path, use_fast=None):
        if use_fast is not None:
            raise ValueError("use_fast is deprecated")
        with io.open(config_file_path, encoding="utf-8") as f:
            init_kwargs = json.load(f)
        # class name corresponds to this configuration
        init_class = init_kwargs.pop("init_class", None)
        if init_class is None:
            init_class = init_kwargs.pop("tokenizer_class", None)

        if init_class:
            if init_class in cls._name_mapping:
                class_name = cls._name_mapping[init_class]
                import_class = import_module(f"paddlenlp.transformers.{class_name}.tokenizer")
                tokenizer_class = None
                try:
                    if tokenizer_class is None:
                        tokenizer_class = getattr(import_class, init_class)
                except:
                    raise ValueError(f"Tokenizer class {init_class} is not currently imported.")
                return tokenizer_class
            else:
                import_class = import_module("paddlenlp.transformers")
                tokenizer_class = getattr(import_class, init_class, None)
                assert tokenizer_class is not None, f"Can't find tokenizer {init_class}"
                return tokenizer_class

        # If no `init_class`, we use pattern recognition to recognize the tokenizer class.
        else:
            # TODO: Potential issue https://github.com/PaddlePaddle/PaddleNLP/pull/3786#discussion_r1024689810
            logger.info("We use pattern recognition to recognize the Tokenizer class.")
            for key, pattern in cls._name_mapping.items():
                if pattern in pretrained_model_name_or_path.lower():
                    init_class = key
                    class_name = cls._name_mapping[init_class]
                    import_class = import_module(f"paddlenlp.transformers.{class_name}.tokenizer")
                    tokenizer_class = getattr(import_class, init_class)
                    break
            return tokenizer_class

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Creates an instance of `AutoTokenizer`. Related resources are loaded by
        specifying name of a built-in pretrained model, or a community-contributed
        pretrained model, or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model or dir path
                to load from. The string can be:

                - Name of built-in pretrained model
                - Name of a community-contributed pretrained model.
                - Local directory path which contains tokenizer related resources
                  and tokenizer config file ("tokenizer_config.json").
            *inputs (tuple): position arguments for model `__init__`. If provided,
                use these as position argument values for tokenizer initialization.
            **kwargs (dict): keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for tokenizer
                initialization.

        Returns:
            PretrainedTokenizer: An instance of `PretrainedTokenizer`.

        Example:
            .. code-block::

                from paddlenlp.transformers import AutoTokenizer

                # Name of built-in pretrained model
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                print(type(tokenizer))
                # <class 'paddlenlp.transformers.bert.tokenizer.BertTokenizer'>

                # Name of community-contributed pretrained model
                tokenizer = AutoTokenizer.from_pretrained('yingyibiao/bert-base-uncased-sst-2-finetuned')
                print(type(tokenizer))
                # <class 'paddlenlp.transformers.bert.tokenizer.BertTokenizer'>

                # Load from local directory path
                tokenizer = AutoTokenizer.from_pretrained('./my_bert/')
                print(type(tokenizer))
                # <class 'paddlenlp.transformers.bert.tokenizer.BertTokenizer'>
        """
        config = kwargs.pop("config", None)
        kwargs["_from_auto"] = True

        use_fast = kwargs.pop("use_fast", True)
        tokenizer_type = kwargs.pop("tokenizer_type", None)
        if tokenizer_type is not None:
            # TODO: Support tokenizer_type
            raise NotImplementedError("tokenizer_type is not supported yet.")

        tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
        config_tokenizer_class = tokenizer_config.get("tokenizer_class")
        if config_tokenizer_class is None:
            if not isinstance(config, PretrainedConfig):
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            config_tokenizer_class = config.tokenizer_class
        if config_tokenizer_class is not None:
            tokenizer_class = None
            if use_fast and not config_tokenizer_class.endswith("Fast"):
                tokenizer_class_candidate = f"{config_tokenizer_class}Fast"
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)
            if tokenizer_class is None:
                tokenizer_class_candidate = config_tokenizer_class
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)
            if tokenizer_class is None:
                raise ValueError(
                    f"Tokenizer class {tokenizer_class_candidate} does not exist or is not currently imported."
                )
            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        # TODO: if model is an encoder decoder

        model_type = config_class_to_model_type(type(config).__name__)
        if model_type is not None:
            tokenizer_class_py = TOKENIZER_MAPPING[type(config)]
            if isinstance(tokenizer_class_py, (list, tuple)):
                if len(tokenizer_class_py) == 2:
                    tokenizer_class_fast = tokenizer_class_py[1]
                    tokenizer_class_py = tokenizer_class_py[0]
                else:
                    tokenizer_class_fast = None
            else:
                tokenizer_class_fast = None
            if tokenizer_class_fast and (use_fast or tokenizer_class_py is None):
                return tokenizer_class_fast.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
            else:
                if tokenizer_class_py is not None:
                    return tokenizer_class_py.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
                else:
                    raise ValueError(
                        "This tokenizer cannot be instantiated. Please make sure you have `sentencepiece` installed "
                        "in order to use this tokenizer."
                    )
        raise RuntimeError(
            f"Can't load tokenizer for '{pretrained_model_name_or_path}'.\n"
            f"Please make sure that '{pretrained_model_name_or_path}' is:\n"
            "- a correct model-identifier of built-in pretrained models,\n"
            "- or a correct model-identifier of community-contributed pretrained models,\n"
            "- or the correct path to a directory containing relevant tokenizer files.\n"
        )
