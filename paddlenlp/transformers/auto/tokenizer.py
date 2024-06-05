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

from ...utils.download import resolve_file_path
from ...utils.import_utils import import_module, is_fast_tokenizer_available
from ...utils.log import logger

__all__ = [
    "AutoTokenizer",
]

TOKENIZER_MAPPING_NAMES = OrderedDict(
    [
        ("AlbertEnglishTokenizer", "albert"),
        ("AlbertChineseTokenizer", "albert"),
        ("BertJapaneseTokenizer", "bert_japanese"),
        ("BigBirdTokenizer", "bigbird"),
        ("BlenderbotSmallTokenizer", "blenderbot_small"),
        ("BlenderbotTokenizer", "blenderbot"),
        ("ChatGLMTokenizer", "chatglm"),
        ("ChatGLMv2Tokenizer", "chatglm_v2"),
        ("ChineseBertTokenizer", "chinesebert"),
        ("ConvBertTokenizer", "convbert"),
        ("CTRLTokenizer", "ctrl"),
        ("DalleBartTokenizer", "dallebart"),
        ("DistilBertTokenizer", "distilbert"),
        ("ElectraTokenizer", "electra"),
        ("ErnieCtmTokenizer", "ernie_ctm"),
        ("ErnieDocTokenizer", "ernie_doc"),
        ("ErnieDocBPETokenizer", "ernie_doc"),
        ("ErnieGramTokenizer", "ernie_gram"),
        ("ErnieLayoutTokenizer", "ernie_layout"),
        ("ErnieMTokenizer", "ernie_m"),
        ("ErnieCodeTokenizer", "ernie_code"),
        ("ErnieTokenizer", "ernie"),
        ("FNetTokenizer", "fnet"),
        ("FunnelTokenizer", "funnel"),
        ("LlamaTokenizer", "llama"),
        ("LayoutXLMTokenizer", "layoutxlm"),
        ("LayoutLMv2Tokenizer", "layoutlmv2"),
        ("LayoutLMTokenizer", "layoutlm"),
        ("LukeTokenizer", "luke"),
        ("MBartTokenizer", "mbart"),
        ("MBart50Tokenizer", "mbart"),
        ("MegatronBertTokenizer", "megatronbert"),
        ("MobileBertTokenizer", "mobilebert"),
        ("MPNetTokenizer", "mpnet"),
        ("NeZhaTokenizer", "nezha"),
        ("NystromformerTokenizer", "nystromformer"),
        ("PPMiniLMTokenizer", "ppminilm"),
        ("ProphetNetTokenizer", "prophetnet"),
        ("ReformerTokenizer", "reformer"),
        ("RemBertTokenizer", "rembert"),
        ("RobertaChineseTokenizer", "roberta"),
        ("RobertaBPETokenizer", "roberta"),
        ("RoFormerTokenizer", "roformer"),
        ("RoFormerv2Tokenizer", "roformerv2"),
        ("SkepTokenizer", "skep"),
        ("SqueezeBertTokenizer", "squeezebert"),
        ("TinyBertTokenizer", "tinybert"),
        ("UnifiedTransformerTokenizer", "unified_transformer"),
        ("UNIMOTokenizer", "unimo"),
        ("XLNetTokenizer", "xlnet"),
        ("XLMTokenizer", "xlm"),
        ("GPTTokenizer", "gpt"),
        ("GPTChineseTokenizer", "gpt"),
        ("T5Tokenizer", "t5"),
        ("BertTokenizer", "bert"),
        ("BartTokenizer", "bart"),
        ("GAUAlphaTokenizer", "gau_alpha"),
        ("CodeGenTokenizer", "codegen"),
        ("CLIPTokenizer", "clip"),
        ("ArtistTokenizer", "artist"),
        ("ChineseCLIPTokenizer", "chineseclip"),
        ("ErnieViLTokenizer", "ernie_vil"),
        ("PegasusChineseTokenizer", "pegasus"),
        ("GLMBertTokenizer", "glm"),
        ("GLMChineseTokenizer", "glm"),
        ("GLMGPT2Tokenizer", "glm"),
        ("BloomTokenizer", "bloom"),
        ("SpeechT5Tokenizer", "speecht5"),
        ("QWenTokenizer", "qwen"),
        ("GemmaTokenizer", "gemma"),
    ]
)

FAST_TOKENIZER_MAPPING_NAMES = OrderedDict(
    [
        ("BertFastTokenizer", "bert"),
        ("ErnieFastTokenizer", "ernie"),
        ("TinyBertFastTokenizer", "tinybert"),
        ("ErnieMFastTokenizer", "ernie_m"),
        ("NystromformerFastTokenizer", "nystromformer"),
    ]
)
# For FastTokenizer
if is_fast_tokenizer_available():
    TOKENIZER_MAPPING_NAMES.update(FAST_TOKENIZER_MAPPING_NAMES)


def get_configurations():
    MAPPING_NAMES = OrderedDict()
    for key, class_name in TOKENIZER_MAPPING_NAMES.items():
        fast_name = ""
        if "Fast" in key:
            fast_name = "fast_"
        import_class = importlib.import_module(f"paddlenlp.transformers.{class_name}.{fast_name}tokenizer")
        tokenizer_name = getattr(import_class, key)
        name = tuple(tokenizer_name.pretrained_init_configuration.keys())
        # FastTokenizer will share the same config with python tokenizer
        # So same config would map more than one tokenizer
        if MAPPING_NAMES.get(name, None) is None:
            MAPPING_NAMES[name] = []
        # (tokenizer_name, is_fast)
        MAPPING_NAMES[name].append((tokenizer_name, fast_name != ""))
    return MAPPING_NAMES


class AutoTokenizer:
    """
    AutoClass can help you automatically retrieve the relevant model given the provided
    pretrained weights/vocabulary.
    AutoTokenizer is a generic tokenizer class that will be instantiated as one of the
    base tokenizer classes when created with the AutoTokenizer.from_pretrained() classmethod.
    """

    MAPPING_NAMES = get_configurations()
    _tokenizer_mapping = MAPPING_NAMES
    _name_mapping = TOKENIZER_MAPPING_NAMES
    _fast_name_mapping = FAST_TOKENIZER_MAPPING_NAMES
    tokenizer_config_file = "tokenizer_config.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path).`"
        )

    @classmethod
    def _get_fast_tokenizer_class(cls, init_class, class_name):
        tokenizer_class = None
        if is_fast_tokenizer_available():
            is_support_fast_tokenizer = False
            init_class_prefix = init_class[:-9]
            for fast_tokenizer_class, name in cls._fast_name_mapping.items():
                fast_tokenizer_class_prefix = fast_tokenizer_class[:-9]
                if name == class_name and fast_tokenizer_class_prefix.startswith(init_class_prefix):
                    is_support_fast_tokenizer = True
                    import_class = import_module(f"paddlenlp.transformers.{class_name}.fast_tokenizer")
                    tokenizer_class = getattr(import_class, fast_tokenizer_class)
                    break
            if not is_support_fast_tokenizer:
                logger.warning(
                    f"The tokenizer {tokenizer_class} doesn't have the fast version."
                    " Please check the map `paddlenlp.transformers.auto.tokenizer.FAST_TOKENIZER_MAPPING_NAMES`"
                    " to see which fast tokenizers are currently supported."
                )
        else:
            logger.warning(
                "Can't find the fast_tokenizer package, "
                "please ensure install fast_tokenizer correctly. "
                "You can install fast_tokenizer by `pip install fast-tokenizer-python`."
            )
        return tokenizer_class

    @classmethod
    def _get_tokenizer_class_from_config(cls, pretrained_model_name_or_path, config_file_path, use_fast):
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
                    if use_fast:
                        tokenizer_class = cls._get_fast_tokenizer_class(init_class, class_name)
                except:
                    # use the non fast tokenizer as default
                    logger.warning(
                        "`use_fast` is set to `True` but the tokenizer class does not have a fast version. "
                        " Falling back to the slow version."
                    )
                try:
                    if tokenizer_class is None:
                        tokenizer_class = getattr(import_class, init_class)
                except:
                    raise ValueError(
                        f"Tokenizer class {init_class} is not currently imported, if you use fast tokenizer, please set use_fast to True."
                    )
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
                    if use_fast:
                        fast_tokenizer_class = cls._get_fast_tokenizer_class(init_class, class_name)
                        tokenizer_class = fast_tokenizer_class if fast_tokenizer_class else tokenizer_class
                    break
            return tokenizer_class

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
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
            *args (tuple): position arguments for model `__init__`. If provided,
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
        # Default not to use fast tokenizer
        use_fast = kwargs.pop("use_fast", False)
        cache_dir = kwargs.get("cache_dir", None)
        subfolder = kwargs.get("subfolder", "")
        if subfolder is None:
            subfolder = ""
        from_aistudio = kwargs.get("from_aistudio", False)
        from_hf_hub = kwargs.get("from_hf_hub", False)

        if "use_faster" in kwargs:
            use_fast = kwargs.pop("use_faster", False)
            logger.warning("The keyword argument `use_faster` is deprecated in future, please use `use_fast` instead")

        all_tokenizer_names = []
        for names, tokenizer_class in cls._tokenizer_mapping.items():
            for name in names:
                all_tokenizer_names.append(name)

        # From built-in pretrained models
        if pretrained_model_name_or_path in all_tokenizer_names:
            for names, tokenizer_classes in cls._tokenizer_mapping.items():
                for pattern in names:
                    if pattern == pretrained_model_name_or_path:
                        actual_tokenizer_class = None
                        # Default setting the python tokenizer to actual_tokenizer_class
                        for tokenizer_class in tokenizer_classes:
                            if not tokenizer_class[1]:
                                actual_tokenizer_class = tokenizer_class[0]
                                break
                        if use_fast:
                            if is_fast_tokenizer_available():
                                is_support_fast_tokenizer = False
                                for tokenizer_class in tokenizer_classes:
                                    if tokenizer_class[1]:
                                        actual_tokenizer_class = tokenizer_class[0]
                                        is_support_fast_tokenizer = True
                                        break
                                if not is_support_fast_tokenizer:
                                    logger.warning(
                                        f"The tokenizer {actual_tokenizer_class} doesn't have the fast version."
                                        " Please check the map `paddlenlp.transformers.auto.tokenizer.FAST_TOKENIZER_MAPPING_NAMES`"
                                        " to see which fast tokenizers are currently supported."
                                    )
                            else:
                                logger.warning(
                                    "Can't find the fast_tokenizer package, "
                                    "please ensure install fast_tokenizer correctly. "
                                    "You can install fast_tokenizer by `pip install fast-tokenizer-python`."
                                )

                        logger.info(f"We are using {tokenizer_class} to load '{pretrained_model_name_or_path}'.")
                        return actual_tokenizer_class.from_pretrained(
                            pretrained_model_name_or_path, *model_args, **kwargs
                        )

        config_file = resolve_file_path(
            pretrained_model_name_or_path,
            cls.tokenizer_config_file,
            subfolder,
            cache_dir=cache_dir,
            from_hf_hub=from_hf_hub,
            from_aistudio=from_aistudio,
        )
        if config_file is not None and os.path.exists(config_file):
            tokenizer_class = cls._get_tokenizer_class_from_config(
                pretrained_model_name_or_path, config_file, use_fast
            )
            logger.info(f"We are using {tokenizer_class} to load '{pretrained_model_name_or_path}'.")
            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        else:
            raise RuntimeError(
                f"Can't load tokenizer for '{pretrained_model_name_or_path}'.\n"
                f"Please make sure that '{pretrained_model_name_or_path}' is:\n"
                "- a correct model-identifier of built-in pretrained models,\n"
                "- or a correct model-identifier of community-contributed pretrained models,\n"
                "- or the correct path to a directory containing relevant tokenizer files.\n"
            )
