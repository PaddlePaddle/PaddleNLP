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
import os
import io
import importlib
import json
from collections import OrderedDict
from paddlenlp.transformers import *
from paddlenlp.utils.downloader import COMMUNITY_MODEL_PREFIX, get_path_from_url
from paddlenlp.utils.env import MODEL_HOME
from paddlenlp.utils.log import logger
from paddlenlp.utils.import_utils import is_faster_tokenizer_available

__all__ = [
    "AutoTokenizer",
]

TOKENIZER_MAPPING_NAMES = OrderedDict([
    ("AlbertEnglishTokenizer", "albert"),
    ("AlbertChineseTokenizer", "albert"),
    ("BertJapaneseTokenizer", "bert_japanese"),
    ("BigBirdTokenizer", "bigbird"),
    ("BlenderbotSmallTokenizer", "blenderbot_small"),
    ("BlenderbotTokenizer", "blenderbot"),
    ("ChineseBertTokenizer", "chinesebert"),
    ("ConvBertTokenizer", "convbert"),
    ("CTRLTokenizer", "ctrl"),
    ("DistilBertTokenizer", "distilbert"),
    ("ElectraTokenizer", "electra"),
    ("ErnieCtmTokenizer", "ernie_ctm"),
    ("ErnieDocTokenizer", "ernie_doc"),
    ("ErnieDocBPETokenizer", "ernie_doc"),
    ("ErnieGramTokenizer", "ernie_gram"),
    ("ErnieMTokenizer", "ernie_m"),
    ("ErnieTokenizer", "ernie"),
    ("FNetTokenizer", "fnet"),
    ("FunnelTokenizer", "funnel"),
    ("LayoutXLMTokenizer", "layoutxlm"),
    ("LayoutLMv2Tokenizer", "layoutlmv2"),
    ("LayoutLMTokenizer", "layoutlm"),
    ("LukeTokenizer", "luke"),
    ("MBartTokenizer", "mbart"),
    ("MegatronBertTokenizer", "megatronbert"),
    ("MobileBertTokenizer", "mobilebert"),
    ("MPNetTokenizer", "mpnet"),
    ("NeZhaTokenizer", "nezha"),
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
    ("T5Tokenizer", 't5'),
    ("BertTokenizer", "bert"),
    ("BartTokenizer", "bart"),
    ("GAUAlphaTokenizer", "gau_alpha"),
    ("CodeGenTokenizer", "codegen"),
])

FASTER_TOKENIZER_MAPPING_NAMES = OrderedDict([
    ("BertFasterTokenizer", "bert"), ("ErnieFasterTokenizer", "ernie"),
    ("TinyBertFasterTokenizer", "tinybert"),
    ("ErnieMFasterTokenizer", "ernie_m")
])
# For FasterTokenizer
if is_faster_tokenizer_available():
    TOKENIZER_MAPPING_NAMES.update(FASTER_TOKENIZER_MAPPING_NAMES)


def get_configurations():
    MAPPING_NAMES = OrderedDict()
    for key, class_name in TOKENIZER_MAPPING_NAMES.items():
        faster_name = ""
        if "Faster" in key:
            faster_name = "faster_"
        import_class = importlib.import_module(
            f"paddlenlp.transformers.{class_name}.{faster_name}tokenizer")
        tokenizer_name = getattr(import_class, key)
        name = tuple(tokenizer_name.pretrained_init_configuration.keys())
        # FasterTokenizer will share the same config with python tokenizer
        # So same config would map more than one tokenizer
        if MAPPING_NAMES.get(name, None) is None:
            MAPPING_NAMES[name] = []
        # (tokenizer_name, is_faster)
        MAPPING_NAMES[name].append((tokenizer_name, faster_name != ""))
    return MAPPING_NAMES


class AutoTokenizer():
    """
    AutoClass can help you automatically retrieve the relevant model given the provided
    pretrained weights/vocabulary.
    AutoTokenizer is a generic tokenizer class that will be instantiated as one of the
    base tokenizer classes when created with the AutoTokenizer.from_pretrained() classmethod.
    """
    MAPPING_NAMES = get_configurations()
    _tokenizer_mapping = MAPPING_NAMES
    _name_mapping = TOKENIZER_MAPPING_NAMES
    tokenizer_config_file = "tokenizer_config.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path).`"
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
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
        # default not to use faster tokenizer
        use_faster = kwargs.pop("use_faster", False)

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
                        if use_faster:
                            if is_faster_tokenizer_available():
                                is_support_faster_tokenizer = False
                                for tokenizer_class in tokenizer_classes:
                                    if tokenizer_class[1]:
                                        actual_tokenizer_class = tokenizer_class[
                                            0]
                                        is_support_faster_tokenizer = True
                                        break
                                if not is_support_faster_tokenizer:
                                    logger.warning(
                                        f"The tokenizer {actual_tokenizer_class} doesn't have the faster version."
                                        " Please check the map `paddlenlp.transformers.auto.tokenizer.FASTER_TOKENIZER_MAPPING_NAMES`"
                                        " to see which faster tokenizers are currently supported."
                                    )
                            else:
                                logger.warning(
                                    "Can't find the faster_tokenizer package, "
                                    "please ensure install faster_tokenizer correctly. "
                                    "You can install faster_tokenizer by `pip install faster_tokenizer`."
                                )

                        logger.info("We are using %s to load '%s'." %
                                    (actual_tokenizer_class,
                                     pretrained_model_name_or_path))
                        return actual_tokenizer_class.from_pretrained(
                            pretrained_model_name_or_path, *model_args,
                            **kwargs)
        # From local dir path
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path,
                                       cls.tokenizer_config_file)
            if os.path.exists(config_file):
                with io.open(config_file, encoding="utf-8") as f:
                    init_kwargs = json.load(f)
                # class name corresponds to this configuration
                init_class = init_kwargs.pop("init_class", None)
                if init_class is None:
                    init_class = init_kwargs.pop("tokenizer_class", None)
                if init_class:
                    class_name = cls._name_mapping[init_class]
                    import_class = importlib.import_module(
                        f"paddlenlp.transformers.{class_name}.tokenizer")
                    tokenizer_class = getattr(import_class, init_class)
                    logger.info(
                        "We are using %s to load '%s'." %
                        (tokenizer_class, pretrained_model_name_or_path))
                    return tokenizer_class.from_pretrained(
                        pretrained_model_name_or_path, *model_args, **kwargs)
                # If no `init_class`, we use pattern recognition to recognize the tokenizer class.
                else:
                    print(
                        'We use pattern recognition to recognize the Tokenizer class.'
                    )
                    for key, pattern in cls._name_mapping.items():
                        if pattern in pretrained_model_name_or_path.lower():
                            init_class = key
                            class_name = cls._name_mapping[init_class]
                            import_class = importlib.import_module(
                                f"paddlenlp.transformers.{class_name}.tokenizer"
                            )
                            tokenizer_class = getattr(import_class, init_class)
                            logger.info("We are using %s to load '%s'." %
                                        (tokenizer_class,
                                         pretrained_model_name_or_path))
                            return tokenizer_class.from_pretrained(
                                pretrained_model_name_or_path, *model_args,
                                **kwargs)
        # Assuming from community-contributed pretrained models
        else:
            community_config_path = os.path.join(COMMUNITY_MODEL_PREFIX,
                                                 pretrained_model_name_or_path,
                                                 cls.tokenizer_config_file)

            default_root = os.path.join(MODEL_HOME,
                                        pretrained_model_name_or_path)
            try:
                resolved_vocab_file = get_path_from_url(community_config_path,
                                                        default_root)
            except RuntimeError as err:
                logger.error(err)
                raise RuntimeError(
                    f"Can't load tokenizer for '{pretrained_model_name_or_path}'.\n"
                    f"Please make sure that '{pretrained_model_name_or_path}' is:\n"
                    "- a correct model-identifier of built-in pretrained models,\n"
                    "- or a correct model-identifier of community-contributed pretrained models,\n"
                    "- or the correct path to a directory containing relevant tokenizer files.\n"
                )

            if os.path.exists(resolved_vocab_file):
                with io.open(resolved_vocab_file, encoding="utf-8") as f:
                    init_kwargs = json.load(f)
                # class name corresponds to this configuration
                init_class = init_kwargs.pop("init_class", None)
                if init_class:
                    class_name = cls._name_mapping[init_class]
                    import_class = importlib.import_module(
                        f"paddlenlp.transformers.{class_name}.tokenizer")
                    tokenizer_class = getattr(import_class, init_class)
                    logger.info(
                        "We are using %s to load '%s'." %
                        (tokenizer_class, pretrained_model_name_or_path))
                    return tokenizer_class.from_pretrained(
                        pretrained_model_name_or_path, *model_args, **kwargs)
                # If no `init_class`, we use pattern recognition to recognize the Tokenizer class.
                else:
                    print(
                        'We use pattern recognition to recognize the Tokenizer class.'
                    )
                    for key, pattern in cls._name_mapping.items():
                        if pattern in pretrained_model_name_or_path.lower():
                            init_class = key
                            class_name = cls._name_mapping[init_class]
                            import_class = importlib.import_module(
                                f"paddlenlp.transformers.{class_name}.tokenizer"
                            )
                            tokenizer_class = getattr(import_class, init_class)
                            logger.info("We are using %s to load '%s'." %
                                        (tokenizer_class,
                                         pretrained_model_name_or_path))
                            return tokenizer_class.from_pretrained(
                                pretrained_model_name_or_path, *model_args,
                                **kwargs)
