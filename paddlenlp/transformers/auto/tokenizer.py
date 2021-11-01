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

__all__ = ["AutoTokenizer", ]

TOKENIZER_MAPPING_NAMES = OrderedDict([
    # Base model mapping
    ("AlbertTokenizer", "albert"),
    ("BartTokenizer", "bart"),
    ("BigBirdTokenizer", "bigbird"),
    ("ConvBertTokenizer", "convbert"),
    ("DistilBertTokenizer", "distilbert"),
    ("ElectraTokenizer", "electra"),
    ("SkepTokenizer", "skep"),
    ("ErnieCtmTokenizer", "ernie-ctm"),
    ("ErnieDocTokenizer", "ernie-doc"),
    ("ErnieGramTokenizer", "ernie-gram"),
    ("ErnieTokenizer", "ernie"),
    ("GPTTokenizer", "gpt"),
    ("MPNetTokenizer", "mpnet"),
    ("NeZhaTokenizer", "nezha"),
    ("RobertaTokenizer", "roberta"),
    ("RoFormerTokenizer", "roformer"),
    ("TinyBertTokenizer", "tinybert"),
    ("BertTokenizer", "bert"),
    ("UnifiedTransformerTokenizer", "unified_transformer"),
    ("UNIMOTokenizer", "unimo"),
    ("XLNetTokenizer", "xlnet"),
])


def get_all_configurations():
    albert = tuple(AlbertPretrainedModel.pretrained_init_configuration.keys())
    bart = tuple(BartPretrainedModel.pretrained_init_configuration.keys())
    bigbird = tuple(BigBirdPretrainedModel.pretrained_init_configuration.keys())
    convbert = tuple(ConvBertPretrainedModel.pretrained_init_configuration.keys(
    ))
    distilbert = tuple(
        DistilBertPretrainedModel.pretrained_init_configuration.keys())
    electra = tuple(ElectraPretrainedModel.pretrained_init_configuration.keys())
    skep = tuple(SkepPretrainedModel.pretrained_init_configuration.keys())
    erniectm = tuple(ErnieCtmPretrainedModel.pretrained_init_configuration.keys(
    ))
    erniedoc = tuple(ErnieDocPretrainedModel.pretrained_init_configuration.keys(
    ))
    erniegram = tuple(ErnieGramModel.pretrained_init_configuration.keys())
    ernie = tuple(ErniePretrainedModel.pretrained_init_configuration.keys())
    gpt = tuple(GPTPretrainedModel.pretrained_init_configuration.keys())
    mpnet = tuple(MPNetPretrainedModel.pretrained_init_configuration.keys())
    nezha = tuple(NeZhaPretrainedModel.pretrained_init_configuration.keys())
    roberta = tuple(RobertaPretrainedModel.pretrained_init_configuration.keys())
    roformer = tuple(RobertaPretrainedModel.pretrained_init_configuration.keys(
    ))
    tinybert = tuple(TinyBertPretrainedModel.pretrained_init_configuration.keys(
    ))
    bert = tuple(BertPretrainedModel.pretrained_init_configuration.keys())
    unifiedtransformer = tuple(
        UnifiedTransformerModel.pretrained_init_configuration.keys())
    unimo = tuple(UNIMOPretrainedModel.pretrained_init_configuration.keys())
    xlnet = tuple(XLNetPretrainedModel.pretrained_init_configuration.keys())

    MAPPING_NAMES = OrderedDict([
        # Base model mapping
        (albert, AlbertTokenizer),
        (bart, BartTokenizer),
        (bigbird, BigBirdTokenizer),
        (convbert, ConvBertTokenizer),
        (distilbert, DistilBertTokenizer),
        (electra, ElectraTokenizer),
        (skep, SkepTokenizer),
        (erniectm, ErnieCtmTokenizer),
        (erniedoc, ErnieDocTokenizer),
        (erniegram, ErnieGramTokenizer),
        (ernie, ErnieTokenizer),
        (gpt, GPTTokenizer),
        (mpnet, MPNetTokenizer),
        (nezha, NeZhaTokenizer),
        (roberta, RobertaTokenizer),
        (roformer, RoFormerTokenizer),
        (tinybert, TinyBertTokenizer),
        (bert, BertTokenizer),
        (unifiedtransformer, UnifiedTransformerTokenizer),
        (unimo, UNIMOTokenizer),
        (xlnet, XLNetTokenizer),
    ])
    return MAPPING_NAMES


class _BaseAutoTokenizerClass:
    # Base class for auto models.
    _tokenizer_mapping = None
    _name_mapping = None
    tokenizer_config_file = "tokenizer_config.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path).`"
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        # From local dir path
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path,
                                       cls.tokenizer_config_file)
            if os.path.exists(config_file):
                with io.open(config_file, encoding="utf-8") as f:
                    init_kwargs = json.load(f)
                # class name corresponds to this configuration
                init_class = init_kwargs.pop("init_class")
                class_name = cls._name_mapping[init_class]
                import_class = importlib.import_module(
                    f"paddlenlp.transformers.{class_name}.tokenizer")
                tokenizer_name = getattr(import_class, init_class)
                return tokenizer_name.from_pretrained(
                    pretrained_model_name_or_path, *model_args, **kwargs)

        else:
            for tokenizer_names, tokenizer_class in cls._tokenizer_mapping.items(
            ):
                # From built-in pretrained models
                for pattern in tokenizer_names:
                    if pattern == pretrained_model_name_or_path:
                        return tokenizer_class.from_pretrained(
                            pretrained_model_name_or_path, **kwargs)

            # Assuming from community-contributed pretrained models
            community_config_path = os.path.join(COMMUNITY_MODEL_PREFIX,
                                                 pretrained_model_name_or_path,
                                                 cls.tokenizer_config_file)

            default_root = os.path.join(MODEL_HOME,
                                        pretrained_model_name_or_path)
            try:
                resolved_vocab_file = get_path_from_url(community_config_path,
                                                        default_root)
                if os.path.exists(resolved_vocab_file):
                    with io.open(resolved_vocab_file, encoding="utf-8") as f:
                        init_kwargs = json.load(f)
                    # class name corresponds to this configuration
                    init_class = init_kwargs.pop("init_class")
                    class_name = cls._name_mapping[init_class]
                    import_class = importlib.import_module(
                        f"paddlenlp.transformers.{class_name}.tokenizer")
                    tokenizer_name = getattr(import_class, init_class)
                    return tokenizer_name.from_pretrained(
                        pretrained_model_name_or_path, *model_args, **kwargs)
            except RuntimeError as err:
                logger.error(err)
                raise RuntimeError(
                    f"Can't load tokenizer for '{pretrained_model_name_or_path}'.\n"
                    f"Please make sure that '{pretrained_model_name_or_path}' is:\n"
                    "- a correct model-identifier of built-in pretrained models,\n"
                    "- or a correct model-identifier of community-contributed pretrained models,\n"
                    "- or the correct path to a directory containing relevant tokenizer files.\n"
                )


class AutoTokenizer(_BaseAutoTokenizerClass):
    MAPPING_NAMES = get_all_configurations()
    _tokenizer_mapping = MAPPING_NAMES
    _name_mapping = TOKENIZER_MAPPING_NAMES


if __name__ == '__main__':
    # From local dir path
    tokenizer = AutoTokenizer.from_pretrained(
        ('/Users/huhuiwen01/Untitled Folder/my_bert'))
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer = AutoTokenizer.from_pretrained(
        ('/Users/huhuiwen01/Untitled Folder/my_bart'))
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer = AutoTokenizer.from_pretrained(
        ('/Users/huhuiwen01/Untitled Folder/my_bigbird'))
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))

    # From built-in pretrained models
    tokenizer = AutoTokenizer.from_pretrained(('bert-base-cased'))
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer = AutoTokenizer.from_pretrained(('rbt3'))
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer = AutoTokenizer.from_pretrained(('plato-mini'))
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer = AutoTokenizer.from_pretrained(('bigbird-base-uncased'))
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))

    # From community-contributed pretrained models
    tokenizer = AutoTokenizer.from_pretrained(
        ('yingyibiao/bert-base-uncased-sst-2-finetuned'))
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
