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
import paddle
import json
from collections import OrderedDict
from paddlenlp.transformers import *
from paddlenlp.utils.downloader import COMMUNITY_MODEL_PREFIX, get_path_from_url
from paddlenlp.utils.env import MODEL_HOME
from paddlenlp.utils.log import logger

__all__ = [
    "AutoModel",
    "AutoModelForPreTraining",
    "AutoModelWithLMHead",
    "AutoModelForMaskedLM",
    "AutoModelForSequenceClassification",
    "AutoModelForTokenClassification",
    "AutoModelForQuestionAnswering",
    "AutoModelForMultipleChoice",
    "AutoEncoder",
    "AutoDecoder",
    "AutoGenerator",
    "AutoDiscriminator",
]


class _BaseAutoModelClass:
    # Base class for auto models.
    _model_mapping = None
    _name_mapping = None
    model_config_file = "model_config.json"

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
                                       cls.model_config_file)
            if os.path.exists(config_file):
                with io.open(config_file, encoding="utf-8") as f:
                    init_kwargs = json.load(f)
                # class name corresponds to this configuration
                init_class = init_kwargs.pop("init_class")
                try:
                    class_name = cls._name_mapping[init_class]
                    import_class = importlib.import_module(
                        f"paddlenlp.transformers.{class_name}.modeling")
                    model_name = getattr(import_class, init_class)
                    return model_name.from_pretrained(
                        pretrained_model_name_or_path, *model_args, **kwargs)
                except KeyError as err:
                    logger.error(err)
                    print(
                        f"The model class is {init_class}, it is not match the Auto Class."
                    )
        else:
            for names, model_class in cls._model_mapping.items():
                # From built-in pretrained models
                for pattern in names:
                    if pattern == pretrained_model_name_or_path:
                        print(pattern, model_class)
                        return model_class.from_pretrained(
                            pretrained_model_name_or_path, **kwargs)

            community_config_path = os.path.join(COMMUNITY_MODEL_PREFIX,
                                                 pretrained_model_name_or_path,
                                                 cls.model_config_file)

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
                    try:
                        class_name = cls._name_mapping[init_class]
                        import_class = importlib.import_module(
                            f"paddlenlp.transformers.{class_name}.modeling")
                        model_name = getattr(import_class, init_class)
                        return model_name.from_pretrained(
                            pretrained_model_name_or_path, *model_args,
                            **kwargs)
                    except KeyError as err:
                        logger.error(err)
                        print(
                            f"The model class is {init_class}, it is not match the Auto Class."
                        )
            except RuntimeError as err:
                logger.error(err)
                raise RuntimeError(
                    f"Can't load weights for '{pretrained_model_name_or_path}'.\n"
                    f"Please make sure that '{pretrained_model_name_or_path}' is:\n"
                    "- a correct model-identifier of built-in pretrained models,\n"
                    "- or a correct model-identifier of community-contributed pretrained models,\n"
                    "- or the correct path to a directory containing relevant modeling files(model_weights and model_config).\n"
                )


MODEL_MAPPING_NAMES = OrderedDict([
    # Base model mapping
    ("AlbertModel", "albert"),
    ("BartModel", "bart"),
    ("BigBirdModel", "bigbird"),
    ("ConvBertModel", "convbert"),
    ("DistilBertModel", "distilbert"),
    ("ElectraModel", "electra"),
    ("SkepModel", "skep"),
    ("ErnieCtmModel", "ernie-ctm"),
    ("ErnieDocModel", "ernie-doc"),
    ("ErnieForGeneration", "ernie-gen"),
    ("ErnieGramModel", "ernie-gram"),
    ("ErnieModel", "ernie"),
    ("GPTModel", "gpt"),
    ("MPNetModel", "mpnet"),
    ("NeZhaModel", "nezha"),
    ("RobertaModel", "roberta"),
    ("RoFormerModel", "roformer"),
    ("TinyBertModel", "tinybert"),
    ("BertModel", "bert"),
    ("UNIMOModel", "unimo"),
    ("UnifiedTransformerModel", "unifiedtransformer"),
    ("XLNetModel", "xlnet"),
])

MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict([
    # Model for pre-training mapping
    ("AlbertForPretraining", "albert"),
    ("BartForConditionalGeneration", "bart"),
    ("BigBirdForPretraining", "bigbird"),
    ("ErnieForPretraining", "ernie"),
    ("GPTForPretraining", "gpt"),
    ("NeZhaForPretraining", "nezha"),
    ("RoFormerForPretraining", "roformer"),
    ("TinyBertForPretraining", "tinybert"),
    ("BertForPretraining", "bert"),
])

MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict([
    # Model with LM heads mapping
    ("AlbertForMaskedLM", "albert"),
    ("DistilBertForMaskedLM", "distilbert"),
    ("GPTLMHeadModel", "gpt"),
    ("MPNetForMaskedLM", "mpnet"),
    ("UnifiedTransformerLMHeadModel", "unifiedtransformer"),
    ("UNIMOLMHeadModel", "unimo"),
])

MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict([
    # Model for Masked LM mapping
    ("AlbertForMaskedLM", "albert"),
    ("BartForConditionalGeneration", "bart"),
    ("DistilBertForMaskedLM", "distilbert"),
    ("MPNetForMaskedLM", "mpnet"),
])

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict([
    # Model for Sequence Classification mapping
    ("AlbertForSequenceClassification", "albert"),
    ("BartForSequenceClassification", "bart"),
    ("BigBirdForSequenceClassification", "bigbird"),
    ("ConvBertForSequenceClassification", "convbert"),
    ("DistilBertForSequenceClassification", "distilbert"),
    ("ElectraForSequenceClassification", "electra"),
    ("SkepForSequenceClassification", "skep"),
    ("ErnieDocForSequenceClassification", "ernie-doc"),
    ("ErnieGramForSequenceClassification", "ernie-gram"),
    ("ErnieForSequenceClassification", "ernie"),
    ("MPNetForSequenceClassification", "mpnet"),
    ("NeZhaForSequenceClassification", "nezha"),
    ("RobertaForSequenceClassification", "roberta"),
    ("RoFormerForSequenceClassification", "roformer"),
    ("TinyBertForSequenceClassification", "tinybert"),
    ("BertForSequenceClassification", "bert"),
    ("XLNetForSequenceClassification", "xlnet"),
])

MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict([
    # Model for Question Answering mapping
    ("BartForQuestionAnswering", "bart"),
    ("ConvBertForQuestionAnswering", "convbert"),
    ("DistilBertForQuestionAnswering", "distilbert"),
    ("ErnieDocForQuestionAnswering", "ernie-doc"),
    ("ErnieGramForQuestionAnswering", "ernie-gram"),
    ("ErnieForQuestionAnswering", "ernie"),
    ("MPNetForQuestionAnswering", "mpnet"),
    ("NeZhaForQuestionAnswering", "nezha"),
    ("RobertaForQuestionAnswering", "roberta"),
    ("RoFormerForQuestionAnswering", "roformer"),
    ("BertForQuestionAnswering", "bert"),
])

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict([
    # Model for Token Classification mapping
    ("AlbertForTokenClassification", "albert"),
    ("ConvBertForTokenClassification", "convbert"),
    ("DistilBertForTokenClassification", "distilbert"),
    ("ElectraForTokenClassification", "electra"),
    ("SkepForTokenClassification", "skep"),
    ("ErnieCtmForTokenClassification", "ernie-ctm"),
    ("ErnieDocForTokenClassification", "ernie-doc"),
    ("ErnieGramForTokenClassification", "ernie-gram"),
    ("ErnieForTokenClassification", "ernie"),
    ("MPNetForTokenClassification", "mpnet"),
    ("NeZhaForTokenClassification", "nezha"),
    ("RobertaForTokenClassification", "roberta"),
    ("BertForTokenClassification", "bert"),
    ("RoFormerForTokenClassification", "roformer"),
    ("XLNetForTokenClassification", "xlnet"),
])

MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict([
    # Model for Multiple Choice mapping
    ("AlbertForMultipleChoice", "albert"),
    ("ConvBertForMultipleChoice", "convbert"),
    ("MPNetForMultipleChoice", "mpnet"),
    ("NeZhaForMultipleChoice", "nezha"),
])

ENCODER_MAPPING_NAMES = OrderedDict([
    # Model for Encoder mapping
    ("BartEncoder", "bart"),
])

DECODER_MAPPING_NAMES = OrderedDict([
    # Model for Decoder mapping
    ("BartDeocder", "bart"),
])

GENERATOR_MAPPING_NAMES = OrderedDict([
    # Model for Generator mapping
    ("ConvBertGenerator", "convbert"),
    ("ElectraGenerator", "electra"),
])

DISCRIMINATOR_MAPPING_NAMES = OrderedDict([
    # Model for Discriminator mapping
    ("ConvBertDiscriminator", "convbert"),
    ("ElectraDiscriminator", "electra"),
])


def get_configurations(key):
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
    erniegen = tuple(ErnieForGeneration.pretrained_init_configuration.keys())
    erniegram = tuple(ErnieGramModel.pretrained_init_configuration.keys())
    ernie = tuple(ErniePretrainedModel.pretrained_init_configuration.keys())
    gpt = tuple(GPTPretrainedModel.pretrained_init_configuration.keys())
    mpnet = tuple(MPNetPretrainedModel.pretrained_init_configuration.keys())
    nezha = tuple(NeZhaPretrainedModel.pretrained_init_configuration.keys())
    roberta = tuple(RobertaPretrainedModel.pretrained_init_configuration.keys())
    roformer = tuple(RoFormerPretrainedModel.pretrained_init_configuration.keys(
    ))
    tinybert = tuple(TinyBertPretrainedModel.pretrained_init_configuration.keys(
    ))
    bert = tuple(BertPretrainedModel.pretrained_init_configuration.keys())
    unifiedtransformer = tuple(
        UnifiedTransformerModel.pretrained_init_configuration.keys())
    unimo = tuple(UNIMOPretrainedModel.pretrained_init_configuration.keys())
    xlnet = tuple(XLNetPretrainedModel.pretrained_init_configuration.keys())

    mapping_key = key
    key_dict = {
        "model": OrderedDict([
            (albert, AlbertModel),
            (bart, BartModel),
            (bigbird, BigBirdModel),
            (convbert, ConvBertModel),
            (distilbert, DistilBertModel),
            (electra, ElectraModel),
            (skep, SkepModel),
            (erniectm, ErnieCtmModel),
            (erniedoc, ErnieDocModel),
            (erniegen, ErnieForGeneration),
            (erniegram, ErnieGramModel),
            (ernie, ErnieModel),
            (gpt, GPTModel),
            (mpnet, MPNetModel),
            (nezha, NeZhaModel),
            (roberta, RobertaModel),
            (roformer, RoFormerModel),
            (tinybert, TinyBertModel),
            (bert, BertModel),
            (unifiedtransformer, UnifiedTransformerModel),
            (unimo, UNIMOModel),
            (xlnet, XLNetModel),
        ]),
        "pretraining": OrderedDict([
            (albert, AlbertForPretraining),
            (bart, BartForConditionalGeneration),
            (bigbird, BigBirdForPretraining),
            #(convbert, ConvBertForTotalPretraining),
            #(electra, ElectraForTotalPretraining),
            (ernie, ErnieForPretraining),
            (gpt, GPTForPretraining),
            (nezha, NeZhaForPretraining),
            (roformer, RoFormerForPretraining),
            (tinybert, TinyBertForPretraining),
            (bert, BertForPretraining),
        ]),
        "lm_head": OrderedDict([
            (albert, AlbertForMaskedLM),
            (bart, BartForConditionalGeneration),
            (bigbird, BigBirdPretrainingHeads),
            (convbert, ConvBertClassificationHead),
            (distilbert, DistilBertForMaskedLM),
            (electra, ElectraClassificationHead),
            (gpt, GPTLMHeadModel),
            (mpnet, MPNetForMaskedLM),
            (nezha, NeZhaPretrainingHeads),
            (roformer, RoFormerPretrainingHeads),
            (bert, BertPretrainingHeads),
            (unifiedtransformer, UnifiedTransformerLMHeadModel),
            (unimo, UNIMOLMHeadModel),
        ]),
        "masked_lm": OrderedDict([
            (albert, AlbertForMaskedLM),
            (bart, BartForConditionalGeneration),
            (distilbert, DistilBertForMaskedLM),
            (mpnet, MPNetForMaskedLM),
        ]),
        "sequence_classification": OrderedDict([
            (albert, AlbertForSequenceClassification),
            (bart, BartForSequenceClassification),
            (bigbird, BigBirdForSequenceClassification),
            (convbert, ConvBertForSequenceClassification),
            (distilbert, DistilBertForSequenceClassification),
            (electra, ElectraForSequenceClassification),
            (skep, SkepForSequenceClassification),
            (erniedoc, ErnieDocForSequenceClassification),
            (erniegram, ErnieGramForSequenceClassification),
            (ernie, ErnieForSequenceClassification),
            (mpnet, MPNetForSequenceClassification),
            (nezha, NeZhaForSequenceClassification),
            (roberta, RobertaForSequenceClassification),
            (roformer, RoFormerForSequenceClassification),
            (tinybert, TinyBertForSequenceClassification),
            (bert, BertForSequenceClassification),
            (xlnet, XLNetForSequenceClassification),
        ]),
        "question_answering": OrderedDict([
            (bart, BartForQuestionAnswering),
            (convbert, ConvBertForQuestionAnswering),
            (distilbert, DistilBertForQuestionAnswering),
            (erniedoc, ErnieDocForQuestionAnswering),
            (erniegram, ErnieGramForQuestionAnswering),
            (ernie, ErnieForQuestionAnswering),
            (mpnet, MPNetForQuestionAnswering),
            (nezha, NeZhaForQuestionAnswering),
            (roberta, RobertaForQuestionAnswering),
            (roformer, RoFormerForQuestionAnswering),
            (bert, BertForQuestionAnswering),
        ]),
        "token_classification": OrderedDict([
            (albert, AlbertForTokenClassification),
            (convbert, ConvBertForTokenClassification),
            (distilbert, DistilBertForTokenClassification),
            (electra, ElectraForTokenClassification),
            (skep, SkepForTokenClassification),
            (erniectm, ErnieCtmForTokenClassification),
            (erniedoc, ErnieDocForTokenClassification),
            (erniegram, ErnieGramForTokenClassification),
            (ernie, ErnieForTokenClassification),
            (mpnet, MPNetForTokenClassification),
            (nezha, NeZhaForTokenClassification),
            (roberta, RobertaForTokenClassification),
            (roformer, RoFormerForTokenClassification),
            (bert, BertForTokenClassification),
            (xlnet, XLNetForTokenClassification),
        ]),
        "multiple_choice": OrderedDict([
            (albert, AlbertForMultipleChoice),
            (convbert, ConvBertForMultipleChoice),
            (mpnet, MPNetForMultipleChoice),
            (nezha, NeZhaForMultipleChoice),
        ]),
        "encoder": OrderedDict([(bart, BartEncoder), ]),
        "decoder": OrderedDict([(bart, BartDecoder), ]),
        "generator": OrderedDict([
            (convbert, ConvBertGenerator),
            (electra, ElectraGenerator),
        ]),
        "discriminator": OrderedDict([
            (convbert, ConvBertDiscriminator),
            (electra, ElectraDiscriminator),
        ]),
    }

    MAPPING_NAMES = key_dict.get(mapping_key)
    return MAPPING_NAMES


class AutoModel(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations("model")
    _model_mapping = MAPPING_NAMES
    _name_mapping = MODEL_MAPPING_NAMES


class AutoModelForPreTraining(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations("pretraining")
    _model_mapping = MAPPING_NAMES
    _name_mapping = MODEL_FOR_PRETRAINING_MAPPING_NAMES


class AutoModelWithLMHead(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations("lm_head")
    _model_mapping = MAPPING_NAMES
    _name_mapping = MODEL_WITH_LM_HEAD_MAPPING_NAMES


class AutoModelForMaskedLM(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations("masked_lm")
    _model_mapping = MAPPING_NAMES
    _name_mapping = MODEL_FOR_MASKED_LM_MAPPING_NAMES


class AutoModelForSequenceClassification(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations("sequence_classification")
    _model_mapping = MAPPING_NAMES
    _name_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES


class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations("question_answering")
    _model_mapping = MAPPING_NAMES
    _name_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES


class AutoModelForTokenClassification(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations("token_classification")
    _model_mapping = MAPPING_NAMES
    _name_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations("multiple_choice")
    _model_mapping = MAPPING_NAMES
    _name_mapping = MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES


class AutoEncoder(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations("encoder")
    _model_mapping = MAPPING_NAMES
    _name_mapping = ENCODER_MAPPING_NAMES


class AutoDecoder(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations("decoder")
    _model_mapping = MAPPING_NAMES
    _name_mapping = DECODER_MAPPING_NAMES


class AutoGenerator(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations("generator")
    _model_mapping = MAPPING_NAMES
    _name_mapping = GENERATOR_MAPPING_NAMES


class AutoDiscriminator(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations("discriminator")
    _model_mapping = MAPPING_NAMES
    _name_mapping = DISCRIMINATOR_MAPPING_NAMES
