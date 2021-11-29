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

__all__ = [
    "AutoModel",
    "AutoModelForPretraining",
    "AutoModelForSequenceClassification",
    "AutoModelForTokenClassification",
    "AutoModelForQuestionAnswering",
    "AutoModelForMultipleChoice",
    "AutoModelWithLMHead",
    "AutoModelForMaskedLM",
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
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        task='model',
                        *model_args,
                        **kwargs):
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        key_dict = {
            'model': MODEL_MAPPING_NAMES,
            'pretraining': PRETRAINING_MAPPING_NAMES,
            'sequence_classification': SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
            'token_classification': TOKEN_CLASSIFICATION_MAPPING_NAMES,
            'question_answering': QUESTION_ANSWERING_MAPPING_NAMES,
            'multiple_choice': MULTIPLE_CHOICE_MAPPING_NAMES,
            'lm_head': LM_HEAD_MAPPING_NAMES,
            'masked_lm': MASKED_LM_MAPPING_NAMES,
            'encoder': ENCODER_MAPPING_NAMES,
            'decoder': DECODER_MAPPING_NAMES,
            'generator': GENERATOR_MAPPING_NAMES,
            'discriminator': DISCRIMINATOR_MAPPING_NAMES,
        }
        cls._name_mapping = key_dict[task]

        all_model_names = []
        for names, model_class in cls._model_mapping.items():
            for name in names:
                all_model_names.append(name)

        # From built-in pretrained models
        if pretrained_model_name_or_path in all_model_names:
            for names, model_name in cls._model_mapping.items():
                # From built-in pretrained models
                for pattern in names:
                    if pattern == pretrained_model_name_or_path:
                        # print(pattern, model_class)
                        class_name = cls._name_mapping[model_name]
                        import_class = importlib.import_module(
                            f"paddlenlp.transformers.{class_name}.modeling")
                        init_class = cls._name_mapping[model_name +
                                                       '_Import_Class']
                        model_class = getattr(import_class, init_class)
                        return model_class.from_pretrained(
                            pretrained_model_name_or_path, *model_args,
                            **kwargs)
        # From local dir path
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path,
                                       cls.model_config_file)
            if os.path.exists(config_file):
                with io.open(config_file, encoding="utf-8") as f:
                    init_kwargs = json.load(f)
                # class name corresponds to this configuration
                init_class = init_kwargs.pop("init_class", None)
                try:
                    class_name = cls._name_mapping[init_class]
                    import_class = importlib.import_module(
                        f"paddlenlp.transformers.{class_name}.modeling")
                    model_name = getattr(import_class, init_class)
                    keyerror = False
                    return model_name.from_pretrained(
                        pretrained_model_name_or_path, *model_args, **kwargs)
                except KeyError as err:
                    keyerror = True
                if keyerror:
                    print(
                        'We use pattern recoginition to recoginize the Model class.'
                    )
                    # 从init_class判断
                    if init_class != None:
                        init_class = init_class.lower()
                        mapping_init_class = init_class
                    else:
                        # 无init_class,从pretrained_model_name_or_path判断
                        pretrained_model_name_or_path = pretrained_model_name_or_path.lower(
                        )
                        mapping_init_class = init_class
                    for key, pattern in cls._name_mapping.items():
                        if pattern in mapping_init_class:
                            init_class = key
                            class_name = cls._name_mapping[init_class]
                            import_class = importlib.import_module(
                                f"paddlenlp.transformers.{class_name}.modeling")
                            model_name = getattr(import_class, init_class)
                            return model_name.from_pretrained(
                                pretrained_model_name_or_path, *model_args,
                                **kwargs, **init_kwargs)
        # Assuming from community-contributed pretrained models
        else:
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
                    init_class = init_kwargs.pop("init_class", None)
                    try:
                        class_name = cls._name_mapping[init_class]
                        import_class = importlib.import_module(
                            f"paddlenlp.transformers.{class_name}.modeling")
                        model_name = getattr(import_class, init_class)
                        keyerror = False
                        return model_name.from_pretrained(
                            pretrained_model_name_or_path, *model_args,
                            **kwargs, **init_kwargs)
                    except KeyError as err:
                        #logger.error(err)
                        keyerror = True
                    if keyerror:
                        print(
                            'We use pattern recoginition to recoginize the Model class.'
                        )
                        # From init_class
                        if init_class != None:
                            init_class = init_class.lower()
                            mapping_init_class = init_class
                        else:
                            # From pretrained_model_name_or_path
                            pretrained_model_name_or_path = pretrained_model_name_or_path.lower(
                            )
                            mapping_init_class = init_class
                        for key, pattern in cls._name_mapping.items():
                            if pattern in mapping_init_class:
                                init_class = key
                                class_name = cls._name_mapping[init_class]
                                import_class = importlib.import_module(
                                    f"paddlenlp.transformers.{class_name}.modeling"
                                )
                                model_name = getattr(import_class, init_class)
                                return model_name.from_pretrained(
                                    pretrained_model_name_or_path, *model_args,
                                    **kwargs)

            except RuntimeError as err:
                logger.error(err)
                raise RuntimeError(
                    f"Can't load weights for '{pretrained_model_name_or_path}'.\n"
                    f"Please make sure that '{pretrained_model_name_or_path}' is:\n"
                    "- a correct model-identifier of built-in pretrained models,\n"
                    "- or a correct model-identifier of community-contributed pretrained models,\n"
                    "- or the correct path to a directory containing relevant modeling files(model_weights and model_config).\n"
                )


MAPPING_NAMES = OrderedDict([
    # Base model mapping
    ("Albert", "albert"),
    ("Bart", "bart"),
    ("BigBird", "bigbird"),
    ("ConvBert", "convbert"),
    ("DistilBert", "distilbert"),
    ("Electra", "electra"),
    ("Skep", "skep"),
    ("ErnieCtm", "ernie-ctm"),
    ("ErnieDoc", "ernie-doc"),
    ("ErnieGram", "ernie-gram"),
    ("Ernie", "ernie"),
    ("GPT", "gpt"),
    ("MPNet", "mpnet"),
    ("NeZha", "nezha"),
    ("Roberta", "roberta"),
    ("RoFormer", "roformer"),
    ("TinyBert", "tinybert"),
    ("Bert", "bert"),
    ("UNIMO", "unimo"),
    ("UnifiedTransformer", "unifiedtransformer"),
    ("XLNet", "xlnet"),
])

# Base model mapping
MODEL_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    MODEL_MAPPING_NAMES[key1] = value
    key2 = key + 'Model_Import_Class'
    import_class = key + 'Model'
    MODEL_MAPPING_NAMES[key2] = import_class

# Model for Pre-training mapping
PRETRAINING_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    PRETRAINING_MAPPING_NAMES[key1] = value
    key2 = key + 'Model_Import_Class'
    import_class = key + 'ForPretraining'
    PRETRAINING_MAPPING_NAMES[key2] = import_class

# Model for Sequence Classification mapping
SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    SEQUENCE_CLASSIFICATION_MAPPING_NAMES[key1] = value
    key2 = key + 'Model_Import_Class'
    import_class = key + 'ForSequenceClassification'
    SEQUENCE_CLASSIFICATION_MAPPING_NAMES[key2] = import_class

# Model for Token Classification mapping
TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    TOKEN_CLASSIFICATION_MAPPING_NAMES[key1] = value
    key2 = key + 'Model_Import_Class'
    import_class = key + 'ForTokenClassification'
    TOKEN_CLASSIFICATION_MAPPING_NAMES[key2] = import_class

# Model for Question Answering mapping
QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    QUESTION_ANSWERING_MAPPING_NAMES[key1] = value
    key2 = key + 'Model_Import_Class'
    import_class = key + 'ForQuestionAnswering'
    QUESTION_ANSWERING_MAPPING_NAMES[key2] = import_class

# Model for Multiple Choice mapping
MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    MULTIPLE_CHOICE_MAPPING_NAMES[key1] = value
    key2 = key + 'Model_Import_Class'
    import_class = key + 'ForMultipleChoice'
    MULTIPLE_CHOICE_MAPPING_NAMES[key2] = import_class

# Model for MaskedLM mapping
MASKED_LM_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    MASKED_LM_MAPPING_NAMES[key1] = value
    key2 = key + 'Model_Import_Class'
    import_class = key + 'ForMaskedLM'
    MASKED_LM_MAPPING_NAMES[key2] = import_class

# Model with LH mapping
LM_HEAD_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    LM_HEAD_MAPPING_NAMES[key1] = value
    key2 = key + 'Model_Import_Class'
    import_class = key + 'LMHeadModel'
    LM_HEAD_MAPPING_NAMES[key2] = import_class

# Encoder mapping
ENCODER_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    ENCODER_MAPPING_NAMES[key1] = value
    key2 = key + 'Model_Import_Class'
    import_class = key + 'LMHeadModel'
    ENCODER_MAPPING_NAMES[key2] = import_class

# Decoder mapping
DECODER_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    DECODER_MAPPING_NAMES[key1] = value
    key2 = key + 'Model_Import_Class'
    import_class = key + 'Decoder'
    DECODER_MAPPING_NAMES[key2] = import_class

# Generator mapping
GENERATOR_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    GENERATOR_MAPPING_NAMES[key1] = value
    key2 = key + 'Model_Import_Class'
    import_class = key + 'Generator'
    GENERATOR_MAPPING_NAMES[key2] = import_class

# Discriminator mapping
DISCRIMINATOR_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    DISCRIMINATOR_MAPPING_NAMES[key1] = value
    key2 = key + 'Model_Import_Class'
    import_class = key + 'Discriminator'
    DISCRIMINATOR_MAPPING_NAMES[key2] = import_class


def get_configurations():
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

    MAPPING_NAMES = OrderedDict([
        (albert, 'AlbertModel'),
        (bart, 'BartModel'),
        (bigbird, 'BigBirdModel'),
        (convbert, 'ConvBertModel'),
        (distilbert, 'DistilBertModel'),
        (electra, 'ElectraModel'),
        (skep, 'SkepModel'),
        (erniectm, 'ErnieCtmModel'),
        (erniedoc, 'ErnieDocModel'),
        #(erniegen, ErnieForGeneration),
        (erniegram, 'ErnieGramModel'),
        (ernie, 'ErnieModel'),
        (gpt, 'GPTModel'),
        (mpnet, 'MPNetModel'),
        (nezha, 'NeZhaModel'),
        (roberta, 'RobertaModel'),
        (roformer, 'RoFormerModel'),
        (tinybert, 'TinyBertModel'),
        (bert, 'BertModel'),
        (unifiedtransformer, 'UnifiedTransformerModel'),
        (unimo, 'UNIMOModel'),
        (xlnet, 'XLNetModel'),
    ])
    return MAPPING_NAMES


class AutoModel(_BaseAutoModelClass):
    """
    AutoClass can help you automatically retrieve the relevant model given the provided
    pretrained weights/vocabulary.
    AutoModel is a generic model class that will be instantiated as one of the base model classes
    when created with the from_pretrained() classmethod.

    Classmethod: from_pretrained():
    Creates an instance of `AutoModel`. Model weights are loaded
    by specifying name of a built-in pretrained model, or a community contributed model,
    or a local file directory path.

    Args:
        pretrained_model_name_or_path (str): Name of pretrained model or dir path
            to load from. The string can be:

            - Name of a built-in pretrained model
            - Name of a community-contributed pretrained model.
            - Local directory path which contains model weights file("model_state.pdparams")
              and model config file ("model_config.json").
        *args (tuple): Position arguments for model `__init__`. If provided,
            use these as position argument values for model initialization.
        **kwargs (dict): Keyword arguments for model `__init__`. If provided,
            use these to update pre-defined keyword argument values for model
            initialization. If the keyword is in `__init__` argument names of
            base model, update argument values of the base model; else update
            argument values of derived model.

    Returns:
        PretrainedModel: An instance of `AutoModel`.

    Example:
        .. code-block::

            from paddlenlp.transformers import AutoModel

            # Name of built-in pretrained model
            model = AutoModel.from_pretrained('bert-base-uncased')

            # Name of community-contributed pretrained model
            model = AutoModel.from_pretrained('yingyibiao/bert-base-uncased-sst-2-finetuned')

            # Load from local directory path
            model = AutoModel.from_pretrained('./my_bert/')
    """
    MAPPING_NAMES = get_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = MODEL_MAPPING_NAMES


class AutoModelForPretraining(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = PRETRAINING_MAPPING_NAMES


class AutoModelForSequenceClassification(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = SEQUENCE_CLASSIFICATION_MAPPING_NAMES


class AutoModelForTokenClassification(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = TOKEN_CLASSIFICATION_MAPPING_NAMES


class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = QUESTION_ANSWERING_MAPPING_NAMES


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = MULTIPLE_CHOICE_MAPPING_NAMES


class AutoModelWithLMHead(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = LM_HEAD_MAPPING_NAMES


class AutoModelForMaskedLM(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = MASKED_LM_MAPPING_NAMES


class AutoEncoder(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = ENCODER_MAPPING_NAMES


class AutoDecoder(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = DECODER_MAPPING_NAMES


class AutoGenerator(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = GENERATOR_MAPPING_NAMES


class AutoDiscriminator(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = DISCRIMINATOR_MAPPING_NAMES
