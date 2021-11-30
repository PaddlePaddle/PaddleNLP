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
    "AutoModel", "AutoModelForPretraining",
    "AutoModelForSequenceClassification", "AutoModelForTokenClassification",
    "AutoModelForQuestionAnswering", "AutoModelForMultipleChoice",
    "AutoModelWithLMHead", "AutoModelForMaskedLM", "AutoModelForCausalLM",
    "AutoEncoder", "AutoDecoder", "AutoGenerator", "AutoDiscriminator",
    "AutoModelForConditionalGeneration", "Auto"
]

MAPPING_NAMES = OrderedDict([
    # Base model mapping
    ("Albert", "albert"),
    ("Bart", "bart"),
    ("BigBird", "bigbird"),
    ("BlenderbotSmall", "blenderbot_small"),
    ("Blenderbot", "blenderbot"),
    ("ConvBert", "convbert"),
    ("DistilBert", "distilbert"),
    ("Electra", "electra"),
    ("Skep", "skep"),
    ("ErnieCtm", "ernie_ctm"),
    ("ErnieDoc", "ernie_doc"),
    ("ErnieGram", "ernie_gram"),
    ("ErnieGen", "ernie_gen"),
    ("Ernie", "ernie"),
    ("GPT", "gpt"),
    ("MPNet", "mpnet"),
    ("NeZha", "nezha"),
    ("Roberta", "roberta"),
    ("RoFormer", "roformer"),
    ("SqueezeBert", "squeezebert"),
    ("TinyBert", "tinybert"),
    ("Bert", "bert"),
    ("UNIMO", "unimo"),
    ("UnifiedTransformer", "unified_transformer"),
    ("XLNet", "xlnet"),
])

# Base model mapping
MODEL_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    key2 = key + 'Model_Import_Class'
    import_class = key + 'Model'
    MODEL_MAPPING_NAMES[key2] = import_class
    MODEL_MAPPING_NAMES[key1] = value

# Model for Pre-training mapping
PRETRAINING_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    key2 = key + 'Model_Import_Class'
    import_class = key + 'ForPretraining'
    PRETRAINING_MAPPING_NAMES[key2] = import_class
    PRETRAINING_MAPPING_NAMES[import_class] = value
    PRETRAINING_MAPPING_NAMES[key1] = value

# Model for Sequence Classification mapping
SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    key2 = key + 'Model_Import_Class'
    import_class = key + 'ForSequenceClassification'
    SEQUENCE_CLASSIFICATION_MAPPING_NAMES[key2] = import_class
    SEQUENCE_CLASSIFICATION_MAPPING_NAMES[import_class] = value
    SEQUENCE_CLASSIFICATION_MAPPING_NAMES[key1] = value

# Model for Token Classification mapping
TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    TOKEN_CLASSIFICATION_MAPPING_NAMES[key1] = value
    key2 = key + 'Model_Import_Class'
    import_class = key + 'ForTokenClassification'
    TOKEN_CLASSIFICATION_MAPPING_NAMES[key2] = import_class
    TOKEN_CLASSIFICATION_MAPPING_NAMES[import_class] = value
    TOKEN_CLASSIFICATION_MAPPING_NAMES[key1] = value

# Model for Question Answering mapping
QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    key2 = key + 'Model_Import_Class'
    import_class = key + 'ForQuestionAnswering'
    QUESTION_ANSWERING_MAPPING_NAMES[key2] = import_class
    QUESTION_ANSWERING_MAPPING_NAMES[import_class] = value
    QUESTION_ANSWERING_MAPPING_NAMES[key1] = value

# Model for Multiple Choice mapping
MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    key2 = key + 'Model_Import_Class'
    import_class = key + 'ForMultipleChoice'
    MULTIPLE_CHOICE_MAPPING_NAMES[key2] = import_class
    MULTIPLE_CHOICE_MAPPING_NAMES[import_class] = value
    MULTIPLE_CHOICE_MAPPING_NAMES[key1] = value

# Model for MaskedLM mapping
MASKED_LM_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    key2 = key + 'Model_Import_Class'
    import_class = key + 'ForMaskedLM'
    MASKED_LM_MAPPING_NAMES[key2] = import_class
    MASKED_LM_MAPPING_NAMES[import_class] = value
    MASKED_LM_MAPPING_NAMES[key1] = value

# Model for CausalLM mapping
CAUSAL_LM_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    key2 = key + 'Model_Import_Class'
    import_class = key + 'ForCausalLM'
    CAUSAL_LM_MAPPING_NAMES[key2] = import_class
    CAUSAL_LM_MAPPING_NAMES[import_class] = value
    CAUSAL_LM_MAPPING_NAMES[key1] = value

# Model with LH mapping
LM_HEAD_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    key2 = key + 'Model_Import_Class'
    import_class = key + 'LMHeadModel'
    LM_HEAD_MAPPING_NAMES[key2] = import_class
    LM_HEAD_MAPPING_NAMES[import_class] = value
    LM_HEAD_MAPPING_NAMES[key1] = value

# Encoder mapping
ENCODER_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    key2 = key + 'Model_Import_Class'
    import_class = key + 'LMHeadModel'
    ENCODER_MAPPING_NAMES[key2] = import_class
    ENCODER_MAPPING_NAMES[import_class] = value
    ENCODER_MAPPING_NAMES[key1] = value

# Decoder mapping
DECODER_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    key2 = key + 'Model_Import_Class'
    import_class = key + 'Decoder'
    DECODER_MAPPING_NAMES[key2] = import_class
    DECODER_MAPPING_NAMES[import_class] = value
    DECODER_MAPPING_NAMES[key1] = value

# Generator mapping
GENERATOR_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    key2 = key + 'Model_Import_Class'
    import_class = key + 'Generator'
    GENERATOR_MAPPING_NAMES[key2] = import_class
    GENERATOR_MAPPING_NAMES[import_class] = value
    GENERATOR_MAPPING_NAMES[key1] = value

# Discriminator mapping
DISCRIMINATOR_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    key2 = key + 'Model_Import_Class'
    import_class = key + 'Discriminator'
    DISCRIMINATOR_MAPPING_NAMES[key2] = import_class
    DISCRIMINATOR_MAPPING_NAMES[import_class] = value
    DISCRIMINATOR_MAPPING_NAMES[key1] = value

# Conditional generation mapping
CONDITIONAL_GENERATION_MAPPING_NAMES = OrderedDict()
for key, value in MAPPING_NAMES.items():
    key1 = key + 'Model'
    key2 = key + 'Model_Import_Class'
    import_class = key + 'ForConditionalGeneration'
    CONDITIONAL_GENERATION_MAPPING_NAMES[key2] = import_class
    CONDITIONAL_GENERATION_MAPPING_NAMES[import_class] = value
    CONDITIONAL_GENERATION_MAPPING_NAMES[key1] = value


def get_configurations():
    CONFIGURATION_MODEL_MAPPING = OrderedDict()
    for key, class_name in MAPPING_NAMES.items():
        import_class = importlib.import_module(
            f"paddlenlp.transformers.{class_name}.modeling")
        model_name = getattr(import_class, key + 'Model')
        name = tuple(model_name.pretrained_init_configuration.keys())
        CONFIGURATION_MODEL_MAPPING[name] = key + 'Model'

    return CONFIGURATION_MODEL_MAPPING


class _BaseAutoModelClass:
    # Base class for auto models.
    _model_mapping = None
    _name_mapping = None
    _task_choice = False
    model_config_file = "model_config.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path).`"
        )

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        task=None,
                        *model_args,
                        **kwargs):
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        if task:
            if cls._task_choice == True:
                key_dict = {
                    'model': MODEL_MAPPING_NAMES,
                    'pretraining': PRETRAINING_MAPPING_NAMES,
                    'sequence_classification':
                    SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
                    'token_classification': TOKEN_CLASSIFICATION_MAPPING_NAMES,
                    'question_answering': QUESTION_ANSWERING_MAPPING_NAMES,
                    'multiple_choice': MULTIPLE_CHOICE_MAPPING_NAMES,
                    'lm_head': LM_HEAD_MAPPING_NAMES,
                    'masked_lm': MASKED_LM_MAPPING_NAMES,
                    'causal_lm': CAUSAL_LM_MAPPING_NAMES,
                    'encoder': ENCODER_MAPPING_NAMES,
                    'decoder': DECODER_MAPPING_NAMES,
                    'generator': GENERATOR_MAPPING_NAMES,
                    'discriminator': DISCRIMINATOR_MAPPING_NAMES,
                    'conditional_generation':
                    CONDITIONAL_GENERATION_MAPPING_NAMES,
                }
                try:
                    cls._name_mapping = key_dict[task]
                except KeyError as err:
                    logger.error(err)
                    raise KeyError(f'We only support {key_dict.keys()}.')
            else:
                print('We only support task choice for AutoModel.')

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
                        class_name = cls._name_mapping[model_name]
                        import_class = importlib.import_module(
                            f"paddlenlp.transformers.{class_name}.modeling")
                        init_class = cls._name_mapping[model_name +
                                                       '_Import_Class']
                        model_class = getattr(import_class, init_class)
                        print(model_class)
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
                class_name = cls._name_mapping.get(init_class, None)
                if class_name:
                    import_class = importlib.import_module(
                        f"paddlenlp.transformers.{class_name}.modeling")
                    model_name = getattr(import_class, init_class)
                    return model_name.from_pretrained(
                        pretrained_model_name_or_path, *model_args, **kwargs)
                else:
                    print(
                        'We use pattern recognition to recognize the Model class.'
                    )
                    # From init_class
                    if init_class:
                        init_class = init_class.lower()
                        mapping_init_class = init_class
                    else:
                        # From pretrained_model_name_or_path
                        pretrained_model_name_or_path = pretrained_model_name_or_path.lower(
                        )
                        mapping_init_class = pretrained_model_name_or_path
                    for key, pattern in cls._name_mapping.items():
                        if pattern in mapping_init_class:
                            init_class = key
                            class_name = cls._name_mapping[init_class]
                            import_class = importlib.import_module(
                                f"paddlenlp.transformers.{class_name}.modeling")
                            model_name = getattr(import_class, init_class)
                            return model_name.from_pretrained(
                                pretrained_model_name_or_path, *model_args,
                                **kwargs)
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
            except RuntimeError as err:
                logger.error(err)
                raise RuntimeError(
                    f"Can't load weights for '{pretrained_model_name_or_path}'.\n"
                    f"Please make sure that '{pretrained_model_name_or_path}' is:\n"
                    "- a correct model-identifier of built-in pretrained models,\n"
                    "- or a correct model-identifier of community-contributed pretrained models,\n"
                    "- or the correct path to a directory containing relevant modeling files(model_weights and model_config).\n"
                )

            if os.path.exists(resolved_vocab_file):
                with io.open(resolved_vocab_file, encoding="utf-8") as f:
                    init_kwargs = json.load(f)
                # class name corresponds to this configuration
                init_class = init_kwargs.pop("init_class", None)
                class_name = cls._name_mapping.get(init_class, None)
                if class_name:
                    import_class = importlib.import_module(
                        f"paddlenlp.transformers.{class_name}.modeling")
                    model_name = getattr(import_class, init_class)
                    return model_name.from_pretrained(
                        pretrained_model_name_or_path, *model_args, **kwargs)
                else:
                    print(
                        'We use pattern recognition to recognize the Model class.'
                    )
                    # From init_class
                    if init_class:
                        init_class = init_class.lower()
                        mapping_init_class = init_class
                    else:
                        # From pretrained_model_name_or_path
                        pretrained_model_name_or_path = pretrained_model_name_or_path.lower(
                        )
                        mapping_init_class = pretrained_model_name_or_path
                    for key, pattern in cls._name_mapping.items():
                        if pattern in mapping_init_class:
                            init_class = key
                            class_name = cls._name_mapping[init_class]
                            import_class = importlib.import_module(
                                f"paddlenlp.transformers.{class_name}.modeling")
                            model_name = getattr(import_class, init_class)
                            return model_name.from_pretrained(
                                pretrained_model_name_or_path, *model_args,
                                **kwargs)


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
    _task_choice = True


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


class AutoModelForCausalLM(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = CAUSAL_LM_MAPPING_NAMES


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


class AutoModelForConditionalGeneration(_BaseAutoModelClass):
    MAPPING_NAMES = get_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = CONDITIONAL_GENERATION_MAPPING_NAMES


Auto = AutoModel
