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
    "AutoModelForMaskedLM", "AutoModelForCausalLM", "AutoEncoder",
    "AutoDecoder", "AutoGenerator", "AutoDiscriminator",
    "AutoModelForConditionalGeneration"
]

MAPPING_NAMES = OrderedDict([
    # Base model mapping
    ("Albert", "albert"),
    ("BigBird", "bigbird"),
    ("BlenderbotSmall", "blenderbot_small"),
    ("Blenderbot", "blenderbot"),
    ("ConvBert", "convbert"),
    ("CTRL", "ctrl"),
    ("DistilBert", "distilbert"),
    ("Electra", "electra"),
    ("Skep", "skep"),
    ("ErnieCtm", "ernie_ctm"),
    ("ErnieDoc", "ernie_doc"),
    ("ErnieGram", "ernie_gram"),
    ("ErnieGen", "ernie_gen"),
    ("Ernie", "ernie"),
    ("GPT", "gpt"),
    ("LayoutXLM", "layoutxlm"),
    ("MBart", "mbart"),
    ("MPNet", "mpnet"),
    ("NeZha", "nezha"),
    ("Roberta", "roberta"),
    ("RoFormer", "roformer"),
    ("SqueezeBert", "squeezebert"),
    ("T5", "t5"),
    ("TinyBert", "tinybert"),
    ("Bert", "bert"),
    ("Bart", "bart"),
    ("UNIMO", "unimo"),
    ("UnifiedTransformer", "unified_transformer"),
    ("XLNet", "xlnet"),
])


def get_name_mapping(task='Model'):
    '''
    Task can be 'Model', 'ForPretraining', 'ForSequenceClassification', 'ForTokenClassification',
    'ForQuestionAnswering', 'ForMultipleChoice', 'ForMaskedLM', 'ForCausalLM', 'Encoder', 'Decoder',
    'Generator', 'Discriminator', 'ForConditionalGeneration'.
    '''
    NAME_MAPPING = OrderedDict()
    for key, value in MAPPING_NAMES.items():
        key1 = key + 'Model'
        key2 = key + 'Model_Import_Class'
        import_class = key + task
        NAME_MAPPING[key2] = import_class
        NAME_MAPPING[import_class] = value
        NAME_MAPPING[key1] = value

    return NAME_MAPPING


def get_init_configurations():
    CONFIGURATION_MODEL_MAPPING = OrderedDict()
    for key, class_name in MAPPING_NAMES.items():
        import_class = importlib.import_module(
            f"paddlenlp.transformers.{class_name}.modeling")
        model_name = getattr(import_class, key + 'Model')
        if key == 'ErnieGen':
            name = tuple(
                model_name.ernie_gen_pretrained_init_configuration.keys())
        else:
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
        if task:
            if cls._task_choice == True:
                cls._name_mapping = get_name_mapping(task)
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
                        mapping_init_class = pretrained_model_name_or_path.lower(
                        )
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
                        mapping_init_class = pretrained_model_name_or_path.lower(
                        )
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
        task (str): Specify a downstream task. `task` can be 'model', 'pretraining', 'sequence_classification',
            'token_classification', 'question_answering', 'multiple_choice', 'lm_head',
            'masked_lm', 'causal_lm', 'encoder', 'decoder', 'generator', 'discriminator',
            'conditional_generation'. We only support specify downstream tasks in AutoModel. Defaults to `None`.
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
    MAPPING_NAMES = get_init_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = get_name_mapping('Model')
    _task_choice = True


class AutoModelForPretraining(_BaseAutoModelClass):
    MAPPING_NAMES = get_init_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = get_name_mapping('ForPretraining')


class AutoModelForSequenceClassification(_BaseAutoModelClass):
    MAPPING_NAMES = get_init_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = get_name_mapping('ForSequenceClassification')


class AutoModelForTokenClassification(_BaseAutoModelClass):
    MAPPING_NAMES = get_init_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = get_name_mapping('ForTokenClassification')


class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    MAPPING_NAMES = get_init_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = get_name_mapping('ForQuestionAnswering')


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    MAPPING_NAMES = get_init_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = get_name_mapping('ForMultipleChoice')


class AutoModelForMaskedLM(_BaseAutoModelClass):
    MAPPING_NAMES = get_init_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = get_name_mapping('ForMaskedLM')


class AutoModelForCausalLM(_BaseAutoModelClass):
    MAPPING_NAMES = get_init_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = get_name_mapping('ForCausalLM')


class AutoEncoder(_BaseAutoModelClass):
    MAPPING_NAMES = get_init_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = get_name_mapping('Encoder')


class AutoDecoder(_BaseAutoModelClass):
    MAPPING_NAMES = get_init_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = get_name_mapping('Decoder')


class AutoGenerator(_BaseAutoModelClass):
    MAPPING_NAMES = get_init_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = get_name_mapping('Generator')


class AutoDiscriminator(_BaseAutoModelClass):
    MAPPING_NAMES = get_init_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = get_name_mapping('Discriminator')


class AutoModelForConditionalGeneration(_BaseAutoModelClass):
    MAPPING_NAMES = get_init_configurations()
    _model_mapping = MAPPING_NAMES
    _name_mapping = get_name_mapping('ForConditionalGeneration')
