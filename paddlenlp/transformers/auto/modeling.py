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
import importlib
from collections import OrderedDict
import paddle
from paddlenlp.transformers import *
from paddlenlp.utils.downloader import COMMUNITY_MODEL_PREFIX, get_path_from_url

CLASS_DOCSTRING = """
    This is a generic model class that will be instantiated as one of the model classes of the library when created
    with the :meth:`~paddlenlp.transformers.BaseAutoModelClass.from_pretrained` class method.
    This class cannot be instantiated directly using ``__init__()`` (throws an error).
"""

FROM_PRETRAINED_DOCSTRING = """
        Instantiate one of the model classes of the library from a pretrained model.
        The model class to instantiate is selected based on the :obj:`model_type` property of the config object (either
        passed as an argument or loaded from :obj:`pretrained_model_name_or_path` if possible), or when it's missing,
        by falling back to using pattern matching on :obj:`pretrained_model_name_or_path`:
        List options
        The model is set in evaluation mode by default using ``model.eval()`` (so for instance, dropout modules are
        deactivated). To train the model, you should first set it back in training mode with ``model.train()``

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                Can be either:
                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
            model_args (additional positional arguments, `optional`):
                Will be passed along to the underlying model ``__init__()`` method.
            config (:class:`~transformers.PretrainedConfig`, `optional`):
                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:
                    - The model is a model provided by the library (loaded with the `model id` string of a pretrained
                      model).
                    - The model was saved using :meth:`~transformers.PreTrainedModel.save_pretrained` and is reloaded
                      by supplying the save directory.
                    - The model is loaded by supplying a local directory as ``pretrained_model_name_or_path`` and a
                      configuration JSON file named `config.json` is found in the directory.
            state_dict (`Dict[str, torch.Tensor]`, `optional`):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using
                :func:`~transformers.PreTrainedModel.save_pretrained` and
                :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_tf (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Load the model weights from a TensorFlow checkpoint save file (see docstring of
                ``pretrained_model_name_or_path`` argument).
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str]`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to only look at local files (e.g., not try downloading the model).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            trust_remote_code (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to :obj:`True` for repositories you trust and in which you have read the code, as it
                will execute code present on the Hub on your local machine.
            kwargs (additional keyword arguments, `optional`):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                :obj:`output_attentions=True`). Behaves differently depending on whether a ``config`` is provided or
                automatically loaded:
                    - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the
                      underlying model's ``__init__`` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class
                      initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of
                      ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute
                      with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration
                      attribute will be passed to the underlying model's ``__init__`` function.
"""
'''
def _get_model_class(config, model_mapping):
    supported_models = model_mapping[type(config)]
    if not isinstance(supported_models, (list, tuple)):
        return supported_models

    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in name_to_model:
            return name_to_model[arch]

    # If not architecture is set in the config or match the supported models, the first element of the tuple is the
    # defaults.
    return supported_models[0]

'''
"""
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                **kwargs)
        if hasattr(config, "auto_map") and cls.__name__ in config.auto_map:
            if not trust_remote_code:
                raise ValueError(
                    f"Loading {pretrained_model_name_or_path} requires you to execute the modeling file in that repo "
                    "on your local machine. Make sure you have read the code there to avoid malicious use, then set "
                    "the option `trust_remote_code=True` to remove this error.")
            if kwargs.get("revision", None) is None:
                logger.warn(
                    "Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure "
                    "no malicious code has been contributed in a newer revision."
                )
            class_ref = config.auto_map[cls.__name__]
            module_file, class_name = class_ref.split(".")
            model_class = get_class_from_dynamic_module(
                pretrained_model_name_or_path, module_file + ".py", class_name,
                **kwargs)
            return model_class.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                **kwargs)
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                **kwargs)
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )
"""


def insert_head_doc(docstring, head_doc=""):
    if len(head_doc) > 0:
        return docstring.replace(
            "one of the model classes of the library ",
            f"one of the model classes of the library (with a {head_doc} head) ",
        )
    return docstring.replace("one of the model classes of the library ",
                             "one of the base model classes of the library ")


'''

def auto_class_update(cls,
                      checkpoint_for_example="bert-base-cased",
                      head_doc=""):
    # Create a new class with the right name from the base class
    model_mapping = cls._model_mapping
    name = cls.__name__
    class_docstring = insert_head_doc(CLASS_DOCSTRING, head_doc=head_doc)
    cls.__doc__ = class_docstring.replace("BaseAutoModelClass", name)

    # Now we need to copy and re-register `from_pretrained` as class methods otherwise we can't
    # have a specific docstrings for them.
    from_pretrained_docstring = FROM_PRETRAINED_DOCSTRING

    from_pretrained = copy_func(_BaseAutoModelClass.from_pretrained)
    from_pretrained_docstring = insert_head_doc(
        from_pretrained_docstring, head_doc=head_doc)
    from_pretrained_docstring = from_pretrained_docstring.replace(
        "BaseAutoModelClass", name)
    from_pretrained_docstring = from_pretrained_docstring.replace(
        "checkpoint_placeholder", checkpoint_for_example)
    shortcut = checkpoint_for_example.split("/")[-1].split("-")[0]
    from_pretrained_docstring = from_pretrained_docstring.replace(
        "shortcut_placeholder", shortcut)
    from_pretrained.__doc__ = from_pretrained_docstring
    from_pretrained = replace_list_option_in_docstrings(
        model_mapping._model_mapping)(from_pretrained)
    cls.from_pretrained = classmethod(from_pretrained)
    return cls
'''


def get_values(model_mapping):
    result = []
    for model in model_mapping.values():
        if isinstance(model, (list, tuple)):
            result += list(model)
        else:
            result.append(model)
    return result


def model_type_to_module_name(key):
    """Converts a key to the corresponding module."""
    return key.replace("-", "_")


class _LazyAutoMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, model_mapping):
        self._model_mapping = model_mapping
        self._modules = {}

    def __getitem__(self, key):
        if key not in self._model_mapping:
            raise KeyError(key)
        value = self._model_mapping[key]
        module_name = model_type_to_module_name(key)
        if module_name not in self._modules:
            #self._modules[module_name] = importlib.import_module(f".{module_name}", "paddlenlp.transformers")
            self._modules[module_name] = importlib.import_module(
                f"paddlenlp.transformers.{module_name}.modeling")
        return getattr(self._modules[module_name], value)

    def keys(self):
        return self._model_mapping.keys()

    def values(self):
        return [self[k] for k in self._model_mapping.keys()]

    def items(self):
        return [(k, self[k]) for k in self._model_mapping.keys()]

    def __iter__(self):
        return iter(self._model_mapping.keys())

    def __contains__(self, item):
        return item in self._model_mapping


class _BaseAutoModelClass:
    # Base class for auto models.
    _model_mapping = None
    model_config_file = "model_config.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path).`"
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        #config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        #kwargs["_from_auto"] = True
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        # From local dir path
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path,
                                       cls.model_config_file)
        else:
            community_config_path = os.path.join(COMMUNITY_MODEL_PREFIX,
                                                 pretrained_model_name_or_path,
                                                 cls.model_config_file)
            # From community-contributed pretrained models

            if os.path.isfile(community_config_path):
                config_file = community_config_path

            # Assuming from built-in pretrained models
            else:
                for pattern, model_class in cls._model_mapping.items():
                    if pattern in pretrained_model_name_or_path:
                        #print(pattern, model_class)
                        return model_class.from_pretrained(
                            pretrained_model_name_or_path, **kwargs)


__all__ = [
    "AutoModel",
    "AutoModelForPreTraining",
    "AutoModelWithLMHead",
    "AutoModelForMaskedLM",
    "AutoModelForSequenceClassification",
    "AutoModelForTokenClassification",
    "AutoModelForQuestionAnswering",
    "AutoModelForMultipleChoice",
]
"""
__all__ = [
    "AlbertPretrainedModel", 
    "AlbertModel",
    "AlbertForPretraining",
    "AlbertForMaskedLM",
    "AlbertForSequenceClassification",
    "AlbertForTokenClassification",
    "AlbertForMultipleChoice",
]

__all__ = [
    'BartModel', 'BartPretrainedModel', 'BartEncoder', 'BartDecoder',
    'BartClassificationHead', 'BartForSequenceClassification',
    'BartForQuestionAnswering', 'BartForConditionalGeneration'
]

__all__ = [
    'BertModel',
    "BertPretrainedModel",
    'BertForPretraining',
    'BertPretrainingCriterion',
    'BertPretrainingHeads',
    'BertForSequenceClassification',
    'BertForTokenClassification',
    'BertForQuestionAnswering',
]

__all__ = [
    'BigBirdModel',
    'BigBirdPretrainedModel',
    'BigBirdForPretraining',
    'BigBirdPretrainingCriterion',
    'BigBirdForSequenceClassification',
    'BigBirdPretrainingHeads',
]

__all__ = [
    "ConvBertModel", "ConvBertPretrainedModel", "ConvBertForTotalPretraining",
    "ConvBertDiscriminator", "ConvBertGenerator", "ConvBertClassificationHead",
    "ConvBertForSequenceClassification", "ConvBertForTokenClassification",
    "ConvBertPretrainingCriterion", "ConvBertForQuestionAnswering",
    "ConvBertForMultipleChoice"
]

__all__ = [
    'DistilBertModel',
    'DistilBertPretrainedModel',
    'DistilBertForSequenceClassification',
    'DistilBertForTokenClassification',
    'DistilBertForQuestionAnswering',
    'DistilBertForMaskedLM',
]

__all__ = [
    'ElectraModel', 'ElectraPretrainedModel', 'ElectraForTotalPretraining',
    'ElectraDiscriminator', 'ElectraGenerator', 'ElectraClassificationHead',
    'ElectraForSequenceClassification', 'ElectraForTokenClassification',
    'ElectraPretrainingCriterion'
]

__all__ = [
    'ErnieModel', 'ErniePretrainedModel', 'ErnieForSequenceClassification',
    'ErnieForTokenClassification', 'ErnieForQuestionAnswering',
    'ErnieForPretraining', 'ErniePretrainingCriterion'
]

__all__ = [
    'ErnieCtmPretrainedModel', 'ErnieCtmModel', 'ErnieCtmWordtagModel',
    'ErnieCtmForTokenClassification'
]

__all__ = [
    'ErnieDocModel',
    'ErnieDocPretrainedModel',
    'ErnieDocForSequenceClassification',
    'ErnieDocForTokenClassification',
    'ErnieDocForQuestionAnswering',
]

__all__ = ["ErnieGenPretrainedModel", "ErnieForGeneration"]

__all__ = [
    'ErnieGramModel',
    'ErnieGramForSequenceClassification',
    'ErnieGramForTokenClassification',
    'ErnieGramForQuestionAnswering',
]

__all__ = [
    'GPTModel',
    "GPTPretrainedModel",
    'GPTForPretraining',
    'GPTPretrainingCriterion',
    'GPTForGreedyGeneration',
    'GPTLMHeadModel',
]

__all__ = [
    "MPNetModel",
    "MPNetPretrainedModel",
    "MPNetForMaskedLM",
    "MPNetForSequenceClassification",
    "MPNetForMultipleChoice",
    "MPNetForTokenClassification",
    "MPNetForQuestionAnswering",
]

__all__ = [
    'NeZhaModel', "NeZhaPretrainedModel", 'NeZhaForPretraining',
    'NeZhaForSequenceClassification', 'NeZhaPretrainingHeads',
    'NeZhaForTokenClassification', 'NeZhaForQuestionAnswering',
    'NeZhaForMultipleChoice'
]

__all__ = [
    'RobertaModel',
    'RobertaPretrainedModel',
    'RobertaForSequenceClassification',
    'RobertaForTokenClassification',
    'RobertaForQuestionAnswering',
]

__all__ = [
    "RoFormerModel",
    "RoFormerPretrainedModel",
    "RoFormerForPretraining",
    "RoFormerPretrainingCriterion",
    "RoFormerPretrainingHeads",
    "RoFormerForSequenceClassification",
    "RoFormerForTokenClassification",
    "RoFormerForQuestionAnswering",
]

__all__ = [
    'SkepModel', 'SkepPretrainedModel', 'SkepForSequenceClassification',
    'SkepForTokenClassification', 'SkepCrfForTokenClassification'
]

__all__ = [
    'TinyBertModel',
    'TinyBertPretrainedModel',
    'TinyBertForPretraining',
    'TinyBertForSequenceClassification',
]

__all__ = [
    "UNIMOPretrainedModel",
    'UNIMOModel',
    'UNIMOLMHeadModel',
]

__all__ = [
    "XLNetPretrainedModel",
    "XLNetModel",
    "XLNetForSequenceClassification",
    "XLNetForTokenClassification",
]

"""

MODEL_MAPPING_NAMES = OrderedDict([
    # Base model mapping
    ("albert", "AlbertModel"),
    ("bart", "BartModel"),
    ("bigbird", "BigBirdModel"),
    ("convbert", "ConvBertModel"),
    ("distilbert", "DistilBertModel"),
    ("electra", "ElectraModel"),
    ("ernie-ctm", "ErnieCtmModel"),
    ("ernie-doc", "ErnieDocModel"),
    ("ernie-gen", "ErnieForGeneration"),
    ("ernie-gram", "ErnieGramModel"),
    ("ernie", "ErnieModel"),
    ("gpt", "GPTModel"),
    ("mpnet", "MPNetModel"),
    ("nezha", "NeZhaModel"),
    ("roberta", "RobertaModel"),
    ("roformer", "RoFormerModel"),
    ("skep", "SkepModel"),
    ("tinybert", "TinyBertModel"),
    ("bert", "BertModel"),
    ("unimo", "UNIMOModel"),
    ("xlnet", "XLNetModel"),
])

MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict([
    # Model for pre-training mapping
    ("albert", "AlbertForPreTraining"),
    ("bart", "BartForConditionalGeneration"),
    ("bigbird", "BigBirdForPreTraining"),
    ("convbert", "ConvBertForTotalPretraining"),
    ("electra", "ElectraForTotalPreTraining"),
    ("ernie", "ErnieForPreTraining"),
    ("gpt", "GPTForPretraining"),
    ("nezha", "NeZhaForPretraining"),
    ("roformer", "RoformerForPretraining"),
    ("tinybert", "TinyBertForPretraining"),
    ("bert", "BertForPreTraining"),
])

MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict([
    # Model with LM heads mapping
    ("albert", "AlbertForMaskedLM"),
    ("bart", "BartForConditionalGeneration"),
    ("bigbird", "BigBirdPretrainingHeads"),
    ("convbert", "ConvBertClassificationHead"),
    ("distilbert", "DistilBertForMaskedLM"),
    ("bert", "BertPretrainingHeads"),
    ("electra", "ElectraClassificationHead"),
    ("gpt", "GPTLMHeadModel"),
    ("mpnet", "MPNetForMaskedLM"),
    ("nezha", "NeZhaPretrainingHeads"),
    ("roformer", "RoFormerPretrainingHeads"),
    ("unimo", "UNIMOLMHeadModel"),
])

MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict([
    # Model for Masked LM mapping
    ("albert", "AlbertForMaskedLM"),
    ("bart", "BartForConditionalGeneration"),
    ("distilbert", "DistilBertForMaskedLM"),
    ("electra", "ElectraForMaskedLM"),
    ("mpnet", "MPNetForMaskedLM"),
    ("roberta", "RobertaForMaskedLM"),
])

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict([
    # Model for Sequence Classification mapping
    ("albert", "AlbertForSequenceClassification"),
    ("bart", "BartForSequenceClassification"),
    ("bigbird", "BigBirdForSequenceClassification"),
    ("convbert", "ConvBertForSequenceClassification"),
    ("distilbert", "DistilBertForSequenceClassification"),
    ("electra", "ElectraForSequenceClassification"),
    ("ernie-doc", "ErnieDocForSequenceClassification"),
    ("ernie-gram", "ErnieGramForSequenceClassification"),
    ("ernie", "ErnieForSequenceClassification"),
    ("gpt", "GPTForSequenceClassification"),
    ("mpnet", "MPNetForSequenceClassification"),
    ("nezha", "NeZhaForSequenceClassification"),
    ("roberta", "RobertaForSequenceClassification"),
    ("roformer", "RoFormerForSequenceClassification"),
    ("skep", "SkepForSequenceClassification"),
    ("tinybert", "TinyBertForSequenceClassification"),
    ("bert", "BertForSequenceClassification"),
    ("xlnet", "XLNetForSequenceClassification"),
])

MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict([
    # Model for Question Answering mapping
    ("bart", "BartForQuestionAnswering"),
    ("convbert", "ConvBertForQuestionAnswering"),
    ("distilbert", "DistilBertForQuestionAnswering"),
    ("ernie-doc", "ErnieDocForQuestionAnswering"),
    ("ernie-gram", "ErnieGramForQuestionAnswering"),
    ("ernie", "ErnieForQuestionAnswering"),
    ("mpnet", "MPNetForQuestionAnswering"),
    ("nezha", "NeZhaForQuestionAnswering"),
    ("roberta", "RobertaForQuestionAnswering"),
    ("bert", "BertForQuestionAnswering"),
    ("roformer", "RoFormerForQuestionAnswering"),
])

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict([
    # Model for Token Classification mapping
    ("albert", "AlbertForTokenClassification"),
    ("bigbird", "BigBirdForTokenClassification"),
    ("convbert", "ConvBertForTokenClassification"),
    ("distilbert", "DistilBertForTokenClassification"),
    ("electra", "ElectraForTokenClassification"),
    ("ernie-ctm", "ErnieCtmForTokenClassification"),
    ("ernie-doc", "ErnieDocForTokenClassification"),
    ("ernie-gram", "ErnieGramForTokenClassification"),
    ("ernie", "ErnieForTokenClassification"),
    ("mpnet", "MPNetForTokenClassification"),
    ("nezha", "NeZhaForTokenClassification"),
    ("roberta", "RobertaForTokenClassification"),
    ("bert", "BertForTokenClassification"),
    ("roformer", "RoformerForTokenClassification"),
    ("skep", "SkepForTokenClassification"),
    ("xlnet", "XlnetForTokenClassification"),
])

MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict([
    # Model for Multiple Choice mapping
    ("albert", "AlbertForMultipleChoice"),
    ("convbert", "ConvbertForMultipleChoice"),
    ("mpnet", "MPNetForMultipleChoice"),
    ("nezha", "NeZhaForMultipleChoice"),
])

MODEL_MAPPING = _LazyAutoMapping(MODEL_MAPPING_NAMES)

MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(
    MODEL_FOR_PRETRAINING_MAPPING_NAMES)

MODEL_WITH_LM_HEAD_MAPPING = _LazyAutoMapping(MODEL_WITH_LM_HEAD_MAPPING_NAMES)

MODEL_FOR_MASKED_LM_MAPPING = _LazyAutoMapping(
    MODEL_FOR_MASKED_LM_MAPPING_NAMES)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES)

MODEL_FOR_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES)

MODEL_FOR_MULTIPLE_CHOICE_MAPPING = _LazyAutoMapping(
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES)


class AutoModel(_BaseAutoModelClass):
    #_model_mapping = MODEL_MAPPING_NAMES
    _model_mapping = MODEL_MAPPING


#AutoModel = auto_class_update(AutoModel)


class AutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PRETRAINING_MAPPING


#AutoModelForPreTraining = auto_class_update(
#    AutoModelForPreTraining, head_doc="pretraining")


# Private on purpose, the public class will add the deprecation warnings.
class AutoModelWithLMHead(_BaseAutoModelClass):
    _model_mapping = MODEL_WITH_LM_HEAD_MAPPING


#AutoModelWithLMHead = auto_class_update(
#    AutoModelWithLMHead, head_doc="language modeling")


class AutoModelForMaskedLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASKED_LM_MAPPING


#AutoModelForMaskedLM = auto_class_update(
#    AutoModelForMaskedLM, head_doc="masked language modeling")


class AutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


#AutoModelForSequenceClassification = auto_class_update(
#    AutoModelForSequenceClassification, head_doc="sequence classification")


class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING


#AutoModelForQuestionAnswering = auto_class_update(
#    AutoModelForQuestionAnswering, head_doc="question answering")


class AutoModelForTokenClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING


#AutoModelForTokenClassification = auto_class_update(
#    AutoModelForTokenClassification, head_doc="token classification")


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MULTIPLE_CHOICE_MAPPING


#AutoModelForMultipleChoice = auto_class_update(
#    AutoModelForMultipleChoice, head_doc="multiple choice")

if __name__ == '__main__':
    print(AutoModel.from_pretrained('albert-base-v1'))
