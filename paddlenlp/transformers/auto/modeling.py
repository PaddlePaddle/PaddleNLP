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
                    - A path or url.
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
            print(value)
        print(getattr(self._modules[module_name], value))
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
        #trust_remote_code = kwargs.pop("trust_remote_code", False)
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
    ("skep", "SkepModel"),
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
    ("tinybert", "TinyBertModel"),
    ("bert", "BertModel"),
    ("unimo", "UNIMOModel"),
    ("xlnet", "XLNetModel"),
])

MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict([
    # Model for pre-training mapping
    ("albert", "AlbertForPretraining"),
    ("bart", "BartForConditionalGeneration"),
    ("bigbird", "BigBirdForPretraining"),
    ("convbert", "ConvBertForTotalPretraining"),
    ("electra", "ElectraForTotalPretraining"),
    ("ernie", "ErnieForPretraining"),
    ("gpt", "GPTForPretraining"),
    ("nezha", "NeZhaForPretraining"),
    ("roformer", "RoFormerForPretraining"),
    ("tinybert", "TinyBertForPretraining"),
    ("bert", "BertForPretraining"),
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
    #("electra", "ElectraForMaskedLM"),
    ("mpnet", "MPNetForMaskedLM"),
    #("roberta", "RobertaForMaskedLM"),
])

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict([
    # Model for Sequence Classification mapping
    ("albert", "AlbertForSequenceClassification"),
    ("bart", "BartForSequenceClassification"),
    ("bigbird", "BigBirdForSequenceClassification"),
    ("convbert", "ConvBertForSequenceClassification"),
    ("distilbert", "DistilBertForSequenceClassification"),
    ("electra", "ElectraForSequenceClassification"),
    ("skep", "SkepForSequenceClassification"),
    ("ernie-doc", "ErnieDocForSequenceClassification"),
    ("ernie-gram", "ErnieGramForSequenceClassification"),
    ("ernie", "ErnieForSequenceClassification"),
    ("mpnet", "MPNetForSequenceClassification"),
    ("nezha", "NeZhaForSequenceClassification"),
    ("roberta", "RobertaForSequenceClassification"),
    ("roformer", "RoFormerForSequenceClassification"),
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
    ("convbert", "ConvBertForTokenClassification"),
    ("distilbert", "DistilBertForTokenClassification"),
    ("electra", "ElectraForTokenClassification"),
    ("skep", "SkepForTokenClassification"),
    ("ernie-ctm", "ErnieCtmForTokenClassification"),
    ("ernie-doc", "ErnieDocForTokenClassification"),
    ("ernie-gram", "ErnieGramForTokenClassification"),
    ("ernie", "ErnieForTokenClassification"),
    ("mpnet", "MPNetForTokenClassification"),
    ("nezha", "NeZhaForTokenClassification"),
    ("roberta", "RobertaForTokenClassification"),
    ("bert", "BertForTokenClassification"),
    ("roformer", "RoFormerForTokenClassification"),
    ("xlnet", "XLNetForTokenClassification"),
])

MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict([
    # Model for Multiple Choice mapping
    ("albert", "AlbertForMultipleChoice"),
    ("convbert", "ConvBertForMultipleChoice"),
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
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
    model = AutoModel.from_pretrained('albert-base-v1')
    print(model)
'''
    inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = {k: paddle.to_tensor([v]) for (k, v) in inputs.items()}
    outputs = model(**inputs)
    print(outputs)

    logits = outputs[0]
'''
