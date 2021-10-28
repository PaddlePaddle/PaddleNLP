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
from paddlenlp.transformers import *
from paddlenlp.utils.downloader import COMMUNITY_MODEL_PREFIX

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
                    - The model was saved using :meth:`~PreTrainedModel.save_pretrained` and is reloaded
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


def tokenizer_type_to_module_name(key):
    """Converts a key to the corresponding module."""
    return key.replace("-", "_")


class _LazyAutoMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, tokenizer_mapping):
        self._tokenizer_mapping = tokenizer_mapping
        self._modules = {}

    def __getitem__(self, key):
        if key not in self._tokenizer_mapping:
            raise KeyError(key)
        value = self._tokenizer_mapping[key]
        #module_name = tokenizer_type_to_module_name(key)
        module_name = key
        if module_name not in self._modules:
            #self._modules[module_name] = importlib.import_module(f".{module_name}", "paddlenlp.transformers")
            self._modules[module_name] = importlib.import_module(
                f"paddlenlp.transformers.{module_name}.tokenizer")
        return getattr(self._modules[module_name], value)

    def keys(self):
        return self._tokenizer_mapping.keys()

    def values(self):
        return [self[k] for k in self._tokenizer_mapping.keys()]

    def items(self):
        return [(k, self[k]) for k in self._tokenizer_mapping.keys()]

    def __iter__(self):
        return iter(self._tokenizer_mapping.keys())

    def __contains__(self, item):
        return item in self._tokenizer_mapping


class _BaseAutoTokenizerClass:
    # Base class for auto models.
    _tokenizer_mapping = None
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
        else:
            community_config_path = os.path.join(COMMUNITY_MODEL_PREFIX,
                                                 pretrained_model_name_or_path,
                                                 cls.model_config_file)

            # From community-contributed pretrained models
            if os.path.isfile(community_config_path):
                config_file = community_config_path

            # Assuming from built-in pretrained models
            else:
                for tokenizer_names, tokenizer_class in cls._tokenizer_mapping.items(
                ):
                    if type(tokenizer_names) == tuple:
                        for pattern in tokenizer_names:
                            if pattern in pretrained_model_name_or_path:
                                return tokenizer_class.from_pretrained(
                                    pretrained_model_name_or_path, **kwargs)
                    else:
                        if tokenizer_names in pretrained_model_name_or_path:
                            return tokenizer_class.from_pretrained(
                                pretrained_model_name_or_path, **kwargs)


__all__ = ["AutoTokenizer", ]

TOKENIZER_MAPPING_NAMES = OrderedDict([
    # Base model mapping
    ("albert", "AlbertTokenizer"),
    ("bart", "BartTokenizer"),
    ("bigbird", "BigBirdTokenizer"),
    ("convbert", "ConvBertTokenizer"),
    ("distilbert", "DistilBertTokenizer"),
    ("electra", "ElectraTokenizer"),
    ("skep", "SkepTokenizer"),
    ("ernie-ctm", "ErnieCtmTokenizer"),
    ("ernie-doc", "ErnieDocTokenizer"),
    ("ernie-gram", "ErnieGramTokenizer"),
    ("ernie", "ErnieTokenizer"),
    ("gpt", "GPTTokenizer"),
    ("mpnet", "MPNetTokenizer"),
    ("nezha", "NeZhaTokenizer"),
    ("roberta", "RobertaTokenizer"),
    ("roformer", "RoFormerTokenizer"),
    ("tinybert", "TinyBertTokenizer"),
    ("bert", "BertTokenizer"),
    ("unimo", "UNIMOTokenizer"),
    ("xlnet", "XLNetTokenizer"),
])


def get_all_model_configurations():
    albert = tuple(AlbertPretrainedModel.pretrained_init_configuration.keys())
    bart = tuple(BartPretrainedModel.pretrained_init_configuration.keys())
    bigbird = tuple(BartPretrainedModel.pretrained_init_configuration.keys())
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
    erniegram = tuple(
        ErnieGramPretrainedModel.pretrained_init_configuration.keys())
    ernie = tuple(ErniePretrainedModel.pretrained_init_configuration.keys())
    gpt = tuple(GPTPretrainedModel.pretrained_init_configuration.keys())
    mpnet = tuple(MPNetPretrainedModel.pretrained_init_configuration.keys())
    nezha = tuple(NeZhaPretrainedModel.pretrained_init_configuration.keys())
    roberta = tuple(RobertaPretrainedModel.pretrained_init_configuration.keys())
    roformer = tuple(NeZhaPretrainedModel.pretrained_init_configuration.keys())
    tinybert = tuple(NeZhaPretrainedModel.pretrained_init_configuration.keys())
    bert = tuple(BertPretrainedModel.pretrained_init_configuration.keys())
    unimo = tuple(UNIMOPretrainedModel.pretrained_init_configuration.keys())
    xlnet = tuple(XLNetPretrainedModel.pretrained_init_configuration.keys())

    TOKENIZER_MAPPING_NAMES1 = OrderedDict([
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
        (unimo, UNIMOTokenizer),
        (xlnet, XLNetTokenizer),
    ])
    return TOKENIZER_MAPPING_NAMES1


TOKENIZER_MAPPING = _LazyAutoMapping(TOKENIZER_MAPPING_NAMES)


class AutoTokenizer(_BaseAutoTokenizerClass):
    #_model_mapping = MODEL_MAPPING_NAMES
    TOKENIZER_MAPPING_NAMES1 = get_all_model_configurations()
    _tokenizer_mapping = TOKENIZER_MAPPING_NAMES1


#AutoModel = auto_class_update(AutoModel)

print(AutoTokenizer.from_pretrained(('rbt3')))
