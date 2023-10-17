# coding=utf-8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Configuration base class and utilities."""
from __future__ import annotations

import copy
import json
import os
import os.path as osp
import re
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import paddle
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from paddlenlp import __version__
from paddlenlp.transformers.utils import resolve_cache_dir
from paddlenlp.utils.env import LEGACY_CONFIG_NAME
from paddlenlp.utils.log import logger

from ..utils import CONFIG_NAME
from ..utils.downloader import (
    COMMUNITY_MODEL_PREFIX,
    get_path_from_url_with_filelock,
    hf_file_exists,
    is_url,
    url_file_exists,
)

_re_configuration_file = re.compile(r"config\.(.*)\.json")


def custom_object_save(obj, folder, config=None):
    """
    Save the modeling files corresponding to a custom model/configuration/tokenizer etc. in a given folder. Optionally
    adds the proper fields in a config.

    Args:
        obj (`Any`): The object for which to save the module files.
        folder (`str` or `os.PathLike`): The folder where to save.
        config (`PretrainedConfig` or dictionary, `optional`):
            A config in which to register the auto_map corresponding to this custom object.
    """
    if obj.__module__ == "__main__":
        logger.warning(
            f"We can't save the code defining {obj} in {folder} as it's been defined in __main__. You should put "
            "this code in a separate module so we can include it in the saved folder and make it easier to share via "
            "the Hub."
        )

    def _set_auto_map_in_config(_config):
        module_name = obj.__class__.__module__
        last_module = module_name.split(".")[-1]
        full_name = f"{last_module}.{obj.__class__.__name__}"
        # Special handling for tokenizers
        if "Tokenizer" in full_name:
            slow_tokenizer_class = None
            fast_tokenizer_class = None
            if obj.__class__.__name__.endswith("Fast"):
                # Fast tokenizer: we have the fast tokenizer class and we may have the slow one has an attribute.
                fast_tokenizer_class = f"{last_module}.{obj.__class__.__name__}"
                if getattr(obj, "slow_tokenizer_class", None) is not None:
                    slow_tokenizer = getattr(obj, "slow_tokenizer_class")
                    slow_tok_module_name = slow_tokenizer.__module__
                    last_slow_tok_module = slow_tok_module_name.split(".")[-1]
                    slow_tokenizer_class = f"{last_slow_tok_module}.{slow_tokenizer.__name__}"
            else:
                # Slow tokenizer: no way to have the fast class
                slow_tokenizer_class = f"{last_module}.{obj.__class__.__name__}"

            full_name = (slow_tokenizer_class, fast_tokenizer_class)

        if isinstance(_config, dict):
            auto_map = _config.get("auto_map", {})
            auto_map[obj._auto_class] = full_name
            _config["auto_map"] = auto_map
        elif getattr(_config, "auto_map", None) is not None:
            _config.auto_map[obj._auto_class] = full_name
        else:
            _config.auto_map = {obj._auto_class: full_name}

    # Add object class to the config auto_map
    if isinstance(config, (list, tuple)):
        for cfg in config:
            _set_auto_map_in_config(cfg)
    elif config is not None:
        _set_auto_map_in_config(config)

    # Copy module file to the output folder.
    object_file = sys.modules[obj.__module__].__file__
    dest_file = Path(folder) / (Path(object_file).name)
    shutil.copy(object_file, dest_file)

    # Gather all relative imports recursively and make sure they are copied as well.
    # TODO(wujingjing): `get_relative_import_files` havn't supported yet.
    # for needed_file in get_relative_import_files(object_file):
    #     dest_file = Path(folder) / (Path(needed_file).name)
    #     shutil.copy(needed_file, dest_file)


def cached_path(
    url_or_filename,
    cache_dir=None,
    force_download=False,
    local_files_only=False,
) -> Optional[str]:
    """
    Given something that might be a URL (or might be a local path), determine which. If it's a URL, download the file
    and cache it, and return the path to the cached file. If it's already a local path, make sure the file exists and
    then return the path

    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-download the file even if it's already cached in the cache dir.
        user_agent: Optional string or dict that will be appended to the user-agent on remote requests.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if is_url(url_or_filename):
        if cache_dir is None:
            raise NotADirectoryError(
                "when download from url<%s>, cache_dir is required, but receive None", url_or_filename
            )

        if force_download:
            # remove the target file under cache_dir
            file_path = osp.join(cache_dir, osp.split(url_or_filename)[-1])
            shutil.rmtree(file_path, ignore_errors=True)

        # URL, so get it from the cache (downloading if necessary)
        output_path = get_path_from_url_with_filelock(
            url_or_filename,
            root_dir=cache_dir,
        )
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    else:
        raise FileNotFoundError(
            "can't find the file<{%s}> which should be valid url or local file path", url_or_filename
        )

    return output_path


def attribute_map(config: PretrainedConfig, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """map the <old-attr> to <new-attr> with configuration

    Args:
        config (PretrainedConfig): the instance of PretrainedConfig
        kwargs (Dict[str, Any]): the kwargs of attribute
    """
    for old_key, new_key in config.attribute_map.items():
        if old_key in kwargs:
            if new_key in kwargs:
                logger.warning(f"receive param<{old_key}> and param<{new_key}>, but the first one will be adopt")
            kwargs[new_key] = kwargs.pop(old_key)
    return kwargs


def convert_to_legacy_config(attribute_map: Dict[str, str], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    works when there are different fields between huggingface and paddle
    Args:
        attribute_map (Dict[str, str]): mapping of between standard config and paddle config
        config (Dict[str, Any]): config of huggingface transformers models
    Returns: the config which can be mapped into config of paddle model
    """
    if "init_args" in config:
        args = []
        for init_arg in config["init_args"]:
            init_arg = convert_to_legacy_config(attribute_map, init_arg)
            args.append(init_arg)
        config["init_args"] = args

    # TODO(wj-Mcat): to improve compatibility for: old local config and new PretrainedConfig, eg:
    # { "init_args": [], "init_class": "", "num_classes": 12 }
    for standard_field, paddle_field in attribute_map.items():
        value = config.pop(standard_field, None) or config.pop(paddle_field, None)
        if value is not None:
            config[paddle_field] = value
    return config


def flatten_model_config(config: dict) -> dict:
    """flatten the model config which can be old-style model config

    Args:
        config (dict): the source of config which can be flatten config or nest config

    Returns:
        dict: the flatten config
    """
    # 1. extract the init_args into the top level
    init_args = config.pop("init_args", [])

    index = 0
    while index < len(init_args):
        if isinstance(init_args[index], dict):
            for key, value in init_args[index].items():
                if key not in config:
                    config[key] = value
            init_args.pop(index)
        else:
            index += 1

    if init_args:
        config["init_args"] = init_args

    # 2. convert `init_class` into `architectures`
    if "init_class" in config:
        config["architectures"] = [config.pop("init_class")]

    return config


def is_standard_config(config: Union[PretrainedConfig, Dict[str, Any]]) -> bool:
    """
    check whether the config is standard
    Args:
        config: the dict data of config
    """
    if isinstance(config, PretrainedConfig):
        return True

    return "init_class" not in config and "architectures" in config


def resolve_hf_config_path(repo_id: str, cache_dir: str, subfolder=None) -> str:
    """resolve config file from hf hub

    Args:
        repo_id (str): the repo name from huggingface hub
        cache_dir (str): the cachedir
        subfolder (str, optional) An optional value corresponding to a folder inside the repo.

    Returns:
        str: the downloaded config file
    """
    if hf_file_exists(repo_id=repo_id, filename=CONFIG_NAME, subfolder=subfolder):
        file_name = CONFIG_NAME
    else:
        raise EntryNotFoundError(f"can not find the paddle/pytorch config file from: https://huggingface.co/{repo_id}")

    return hf_hub_download(
        repo_id=repo_id,
        filename=file_name,
        cache_dir=cache_dir,
        subfolder=subfolder,
        library_name="PaddleNLP",
        library_version=__version__,
    )


class PretrainedConfig:
    r"""
    Base class for all configuration classes. Handles a few parameters common to all models' configurations as well as
    methods for loading/downloading/saving configurations.

    <Tip>

    A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to
    initialize a model does **not** load the model weights. It only affects the model's configuration.

    </Tip>

    Class attributes (overridden by derived classes):

    - **model_type** (`str`) -- An identifier for the model type, serialized into the JSON file, and used to recreate
      the correct object in [`~paddlenlp.AutoConfig`].
    - **is_composition** (`bool`) -- Whether the config class is composed of multiple sub-configs. In this case the
      config has to be initialized from two or more configs of type [`~paddlenlp.PretrainedConfig`] like:
      [`~paddlenlp.EncoderDecoderConfig`] or [`~RagConfig`].
    - **keys_to_ignore_at_inference** (`List[str]`) -- A list of keys to ignore by default when looking at dictionary
      outputs of the model during inference.
    - **attribute_map** (`Dict[str, str]`) -- A dict that maps model specific attribute names to the standardized
      naming of attributes.

    Common attributes (present in all subclasses):

    - **vocab_size** (`int`) -- The number of tokens in the vocabulary, which is also the first dimension of the
      embeddings matrix (this attribute may be missing for models that don't have a text modality like ViT).
    - **hidden_size** (`int`) -- The hidden size of the model.
    - **num_attention_heads** (`int`) -- The number of attention heads used in the multi-head attention layers of the
      model.
    - **num_hidden_layers** (`int`) -- The number of blocks in the model.

    Arg:
        name_or_path (`str`, *optional*, defaults to `""`):
            Store the string that was passed to [`PreTrainedModel.from_pretrained`] or
            [`PreTrainedModel.from_pretrained`] as `pretrained_model_name_or_path` if the configuration was created
            with such a method.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return all hidden-states.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not the model should returns all attentions.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return a [`~paddlenlp.transformers.model_outputs.ModelOutput`] instead of a plain tuple.
        is_encoder_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as an encoder/decoder or not.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as decoder or not (in which case it's used as an encoder).
        cross_attention_hidden_size** (`bool`, *optional*):
            The hidden size of the cross-attention layer in case the model is used as a decoder in an encoder-decoder
            setting and the cross-attention hidden dimension differs from `self.config.hidden_size`.
        add_cross_attention (`bool`, *optional*, defaults to `False`):
            Whether cross-attention layers should be added to the model. Note, this option is only relevant for models
            that can be used as decoder models within the [`EncoderDecoderModel`] class, which consists of all models
            in `AUTO_MODELS_FOR_CAUSAL_LM`.
        tie_encoder_decoder (`bool`, *optional*, defaults to `False`):
            Whether all encoder weights should be tied to their equivalent decoder weights. This requires the encoder
            and decoder model to have the exact same parameter names.
        prune_heads (`Dict[int, List[int]]`, *optional*, defaults to `{}`):
            Pruned heads of the model. The keys are the selected layer indices and the associated values, the list of
            heads to prune in said layer.

            For instance `{1: [0, 2], 2: [2, 3]}` will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        chunk_size_feed_forward (`int`, *optional*, defaults to `0`):
            The chunk size of all feed forward layers in the residual attention blocks. A chunk size of `0` means that
            the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes `n` <
            sequence_length embeddings at a time. For more information on feed forward chunking, see [How does Feed
            Forward Chunking work?](../glossary.html#feed-forward-chunking).

        > Parameters for sequence generation

        max_length (`int`, *optional*, defaults to 20):
            Maximum length that will be used by default in the `generate` method of the model.
        min_length (`int`, *optional*, defaults to 10):
            Minimum length that will be used by default in the `generate` method of the model.
        do_sample (`bool`, *optional*, defaults to `False`):
            Flag that will be used by default in the `generate` method of the model. Whether or not to use sampling ;
            use greedy decoding otherwise.
        early_stopping (`bool`, *optional*, defaults to `False`):
            Flag that will be used by default in the `generate` method of the model. Whether to stop the beam search
            when at least `num_beams` sentences are finished per batch or not.
        num_beams (`int`, *optional*, defaults to 1):
            Number of beams for beam search that will be used by default in the `generate` method of the model. 1 means
            no beam search.
        num_beam_groups (`int`, *optional*, defaults to 1):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams
            that will be used by default in the `generate` method of the model. 1 means no group beam search.
        diversity_penalty (`float`, *optional*, defaults to 0.0):
            Value to control diversity for group beam search. that will be used by default in the `generate` method of
            the model. 0 means no diversity penalty. The higher the penalty, the more diverse are the outputs.
        temperature (`float`, *optional*, defaults to 1):
            The value used to module the next token probabilities that will be used by default in the `generate` method
            of the model. Must be strictly positive.
        top_k (`int`, *optional*, defaults to 50):
            Number of highest probability vocabulary tokens to keep for top-k-filtering that will be used by default in
            the `generate` method of the model.
        top_p (`float`, *optional*, defaults to 1):
            Value that will be used by default in the `generate` method of the model for `top_p`. If set to float < 1,
            only the most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.
        repetition_penalty (`float`, *optional*, defaults to 1):
            Parameter for repetition penalty that will be used by default in the `generate` method of the model. 1.0
            means no penalty.
        length_penalty (`float`, *optional*, defaults to 1):
            Exponential penalty to the length that will be used by default in the `generate` method of the model.
        no_repeat_ngram_size (`int`, *optional*, defaults to 0) -- Value that will be used by default in the
            `generate` method of the model for `no_repeat_ngram_size`. If set to int > 0, all ngrams of that size can
            only occur once.
        encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 0) -- Value that will be used by
            default in the `generate` method of the model for `encoder_no_repeat_ngram_size`. If set to int > 0, all
            ngrams of that size that occur in the `encoder_input_ids` cannot occur in the `decoder_input_ids`.
        bad_words_ids (`List[int]`, *optional*):
            List of token ids that are not allowed to be generated that will be used by default in the `generate`
            method of the model. In order to get the tokens of the words that should not appear in the generated text,
            use `tokenizer.encode(bad_word, add_prefix_space=True)`.
        num_return_sequences (`int`, *optional*, defaults to 1):
            Number of independently computed returned sequences for each element in the batch that will be used by
            default in the `generate` method of the model.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether the model should return the logits when used for generation.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether the model should return a [`~paddlenlp.transformers.model_outputs.ModelOutput`] instead of a `paddlenlp.Tensor`.
        forced_bos_token_id (`int`, *optional*):
            The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful for
            multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be the target
            language token.
        forced_eos_token_id (`int`, *optional*):
            The id of the token to force as the last generated token when `max_length` is reached.
        remove_invalid_values (`bool`, *optional*):
            Whether to remove possible _nan_ and _inf_ outputs of the model to prevent the generation method to crash.
            Note that using `remove_invalid_values` can slow down generation.

        > Parameters for fine-tuning tasks

        architectures (`List[str]`, *optional*):
            Model architectures that can be used with the model pretrained weights.
        finetuning_task (`str`, *optional*):
            Name of the task used to fine-tune the model. This can be used when converting from an original checkpoint.
        id2label (`Dict[int, str]`, *optional*):
            A map from index (for instance prediction index, or target index) to label.
        label2id (`Dict[str, int]`, *optional*): A map from label to index for the model.
        num_labels (`int`, *optional*):
            Number of labels to use in the last layer added to the model, typically for a classification task.
        task_specific_params (`Dict[str, Any]`, *optional*):
            Additional keyword arguments to store for the current task.
        problem_type (`str`, *optional*):
            Problem type for `XxxForSequenceClassification` models. Can be one of `"regression"`,
            `"single_label_classification"` or `"multi_label_classification"`.

        > Parameters linked to the tokenizer

        tokenizer_class (`str`, *optional*):
            The name of the associated tokenizer class to use (if none is set, will use the tokenizer associated to the
            model by default).
        prefix (`str`, *optional*):
            A specific prompt that should be added at the beginning of each text before calling the model.
        bos_token_id (`int`, *optional*): The id of the _beginning-of-stream_ token.
        pad_token_id (`int`, *optional*): The id of the _padding_ token.
        eos_token_id (`int`, *optional*): The id of the _end-of-stream_ token.
        decoder_start_token_id (`int`, *optional*):
            If an encoder-decoder model starts decoding with a different token than _bos_, the id of that token.
        sep_token_id (`int`, *optional*): The id of the _separation_ token.

        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
        dtype (`str`, *optional*):
            The `dtype` of the weights. This attribute can be used to initialize the model to a non-default `dtype`
            (which is normally `float32`) and thus allow for optimal storage allocation. For example, if the saved
            model is `float16`, ideally we want to load it back using the minimal amount of memory needed to load
            `float16` weights. Since the config object is stored in plain text, this attribute contains just the
            floating type string without the `paddle.` prefix. For example, for `paddle.float16` ``dtype` is the
            `"float16"` string.

            This attribute is currently not being used during model loading time, but this may change in the future
            versions. But we can already start preparing for the future by saving the dtype with save_pretrained.
    """
    model_type: str = ""
    is_composition: bool = False

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # TODO(wj-Mcat): this comment should be removed after this feature is accepted by PaddleNLP teams
    # `pretrained_init_configuration` can be `dict` or `url`: eg:
    #     {
    #         "bert-base-uncased": {
    #             "vocab_size": 30522,
    #             "hidden_size": 768,
    #         },
    #         "bert-large-uncased": "https://bj.bcebos.com/paddlenlp/models/transformers/model_config.json"
    #     }
    #
    # advantages:
    #     1. reuse the concept: `pretrained_init_configuration` and extend it
    #     2. make code more concise when support resource file
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    pretrained_init_configuration = {}

    # global attribute mapping
    attribute_map: Dict[str, str] = {"num_classes": "num_labels"}

    _auto_class: Optional[str] = None

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)
        assert hasattr(self, key)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)

    def __init__(self, **kwargs):
        # Attributes with defaults
        # map the old attr to new atr, eg: num_classes -> num_labels
        kwargs = attribute_map(self, kwargs=kwargs)

        self.return_dict = kwargs.pop("return_dict", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.use_cache = kwargs.pop("use_cache", False)
        self.use_flash_attention = kwargs.pop("use_flash_attention", False)
        self.is_causal = kwargs.pop("is_causal", True)
        self.quantization_config = kwargs.pop("quantization_config", None)

        self.pruned_heads = kwargs.pop("pruned_heads", {})
        self.tie_word_embeddings = kwargs.pop(
            "tie_word_embeddings", True
        )  # Whether input and output word embeddings should be tied for all MLM, LM and Seq2Seq models.

        # parameter for model dtype
        if "torch_dtype" in kwargs:
            self.dtype = kwargs.pop("torch_dtype")
        else:
            self.dtype = kwargs.pop("dtype", paddle.get_default_dtype())

        # Parameters for tensor parallel
        self.tensor_parallel_degree = kwargs.pop("tensor_parallel_degree", -1)
        self.tensor_parallel_rank = kwargs.pop("tensor_parallel_rank", 0)
        # If set to True, this option is used with fleet.meta_parallel.ParallelCrossEntropy
        # to calculate cross-entropy loss for parallel model.
        self.tensor_parallel_output = kwargs.pop("tensor_parallel_output", False)
        # Temporary switch to control hook vs. PyLayer implementation of recompute
        self.recompute_use_reentrant = kwargs.pop("recompute_use_reentrant", False)

        # Is decoder is used in encoder-decoder models to differentiate encoder from decoder
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.is_decoder = kwargs.pop("is_decoder", False)
        self.cross_attention_hidden_size = kwargs.pop("cross_attention_hidden_size", None)
        self.add_cross_attention = kwargs.pop("add_cross_attention", False)
        self.tie_encoder_decoder = kwargs.pop("tie_encoder_decoder", False)

        # Parameters for sequence generation
        self.max_length = kwargs.pop("max_length", 20)
        self.min_length = kwargs.pop("min_length", 0)
        self.do_sample = kwargs.pop("do_sample", False)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
        self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.typical_p = kwargs.pop("typical_p", 1.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", 0)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forward", 0)
        self.output_scores = kwargs.pop("output_scores", False)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.remove_invalid_values = kwargs.pop("remove_invalid_values", False)
        self.exponential_decay_length_penalty = kwargs.pop("exponential_decay_length_penalty", None)

        # Fine-tuning task arguments
        self.architectures = kwargs.pop("architectures", None)
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        self.id2label = kwargs.pop("id2label", None)
        self.label2id = kwargs.pop("label2id", None)
        if self.id2label is not None:
            num_labels = kwargs.pop("num_labels", None)
            if num_labels is not None and len(self.id2label) != num_labels:
                logger.warning(
                    f"You passed along `num_labels={num_labels}` with an incompatible id to label map: "
                    f"{self.id2label}. The number of labels wil be overwritten to {self.num_labels}."
                )
            self.id2label = dict((int(key), value) for key, value in self.id2label.items())
            # Keys are always strings in JSON so convert ids to int here.
        else:
            self.num_labels = kwargs.pop("num_labels", 2)
        self.num_choices = kwargs.pop("num_choices", None)

        self.classifier_dropout = kwargs.pop("classifier_dropout", None)

        # Tokenizer arguments TODO: eventually tokenizer and models should share the same config
        self.tokenizer_class = kwargs.pop("tokenizer_class", None)
        self.prefix = kwargs.pop("prefix", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.sep_token_id = kwargs.pop("sep_token_id", None)

        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # task specific arguments
        self.task_specific_params = kwargs.pop("task_specific_params", None)

        # regression / multi-label classification
        self.problem_type = kwargs.pop("problem_type", None)
        allowed_problem_types = ("regression", "single_label_classification", "multi_label_classification")
        if self.problem_type is not None and self.problem_type not in allowed_problem_types:
            raise ValueError(
                f"The config parameter `problem_type` was not understood: received {self.problem_type} "
                "but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid."
            )

        # Name or path to the pretrained checkpoint
        self._name_or_path = str(kwargs.pop("name_or_path", ""))

        # Drop the transformers version info
        self.paddlenlp_version = kwargs.pop("paddlenlp_version", None)

        # Deal with gradient checkpointing
        if kwargs.get("gradient_checkpointing", False):
            warnings.warn(
                "Passing `gradient_checkpointing` to a config initialization is deprecated and will be removed in v5 "
                "Transformers. Using `model.gradient_checkpointing_enable()` instead, or if you are using the "
                "`Trainer` API, pass `gradient_checkpointing=True` in your `TrainingArguments`."
            )

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    @property
    def name_or_path(self) -> str:
        return getattr(self, "_name_or_path", None)

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)  # Make sure that name_or_path is a string (for JSON encoding)

    @property
    def use_return_dict(self) -> bool:
        """
        `bool`: Whether or not return [`~paddlenlp.transformers.model_outputs.ModelOutput`] instead of tuples.
        """
        return self.return_dict

    @property
    def num_labels(self) -> int:
        """
        `int`: The number of labels for classification models.
        """
        return len(self.id2label)

    @num_labels.setter
    def num_labels(self, num_labels: int):
        if not hasattr(self, "id2label") or self.id2label is None or len(self.id2label) != num_labels:
            self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
            self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            kwargs:
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f"Configuration saved in {output_config_file}")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        from_hf_hub: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a [`PretrainedConfig`] (or a derived class) from a pretrained model configuration.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
                  paddlenlp bos server. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a configuration file saved using the
                  [`~PretrainedConfig.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved configuration JSON *file*, e.g., `./my_model_directory/configuration.json`.
            from_hf_hub (bool, *optional*):
                load config from huggingface hub: https://huggingface.co/models
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        <Tip>

        Passing `use_auth_token=True` is required when you want to use a private model.

        </Tip>

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from this pretrained model.

        Examples:

        ```python
        # We can't instantiate directly the base class *PretrainedConfig* so let's show the examples on a
        # derived class: BertConfig
        config = BertConfig.from_pretrained(
            "bert-base-uncased"
        )  # Download configuration from huggingface.co and cache.
        config = BertConfig.from_pretrained(
            "./test/saved_model/"
        )  # E.g. config (or model) was saved using *save_pretrained('./test/saved_model/')*
        config = BertConfig.from_pretrained("./test/saved_model/my_configuration.json")
        config = BertConfig.from_pretrained("bert-base-uncased", output_attentions=True, foo=False)
        assert config.output_attentions == True
        config, unused_kwargs = BertConfig.from_pretrained(
            "bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
        )
        assert config.output_attentions == True
        assert unused_kwargs == {"foo": False}
        ```"""
        kwargs.update({"from_hf_hub": from_hf_hub, "cache_dir": cache_dir})
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        [`PretrainedConfig`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.

        """
        original_kwargs = copy.deepcopy(kwargs)
        cache_dir = kwargs.pop("cache_dir", None)
        from_hf_hub = kwargs.pop("from_hf_hub", False)
        cache_dir = resolve_cache_dir(pretrained_model_name_or_path, from_hf_hub, cache_dir)

        # Get config dict associated with the base config file
        config_dict, kwargs = cls._get_config_dict(
            pretrained_model_name_or_path, cache_dir=cache_dir, from_hf_hub=from_hf_hub, **kwargs
        )

        # That config file may point us toward another config file to use.
        if "configuration_files" in config_dict:
            original_kwargs["cache_dir"] = cache_dir
            configuration_file = get_configuration_file(config_dict["configuration_files"])
            config_dict, kwargs = cls._get_config_dict(
                pretrained_model_name_or_path, _configuration_file=configuration_file, **original_kwargs
            )

        return config_dict, kwargs

    @classmethod
    def _get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cache_dir = kwargs.pop("cache_dir", None)
        from_hf_hub = kwargs.pop("from_hf_hub", False)
        subfolder = kwargs.pop("subfolder", None)

        force_download = kwargs.pop("force_download", False)
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        resolved_config_file = None

        # 0. init from pretrained_init_configuration
        if pretrained_model_name_or_path in cls.pretrained_init_configuration:
            # which can be: dict or url
            pretrained_model_name_or_path = cls.pretrained_init_configuration[pretrained_model_name_or_path]

            if isinstance(pretrained_model_name_or_path, dict):
                return pretrained_model_name_or_path, kwargs

        # 1. get the configuration file from local file, eg: /cache/path/model_config.json
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_config_file = pretrained_model_name_or_path

        # 2. get the configuration file from url, eg: https://ip/path/to/model_config.json
        elif is_url(pretrained_model_name_or_path):
            resolved_config_file = get_path_from_url_with_filelock(
                pretrained_model_name_or_path, cache_dir, check_exist=not force_download
            )
        # 3. get the configuration file from local dir with default name, eg: /local/path
        elif os.path.isdir(pretrained_model_name_or_path):
            configuration_file = kwargs.pop("_configuration_file", CONFIG_NAME)
            configuration_file = os.path.join(pretrained_model_name_or_path, configuration_file)
            if os.path.exists(configuration_file):
                resolved_config_file = configuration_file
            else:
                # try to detect old-school config file
                configuration_file = os.path.join(pretrained_model_name_or_path, LEGACY_CONFIG_NAME)
                if os.path.exists(configuration_file):
                    resolved_config_file = configuration_file
                else:
                    raise FileNotFoundError(
                        "please make sure there is `model_config.json` under the dir, or you can pass the `_configuration_file` "
                        "param into `from_pretarined` method to specific the configuration file name"
                    )  # 4. load it as the community resource file

        # 4. get the configuration file from HF hub
        elif from_hf_hub:
            resolved_config_file = resolve_hf_config_path(
                repo_id=pretrained_model_name_or_path, cache_dir=cache_dir, subfolder=subfolder
            )
        else:
            community_url = "/".join([COMMUNITY_MODEL_PREFIX, pretrained_model_name_or_path, CONFIG_NAME])
            if url_file_exists(community_url):
                return cls._get_config_dict(community_url, cache_dir=cache_dir, **kwargs)

            community_url = "/".join([COMMUNITY_MODEL_PREFIX, pretrained_model_name_or_path, LEGACY_CONFIG_NAME])
            if url_file_exists(community_url):
                return cls._get_config_dict(community_url, cache_dir=cache_dir, **kwargs)

            raise FileNotFoundError(f"configuration file<{CONFIG_NAME}> or <{LEGACY_CONFIG_NAME}> not found")

        try:
            logger.info(f"Loading configuration file {resolved_config_file}")
            # Load config dict
            config_dict = cls._dict_from_json_file(resolved_config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(f"Config file<'{resolved_config_file}'> is not a valid JSON file.")

        return config_dict, kwargs

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        """
        Instantiates a [`PretrainedConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the [`~PretrainedConfig.get_config_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # do standard config map: there are some old-school pretrained-config not refactored.
        config_dict = convert_to_legacy_config(cls.attribute_map, config_dict)

        config_dict = flatten_model_config(config_dict)

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

        # Update config with kwargs if needed
        if "num_labels" in kwargs and "id2label" in kwargs:
            num_labels = kwargs["num_labels"]
            id2label = kwargs["id2label"] if kwargs["id2label"] is not None else []
            if len(id2label) != num_labels:
                raise ValueError(
                    f"You passed along `num_labels={num_labels }` with an incompatible id to label map: "
                    f"{kwargs['id2label']}. Since those arguments are inconsistent with each other, you should remove "
                    "one of them."
                )
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                if key != "dtype":
                    to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> "PretrainedConfig":
        """
        Instantiates a [`PretrainedConfig`] from the path to a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from that JSON file.

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = PretrainedConfig().to_dict()

        # get class specific config dict
        class_config_dict = self.__class__().to_dict() if not self.is_composition else {}

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if (
                key not in default_config_dict
                or key == "paddlenlp_version"
                or value != default_config_dict[key]
                or (key in class_config_dict and value != class_config_dict[key])
            ):
                serializable_config_dict[key] = value

        return serializable_config_dict

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        if "_auto_class" in output:
            del output["_auto_class"]

        return output

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def update(self, config_dict: Dict[str, Any]):
        """
        Updates attributes of this class with attributes from `config_dict`.

        Args:
            config_dict (`Dict[str, Any]`): Dictionary of attributes that should be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)

    def update_from_string(self, update_str: str):
        """
        Updates attributes of this class with attributes from `update_str`.

        The expected format is ints, floats and strings as is, and for booleans use `true` or `false`. For example:
        "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"

        The keys to change have to already exist in the config object.

        Args:
            update_str (`str`): String with attributes that should be updated for this class.

        """

        d = dict(x.split("=") for x in update_str.split(","))
        for k, v in d.items():
            if not hasattr(self, k):
                raise ValueError(f"key {k} isn't in the original config dict")

            old_v = getattr(self, k)
            if isinstance(old_v, bool):
                if v.lower() in ["true", "1", "y", "yes"]:
                    v = True
                elif v.lower() in ["false", "0", "n", "no"]:
                    v = False
                else:
                    raise ValueError(f"can't derive true or false from {v} (key {k})")
            elif isinstance(old_v, int):
                v = int(v)
            elif isinstance(old_v, float):
                v = float(v)
            elif not isinstance(old_v, str):
                raise ValueError(
                    f"You can only update int, float, bool or string values in the config, got {v} for key {k}"
                )

            setattr(self, k, v)

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoConfig"):
        """
        Register this class with a given auto class. This should only be used for custom configurations as the ones in
        the library are already mapped with `AutoConfig`.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoConfig"`):
                The auto class to register this new configuration with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    def get(self, key, default=None):
        """
        Return the value for key if config class has the attribute , else default.
        If default is not given, it defaults to None, so that this method never raises a AttributeError.
        """
        try:
            value = self.__getattribute__(key)
        except AttributeError:
            return default
        else:
            return value


def get_configuration_file(configuration_files: List[str]) -> str:
    """
    Get the configuration file to use for this version of paddlenlp.

    # TODO: there is not supported actual application models, but useful.
        this method has not been tested, so be caution to use this feature.

    Args:
        configuration_files (`List[str]`): The list of available configuration files.

    Returns:
        `str`: The configuration file to use.
    """
    configuration_files_map = {}
    for file_name in configuration_files:
        search = _re_configuration_file.search(file_name)
        if search is not None:
            v = search.groups()[0]
            configuration_files_map[v] = file_name
    available_versions = sorted(configuration_files_map.keys())

    # Defaults to FULL_CONFIGURATION_FILE and then try to look at some newer versions.
    configuration_file = CONFIG_NAME

    # FIXME: (wj-Mcat) remove the hard dependency of `packaging` which can compare
    # the version of package, also be uesed in `transfromer`.
    # **But**, we don't support version compare function now. so remove the hard dependency.
    from packaging import version

    paddlenlp_version = version.parse(__version__)
    for v in available_versions:
        if version.parse(v) <= paddlenlp_version:
            configuration_file = configuration_files_map[v]
        else:
            # No point going further since the versions are sorted.
            break

    return configuration_file
