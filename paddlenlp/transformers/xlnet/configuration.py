# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
""" XLNet configuration"""
from __future__ import annotations

import logging
import warnings
from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)
__all__ = ["XLNET_PRETRAINED_INIT_CONFIGURATION", "XLNetConfig", "XLNET_PRETRAINED_RESOURCE_FILES_MAP"]
XLNET_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "xlnet-base-cased": "https://bj.bcebos.com/paddlenlp/models/transformers/xlnet/xlnet-base-cased.pdparams",
        "xlnet-large-cased": "https://bj.bcebos.com/paddlenlp/models/transformers/xlnet/xlnet-large-cased.pdparams",
        "chinese-xlnet-base": "https://bj.bcebos.com/paddlenlp/models/transformers/xlnet/chinese-xlnet-base.pdparams",
        "chinese-xlnet-mid": "https://bj.bcebos.com/paddlenlp/models/transformers/xlnet/chinese-xlnet-mid.pdparams",
        "chinese-xlnet-large": "https://bj.bcebos.com/paddlenlp/models/transformers/xlnet/chinese-xlnet-large.pdparams",
    }
}

XLNET_PRETRAINED_INIT_CONFIGURATION = {
    "xlnet-base-cased": {
        "attn_type": "bi",
        "bi_data": False,
        "clamp_len": -1,
        "d_head": 64,
        "d_inner": 3072,
        "d_model": 768,
        "dropout": 0.1,
        "classifier_dropout": 0.1,
        "ff_activation": "gelu",
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "mem_len": None,
        "n_head": 12,
        "n_layer": 12,
        "reuse_len": None,
        "same_length": False,
        "vocab_size": 32000,
    },
    "xlnet-large-cased": {
        "attn_type": "bi",
        "bi_data": False,
        "clamp_len": -1,
        "d_head": 64,
        "d_inner": 4096,
        "d_model": 1024,
        "dropout": 0.1,
        "classifier_dropout": 0.1,
        "ff_activation": "gelu",
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "mem_len": None,
        "n_head": 16,
        "n_layer": 24,
        "reuse_len": None,
        "same_length": False,
        "vocab_size": 32000,
    },
    "chinese-xlnet-base": {
        "attn_type": "bi",
        "bi_data": False,
        "clamp_len": -1,
        "d_head": 64,
        "d_inner": 3072,
        "d_model": 768,
        "dropout": 0.1,
        "classifier_dropout": 0.1,
        "ff_activation": "relu",
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "mem_len": None,
        "n_head": 12,
        "n_layer": 12,
        "reuse_len": None,
        "same_length": False,
        "vocab_size": 32000,
    },
    "chinese-xlnet-mid": {
        "attn_type": "bi",
        "bi_data": False,
        "clamp_len": -1,
        "d_head": 64,
        "d_inner": 3072,
        "d_model": 768,
        "dropout": 0.1,
        "classifier_dropout": 0.1,
        "ff_activation": "relu",
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "mem_len": None,
        "n_head": 12,
        "n_layer": 24,
        "reuse_len": None,
        "same_length": False,
        "vocab_size": 32000,
    },
    "chinese-xlnet-large": {
        "attn_type": "bi",
        "bi_data": False,
        "clamp_len": -1,
        "d_head": 64,
        "d_inner": 4096,
        "d_model": 1024,
        "dropout": 0.1,
        "classifier_dropout": 0.1,
        "ff_activation": "relu",
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "mem_len": None,
        "n_head": 16,
        "n_layer": 24,
        "reuse_len": None,
        "same_length": False,
        "vocab_size": 32000,
    },
}


class XLNetConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`XLNetModel`]. It is used to
    instantiate a XLNet model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [xlnet-large-cased] architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the XLNet model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`XLNetModel`].
        d_model (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        n_layer (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        d_inner (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        ff_activation (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the If string, `"gelu"`, `"relu"`, `"silu"` and
            `"gelu_new"` are supported.
        untie_r (`bool`, *optional*, defaults to `True`):
            Whether or not to untie relative position biases
        attn_type (`str`, *optional*, defaults to `"bi"`):
            The attention type used by the model. Set `"bi"` for XLNet, `"uni"` for Transformer-XL.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        mem_len (`int` or `None`, *optional*):
            The number of tokens to cache. The key/value pairs that have already been pre-computed in a previous
            forward pass won't be re-computed.
        reuse_len (`int`, *optional*):
            The number of tokens in the current batch to be cached and reused in the future.
        bi_data (`bool`, *optional*, defaults to `False`):
            Whether or not to use bidirectional input pipeline. Usually set to `True` during pretraining and `False`
            during finetuning.
        clamp_len (`int`, *optional*, defaults to -1):
            Clamp all relative distances larger than clamp_len. Setting this attribute to -1 means no clamping.
        same_length (`bool`, *optional*, defaults to `False`):
            Whether or not to use the same attention length for each token.
        summary_type (`str`, *optional*, defaults to "last"):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Has to be one of the following options:

                - `"last"`: Take the last token hidden state (like XLNet).
                - `"first"`: Take the first token hidden state (like BERT).
                - `"mean"`: Take the mean of all tokens hidden states.
                - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - `"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Whether or not to add a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (`boo`, *optional*, defaults to `True`):
            Used in the sequence classification and multiple choice models.

            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_last_dropout (`float`, *optional*, defaults to 0.1):
            Used in the sequence classification and multiple choice models.
            The dropout ratio to be used after the projection and activation.
        start_n_top (`int`, *optional*, defaults to 5):
            Used in the SQuAD evaluation script.
        end_n_top (`int`, *optional*, defaults to 5):
            Used in the SQuAD evaluation script.
        use_mems_eval (`bool`, *optional*, defaults to `True`):
            Whether or not the model should make use of the recurrent memory mechanism in evaluation mode.
        use_mems_train (`bool`, *optional*, defaults to `False`):
            Whether or not the model should make use of the recurrent memory mechanism in train mode.

    Examples:

    ```python
    >>> from transformers import XLNetConfig, XLNetModel

    >>> # Initializing a XLNet configuration
    >>> configuration = XLNetConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = XLNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xlnet"
    keys_to_ignore_at_inference = ["mems"]
    pretrained_init_configuration = XLNET_PRETRAINED_INIT_CONFIGURATION
    # attribute_map: Dict[str, str] = {"hidden_size": "classifier_dropout", "num_classes": "num_labels"}
    attribute_map: Dict[str, str] = {
        "n_token": "vocab_size",  # Backward compatibility
        "hidden_size": "d_model",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
        "num_classes": "num_labels",
    }

    def __init__(
        self,
        vocab_size=32000,
        d_model=1024,
        n_layer=24,
        n_head=16,
        d_inner=4096,
        ff_activation="gelu",
        untie_r=True,
        attn_type="bi",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        dropout=0.1,
        classfier_dropout=0.1,
        mem_len=512,
        reuse_len=None,
        use_mems_eval=True,
        use_mems_train=False,
        bi_data=False,
        clamp_len=-1,
        same_length=False,
        summary_type="last",
        summary_use_proj=True,
        summary_activation="tanh",
        summary_last_dropout=0.1,
        start_n_top=5,
        end_n_top=5,
        pad_token_id=5,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        """Constructs XLNetConfig."""
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        if d_model % n_head != 0:
            raise ValueError(f"'d_model % n_head' ({d_model % n_head}) should be equal to 0")
        if "d_head" in kwargs:
            if kwargs["d_head"] != d_model // n_head:
                raise ValueError(
                    f"`d_head` ({kwargs['d_head']}) should be equal to `d_model // n_head` ({d_model // n_head})"
                )
        self.d_head = d_model // n_head
        self.ff_activation = ff_activation
        self.d_inner = d_inner
        self.untie_r = untie_r
        self.attn_type = attn_type

        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.dropout = dropout
        self.classfier_dropout = classfier_dropout
        self.mem_len = mem_len
        self.reuse_len = reuse_len
        self.bi_data = bi_data
        self.clamp_len = clamp_len
        self.same_length = same_length

        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_last_dropout = summary_last_dropout
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top

        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        if "use_cache" in kwargs:
            warnings.warn(
                "The `use_cache` argument is deprecated and will be removed in a future version, use `use_mems_eval`"
                " instead.",
                FutureWarning,
            )
            use_mems_eval = kwargs["use_cache"]

        self.use_mems_eval = use_mems_eval
        self.use_mems_train = use_mems_train

    @property
    def max_position_embeddings(self):
        logger.info(f"The model {self.model_type} is one of the few models that has no sequence length limit.")
        return -1

    @max_position_embeddings.setter
    def max_position_embeddings(self, value):
        # Message copied from Transformer-XL documentation
        raise NotImplementedError(
            f"The model {self.model_type} is one of the few models that has no sequence length limit."
        )
