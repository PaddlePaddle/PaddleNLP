# coding=utf-8
# Copyright 2020 The Trax Authors and The HuggingFace Inc. team.
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
""" Reformer model configuration"""
from __future__ import annotations

from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["REFORMER_PRETRAINED_INIT_CONFIGURATION", "ReformerConfig", "REFORMER_PRETRAINED_RESOURCE_FILES_MAP"]

REFORMER_PRETRAINED_INIT_CONFIGURATION = {
    "reformer-enwik8": {
        "tie_word_embeddings": False,
        "is_decoder": True,
        "chunk_size_feed_forward": 0,
        "pad_token_id": 0,
        "hash_seed": None,
        "vocab_size": 258,
        "attention_head_size": 128,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_hashes": 4,
        "num_hidden_layers": 12,
        "num_buckets": 512,
        "lsh_attn_chunk_length": 256,
        "local_attn_chunk_length": 128,
        "lsh_num_chunks_after": 0,
        "lsh_num_chunks_before": 1,
        "local_num_chunks_after": 0,
        "local_num_chunks_before": 1,
        "hidden_act": "relu",
        "feed_forward_size": 4096,
        "hidden_dropout_prob": 0.2,
        "lsh_attention_probs_dropout_prob": 0.1,
        "local_attention_probs_dropout_prob": 0.2,
        "max_position_embeddings": 65536,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "axial_pos_embds": True,
        "axial_pos_shape": [128, 512],
        "axial_pos_embds_dim": [256, 768],
        "axial_norm_std": 1.0,
        "chunk_size_lm_head": 0,
        "attn_layers": [
            "local",
            "local",
            "lsh",
            "local",
            "local",
            "local",
            "lsh",
            "local",
            "local",
            "local",
            "lsh",
            "local",
        ],
    },
    "reformer-crime-and-punishment": {
        "tie_word_embeddings": False,
        "is_decoder": True,
        "chunk_size_feed_forward": 0,
        "pad_token_id": 0,
        "num_hidden_layers": 6,
        "hash_seed": None,
        "vocab_size": 320,
        "attention_head_size": 64,
        "hidden_size": 256,
        "num_attention_heads": 2,
        "num_hashes": 1,
        "num_buckets": [64, 128],
        "lsh_attn_chunk_length": 64,
        "local_attn_chunk_length": 64,
        "lsh_num_chunks_after": 0,
        "lsh_num_chunks_before": 1,
        "local_num_chunks_after": 0,
        "local_num_chunks_before": 1,
        "hidden_act": "relu",
        "feed_forward_size": 512,
        "hidden_dropout_prob": 0.05,
        "lsh_attention_probs_dropout_prob": 0.0,
        "local_attention_probs_dropout_prob": 0.05,
        "max_position_embeddings": 524288,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "axial_pos_embds": True,
        "axial_pos_shape": [512, 1024],
        "axial_pos_embds_dim": [64, 192],
        "axial_norm_std": 1.0,
        "chunk_size_lm_head": 0,
        "attn_layers": ["local", "lsh", "local", "lsh", "local", "lsh"],
    },
}

REFORMER_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "reformer-enwik8": "http://paddlenlp.bj.bcebos.com/models/transformers/reformer/reformer-enwik8/model_state.pdparams",
        "reformer-crime-and-punishment": "http://paddlenlp.bj.bcebos.com/models/transformers/reformer/reformer-crime-and-punishment/model_state.pdparams",
    }
}


class ReformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ReformerModel`]. It is used to instantiate a
    Reformer model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the ReFormer
    [google/reformer-crime-and-punishment](https://huggingface.co/google/reformer-crime-and-punishment) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        tie_word_embeddings (bool, optional):
            Whether to tie input and output embeddings. Defaults to `False`.
        is_decoder (bool, optional):
            Whether or not to use a causal mask in addition to the `attention_mask` passed to `ReformerModel`. When using the Reformer for causal language modeling, this argument should be set to `True`. Defaults to `True`.
        chunk_size_feed_forward (int, optional):
            The chunk size of all feed forward layers in the residual attention blocks. A chunk size of `0` means
            that the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes
            `n` < sequence_length embeddings at a time. Defaults to `0`.
        pad_token_id (int, optional):
            The id of the `padding` token. Defaults to `0`.
        hash_seed (int, optional):
            Seed that can be used to make local sensitive hashing in `LSHSelfAttention` deterministic. This should
            only be set for testing purposed. For evaluation and training purposes `hash_seed` should be left as
            `None` to ensure fully random rotations in local sensitive hashing scheme. Defaults to `None`.
        vocab_size (int, optional):
            Vocabulary size of `inputs_ids` in `ReformerModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `ReformerModel`. Defaults to `258`.
        attention_head_size (int, optional):
            Dimensionality of the projected key, query and value vectors. Defaults to `128`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer, encoder layer.Defaults to `1024`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to `8`.
        num_hashes (int, optional):
            Number of hashing rounds (e.g., number of random rotations) in Local Sensitive Hashing scheme. The higher `num_hashes`, the more accurate the `LSHSelfAttention` becomes, but also the more memory and time intensive the hashing becomes. Defaults to `4`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `12`.
        num_buckets (int or List[int], optional):
            Number of buckets, the key query vectors can be "hashed into" using the locality sensitive hashing scheme.
            Each query key vector is hashed into a hash in `1, ..., num_buckets`. The number of buckets can also be factorized into a list for improved memory complexity. In this case, each query key vector is hashed into a hash in `1-1, 1-2, ..., num_buckets[0]-1, ..., num_buckets[0]-num_buckets[1]` if `num_buckets` is factorized into two factors. The number of buckets (or the product the factors) should approximately equal sequence length / lsh_chunk_length. If `num_buckets` not set, a good value is calculated on the fly. Defaults to `512`.
        lsh_attn_chunk_length (int, optional):
            Length of chunk which attends to itself in `LSHSelfAttention`. Chunking reduces memory complexity from sequence length x sequence length (self attention) to chunk length x chunk length x sequence length / chunk length (chunked self attention).Defaults to `256`.
        local_attn_chunk_length (int, optional):
            Length of chunk which attends to itself in `LocalSelfAttention`. Chunking reduces memory complexity from sequence length x sequence length (self attention) to chunk length x chunk length x sequence length / chunk length (chunked self attention).Defaults to `128`.
        lsh_num_chunks_after (int, optional):
            Number of following neighbouring chunks to attend to in `LSHSelfAttention` layer to itself. Defaults to `0`.
        lsh_num_chunks_before (int, optional):
            Number of previous neighbouring chunks to attend to in `LSHSelfAttention` layer to itself. Defaults to `1`.
        local_num_chunks_after (int, optional):
            Number of following neighbouring chunks to attend to in `LocalSelfAttention` layer to itself. Defaults to `0`.
        local_num_chunks_before (int, optional):
            Number of previous neighbouring chunks to attend to in `LocalSelfAttention` layer to itself. Defaults to `1`.
        hidden_act (str, optional):
            The non-linear activation function (function or string) in the feed forward layer in the residual attention block. If string, `"gelu"`, `"relu"`, `"tanh"`, `"mish"` and `"gelu_new"` are supported. Defaults to `"relu"`.
        feed_forward_size (int, optional):
            Dimensionality of the feed_forward layer in the residual attention block. Defaults to `4096`.
        hidden_dropout_prob (float, optional):
            The dropout ratio for all fully connected layers in the embeddings and encoder. Defaults to `0.2`.
        lsh_attention_probs_dropout_prob (float, optional):
            The dropout ratio for the attention probabilities in `LSHSelfAttention`. Defaults to `0.1`.
        local_attention_probs_dropout_prob (float, optional):
            The dropout ratio for the attention probabilities in `LocalSelfAttention`. Defaults to `0.2`.
        max_position_embeddings (int, optional):
            The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048). Defaults to `65536`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer. Defaults to `0.02`.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`ReformerPretrainedModel._init_weights()` for how weights are initialized in `ReformerModel`.

        layer_norm_eps (float, optional):
            The epsilon used by the layer normalization layers. Defaults to `1e-12`.

        axial_pos_embds (bool, optional):
            Whether or not to use axial position embeddings. Defaults to `True`.
        axial_pos_shape (List[int], optional):
            The position dims of the axial position encodings. During training, the product of the position dims has to be equal to the sequence length. Defaults to `[128, 512]`.
        axial_pos_embds_dim (List[int], optional):
            The embedding dims of the axial position encodings. The sum of the embedding dims has to be equal to the
            hidden size. Defaults to `[256, 768]`.
        axial_norm_std (float, optional):
            The standard deviation of the normal_initializer for initializing the weight matrices of the axial
            positional encodings. Defaults to `1.0`.
        chunk_size_lm_head (int, optional):
            The chunk size of the final language model feed forward head layer. A chunk size of 0 means that the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes n <
            sequence_length embeddings at a time. Defaults to `0`.
        attn_layers (List[str], optional):
            List of attention layer types in ascending order. It can be chosen between a LSHSelfAttention layer
            (`"lsh"`) and a LocalSelfAttention layer (`"local"`). Defaults to `["local", "local", "lsh", "local", "local", "local", "lsh", "local", "local", "local", "lsh", "local"]`.

    """
    model_type = "reformer"
    attribute_map: Dict[str, str] = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "num_classes": "num_labels",
    }
    pretrained_init_configuration = REFORMER_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        axial_pos_shape=[128, 512],
        axial_pos_embds_dim=[256, 768],
        hidden_dropout_prob=0.2,
        attn_layers=[
            "local",
            "local",
            "lsh",
            "local",
            "local",
            "local",
            "lsh",
            "local",
            "local",
            "local",
            "lsh",
            "local",
        ],
        lsh_attn_chunk_length=256,
        local_attn_chunk_length=128,
        hidden_size=1024,
        max_position_embeddings=65536,
        axial_pos_embds=True,
        vocab_size=258,
        num_hashes=4,
        num_buckets=512,
        lsh_num_chunks_before=1,
        lsh_num_chunks_after=0,
        hash_seed=None,
        is_decoder=True,
        lsh_attention_probs_dropout_prob=0.1,
        num_attention_heads=8,
        attention_head_size=128,
        local_num_chunks_before=1,
        local_num_chunks_after=0,
        pad_token_id=0,
        local_attention_probs_dropout_prob=0.2,
        layer_norm_eps=1e-12,
        hidden_act="relu",
        feed_forward_size=4096,
        chunk_size_feed_forward=0,
        chunk_size_lm_head=0,
        tie_word_embeddings=False,
        initializer_range=0.02,
        axial_norm_std=1.0,
        use_cache=True,
        classifier_dropout=None,
        num_hidden_layers=12,
        **kwargs
    ):

        self.axial_pos_shape = tuple(axial_pos_shape)
        self.axial_pos_embds_dim = tuple(axial_pos_embds_dim)
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_layers = attn_layers
        self.lsh_attn_chunk_length = lsh_attn_chunk_length
        self.local_attn_chunk_length = local_attn_chunk_length
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.axial_pos_embds = axial_pos_embds
        self.vocab_size = vocab_size
        self.num_hashes = num_hashes
        self.num_buckets = tuple(num_buckets) if isinstance(num_buckets, list) else num_buckets
        self.lsh_num_chunks_before = lsh_num_chunks_before
        self.lsh_num_chunks_after = lsh_num_chunks_after
        self.hash_seed = hash_seed
        self.is_decoder = is_decoder
        self.lsh_attention_probs_dropout_prob = lsh_attention_probs_dropout_prob
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.local_num_chunks_before = local_num_chunks_before
        self.local_num_chunks_after = local_num_chunks_after
        self.pad_token_id = pad_token_id
        self.local_attention_probs_dropout_prob = local_attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.feed_forward_size = feed_forward_size
        self.chunk_size_lm_head = chunk_size_lm_head
        self.tie_word_embeddings = tie_word_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.initializer_range = initializer_range
        self.axial_norm_std = axial_norm_std
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        super().__init__(
            pad_token_id=pad_token_id,
            is_decoder=is_decoder,
            tie_word_embeddings=tie_word_embeddings,
            chunk_size_feed_forward=chunk_size_feed_forward,
            **kwargs,
        )
