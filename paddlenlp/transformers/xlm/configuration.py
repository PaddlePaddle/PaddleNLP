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
""" XLM configuration"""
from __future__ import annotations

# from .onnx import OnnxConfig
import logging
from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

__all__ = ["XLM_PRETRAINED_INIT_CONFIGURATION", "XLM_PRETRAINED_RESOURCE_FILES_MAP", "XLMConfig"]


XLM_PRETRAINED_INIT_CONFIGURATION = {
    "xlm-mlm-en-2048": {
        "is_encoder": True,
        "causal": False,
        "n_langs": 1,
        "use_lang_embeddings": True,
        "vocab_size": 30145,
        "pad_token_id": 2,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_hidden_layers": 12,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "use_sinusoidal_embeddings": False,
        "layer_norm_eps": 1e-12,
        "hidden_act": "gelu",
        "embed_init_std": 0.015625,
        "init_std": 0.02,
        "lang_id": 0,
        "lang2id": None,
    },
    "xlm-mlm-ende-1024": {
        "is_encoder": True,
        "causal": False,
        "n_langs": 2,
        "use_lang_embeddings": True,
        "vocab_size": 64699,
        "pad_token_id": 2,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_hidden_layers": 6,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "use_sinusoidal_embeddings": False,
        "layer_norm_eps": 1e-12,
        "hidden_act": "gelu",
        "embed_init_std": 0.02209708691207961,
        "init_std": 0.02,
        "lang_id": 1,
        "lang2id": {"de": 0, "en": 1},
    },
    "xlm-mlm-enfr-1024": {
        "is_encoder": True,
        "causal": False,
        "n_langs": 2,
        "use_lang_embeddings": True,
        "vocab_size": 64139,
        "pad_token_id": 2,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_hidden_layers": 6,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "use_sinusoidal_embeddings": False,
        "layer_norm_eps": 1e-12,
        "hidden_act": "gelu",
        "embed_init_std": 0.02209708691207961,
        "init_std": 0.02,
        "lang_id": 0,
        "lang2id": {"en": 0, "fr": 1},
    },
    "xlm-mlm-enro-1024": {
        "is_encoder": True,
        "causal": False,
        "n_langs": 2,
        "use_lang_embeddings": True,
        "vocab_size": 64592,
        "pad_token_id": 2,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_hidden_layers": 6,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "use_sinusoidal_embeddings": False,
        "layer_norm_eps": 1e-12,
        "hidden_act": "gelu",
        "embed_init_std": 0.02209708691207961,
        "init_std": 0.02,
        "lang_id": 0,
        "lang2id": {"en": 0, "ro": 1},
    },
    "xlm-mlm-tlm-xnli15-1024": {
        "is_encoder": True,
        "causal": False,
        "n_langs": 15,
        "use_lang_embeddings": True,
        "vocab_size": 95000,
        "pad_token_id": 2,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_hidden_layers": 12,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "use_sinusoidal_embeddings": False,
        "layer_norm_eps": 1e-12,
        "hidden_act": "gelu",
        "embed_init_std": 0.02209708691207961,
        "init_std": 0.02,
        "lang_id": 4,
        "lang2id": {
            "ar": 0,
            "bg": 1,
            "de": 2,
            "el": 3,
            "en": 4,
            "es": 5,
            "fr": 6,
            "hi": 7,
            "ru": 8,
            "sw": 9,
            "th": 10,
            "tr": 11,
            "ur": 12,
            "vi": 13,
            "zh": 14,
        },
    },
    "xlm-mlm-xnli15-1024": {
        "is_encoder": True,
        "causal": False,
        "n_langs": 15,
        "use_lang_embeddings": True,
        "vocab_size": 95000,
        "pad_token_id": 2,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_hidden_layers": 12,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "use_sinusoidal_embeddings": False,
        "layer_norm_eps": 1e-12,
        "hidden_act": "gelu",
        "embed_init_std": 0.02209708691207961,
        "init_std": 0.02,
        "lang_id": 4,
        "lang2id": {
            "ar": 0,
            "bg": 1,
            "de": 2,
            "el": 3,
            "en": 4,
            "es": 5,
            "fr": 6,
            "hi": 7,
            "ru": 8,
            "sw": 9,
            "th": 10,
            "tr": 11,
            "ur": 12,
            "vi": 13,
            "zh": 14,
        },
    },
    "xlm-clm-enfr-1024": {
        "is_encoder": True,
        "causal": False,
        "n_langs": 2,
        "use_lang_embeddings": True,
        "vocab_size": 64139,
        "pad_token_id": 2,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_hidden_layers": 6,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "use_sinusoidal_embeddings": False,
        "layer_norm_eps": 1e-12,
        "hidden_act": "gelu",
        "embed_init_std": 0.02209708691207961,
        "init_std": 0.02,
        "lang_id": 0,
        "lang2id": {"en": 0, "fr": 1},
    },
    "xlm-clm-ende-1024": {
        "is_encoder": True,
        "causal": False,
        "n_langs": 2,
        "use_lang_embeddings": True,
        "vocab_size": 64699,
        "pad_token_id": 2,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_hidden_layers": 6,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "use_sinusoidal_embeddings": False,
        "layer_norm_eps": 1e-12,
        "hidden_act": "gelu",
        "embed_init_std": 0.02209708691207961,
        "init_std": 0.02,
        "lang_id": 1,
        "lang2id": {"de": 0, "en": 1},
    },
    "xlm-mlm-17-1280": {
        "is_encoder": True,
        "causal": False,
        "n_langs": 17,
        "use_lang_embeddings": False,
        "vocab_size": 200000,
        "pad_token_id": 2,
        "hidden_size": 1280,
        "num_attention_heads": 16,
        "num_hidden_layers": 16,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "use_sinusoidal_embeddings": False,
        "layer_norm_eps": 1e-12,
        "hidden_act": "gelu",
        "embed_init_std": 0.01976423537605237,
        "init_std": 0.02,
        "lang_id": 2,
        "lang2id": {
            "ar": 0,
            "de": 1,
            "en": 2,
            "es": 3,
            "fr": 4,
            "hi": 5,
            "it": 6,
            "ja": 7,
            "ko": 8,
            "nl": 9,
            "pl": 10,
            "pt": 11,
            "ru": 12,
            "sv": 13,
            "tr": 14,
            "vi": 15,
            "zh": 16,
        },
    },
    "xlm-mlm-100-1280": {
        "is_encoder": True,
        "causal": False,
        "n_langs": 100,
        "use_lang_embeddings": False,
        "vocab_size": 200000,
        "pad_token_id": 2,
        "hidden_size": 1280,
        "num_attention_heads": 16,
        "num_hidden_layers": 16,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "use_sinusoidal_embeddings": False,
        "layer_norm_eps": 1e-12,
        "hidden_act": "gelu",
        "embed_init_std": 0.01976423537605237,
        "init_std": 0.02,
        "lang_id": 23,
        "lang2id": {
            "af": 0,
            "als": 1,
            "am": 2,
            "an": 3,
            "ang": 4,
            "ar": 5,
            "arz": 6,
            "ast": 7,
            "az": 8,
            "bar": 9,
            "be": 10,
            "bg": 11,
            "bn": 12,
            "br": 13,
            "bs": 14,
            "ca": 15,
            "ceb": 16,
            "ckb": 17,
            "cs": 18,
            "cy": 19,
            "da": 20,
            "de": 21,
            "el": 22,
            "en": 23,
            "eo": 24,
            "es": 25,
            "et": 26,
            "eu": 27,
            "fa": 28,
            "fi": 29,
            "fr": 30,
            "fy": 31,
            "ga": 32,
            "gan": 33,
            "gl": 34,
            "gu": 35,
            "he": 36,
            "hi": 37,
            "hr": 38,
            "hu": 39,
            "hy": 40,
            "ia": 41,
            "id": 42,
            "is": 43,
            "it": 44,
            "ja": 45,
            "jv": 46,
            "ka": 47,
            "kk": 48,
            "kn": 49,
            "ko": 50,
            "ku": 51,
            "la": 52,
            "lb": 53,
            "lt": 54,
            "lv": 55,
            "mk": 56,
            "ml": 57,
            "mn": 58,
            "mr": 59,
            "ms": 60,
            "my": 61,
            "nds": 62,
            "ne": 63,
            "nl": 64,
            "nn": 65,
            "no": 66,
            "oc": 67,
            "pl": 68,
            "pt": 69,
            "ro": 70,
            "ru": 71,
            "scn": 72,
            "sco": 73,
            "sh": 74,
            "si": 75,
            "simple": 76,
            "sk": 77,
            "sl": 78,
            "sq": 79,
            "sr": 80,
            "sv": 81,
            "sw": 82,
            "ta": 83,
            "te": 84,
            "th": 85,
            "tl": 86,
            "tr": 87,
            "tt": 88,
            "uk": 89,
            "ur": 90,
            "uz": 91,
            "vi": 92,
            "war": 93,
            "wuu": 94,
            "yi": 95,
            "zh": 96,
            "zh_classical": 97,
            "zh_min_nan": 98,
            "zh_yue": 99,
        },
    },
}

XLM_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "xlm-mlm-en-2048": "https://bj.bcebos.com/paddlenlp/models/transformers/xlm/xlm-mlm-en-2048/model_state.pdparams",
        "xlm-mlm-ende-1024": "https://bj.bcebos.com/paddlenlp/models/transformers/xlm/xlm-mlm-ende-1024/model_state.pdparams",
        "xlm-mlm-enfr-1024": "https://bj.bcebos.com/paddlenlp/models/transformers/xlm/xlm-mlm-enfr-1024/model_state.pdparams",
        "xlm-mlm-enro-1024": "https://bj.bcebos.com/paddlenlp/models/transformers/xlm/xlm-mlm-enro-1024/model_state.pdparams",
        "xlm-mlm-tlm-xnli15-1024": "https://bj.bcebos.com/paddlenlp/models/transformers/xlm/xlm-mlm-tlm-xnli15-1024/model_state.pdparams",
        "xlm-mlm-xnli15-1024": "https://bj.bcebos.com/paddlenlp/models/transformers/xlm/xlm-mlm-xnli15-1024/model_state.pdparams",
        "xlm-clm-enfr-1024": "https://bj.bcebos.com/paddlenlp/models/transformers/xlm/xlm-clm-enfr-1024/model_state.pdparams",
        "xlm-clm-ende-1024": "https://bj.bcebos.com/paddlenlp/models/transformers/xlm/xlm-clm-ende-1024/model_state.pdparams",
        "xlm-mlm-17-1280": "https://bj.bcebos.com/paddlenlp/models/transformers/xlm/xlm-mlm-17-1280/model_state.pdparams",
        "xlm-mlm-100-1280": "https://bj.bcebos.com/paddlenlp/models/transformers/xlm/xlm-mlm-100-1280/model_state.pdparams",
    }
}


class XLMConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`XLMModel`]. It is used to
    instantiate a XLM model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [xlm-mlm-en-2048] architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30145):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`XLMModel`] .
        emb_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the encoder layers and the pooler layer.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention mechanism
        gelu_activation (`bool`, *optional*, defaults to `True`):
            Whether or not to use *gelu* for the activations instead of *relu*.
        sinusoidal_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to use sinusoidal positional embeddings instead of absolute positional embeddings.
        causal (`bool`, *optional*, defaults to `False`):
            Whether or not the model should behave in a causal manner. Causal models use a triangular attention mask in
            order to only attend to the left-side context instead if a bidirectional context.
        asm (`bool`, *optional*, defaults to `False`):
            Whether or not to use an adaptive log softmax projection layer instead of a linear layer for the prediction
            layer.
        n_langs (`int`, *optional*, defaults to 1):
            The number of languages the model handles. Set to 1 for monolingual models.
        use_lang_emb (`bool`, *optional*, defaults to `True`)
            Whether to use language embeddings. Some models use additional language embeddings, see [the multilingual
            models page]
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        embed_init_std (`float`, *optional*, defaults to 2048^-0.5):
            The standard deviation of the truncated_normal_initializer for initializing the embedding matrices.
        init_std (`int`, *optional*, defaults to 50257):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices except the
            embedding matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        bos_index (`int`, *optional*, defaults to 0):
            The index of the beginning of sentence token in the vocabulary.
        eos_index (`int`, *optional*, defaults to 1):
            The index of the end of sentence token in the vocabulary.
        pad_index (`int`, *optional*, defaults to 2):
            The index of the padding token in the vocabulary.
        unk_index (`int`, *optional*, defaults to 3):
            The index of the unknown token in the vocabulary.
        mask_index (`int`, *optional*, defaults to 5):
            The index of the masking token in the vocabulary.
        is_encoder(`bool`, *optional*, defaults to `True`):
            Whether or not the initialized model should be a transformer encoder or decoder as seen in Vaswani et al.
        summary_type (`string`, *optional*, defaults to "first"):
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
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Used in the sequence classification and multiple choice models.

            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            Used in the sequence classification and multiple choice models.

            The dropout ratio to be used after the projection and activation.
        start_n_top (`int`, *optional*, defaults to 5):
            Used in the SQuAD evaluation script.
        end_n_top (`int`, *optional*, defaults to 5):
            Used in the SQuAD evaluation script.
        mask_token_id (`int`, *optional*, defaults to 0):
            Model agnostic parameter to identify masked tokens when generating text in an MLM context.
        lang_id (`int`, *optional*, defaults to 1):
            The ID of the language used by the model. This parameter is used when generating text in a given language.

    Examples:

    ```python
    >>> from transformers import XLMConfig, XLMModel

    >>> # Initializing a XLM configuration
    >>> configuration = XLMConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = XLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xlm"
    pretrained_init_configuration = XLM_PRETRAINED_INIT_CONFIGURATION
    attribute_map: Dict[str, str] = {
        "dropout_prob": "dropout",
        "attention_probs_dropout_prob": "attention_dropout",
        "hidden_size": "emb_dim",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
        "n_words": "vocab_size",  # For backward compatibility
        "use_lang_embeddings": "use_lang_emb",
        "use_sinusoidal_embeddings": "sinusoidal_embeddings",
        "hidden_dropout_prob": "dropout",
        "num_classes": "num_labels",
    }

    def __init__(
        self,
        vocab_size=30145,
        emb_dim=2048,
        n_layers=12,
        n_heads=16,
        dropout=0.1,
        attention_dropout=0.1,
        gelu_activation=True,
        hidden_act="gelu",
        sinusoidal_embeddings=False,
        causal=False,
        asm=False,
        n_langs=1,
        use_lang_emb=True,
        max_position_embeddings=512,
        embed_init_std=2048**-0.5,
        layer_norm_eps=1e-12,
        init_std=0.02,
        bos_index=0,
        eos_index=1,
        pad_index=2,
        unk_index=3,
        mask_index=5,
        is_encoder=True,
        summary_type="first",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        start_n_top=5,
        end_n_top=5,
        mask_token_id=0,
        lang_id=0,
        pad_token_id=2,
        bos_token_id=0,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, **kwargs)

        """Constructs XLMConfig."""
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.gelu_activation = gelu_activation
        self.hidden_act = hidden_act
        self.sinusoidal_embeddings = sinusoidal_embeddings
        self.causal = causal
        self.asm = asm
        self.n_langs = n_langs
        self.use_lang_emb = use_lang_emb
        self.layer_norm_eps = layer_norm_eps
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.mask_index = mask_index
        self.is_encoder = is_encoder
        self.max_position_embeddings = max_position_embeddings
        self.embed_init_std = embed_init_std
        self.init_std = init_std
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_proj_to_labels = summary_proj_to_labels
        self.summary_first_dropout = summary_first_dropout
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top
        self.mask_token_id = mask_token_id
        self.lang_id = lang_id

        if "n_words" in kwargs:
            self.n_words = kwargs["n_words"]
