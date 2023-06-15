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
""" CODEGEN model configuration"""
from __future__ import annotations

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["CODEGEN_PRETRAINED_INIT_CONFIGURATION", "CodeGenConfig", "CODEGEN_PRETRAINED_RESOURCE_FILES_MAP"]

CODEGEN_PRETRAINED_INIT_CONFIGURATION = {}
CODEGEN_PRETRAINED_RESOURCE_FILES_MAP = {"model_state": {}}


class CodeGenConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CodeGenModel`]. It is used to instantiate a
    CodeGen model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CodeGen
    Salesforce/codegen-350M-mono architecture. Configuration objects
    inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the documentation from
    [`PretrainedConfig`] for more information.


    Args:
        vocab_size (int, optional):
            Vocabulary size of `inputs_ids` in `CodeGenModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `CodeGenModel`.
            Defaulta to `50400`.
        n_embed (int, optional):
            Dimensionality of the embedding layer, decoder layer. Defaults to `4096`.
        n_layer (int, optional):
            Number of hidden layers. Defaults to `28`.
        n_head (int, optional):
            Number of attention heads for each attention layer in the Transformer decoder.
            Defaults to `16`.
        n_ctx (int, optional):
            Dimensionality of the causal mask (usually same as n_positions).
            Defaults to `2048`.
        n_positions (int, optional):
            The maximum sequence length that this model might ever be used with.
            Defaults to `2048`.
        attn_pdrop (float, optional):
            The dropout probability used in MultiHeadAttention in all decoder layers to drop some attention target.
            Defaults to `0.0`.
        resid_pdrop (float, optional):
            The dropout probability for all residual layers in the decoder.
            Defaults to `0.0`.
        embd_pdrop (float, optional):
            The dropout probability used in embedding layers. Defaults to `0.0`.
        rotary_dim (int, optional):
            Dimensionality of rotay position embeddings.
            Defaults to `64`.
        activation_function (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions are supported.
            Defaults to `"gelu_new"`.
        layer_norm_epsilon (float, optional):
            The epsilon to use in the layer normalization layers.
            Defaults to `1e-05`.
        initializer_range (float, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Default to `0.02`.
    ```"""
    model_type = "codegen"
    pretrained_init_configuration = CODEGEN_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 50400,
        bos_token_id: int = 1,
        eos_token_id: int = 50256,
        pad_token_id: int = 50256,
        n_embd: int = 4096,
        n_layer: int = 28,
        n_head: int = 16,
        n_ctx: int = 2048,
        n_positions: int = 2048,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        embd_pdrop: float = 0.0,
        rotary_dim: int = 64,
        activation_function: str = "gelu_new",
        layer_norm_epsilon: float = 1e-05,
        initializer_range: float = 0.02,
        n_inner: int = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.rotary_dim = rotary_dim
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
