# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

""" ChatGLM model configuration """

from ..configuration_utils import PretrainedConfig

__all__ = [
    "ChatGLMConfig",
    "CHATGLM_PRETRAINED_RESOURCE_FILES_MAP",
]


CHATGLM_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "THUDM/chatglm-6b": "https://paddlenlp.bj.bcebos.com/models/community/THUDM/chatglm-6b/model_state.pdparams",
        "THUDM/chatglm-6b-v1.1": "https://paddlenlp.bj.bcebos.com/models/community/THUDM/chatglm-6b-v1.1/model_state.pdparams",
    }
}


class ChatGLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~ChatGLMModel`].
    It is used to instantiate an ChatGLM model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the ChatGLM-6B [THUDM/ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 150528):
            Vocabulary size of the ChatGLM-6B model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~ChatGLMModel`] or
            [`~TFChatGLMModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        inner_hidden_size (`int`, *optional*, defaults to 16384):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        max_sequence_length (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        layernorm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models).
        Example:

    ```python
    >>> from configuration import ChatGLMConfig
    >>> from modeling import ChatGLMModel

    >>> # Initializing a ChatGLM-6B THUDM/ChatGLM-6B style configuration
    >>> configuration = ChatGLMConfig()

    >>> # Initializing a model from the THUDM/ChatGLM-6B style configuration
    >>> model = ChatGLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "chatglm"
    attribute_map = {"num_layers": "num_hidden_layers"}

    def __init__(
        self,
        vocab_size=130528,
        hidden_size=4096,
        num_hidden_layers=28,
        num_attention_heads=32,
        layernorm_epsilon=1e-5,
        use_cache=False,
        bos_token_id=130004,
        eos_token_id=130005,
        pad_token_id=3,
        mask_token_id=130000,
        gmask_token_id=130001,
        max_sequence_length=2048,
        inner_hidden_size=16384,
        position_encoding_2d=True,
        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        output_predict=True,
        attention_scale=True,
        activation="gelu",
        num_image_tokens=0,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.layernorm_epsilon = layernorm_epsilon
        self.inner_hidden_size = inner_hidden_size
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.gmask_token_id = gmask_token_id
        self.position_encoding_2d = position_encoding_2d
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.output_predict = output_predict
        self.attention_scale = attention_scale
        self.activation = activation
        self.num_image_tokens = num_image_tokens
