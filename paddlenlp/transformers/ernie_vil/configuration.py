# coding=utf-8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" ErnieViL model configuration"""

import copy
import os
from typing import Optional, Union

from ...utils.log import logger
from ..configuration_utils import PretrainedConfig

__all__ = [
    "ErnieViLTextConfig",
    "ErnieViLVisionConfig",
    "ErnieViLConfig",
]


class ErnieViLTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ErnieViLTextModel`]. It is used to
    instantiate a ERNIE model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ERNIE
    ernie-3.0-medium-zh architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 40000):
            Vocabulary size of the ERNIE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ErnieModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 0):
            The vocabulary size of the `token_type_ids` passed when calling [`ErnieModel`].
        task_type_vocab_size (`int`, *optional*, defaults to 3):
            The vocabulary size of the `task_ids`.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        task_id (`int`, *optional*, defaults to 0):
            Task id.
        use_task_id (`bool`, *optional*, defaults to `False`):
            Whether or not use task_id.
        pad_token_id (`int`, *optional*, defaults to 0):
            The index of padding token in the token vocabulary.

    Examples:

    ```python
    >>> from paddlenlp.transformers import ErnieViLTextConfig, ErnieViLTextModel

    >>> configuration = ErnieViLTextConfig()

    >>> model = ErnieViLTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    model_type = "ernie_vil_text_model"

    def __init__(
        self,
        vocab_size: int = 40000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 2048,
        task_type_vocab_size: int = 3,
        type_vocab_size: int = 0,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        layer_norm_eps=1e-5,
        task_id: int = 0,
        use_task_id: bool = False,
        fuse: bool = False,
        use_cache: bool = False,
        **kwargs
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.task_id = task_id
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.task_type_vocab_size = task_type_vocab_size
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.fuse = fuse
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.use_task_id = use_task_id

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        from_hf_hub: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs
    ) -> PretrainedConfig:
        kwargs.update({"from_hf_hub": from_hf_hub, "cache_dir": cache_dir})
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the text config dict if we are loading from ErnieViLConfig
        if config_dict.get("model_type") == "ernie_vil":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )
        return cls.from_dict(config_dict, **kwargs)


class ErnieViLVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ErnieViLVisionModel`]. It is used to instantiate an ErnieViL
    model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*,
            defaults to 1e-6): The epsilon used by the layer normalization layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float``, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from paddlenlp.transformers import ErnieViLVisionConfig, ErnieViLVisionModel

    >>> configuration = ErnieViLVisionConfig()

    >>> model = ErnieViLVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "ernie_vil_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        hidden_act="quick_gelu",
        layer_norm_eps=0.000001,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        from_hf_hub: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs
    ) -> PretrainedConfig:
        kwargs.update({"from_hf_hub": from_hf_hub, "cache_dir": cache_dir})
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from ErnieViLConfig
        if config_dict.get("model_type") == "ernie_vil":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ErnieViLConfig(PretrainedConfig):
    r"""
    [`ErnieViLConfig`] is the configuration class to store the configuration of a [`ErnieViLModel`]. It is used to instantiate
    ErnieViL model according to the specified arguments, defining the text model and vision model configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ErnieViLTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ErnieViLVisionConfig`].
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original ErnieViL implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from paddlenlp.transformers import ErnieViLConfig, ErnieViLModel

    >>> configuration = ErnieViLConfig()

    >>> model = ErnieViLModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # Initializing a ErnieViLText and ErnieViLVision configuration
    >>> config_text = ErnieViLTextConfig()
    >>> config_vision = ErnieViLVisionConfig()

    >>> config = ErnieViLConfig.from_text_vision_configs(config_text, config_vision)
    ```
    """

    model_type = "ernie_vil"
    is_composition = True

    def __init__(self, text_config=None, vision_config=None, logit_scale_init_value=2.6592, **kwargs):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

        # If `_config_dict` exist, we use them for the backward compatibility.
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)
        if text_config_dict is not None:
            text_config = text_config_dict
        if vision_config_dict is not None:
            vision_config = vision_config_dict

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the ErnieViLTextConfig with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the ErnieViLVisionConfig with default values.")

        self.text_config = ErnieViLTextConfig(**text_config)
        self.vision_config = ErnieViLVisionConfig(**vision_config)

        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: ErnieViLTextConfig, vision_config: ErnieViLVisionConfig, **kwargs):
        r"""
        Instantiate a [`ErnieViLConfig`] (or a derived class) from ernie_vil text model configuration and ernie_vil vision model
        configuration.

        Returns:
            [`ErnieViLConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["text_config"] = self.text_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
