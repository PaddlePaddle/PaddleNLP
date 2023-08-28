# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import copy
import os
from typing import Optional, Union

from ...utils.log import logger
from ..configuration_utils import PretrainedConfig

__all__ = [
    "ClapTextConfig",
    "ClapAudioConfig",
    "ClapConfig",
]


class ClapTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ClapTextModel`]. It is used to instantiate a CLAP
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the CLAP
    [calp-hsat-fused](https://huggingface.co/laion/clap-hsat-fused) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the CLAP model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ClapTextModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"relu"`,
            `"relu"`, `"silu"` and `"relu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`ClapTextModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        projection_hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the projection layer. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        projection_dim (`int`, *optional*, defaults to 512)
            Dimension of the projection head of the `ClapTextModelWithProjection`.

    Examples:

    ```python
    >>> from paddlenlp.transformers import ClapTextConfig, ClapTextModel

    >>> # Initializing a CLAP text configuration
    >>> configuration = ClapTextConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ClapTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "clap_text_model"

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=1,
        initializer_range=0.02,
        initializer_factor=1.0,
        layer_norm_eps=1e-12,
        projection_dim=512,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        projection_hidden_act="relu",
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.projection_hidden_act = projection_hidden_act
        self.projection_dim = projection_dim

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        from_hf_hub: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs
    ) -> "PretrainedConfig":
        kwargs.update({"from_hf_hub": from_hf_hub, "cache_dir": cache_dir})
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the text config dict if we are loading from ClapConfig
        if config_dict.get("model_type") == "clap":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ClapAudioConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ClapAudioModel`]. It is used to instantiate a
    CLAP audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the CLAP
    [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        window_size (`int`, *optional*, defaults to 8):
            Image size of the spectrogram
        num_mel_bins (`int`, *optional*, defaults to 64):
            Number of mel features used per frames. Should correspond to the value used in the `ClapProcessor` class.
        spec_size (`int`, *optional*, defaults to 256):
            Desired input size of the spectrogram that the model supports. It can be different from the output of the
            `ClapFeatureExtractor`, in which case the input features will be resized. Corresponds to the `image_size`
            of the audio models.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        patch_size (`int`, *optional*, defaults to 4):
            Patch size for the audio spectrogram
        patch_stride (`list`, *optional*, defaults to `[4, 4]`):
            Patch stride for the audio spectrogram
        num_classes (`int`, *optional*, defaults to 527):
            Number of classes used for the head training
        hidden_size (`int`, *optional*, defaults to 768):
            Hidden size of the output of the audio encoder. Correspond to the dimension of the penultimate layer's
            output,which is sent to the projection MLP layer.
        projection_dim (`int`, *optional*, defaults to 512):
            Hidden size of the projection layer.
        depths (`list`, *optional*, defaults to `[2, 2, 6, 2]`):
            Depths used for the Swin Layers of the audio model
        num_attention_heads (`list`, *optional*, defaults to `[4, 8, 16, 32]`):
            Number of attention heads used for the Swin Layers of the audio model
        enable_fusion (`bool`, *optional*, defaults to `False`):
            Whether or not to enable patch fusion. This is the main contribution of the authors, and should give the
            best results.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the encoder.
        fusion_type (`[type]`, *optional*):
            Fusion type used for the patch fusion.
        patch_embed_input_channels (`int`, *optional*, defaults to 1):
            Number of channels used for the input spectrogram
        flatten_patch_embeds (`bool`, *optional*, defaults to `True`):
            Whether or not to flatten the patch embeddings
        patch_embeds_hidden_size (`int`, *optional*, defaults to 96):
            Hidden size of the patch embeddings. It is used as the number of output channels.
        enable_patch_layer_norm (`bool`, *optional*, defaults to `True`):
            Whether or not to enable layer normalization for the patch embeddings
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Drop path rate for the patch fusion
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to add a bias to the query, key, value projections.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of the mlp hidden dim to embedding dim.
        aff_block_r (`int`, *optional*, defaults to 4):
            downsize_ratio used in the AudioFF block
        num_hidden_layers (`int`, *optional*, defaults to 4):
            Number of hidden layers in the Transformer encoder.
        projection_hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the projection layer. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        layer_norm_eps (`[type]`, *optional*, defaults to `1e-5`):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from paddlenlp.transformers import ClapAudioConfig, ClapAudioModel

    >>> # Initializing a ClapAudioConfig with laion/clap-htsat-fused style configuration
    >>> configuration = ClapAudioConfig()

    >>> # Initializing a ClapAudioModel (with random weights) from the laion/clap-htsat-fused style configuration
    >>> model = ClapAudioModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clap_audio_model"

    def __init__(
        self,
        window_size=8,
        num_mel_bins=64,
        spec_size=256,
        hidden_act="gelu",
        patch_size=4,
        patch_stride=[4, 4],
        num_classes=527,
        hidden_size=768,
        projection_dim=512,
        depths=[2, 2, 6, 2],
        num_attention_heads=[4, 8, 16, 32],
        enable_fusion=False,
        hidden_dropout_prob=0.1,
        fusion_type=None,
        patch_embed_input_channels=1,
        flatten_patch_embeds=True,
        patch_embeds_hidden_size=96,
        enable_patch_layer_norm=True,
        drop_path_rate=0.0,
        attention_probs_dropout_prob=0.0,
        qkv_bias=True,
        mlp_ratio=4.0,
        aff_block_r=4,
        num_hidden_layers=4,
        projection_hidden_act="relu",
        layer_norm_eps=1e-5,
        initializer_factor=1.0,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)
        self.window_size = window_size
        self.num_mel_bins = num_mel_bins
        self.spec_size = spec_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.depths = depths
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.window_size = window_size
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.projection_dim = projection_dim
        self.flatten_patch_embeds = flatten_patch_embeds
        self.patch_embeds_hidden_size = patch_embeds_hidden_size
        self.enable_patch_layer_norm = enable_patch_layer_norm
        self.drop_path_rate = drop_path_rate
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.qkv_bias = qkv_bias
        self.mlp_ratio = mlp_ratio
        self.patch_embed_input_channels = patch_embed_input_channels
        self.aff_block_r = aff_block_r
        self.layer_norm_eps = layer_norm_eps
        self.initializer_factor = initializer_factor
        self.projection_hidden_act = projection_hidden_act

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        from_hf_hub: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs
    ) -> "PretrainedConfig":
        kwargs.update({"from_hf_hub": from_hf_hub, "cache_dir": cache_dir})
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the audio config dict if we are loading from ClapConfig
        if config_dict.get("model_type") == "clap":
            config_dict = config_dict["audio_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ClapConfig(PretrainedConfig):
    r"""
    [`ClapConfig`] is the configuration class to store the configuration of a [`ClapModel`]. It is used to instantiate
    a CLAP model according to the specified arguments, defining the text model and audio model configs. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the CLAP
    [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClapTextConfig`].
        audio_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClapAudioConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and audio projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLAP implementation.
        projection_hidden_act (`str`, *optional*, defaults to `"relu"`):
            Activation function for the projection layers.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            Factor to scale the initialization of the model weights.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from paddlenlp.transformers import ClapConfig, ClapModel

    >>> # Initializing a ClapConfig with laion-ai/base style configuration
    >>> configuration = ClapConfig()

    >>> # Initializing a ClapModel (with random weights) from the laion-ai/base style configuration
    >>> model = ClapModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a ClapConfig from a ClapTextConfig and a ClapAudioConfig
    >>> from paddlenlp.transformers import ClapTextConfig, ClapAudioConfig

    >>> # Initializing a ClapText and ClapAudioConfig configuration
    >>> config_text = ClapTextConfig()
    >>> config_audio = ClapAudioConfig()

    >>> config = ClapConfig.from_text_audio_configs(config_text, config_audio)
    ```"""

    model_type = "clap"
    is_composition = True

    def __init__(
        self,
        text_config=None,
        audio_config=None,
        logit_scale_init_value=(1 / 0.07),
        projection_dim=512,
        projection_hidden_act="relu",
        initializer_factor=1.0,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the ClapTextConfig with default values.")

        if audio_config is None:
            audio_config = {}
            logger.info("audio_config is None. initializing the ClapAudioConfig with default values.")

        self.text_config = ClapTextConfig(**text_config)
        self.audio_config = ClapAudioConfig(**audio_config)
        self.text_config.projection_dim = projection_dim
        self.audio_config.projection_dim = projection_dim

        self.text_config.projection_hidden_act = projection_hidden_act
        self.audio_config.projection_hidden_act = projection_hidden_act

        self.projection_dim = projection_dim
        self.projection_hidden_act = projection_hidden_act
        self.hidden_size = self.text_config.hidden_size

        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = initializer_factor
        self.num_hidden_layers = self.text_config.num_hidden_layers + len(self.audio_config.depths)

    @classmethod
    def from_text_audio_configs(cls, text_config: ClapTextConfig, audio_config: ClapAudioConfig, **kwargs):
        r"""
        Instantiate a [`ClapConfig`] (or a derived class) from clap text model configuration and clap audio model
        configuration.

        Returns:
            [`ClapConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), audio_config=audio_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["text_config"] = self.text_config.to_dict()
        output["audio_config"] = self.audio_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
