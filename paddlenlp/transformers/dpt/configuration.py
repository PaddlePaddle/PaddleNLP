# coding=utf-8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" DPT model configuration"""

import copy

from ...utils.log import logger
from ..bit.configuration import BitConfig
from ..configuration_utils import PretrainedConfig

__all__ = ["DPTConfig"]


class DPTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DPTModel`]. It is used to instantiate an DPT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DPT
    [Intel/dpt-large](https://huggingface.co/Intel/dpt-large) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 384):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        backbone_out_indices (`List[int]`, *optional*, defaults to `[2, 5, 8, 11]`):
            Indices of the intermediate hidden states to use from backbone.
        readout_type (`str`, *optional*, defaults to `"project"`):
            The readout type to use when processing the readout token (CLS token) of the intermediate hidden states of
            the ViT backbone. Can be one of [`"ignore"`, `"add"`, `"project"`].
            - "ignore" simply ignores the CLS token.
            - "add" passes the information from the CLS token to all other tokens by adding the representations.
            - "project" passes information to the other tokens by concatenating the readout to all other tokens before
              projecting the
            representation to the original feature dimension D using a linear layer followed by a GELU non-linearity.
        is_hybrid (`bool`, *optional*, defaults to `False`):
            Whether to use a hybrid backbone. Useful in the context of loading DPT-Hybrid models.
        reassemble_factors (`List[int]`, *optional*, defaults to `[4, 2, 1, 0.5]`):
            The up/downsampling factors of the reassemble layers.
        neck_hidden_sizes (`List[str]`, *optional*, defaults to [96, 192, 384, 768]):
            The hidden sizes to project to for the feature maps of the backbone.
        fusion_hidden_size (`int`, *optional*, defaults to 256):
            The number of channels before fusion.
        head_in_index (`int`, *optional*, defaults to -1):
            The index of the features to use in the heads.
        use_batch_norm_in_fusion_residual (`bool`, *optional*, defaults to `False`):
            Whether to use batch normalization in the pre-activate residual units of the fusion blocks.
        use_auxiliary_head (`bool`, *optional*, defaults to `True`):
            Whether to use an auxiliary head during training.
        auxiliary_loss_weight (`float`, *optional*, defaults to 0.4):
            Weight of the cross-entropy loss of the auxiliary head.
        semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
            The index that is ignored by the loss function of the semantic segmentation model.
        semantic_classifier_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the semantic classification head.
        backbone_featmap_shape (`List[int]`, *optional*, defaults to `[1, 1024, 24, 24]`):
            Used only for the `hybrid` embedding type. The shape of the feature maps of the backbone.
        neck_ignore_stages (`List[int]`, *optional*, defaults to `[0, 1]`):
            Used only for the `hybrid` embedding type. The stages of the readout layers to ignore.
        backbone_config (`Union[Dict[str, Any], PretrainedConfig]`, *optional*):
            Used only for the `hybrid` embedding type. The configuration of the backbone in a dictionary.

    Examples:

    ```python
    >>> from paddlenlp.transformers import DPTModel, DPTConfig

    >>> # Initializing a DPT dpt-large style configuration
    >>> configuration = DPTConfig()

    >>> # Initializing a model from the dpt-large style configuration
    >>> model = DPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    model_type = "dpt"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=384,
        patch_size=16,
        num_channels=3,
        is_hybrid=False,
        qkv_bias=True,
        backbone_out_indices=[2, 5, 8, 11],
        readout_type="project",
        reassemble_factors=[4, 2, 1, 0.5],
        neck_hidden_sizes=[96, 192, 384, 768],
        fusion_hidden_size=256,
        head_in_index=-1,
        use_batch_norm_in_fusion_residual=False,
        use_auxiliary_head=True,
        auxiliary_loss_weight=0.4,
        semantic_loss_ignore_index=255,
        semantic_classifier_dropout=0.1,
        backbone_featmap_shape=[1, 1024, 24, 24],
        neck_ignore_stages=[0, 1],
        backbone_config=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.is_hybrid = is_hybrid

        if self.is_hybrid:
            if backbone_config is None:
                logger.info("Initializing the config with a `BiT` backbone.")
                backbone_config = {
                    "global_padding": "same",
                    "layer_type": "bottleneck",
                    "depths": [3, 4, 9],
                    "out_features": ["stage1", "stage2", "stage3"],
                    "embedding_dynamic_padding": True,
                }
                self.backbone_config = BitConfig(**backbone_config)
            elif isinstance(backbone_config, dict):
                logger.info("Initializing the config with a `BiT` backbone.")
                self.backbone_config = BitConfig(**backbone_config)
            elif isinstance(backbone_config, PretrainedConfig):
                self.backbone_config = backbone_config
            else:
                raise ValueError(
                    f"backbone_config must be a dictionary or a `PretrainedConfig`, got {backbone_config.__class__}."
                )

            self.backbone_featmap_shape = backbone_featmap_shape
            self.neck_ignore_stages = neck_ignore_stages

            if readout_type != "project":
                raise ValueError("Readout type must be 'project' when using `DPT-hybrid` mode.")
        else:
            self.backbone_config = None
            self.backbone_featmap_shape = None
            self.neck_ignore_stages = []

        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.backbone_out_indices = backbone_out_indices
        if readout_type not in ["ignore", "add", "project"]:
            raise ValueError("Readout_type must be one of ['ignore', 'add', 'project']")
        self.readout_type = readout_type
        self.reassemble_factors = reassemble_factors
        self.neck_hidden_sizes = neck_hidden_sizes
        self.fusion_hidden_size = fusion_hidden_size
        self.head_in_index = head_in_index
        self.use_batch_norm_in_fusion_residual = use_batch_norm_in_fusion_residual
        # auxiliary head attributes (semantic segmentation)
        self.use_auxiliary_head = use_auxiliary_head
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
        self.semantic_classifier_dropout = semantic_classifier_dropout

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`]. Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)

        if output["backbone_config"] is not None:
            output["backbone_config"] = self.backbone_config.to_dict()

        output["model_type"] = self.__class__.model_type
        return output
