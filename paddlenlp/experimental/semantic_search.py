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
from __future__ import annotations

import inspect
from typing import Dict, List

import paddle
from paddle import nn

from paddlenlp.transformers import PretrainedConfig, PretrainedModel
from paddlenlp.transformers.auto.modeling import get_name_mapping
from paddlenlp.utils.import_utils import import_module


def _load_text_image_model_names() -> List[str]:
    """load image model names

    Returns:
        List[str]: the names of text & image model
    """
    return ["CLIPModel", "CLIPChineseModel", "ErnieViLModel"]


def _load_text_model_names() -> List[str]:
    """load text model names

    Returns:
        List[str]: the names of text model
    """
    model = get_name_mapping("Model")
    return [
        value for key, value in model.items() if key.endswith("Model_Import_Class")
    ] + _load_text_image_model_names()


def _load_image_model_names() -> List[str]:
    """load image model names

    Returns:
        List[str]: the names of image model
    """
    return [
        "ErnieViLVisionModel",
        "CLIPViLVisionModel",
        "ChineseCLIPViLVisionModel",
    ] + _load_text_image_model_names()


class AutoConfig:
    """TODO: to construct AutoConfig to resolve `model-clas`<architectures>"""

    @staticmethod
    def from_pretrained(name_or_path: str, mode: str | list[str] | None = "text_image", **kwargs):

        # auto model
        transformer_module = import_module("paddlenlp.transformers")
        for name in dir(transformer_module):
            module = getattr(transformer_module, name)
            if inspect.isclass(module) and issubclass(module, PretrainedConfig):
                init_configuration = getattr(module, "pretrained_init_configuration")
                if name_or_path in init_configuration:
                    config = init_configuration[name_or_path]
                    config.update(kwargs)
                    return module(**init_configuration[name_or_path])

        # clip model
        from paddlenlp.transformers.clip.modeling import (
            CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
        )

        if name_or_path in CLIP_PRETRAINED_MODEL_ARCHIVE_LIST:
            if "text_image" in mode:
                from paddlenlp.transformers.clip.configuration import CLIPConfig

                return CLIPConfig.from_pretrained(name_or_path, **kwargs)
            if "text" in mode:
                from paddlenlp.transformers.clip.configuration import CLIPTextConfig

                return CLIPTextConfig.from_pretrained(name_or_path, **kwargs)
            if "image" in mode:
                from paddlenlp.transformers.clip.configuration import CLIPVisionConfig

                return CLIPVisionConfig.from_pretrained(name_or_path, **kwargs)


class EncoderConfig(PretrainedConfig):

    mode_mapping: Dict[str, List[str]] = {
        "text": list(_load_text_model_names()),
        "text_image": _load_text_image_model_names(),
        "image": _load_image_model_names(),
    }

    def __init__(
        self,
        pretrained_model_name_or_path,
        architecture: str,
        mode: str | list[str] | None = None,
        embedding_type: str = "pooler",
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.architecture = architecture
        self.mode = mode
        self.embedding_type = embedding_type

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, mode: str | list[str] | None = None, **kwargs):
        # auto model
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        # resolve the architectures from PretrainedModel
        if not config.architectures:
            for model_name in _load_text_model_names():
                model_module = import_module(f"paddlenlp.transformers.{model_name}")
                if not inspect.isclass(model_module):
                    continue
                # TODO(wj-Mcat): assume that `pretrained_model_name_or_path` is name not dir
                if (
                    issubclass(model_module, PretrainedModel)
                    and pretrained_model_name_or_path in model_module.pretrained_init_configuration
                ):
                    config.architectures = [model_name]
                    break

        architecture = None
        if config.architectures:
            architecture = config.architectures[0]

        if not mode and config.architectures:
            architecture = config.architectures[0]
            if architecture in cls.mode_mapping["text_image"]:
                mode = "text_image"
            elif architecture in cls.mode_mapping["text"]:
                mode = "text"
            elif architecture in cls.mode_mapping["image"]:
                mode = "image"

        # TODO(wj-Mcat): check the configuration of EncoderConfig
        return cls(pretrained_model_name_or_path, architecture, mode=mode, **kwargs)


class Encoder(nn.Layer):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config: EncoderConfig = config

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: str, mode: str | list[str] | None = None, **kwargs):
        config = EncoderConfig.from_pretrained(pretrained_model_name_or_path, mode=mode, **kwargs)
        if not config.mode:
            raise ValueError(f"mode of model from <{pretrained_model_name_or_path}> should not be None.")
        if "text" in config.mode:
            return TextEncoder(config)
        if "image" in config.mode:
            return ImageEncoder(config)
        raise ValueError(f"can not detect the mode of model from <{pretrained_model_name_or_path}>")


class _TextFeatureMixin:
    # map the special feature method
    feature_method_mapping = {}
    text_method_name = None

    def get_text_features(self, *args, **kwargs):
        kwargs["return_dict"] = True
        text_method_name = self.text_method_name or "forward"

        # resolve text features
        model_class_name = self.model.__class__.__name__

        method_name = self.feature_method_mapping.get(model_class_name, text_method_name)
        method = getattr(self.model, method_name, None)
        if method is None:
            raise ValueError(f"<{model_class_name}> not supported `{text_method_name}`")

        if not inspect.ismethod(method):
            raise ValueError(f"<{method}> is not the method")

        output = method(*args, **kwargs)

        # some `get_text_features` not support return_dict params
        if paddle.is_tensor(output):
            return output

        if self.config.embedding_type == "cls":
            return output.last_hidden_state[:, 0, :]
        if self.config.embedding_type == "pooler":
            return output.pooler_output


class TextEncoder(_TextFeatureMixin, Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        model_class = import_module(f"paddlenlp.transformers.{config.architecture}")
        self.model = model_class.from_pretrained(config.pretrained_model_name_or_path)

    def forward(self, *args, **kwargs):
        return self.get_text_features(*args, **kwargs)


class _ImageFeatureMixin:
    # map the special feature method
    feature_method_mapping = {}
    image_method_name: str | None = None

    def get_image_features(self, *args, **kwargs):
        image_method_name = self.image_method_name or "forward"
        # resolve image features
        model_class_name = self.model.__class__.__name__

        method_name = self.feature_method_mapping.get(model_class_name, image_method_name)
        method = getattr(self.model, method_name, None)
        if method is None:
            raise ValueError(f"<{model_class_name}> not supported `get_image_features`")

        if not inspect.ismethod(method):
            raise ValueError(f"<{method}> is not the method")

        output = method(*args, **kwargs)
        # some `get_image_features` not support return_dict params
        if paddle.is_tensor(output):
            return output

        if self.config.embedding_type == "cls":
            return output.last_hidden_state[:, 0, :]
        if self.config.embedding_type == "pooler":
            return output.pooler_output


class ImageEncoder(_ImageFeatureMixin, Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        model_class = import_module(f"paddlenlp.transformers.{config.architecture}")
        self.model = model_class.from_pretrained(config.pretrained_model_name_or_path)

    def forward(self, *args, **kwargs):
        # resolve text features
        return self.image_features(*args, **kwargs)


class EncoderForRetrieval(PretrainedModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, second_pretrained_model_name_or_path: str | None = None):
        config = EncoderConfig.from_pretrained(pretrained_model_name_or_path)
        second_config = None

        if second_pretrained_model_name_or_path:
            second_config = EncoderConfig.from_pretrained(second_pretrained_model_name_or_path)

        second_mode = second_config.mode if second_config else config.mode
        modes = [config.mode, second_mode]
        if modes == ["text", "text"]:
            return EncoderForTextTextRetrieval(config, second_config)

        if modes == ["text", "image"]:
            assert (
                second_config is not None
            ), "when init `EncoderForTextImageRetrieval` model, `second_config`<image config> is required"
            return EncoderForTextImageRetrieval(config, second_config)

        if modes == ["image", "text"]:
            assert (
                second_config is not None
            ), "when init `EncoderForTextImageRetrieval` model, `second_config`<text config> is required"
            return EncoderForTextImageRetrieval(config, second_config)

        if modes == ["image", "image"]:
            return EncoderForImageImageRetrieval(config, second_config)

        raise ValueError(f"invalide modes<{','.join(modes)}>")


class EncoderForTextTextRetrieval(PretrainedModel, _TextFeatureMixin):
    """Encoder for Text-Image feature Retrieval"""

    def __init__(self, config: EncoderConfig, second_config: EncoderConfig | None = None):
        super().__init__()
        self.model = Encoder.from_pretrained(config.pretrained_model_name_or_path)

        if not second_config:
            self.second_model = self.model
        else:
            self.second_model = Encoder.from_pretrained(second_config.pretrained_model_name_or_path)

        self.dropout = nn.Dropout(config.dropout)
        if config.projection_dim:
            self.projection_layer = nn.Linear(config.projection_dim)


class EncoderForImageImageRetrieval(PretrainedModel, _TextFeatureMixin, _ImageFeatureMixin):
    """Encoder for Text-Image feature Retrieval"""

    get_features_mapping = {
        "CLIPModel": {
            "get_text_features": "get_text_features",
            "get_image_features": "get_image_features",
        }
    }
    text_method_name = "get_text_features"
    image_method_name = "get_image_features"

    def __init__(self, config: EncoderConfig, second_config: EncoderConfig | None = None):
        super().__init__()
        self.model = Encoder.from_pretrained(config.pretrained_model_name_or_path)
        if not second_config:
            self.second_model = self.model
        else:
            self.second_model = Encoder.from_pretrained(second_config.pretrained_model_name_or_path)

        self.dropout = nn.Dropout(config.dropout)
        if config.projection_dim:
            self.projection_layer = nn.Linear(config.projection_dim)


class EncoderForTextImageRetrieval(PretrainedModel, _TextFeatureMixin, _ImageFeatureMixin):
    """Encoder for Text-Image feature Retrieval"""

    get_features_mapping = {
        "CLIPModel": {
            "get_text_features": "get_text_features",
            "get_image_features": "get_image_features",
        }
    }
    text_method_name = "get_text_features"
    image_method_name = "get_image_features"

    def __init__(self, config: EncoderConfig, second_config: EncoderConfig):
        super().__init__()
        self.model = Encoder.from_pretrained(config.pretrained_model_name_or_path)
        self.second_model = Encoder.from_pretrained(second_config.pretrained_model_name_or_path)
        self.dropout = nn.Dropout(config.dropout)

        if config.projection_dim:
            self.projection_layer = nn.Linear(config.projection_dim)
