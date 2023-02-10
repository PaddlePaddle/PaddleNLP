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

from paddle import nn

from paddlenlp.transformers import AutoModel, PretrainedConfig
from paddlenlp.transformers.auto.modeling import get_name_mapping
from paddlenlp.utils.import_utils import import_module


def _load_text_model():
    model = get_name_mapping("Model")
    return {value for key, value in model.items() if key.endswith("Model_Import_Class")}


def _load_model_class_mapping(config: EncoderConfig):
    pass


class AutoConfig:
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
    def __init__(
        self,
        model_name,
        mode: str | list[str] = None,
        embedding_type: str = "pool",
        dropout: float | None = None,
        type="text",
        architecture: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.dropout = dropout
        self.mode = mode
        self.embedding_type = embedding_type
        self.architecture = architecture

    @staticmethod
    def from_pretrained(name_or_path: str, mode: str | list[str] | None = "text_image", **kwargs):
        # auto model
        transformer_module = import_module("paddlenlp.transformers")
        for name in dir(transformer_module):
            module = getattr(transformer_module, name)
            if inspect.isclass(module) and issubclass(module, PretrainedConfig):
                init_configuration = getattr(module, "pretrained_init_configuration")
                if name_or_path in init_configuration:
                    config = EncoderConfig(model_name=name_or_path, mode=mode, **kwargs)
                    return config

        # clip model
        from paddlenlp.transformers.clip.modeling import (
            CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
        )

        if name_or_path in CLIP_PRETRAINED_MODEL_ARCHIVE_LIST:
            if "text_image" in mode:
                return EncoderConfig(name_or_path, mode="text_image", architecture="CLIPModel")
            if "text" in mode:
                return EncoderConfig(model_name=name_or_path, mode="text", architecture="CLIPTextModel", **kwargs)
            if "image" in mode:
                return EncoderConfig.from_pretrained(
                    name_or_path, mode="image", architecture="CLIPVisionModel", **kwargs
                )


class Encoder(nn.Layer):
    mode_mapping = {
        "image": {"CLIPVisionModel", "ChineseCLIPVisionModel", "ErnieViLVisionModel"},
        "text_image": {"CLIPModel", "ChineseCLIPModel", "ErnieViLModel"},
        "text": {"CLIPTextModel", "ChineseCLIPTextModel", "ErnieViLTextModel"}.union(_load_text_model()),
    }

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config: EncoderConfig = config

    @staticmethod
    def from_pretrained(name_or_path: str, mode: str | list[str] = "text", **kwargs):
        config = EncoderConfig.from_pretrained(name_or_path, mode=mode, **kwargs)
        if "text_image" in config.mode:
            return TextImageEncoder(config)
        if "text" in config.mode:
            return TextEncoder(config)
        if "image" in config.mode:
            return ImageEncoder(config)


class TextEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        model_class = import_module(f"paddlenlp.transformers.{config.architecture}")
        self.model = model_class.from_pretrained(config.model_name)

    def forward(self, *args, **kwargs):
        return self.get_text_features(*args, **kwargs)

    def get_text_features(self, *args, **kwargs):
        # resolve text features
        kwargs["return_dict"] = True
        output = self.model(*args, **kwargs)
        if self.config.embedding_type == "cls":
            return output.last_hidden_state[:, 0, :]
        if self.config.embedding_type == "pool":
            return output.pooler_output

        raise NotImplementedError(f"embedding_type<{self.config.embedding_type}> not implemented")


class ImageEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        model_class = import_module(f"paddlenlp.transformers.{config.architecture}")
        self.model = model_class.from_pretrained(config.model_name)

    def forward(self, *args, **kwargs):
        # resolve text features
        return self.image_features(*args, **kwargs)

    def get_image_features(self, *args, **kwargs):
        # resolve text features
        kwargs["return_dict"] = True
        output = self.model(*args, **kwargs)
        if self.config.embedding_type == "cls":
            return output.last_hidden_state[:, 0, :]
        if self.config.embedding_type == "pool":
            return output.pooler_output

        raise NotImplementedError(f"embedding_type<{self.config.embedding_type}> not implemented")


class TextImageEncoder(TextEncoder, ImageEncoder):
    def __init__(self, config: EncoderConfig):
        super().super().__init__()
        self.model = AutoModel.from_pretrained(config.name_or_path)

    def forward(self, input_ids):
        return self.model(input_ids)

    def get_text_features(self, *args, **kwargs):
        # resolve text features
        kwargs["return_dict"] = True
        output = self.model(*args, **kwargs)
        if self.config.embedding_type == "cls":
            return output.last_hidden_state[:, 0, :]
        if self.config.embedding_type == "pool":
            return output.pooler_output

        raise NotImplementedError(f"embedding_type<{self.config.embedding_type}> not implemented")

    def get_image_features(self, *args, **kwargs):
        # resolve text features
        kwargs["return_dict"] = True
        output = self.model(*args, **kwargs)
        if self.config.embedding_type == "cls":
            return output.last_hidden_state[:, 0, :]
        if self.config.embedding_type == "pool":
            return output.pooler_output

        raise NotImplementedError(f"embedding_type<{self.config.embedding_type}> not implemented")


class EncoderForTextImageRetrieval(TextImageEncoder):
    """Encoder for Text-Image feature Retrieval"""

    get_features_mapping = {
        "CLIPModel": {
            "get_text_features": "get_text_features",
            "get_image_features": "get_image_features",
        }
    }

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.model = AutoModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.projection_layer = nn.Linear(config.projection_dim)

    # def forward(self, **kwargs):
    #     kwargs["return_dict"] = True
    #     output = self.model(**kwargs)
    # loss = output["loss"]

    def get_text_features(self, **text_inputs):
        # resolve text features
        attribute = getattr(self.model, "get_text_features", None)
        if attribute is not None:
            if not inspect.ismethod(attribute):
                raise ValueError("`get_text_features` is not a method")

            return self.model.get_text_features(**text_inputs)

        model_class_name = self.model.__class__.__name__

        # TODO(wj-Mcat): should support detecting features
        if model_class_name not in self.get_features_mapping:
            raise ValueError(f"<{model_class_name}> not supported `get_text_features`")

        attribute = getattr(self.model, self.get_features_mapping[model_class_name]["get_text_features"])
        return attribute(**text_inputs)

    def get_image_features(self, **image_inputs):
        # resolve image features
        attribute = getattr(self.model, "get_image_features", None)
        if attribute is not None:
            if not inspect.ismethod(attribute):
                raise ValueError("`get_image_features` is not a method")

            return self.model.get_image_features(**image_inputs)

        model_class_name = self.model.__class__.__name__

        # TODO(wj-Mcat): should support detecting features
        if model_class_name not in self.get_features_mapping:
            raise ValueError(f"<{model_class_name}> not supported `get_image_features`")

        attribute = getattr(self.model, self.get_features_mapping[model_class_name]["get_image_features"])
        return attribute(**image_inputs)
