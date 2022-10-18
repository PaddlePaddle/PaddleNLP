# coding=utf-8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Inc. team.
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
"""
 Processing saving/loading class for common processors.
"""

import paddlenlp.transformers
import os


class ProcessorMixin(object):
    """
    This is a mixin used to provide saving/loading functionality for all processor classes.
    """

    attributes = ["feature_extractor", "tokenizer"]
    # Names need to be attr_class for attr in attributes
    feature_extractor_class = None
    tokenizer_class = None
    _auto_class = None

    # args have to match the attributes class attribute
    def __init__(self, *args, **kwargs):
        # Sanitize args and kwargs
        for key in kwargs:
            if key not in self.attributes:
                raise TypeError(f"Unexepcted keyword argument {key}.")
        for arg, attribute_name in zip(args, self.attributes):
            if attribute_name in kwargs:
                raise TypeError(
                    f"Got multiple values for argument {attribute_name}.")
            else:
                kwargs[attribute_name] = arg

        if len(kwargs) != len(self.attributes):
            raise ValueError(
                f"This processor requires {len(self.attributes)} arguments: {', '.join(self.attributes)}. Got "
                f"{len(args)} arguments instead.")

        # Check each arg is of the proper class (this will also catch a user initializing in the wrong order)
        for attribute_name, arg in kwargs.items():
            setattr(self, attribute_name, arg)

    def __repr__(self):
        attributes_repr = [
            f"- {name}: {repr(getattr(self, name))}" for name in self.attributes
        ]
        attributes_repr = "\n".join(attributes_repr)
        return f"{self.__class__.__name__}:\n{attributes_repr}"

    def save_pretrained(self, save_directory, **kwargs):
        """
        Saves the attributes of this processor (feature extractor, tokenizer...) in the specified directory so that it
        can be reloaded using the [`~ProcessorMixin.from_pretrained`] method.

        <Tip>

        This class method is simply calling [`~feature_extraction_utils.FeatureExtractionMixin.save_pretrained`] and
        [`~tokenization_utils_base.PreTrainedTokenizer.save_pretrained`]. Please refer to the docstrings of the methods
        above for more information.

        </Tip>

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
            kwargs:
                Additional key word arguments.
        """
        os.makedirs(save_directory, exist_ok=True)

        for attribute_name in self.attributes:
            attribute = getattr(self, attribute_name)
            # Include the processor class in the attribute config so this processor can then be reloaded with the
            # `AutoProcessor` API.
            if hasattr(attribute, "_set_processor_class"):
                attribute._set_processor_class(self.__class__.__name__)
            attribute.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate a processor associated with a pretrained model.

        <Tip>

        This class method is simply calling the feature extractor
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`] and the tokenizer
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`] methods. Please refer to the docstrings of the
        methods above for more information.

        </Tip>

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:
            
                - a string, the name of a community-contributed pretrained or built-in pretrained model.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~SequenceFeatureExtractor.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            **kwargs
                Additional keyword arguments passed along to both
                [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`] and
                [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`].
        """
        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path,
                                                  **kwargs)
        return cls(*args)

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path,
                                       **kwargs):
        args = []
        for attribute_name in cls.attributes:
            class_name = getattr(cls, f"{attribute_name}_class")
            attribute_class = getattr(paddlenlp.transformers, class_name)
            args.append(
                attribute_class.from_pretrained(pretrained_model_name_or_path,
                                                **kwargs))
        return args
