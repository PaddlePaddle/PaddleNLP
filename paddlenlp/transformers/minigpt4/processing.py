# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The Salesforce Team Authors and The HuggingFace Team. All rights reserved.
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
Processor class for MiniGPT4.
"""

from typing import List, Optional, Union

import numpy as np
import paddle
from PIL import Image

from ..image_processing_utils import BatchFeature
from ..image_utils import ImageInput
from ..processing_utils import ProcessorMixin
from ..tokenizer_utils_base import BatchEncoding, TensorType, TextInput

__all__ = [
    "MiniGPT4Processor",
]


class MiniGPT4Processor(ProcessorMixin):
    r"""
    Constructs a MiniGPT4 processor which wraps a MiniGPT4 image processor and an llama tokenizer into a single processor.
    [`MiniGPT4Processor`] offers all the functionalities of [`MiniGPT4ImageProcessor`] and [`LlamaTokenizer`]. See the docstring
    of [`~MiniGPT4ImageProcessor.__call__`] and [`~LlamaTokenizer.decode`] for more information.

    Args:
        image_processor (`MiniGPT4ImageProcessor`):
            An instance of [`MiniGPT4ImageProcessor`]. The image processor is a required input.
        tokenizer (`LlamaTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.

    Examples:
    ```python
    >>> import requests
    >>> from PIL import Image

    >>> import paddle
    >>> from paddlenlp.transformers import MiniGPT4Processor

    >>> # load processor
    >>> minigpt4_13b_path = "model_name"
    >>> processor = MiniGPT4Processor.from_pretrained(minigpt4_13b_path)
    >>> print("load processor and model done!")

    >>> # prepare model inputs for MiniGPT4
    >>> url = "https://paddlenlp.bj.bcebos.com/data/images/mugs.png"
    >>> image = Image.open(requests.get(url, stream=True).raw)
    >>> text = "describe this image"
    >>> prompt = "Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###Human: <Img><ImageHere></Img> <TextHere>###Assistant:"
    >>> res = processor([image], text, prompt)
    ```"""
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "MiniGPT4ImageProcessor"
    tokenizer_class = "LlamaTokenizer"

    def __init__(self, image_processor, tokenizer):
        tokenizer.return_token_type_ids = False
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self.default_prompt = "###Human: <Img><ImageHere></Img> <TextHere>###Assistant: "
        self.image_tag = "<ImageHere>"
        self.text_tag = "<TextHere>"

    def process_images(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PADDLE,
        **kwargs,
    ) -> BatchFeature:
        """
        This method uses [`MiniGPT4ImageProcessor.__call__`] method to prepare image(s) for the model.
        Please refer to the docstring of the method for more information.
        """
        if not images:
            raise ValueError("You have to input correct images.")

        if isinstance(images, (Image.Image, np.ndarray, paddle.Tensor)):
            images = [images]

        # processing with image processor
        processed_images = self.image_processor(images, return_tensors=return_tensors)

        return processed_images

    def process_texts(
        self,
        texts: Union[TextInput, List[TextInput]],
        prompts: Union[TextInput, List[TextInput]] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PADDLE,
        **kwargs,
    ):
        prompts = prompts if prompts is not None else [self.default_prompt]

        if (not isinstance(texts, TextInput)) and (not isinstance(texts, list)):
            raise TypeError("Unsupported type for texts: {}, only str and list type supported.".format(type(texts)))
        if prompts is not None and (not isinstance(prompts, TextInput)) and (not isinstance(prompts, list)):
            raise TypeError(
                "Unsupported type for prompts: {}, only str and list type supported.".format(type(prompts))
            )

        if isinstance(prompts, list):
            if isinstance(texts, list) and len(prompts) != len(texts):
                raise ValueError(
                    "The length of prompts not is equal to texts' length: {} != {}".format(len(prompts), len(texts))
                )
            elif isinstance(texts, TextInput):
                texts = [texts] * len(prompts)
        else:
            if isinstance(texts, TextInput):
                texts = [texts]
                prompts = [prompts]
            else:
                prompts = [prompts] * len(texts)

        assemble_texts = []
        for text, prompt in zip(texts, prompts):
            if self.image_tag not in text:
                if self.image_tag not in prompt:
                    raise ValueError(
                        "A prompt should contain a image tag `{}` to insert image embeddings. if you don't want to use prompt function, you have to input a text with the image tag `{}`.".format(
                            self.image_tag, self.image_tag
                        )
                    )
                if self.text_tag not in prompt:
                    raise ValueError(
                        "A prompt should contain a text tag `{}` to insert text information.".format(self.text_tag)
                    )
                assemble_texts.append(prompt.replace(self.text_tag, text))
            else:
                assemble_texts.append(text)

        # processing with text tokenizer
        first_texts, second_texts = zip(*[assemble_text.split(self.image_tag) for assemble_text in assemble_texts])
        first_text_encoding = self.tokenizer(
            text=first_texts, return_tensors=return_tensors, add_special_tokens=True, **kwargs
        )
        second_text_encoding = self.tokenizer(
            text=second_texts, return_tensors=return_tensors, add_special_tokens=False, **kwargs
        )

        encoded_texts = BatchEncoding(
            {
                "first_input_ids": first_text_encoding["input_ids"],
                "first_attention_mask": first_text_encoding["attention_mask"],
                "second_input_ids": second_text_encoding["input_ids"],
                "second_attention_mask": second_text_encoding["attention_mask"],
            }
        )
        return encoded_texts

    def __call__(
        self,
        images: ImageInput = None,
        text: str = None,
        prompt: str = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PADDLE,
        **kwargs,
    ) -> BatchFeature:
        """
        This method uses [`MiniGPT4ImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`LlamaTokenizer.__call__`] to prepare text for the model.
        Please refer to the docstring of the above two methods for more information.
        """
        prompt = prompt if prompt is not None else self.default_prompt

        if images is None and text is None:
            raise ValueError("Images and text are None, you have to specify either images or texts.")
        if images is not None and not isinstance(images, (Image.Image, np.ndarray, paddle.Tensor, list)):
            raise TypeError(
                "A type in [Image.Image, np.ndarray, paddle.Tensor, list] for images is expected, but received {}.".format(
                    type(images)
                )
            )
        if text is not None and not isinstance(text, str):
            raise TypeError("A str type of text is expected, but received {}.".format(type(text)))
        if prompt is not None and not isinstance(prompt, str):
            raise TypeError("A str type of prompt is expected, but received {}.".format(type(prompt)))

        if images is not None and not isinstance(images, list):
            images = [images]
        if text is not None and images is not None:
            texts = [text] * len(images)
            prompts = [prompt] * len(images)
        elif text is not None and images is None:
            texts = [text]
            prompts = [prompt]

        # image-only mode
        if text is None:
            # processing with image processor
            processed_features = self.process_images(images, return_tensors=return_tensors, **kwargs)
            return processed_features

        # text-only mode
        if images is None:
            # processing with text tokenizer
            encoded_texts = self.process_texts(texts, prompts, **kwargs)
            return encoded_texts

        # text-image mode
        processed_features = self.image_processor(images, return_tensors=return_tensors)
        encoded_texts = self.process_texts(texts, prompts, **kwargs)
        processed_features.update(encoded_texts)

        return processed_features

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """

        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
