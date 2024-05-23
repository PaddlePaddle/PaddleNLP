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
Processor class for VisualGLM.
"""

import re
from typing import List, Optional, Union

import numpy as np
import paddle
from PIL import Image

from ..image_processing_utils import BatchFeature
from ..image_utils import ImageInput
from ..processing_utils import ProcessorMixin
from ..tokenizer_utils_base import BatchEncoding, TensorType, TextInput

__all__ = [
    "VisualGLMProcessor",
]


class VisualGLMProcessor(ProcessorMixin):
    r"""
    Constructs a VisualGLM processor which wraps a VisualGLM image processor and an llama tokenizer into a single processor.
    [`VisualGLMProcessor`] offers all the functionalities of [`VisualGLMImageProcessor`] and [`LlamaTokenizer`]. See the docstring
    of [`~VisualGLMImageProcessor.__call__`] and [`~LlamaTokenizer.decode`] for more information.

    Args:
        image_processor (`VisualGLMImageProcessor`):
            An instance of [`VisualGLMImageProcessor`]. The image processor is a required input.
        tokenizer (`LlamaTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.

    Examples:
    ```python
    >>> import requests
    >>> from PIL import Image

    >>> import paddle
    >>> from paddlenlp.transformers import VisualGLMProcessor

    >>> # load processor
    >>> minigpt4_13b_path = "model_name"
    >>> processor = VisualGLMProcessor.from_pretrained(minigpt4_13b_path)
    >>> print("load processor and model done!")

    >>> # prepare model inputs for VisualGLM
    >>> url = "https://paddlenlp.bj.bcebos.com/data/images/mugs.png"
    >>> image = Image.open(requests.get(url, stream=True).raw)
    >>> text = "describe this image"
    >>> prompt = "Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###Human: <Img><ImageHere></Img> <TextHere>###Assistant:"
    >>> res = processor([image], text, prompt)
    ```"""
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "VisualGLMImageProcessor"
    tokenizer_class = "ChatGLMTokenizer"

    def __init__(self, image_processor, tokenizer):
        tokenizer.return_token_type_ids = False
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self.default_prompt = "<img><ImageHere></img>"
        self.image_tag = "<ImageHere>"
        self.num_query_tokens = 32

    def process_images(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PADDLE,
        **kwargs,
    ) -> BatchFeature:
        """
        This method uses [`VisualGLMImageProcessor.__call__`] method to prepare image(s) for the model.
        Please refer to the docstring of the method for more information.
        """
        if not images:
            raise ValueError("You have to input correct images.")

        if isinstance(images, (Image.Image, np.ndarray, paddle.Tensor)):
            images = [images]

        processed_images = self.image_processor(images, return_tensors=return_tensors)

        return processed_images

    def process_texts(
        self,
        texts: Union[TextInput, List[TextInput]],
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PADDLE,
        **kwargs,
    ) -> BatchEncoding:
        if not texts:
            raise ValueError("You have to input correct texts.")

        if isinstance(texts, TextInput):
            texts = [texts]

        processed_texts = self.tokenizer(text=texts, return_tensors=return_tensors, **kwargs)
        return BatchEncoding(processed_texts)

    def build_inputs_with_image(
        self,
        image: Union[Image.Image, np.ndarray, paddle.Tensor],
        query: str,
        history: Optional[str] = None,
    ):
        # construct prompt with inputs
        if image is not None:
            prompt = self.default_prompt
        else:
            prompt = ""
        for old_query, response in history:
            prompt += "问：{}\n答：{}\n".format(old_query, response)
        prompt += "问：{}\n答：".format(query)

        if image is not None:
            image_start_position = prompt.rfind(self.image_tag)
            image_end_position = image_start_position + len(self.image_tag)
            first_text_input = self.tokenizer.encode(prompt[:image_start_position], add_special_tokens=False)
            image_input = [self.tokenizer.unk_token_id] * self.num_query_tokens
            second_text_input = self.tokenizer.encode(prompt[image_end_position:], add_special_tokens=False)
            all_input_ids = first_text_input["input_ids"] + image_input + second_text_input["input_ids"]
            all_input_ids = self.tokenizer.build_inputs_with_special_tokens(all_input_ids)

            # processing image
            processed_image = self.process_images(image)

            inputs = {
                "input_ids": paddle.to_tensor(all_input_ids, dtype="int64").unsqueeze(0),
                "pre_image_length": len(first_text_input["input_ids"]),
                "pixel_values": processed_image["pixel_values"],
            }
        else:
            inputs = self.tokenizer([prompt], return_tensors="pd")
            inputs["pre_image_length"] = 0

        return inputs

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, paddle.Tensor],
        query: str,
        history: Optional[str] = [],
        **kwargs,
    ):
        if image is None:
            raise ValueError("Image should not be None.")
        if query is None:
            raise ValueError("Query should not be None.")
        if not isinstance(query, str):
            raise TypeError("A string type of query is expected, but acceived {}.".format(type(query)))
        if not isinstance(history, list):
            raise TypeError(
                "A list type of history is expected with each item [query, response] in it, but acceived {}.".format(
                    type(history)
                )
            )

        inputs = self.build_inputs_with_image(image, query, history=history)

        return inputs

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

    def process_response(self, response):
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response

    def get_responses(self, *args, **kwargs):
        processed_responses = []
        responses = self.batch_decode(*args, **kwargs)

        for response in responses:
            response = self.process_response(response)
            processed_responses.append(response)

        return processed_responses

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
