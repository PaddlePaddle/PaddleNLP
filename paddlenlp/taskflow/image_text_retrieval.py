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

import paddle
from PIL import Image

from ..transformers import AutoModel, AutoProcessor
from .task import Task


class ImageTextRetrievalTask(Task):
    """
    The text_to_image generation model to generate the image.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._seed = None
        # we do not use batch
        self._batch_size = 1
        self._construct_tokenizer(model_name=model)
        self._construct_model(model)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        self._model = AutoModel.from_pretrained(model)
        self._model.eval()

    def _construct_tokenizer(self, model_name):
        """
        Construct the tokenizer for the predictor.
        """
        self._processor = AutoProcessor.from_pretrained(model_name)

    def _batchify(self, data, batch_size):
        """
        Generate input batches.
        """

        def _parse_batch(batch_examples):
            if isinstance(batch_examples[0], str):
                batch_texts = batch_examples
                batch_images = None
            else:
                batch_texts = None
                batch_images = batch_examples

            tokenized_inputs = self._processor(
                text=batch_texts, images=batch_images, return_tensors="pd", padding="max_length", truncation=True
            )
            return tokenized_inputs

        # Seperates data into some batches.
        one_batch = []
        for example in data:
            one_batch.append(example)
            if len(one_batch) == batch_size:
                yield _parse_batch(one_batch)
                one_batch = []
        if one_batch:
            yield _parse_batch(one_batch)

    def _check_input_text(self, inputs):
        """
        Check whether the input text meet the requirement.
        """
        inputs = inputs[0]
        if isinstance(inputs, (str, Image.Image)):
            if len(inputs) == 0:
                raise ValueError("Invalid inputs, input text/image should not be empty, please check your input.")
            inputs = [inputs]
        elif isinstance(inputs, list):
            # and len(inputs[0].strip()) > 0
            if not (isinstance(inputs[0], (str, Image.Image))):
                raise TypeError(
                    "Invalid inputs, input text/image should be list of str/PIL.image, and first element of list should not be empty."
                )
        else:
            raise TypeError(
                "Invalid inputs, input text should be str or list of str, but type of {} found!".format(type(inputs))
            )
        return inputs

    def _preprocess(self, inputs):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        batches = self._batchify(inputs, self._batch_size)
        outputs = {"batches": batches, "text": inputs}
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """
        all_feats = []
        for batch_inputs in inputs["batches"]:
            if "input_ids" in batch_inputs:
                text_features = self._model.get_text_features(input_ids=batch_inputs["input_ids"])
                all_feats.append(text_features)
            if "pixel_values" in batch_inputs:
                image_features = self._model.get_image_features(pixel_values=batch_inputs["pixel_values"])
                all_feats.append(image_features)
        inputs.update({"features": all_feats})
        return inputs

    def _postprocess(self, inputs):
        inputs["features"] = paddle.concat(inputs["features"], axis=0)
        return inputs

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
        ]
