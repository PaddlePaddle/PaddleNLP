# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import numpy as np
from PIL import Image
from ..transformers import AutoModelForImageGeneration, AutoTokenizer
from .task import Task

usage = r"""
           from paddlenlp import Taskflow 

           text2imagegen = Taskflow("text2image_generation")
           images = text2imagegen("风阁水帘今在眼，且来先看早梅红")
           images[0].save("figure.png")
           
         """


class Text2ImageGenerationTask(Task):
    """
    The text2image generation model to generate the image. 
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    def __init__(self, task, model="pai-painter-painting-base-zh", **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._batch_size = kwargs.get("batch_size", 1)
        self._temperature = kwargs.get("temperature", 1.)
        self._top_k = kwargs.get("top_k", 32)
        self._top_p = kwargs.get("top_p", 1.)
        self._condition_scale = kwargs.get("condition_scale", 10.)
        self._num_return_images = kwargs.get("num_return_images", 4)
        self._use_faster = kwargs.get("use_faster", False)
        self._use_fp16_decoding = kwargs.get("use_fp16_decoding", False)
        self._construct_tokenizer(model)
        self._construct_model(model)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        self._model = AutoModelForImageGeneration.from_pretrained(model)
        self._model.eval()

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(model)

    def _batchify(self, data, batch_size):
        """
        Generate input batches.
        """

        def _parse_batch(batch_examples):
            tokenizerd_inputs = self._tokenizer(batch_examples,
                                                return_tensors="pd",
                                                padding="max_length",
                                                truncation=True)
            if self._model.base_model_prefix == "dallebart":
                tokenizerd_inputs["condition_scale"] = self._condition_scale
            return tokenizerd_inputs

        # Seperates data into some batches.
        one_batch = []
        for example in data:
            one_batch.append(example)
            if len(one_batch) == batch_size:
                yield _parse_batch(one_batch)
                one_batch = []
        if one_batch:
            yield _parse_batch(one_batch)

    def _preprocess(self, inputs):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        batches = self._batchify(inputs, self._batch_size)
        outputs = {'batches': batches, 'text': inputs}
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """
        all_images = []

        for batch_inputs in inputs["batches"]:
            images = self._model.generate(
                **batch_inputs,
                temperature=self._temperature,
                top_k=self._top_k,
                top_p=self._top_p,
                num_return_sequences=self._num_return_images,
                use_faster=self._use_faster,
                use_fp16_decoding=self._use_fp16_decoding)
            all_images.append(images.numpy())
        inputs['images'] = np.concatenate(all_images, axis=0)
        return inputs

    def _postprocess(self, inputs):
        """
        The model output is images, this function will convert the model output to PIL Image.
        """
        batch_out = []
        generated_images = inputs['images']
        # [batch_size, num_return_sequences, 256, 256, 3] -> [batch_size, 256, num_return_sequences*256, 3]
        generated_images = generated_images.transpose([0, 2, 1, 3, 4]).reshape([
            -1, generated_images.shape[-3],
            self._num_return_images * generated_images.shape[-2],
            generated_images.shape[-1]
        ])
        for generated_image in generated_images:
            batch_out.append(Image.fromarray(generated_image))

        return batch_out

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64",
                                    name='input_ids'),
        ]
