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
import random
import paddle
import numpy as np
from PIL import Image
from ..transformers import AutoModelForImageGeneration, AutoTokenizer
from .task import Task

usage = r"""
           from paddlenlp import Taskflow 

           text_to_image = Taskflow("text_to_image")
           image_list = text_to_image("风阁水帘今在眼，且来先看早梅红")
           
           for batch_index, batch_image in enumerate(image_list):
               # len(batch_image) == 2 (num_return_images)
               for image_index_in_returned_images, each_image in enumerate(batch_image):
                    each_image.save(f"figure_{batch_index}_{image_index_in_returned_images}.png")
           
         """


class TextToImageGenerationTask(Task):
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
        self._temperature = kwargs.get("temperature", 1.0)
        self._top_k = kwargs.get("top_k", 32)
        self._top_p = kwargs.get("top_p", 1.0)
        self._condition_scale = kwargs.get("condition_scale", 10.0)
        self._num_return_images = kwargs.get("num_return_images", 2)
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
            tokenizerd_inputs = self._tokenizer(
                batch_examples, return_tensors="pd", padding="max_length", truncation=True
            )
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
        outputs = {"batches": batches, "text": inputs}
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """
        all_images = []
        for batch_inputs in inputs["batches"]:
            # set seed for reproduce
            if self._seed is None:
                self._seed = random.randint(0, 2**32)
            paddle.seed(self._seed)
            images = self._model.generate(
                **batch_inputs,
                temperature=self._temperature,
                top_k=self._top_k,
                top_p=self._top_p,
                num_return_sequences=self._num_return_images,
                use_faster=self._use_faster,
                use_fp16_decoding=self._use_fp16_decoding,
            )
            all_images.append(images.numpy())

        inputs["images"] = np.concatenate(all_images, axis=0)
        return inputs

    def _postprocess(self, inputs):
        """
        The model output is images, this function will convert the model output to PIL Image with argument.
        """
        batch_out = []
        for generated_image, prompt in zip(inputs["images"], inputs["text"]):
            image_list = []
            for image_index, image in enumerate(generated_image):
                pil_image = Image.fromarray(image)
                # set argument
                pil_image.argument = {
                    "input": prompt,
                    "batch_size": self._batch_size,
                    "seed": self._seed,
                    "temperature": self._temperature,
                    "top_k": self._top_k,
                    "top_p": self._top_p,
                    "condition_scale": self._condition_scale,
                    "num_return_images": self._num_return_images,
                    "use_faster": self._use_faster,
                    "use_fp16_decoding": self._use_fp16_decoding,
                    "image_index_in_returned_images": image_index,
                }
                image_list.append(pil_image)
            batch_out.append(image_list)

        return batch_out

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
        ]

    def set_argument(self, argument: dict):
        for k, v in argument.items():
            if k == "input" or k == "image_index_in_returned_images":
                continue
            setattr(self, f"_{k}", v)


class TextToImageDiscoDiffusionTask(Task):
    """
    The text_to_image disco diffusion model to generate the image.
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
        self._num_inference_steps = kwargs.get("num_inference_steps", 250)
        self._num_return_images = kwargs.get("num_return_images", 1)
        self._width = kwargs.get("width", 1280)
        self._height = kwargs.get("height", 768)
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
            tokenizerd_inputs = self._tokenizer(
                batch_examples,
                return_tensors="pd",
                padding="max_length",
                max_length=self._tokenizer.model_max_length,
                truncation=True,
            )
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
        outputs = {"batches": batches, "text": inputs}
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """
        all_images = []

        for batch_inputs, prompt in zip(inputs["batches"], inputs["text"]):
            image_per_prompt = []
            for _ in range(self._num_return_images):
                if self._seed is None:
                    seed = random.randint(0, 2**32)
                else:
                    seed = self._seed
                image = self._model.generate(
                    **batch_inputs,
                    seed=seed,
                    steps=self._num_inference_steps,
                    width_height=[self._width, self._height],
                )[0]
                argument = dict(
                    seed=seed,
                    height=self._height,
                    width=self._width,
                    num_inference_steps=self._num_inference_steps,
                    input=prompt,
                    num_return_images=1,
                )
                image.argument = argument
                image_per_prompt.append(image)
            all_images.append(image_per_prompt)
        inputs["images"] = all_images
        return inputs

    def _postprocess(self, inputs):
        return inputs["images"]

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
        ]

    def set_argument(self, argument: dict):
        for k, v in argument.items():
            if k == "input":
                continue
            setattr(self, f"_{k}", v)


class TextToImageStableDiffusionTask(Task):
    """
    The text_to_image diffusion model to generate the image.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._seed = None
        self._latents = None
        # we do not use batch
        self._batch_size = 1
        self._mode = kwargs.get("mode", "text2image")
        self._num_inference_steps = kwargs.get("num_inference_steps", 50)
        self._guidance_scale = kwargs.get("guidance_scale", 7.5)
        self._num_return_images = kwargs.get("num_return_images", 2)
        self._width = kwargs.get("width", 512)
        self._height = kwargs.get("height", 512)
        self._strength = kwargs.get("strength", 0.75)
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
            batch_examples = batch_examples[0]
            prompt = batch_examples if isinstance(batch_examples, str) else batch_examples[0]
            tokenizerd_inputs = self._tokenizer(
                prompt,
                return_tensors="pd",
                padding="max_length",
                max_length=self._tokenizer.model_max_length,
                truncation=True,
            )

            if self._mode in ["image2image", "inpaint"]:
                tokenizerd_inputs["init_image"] = batch_examples[1]
            if self._mode == "inpaint":
                tokenizerd_inputs["mask_image"] = batch_examples[2]

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

    def _check_input_text(self, inputs):
        inputs = inputs[0]
        if self._mode == "text2image":
            if isinstance(inputs, str):
                if len(inputs) == 0:
                    raise ValueError(
                        "Invalid inputs, input text should not be empty text, please check your input.".format(
                            type(inputs)
                        )
                    )
                inputs = [inputs]
            elif isinstance(inputs, list):
                if not (isinstance(inputs[0], str) and len(inputs[0].strip()) > 0):
                    raise TypeError(
                        "Invalid inputs, input text should be List[str], and first element of list should not be empty text.".format(
                            type(inputs[0])
                        )
                    )
            else:
                raise TypeError(
                    "Invalid inputs, input text should be str or List[str], but type of {} found!".format(type(inputs))
                )
        elif self._mode == "image2image":
            if isinstance(inputs, list):
                if isinstance(inputs[0], str):
                    if len(inputs) != 2:
                        raise ValueError(
                            "Invalid inputs, inputs should be ['prompt_text', 'init_image_path'], please check your input."
                        )
                    inputs = [inputs]
                if isinstance(inputs[0], list):
                    if len(inputs[0]) != 2:
                        raise ValueError(
                            "Invalid inputs, inputs should be [['prompt_text', 'init_image_path'],], please check your input."
                        )
            else:
                raise TypeError(
                    "Invalid inputs, input text should be `List[str, str]` or `List[List[str, str]]`, but type of {} found!".format(
                        type(inputs)
                    )
                )
        else:
            if isinstance(inputs, list):
                if isinstance(inputs[0], str):
                    if len(inputs) != 3:
                        raise ValueError(
                            "Invalid inputs, inputs should be ['prompt_text', 'init_image_path', 'mask_image_path'], please check your input."
                        )
                    inputs = [inputs]
                if isinstance(inputs[0], list):
                    if len(inputs[0]) != 3:
                        raise ValueError(
                            "Invalid inputs, inputs should be [['prompt_text', 'init_image_path', 'mask_image_path'],], please check your input."
                        )
            else:
                raise TypeError(
                    "Invalid inputs, input text should be `List[str, str, str]` or `List[List[str, str, str]]`, but type of {} found!".format(
                        type(inputs)
                    )
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
        all_images = []

        for batch_inputs, prompt in zip(inputs["batches"], inputs["text"]):
            image_per_prompt = []
            for _ in range(self._num_return_images):
                if self._seed is None:
                    seed = random.randint(0, 2**32)
                else:
                    seed = self._seed

                image = self._model.generate(
                    **batch_inputs,
                    mode=self._mode,
                    seed=seed,
                    strength=self._strength,
                    height=self._height,
                    width=self._width,
                    num_inference_steps=self._num_inference_steps,
                    guidance_scale=self._guidance_scale,
                    latents=self._latents,
                )[0]

                # set argument for reproduce
                if self._mode == "text2image":
                    argument = dict(
                        mode=self._mode,
                        seed=seed,
                        height=self._height,
                        width=self._width,
                        num_inference_steps=self._num_inference_steps,
                        guidance_scale=self._guidance_scale,
                        latents=self._latents,
                    )
                else:
                    argument = dict(
                        mode=self._mode,
                        seed=seed,
                        strength=self._strength,
                        num_inference_steps=self._num_inference_steps,
                        guidance_scale=self._guidance_scale,
                    )
                argument["num_return_images"] = 1
                argument["input"] = prompt
                image.argument = argument

                image_per_prompt.append(image)
            all_images.append(image_per_prompt)
        inputs["images"] = all_images
        return inputs

    def _postprocess(self, inputs):
        return inputs["images"]

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
        ]

    def set_argument(self, argument: dict):
        for k, v in argument.items():
            if k == "input":
                continue
            setattr(self, f"_{k}", v)
