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

import os

import numpy as np
import paddle
from PIL import Image

from ..transformers import AutoModel, AutoProcessor
from ..utils.log import logger
from .task import Task
from .utils import dygraph_mode_guard, static_mode_guard

usage = r"""
            from paddlenlp import Taskflow
            from PIL import Image
            # Multi modal feature_extraction with ernie_vil-2.0-base-zh
            vision_language = Taskflow("feature_extraction", model='PaddlePaddle/ernie_vil-2.0-base-zh')
            image_embeds = vision_language([Image.open("demo/000000039769.jpg")])
            print(image_embeds)
            '''
            Tensor(shape=[1, 768], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                    [[-0.59475428, -0.69795364,  0.22144008,  0.88066685, -0.58184201,
                        -0.73454666,  0.95557910, -0.61410815,  0.23474170,  0.13301648,
                        0.86196446,  0.12281934,  0.69097638,  1.47614217,  0.07238606,
                        ...
            '''
            text_embeds = vision_language(["猫的照片","狗的照片"])
            text_features = text_embeds["features"]
            print(text_features)
            '''
            Tensor(shape=[2, 768], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                    [[ 0.04250504, -0.41429776,  0.26163983, ...,  0.26221892,
                        0.34387422,  0.18779707],
            '''
            image_features /= image_features.norm(axis=-1, keepdim=True)
            text_features /= text_features.norm(axis=-1, keepdim=True)
            logits_per_image = 100 * image_features @ text_features.t()
            probs = F.softmax(logits_per_image, axis=-1)
            print(probs)
            '''
            Tensor(shape=[1, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                [[0.99833173, 0.00166824]])
            '''
         """


class MultimodalFeatureExtractionTask(Task):
    """
    Feature extraction task using no model head. This task extracts the hidden states from the base
    model, which can be used as features in retrieval and clustering tasks.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "config": "config.json",
        "vocab_file": "vocab.txt",
        "preprocessor_config": "preprocessor_config.json",
        "special_tokens_map": "special_tokens_map.json",
        "tokenizer_config": "tokenizer_config.json",
    }
    resource_files_urls = {
        "PaddlePaddle/ernie_vil-2.0-base-zh": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/models/community/PaddlePaddle/ernie_vil-2.0-base-zh/model_state.pdparams",
                "38d8c8e01f74ba881e87d9a3f669e5ae",
            ],
            "config": [
                "https://paddlenlp.bj.bcebos.com/models/community/PaddlePaddle/ernie_vil-2.0-base-zh/config.json",
                "caf929b450d5638e8df2a95c936519e7",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/models/community/PaddlePaddle/ernie_vil-2.0-base-zh/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8",
            ],
            "preprocessor_config": [
                "https://paddlenlp.bj.bcebos.com/models/community/PaddlePaddle/ernie_vil-2.0-base-zh/preprocessor_config.json",
                "9a2e8da9f41896fedb86756b79355ee2",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/models/community/PaddlePaddle/ernie_vil-2.0-base-zh/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/models/community/PaddlePaddle/ernie_vil-2.0-base-zh/tokenizer_config.json",
                "da5385c23c8f522d33fc3aac829e4375",
            ],
        },
        "OFA-Sys/chinese-clip-vit-base-patch16": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/models/community/OFA-Sys/chinese-clip-vit-base-patch16/model_state.pdparams",
                "d594c94833b8cfeffc4f986712b3ef79",
            ],
            "config": [
                "https://paddlenlp.bj.bcebos.com/models/community/OFA-Sys/chinese-clip-vit-base-patch16/config.json",
                "3611b5c34ad69dcf91e3c1d03b01a93a",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/models/community/OFA-Sys/chinese-clip-vit-base-patch16/vocab.txt",
                "3b5b76c4aef48ecf8cb3abaafe960f09",
            ],
            "preprocessor_config": [
                "https://paddlenlp.bj.bcebos.com/models/community/OFA-Sys/chinese-clip-vit-base-patch16/preprocessor_config.json",
                "ba1fb66c75b18b3c9580ea5120e01ced",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/models/community/OFA-Sys/chinese-clip-vit-base-patch16/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/models/community/OFA-Sys/chinese-clip-vit-base-patch16/tokenizer_config.json",
                "573ba0466e15cdb5bd423ff7010735ce",
            ],
        },
        "OFA-Sys/chinese-clip-vit-large-patch14": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/models/community/OFA-Sys/chinese-clip-vit-large-patch14/model_state.pdparams",
                "5c0dde02d68179a9cc566173e53966c0",
            ],
            "config": [
                "https://paddlenlp.bj.bcebos.com/models/community/OFA-Sys/chinese-clip-vit-large-patch14/config.json",
                "a5e35843aa87ab1106e9f60f1e16b96d",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/models/community/OFA-Sys/chinese-clip-vit-large-patch14/vocab.txt",
                "3b5b76c4aef48ecf8cb3abaafe960f09",
            ],
            "preprocessor_config": [
                "https://paddlenlp.bj.bcebos.com/models/community/OFA-Sys/chinese-clip-vit-large-patch14/preprocessor_config.json",
                "ba1fb66c75b18b3c9580ea5120e01ced",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/models/community/OFA-Sys/chinese-clip-vit-large-patch14/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/models/community/OFA-Sys/chinese-clip-vit-large-patch14/tokenizer_config.json",
                "573ba0466e15cdb5bd423ff7010735ce",
            ],
        },
        "OFA-Sys/chinese-clip-vit-large-patch14-336px": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/models/community/OFA-Sys/chinese-clip-vit-large-patch14-336px/model_state.pdparams",
                "ee3eb7f9667cfb06338bea5757c5e0d7",
            ],
            "config": [
                "https://paddlenlp.bj.bcebos.com/models/community/OFA-Sys/chinese-clip-vit-large-patch14-336px/config.json",
                "cb2794d99bea8c8f45901d177e663e1e",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/models/community/OFA-Sys/chinese-clip-vit-large-patch14-336px/vocab.txt",
                "3b5b76c4aef48ecf8cb3abaafe960f09",
            ],
            "preprocessor_config": [
                "https://paddlenlp.bj.bcebos.com/models/community/OFA-Sys/chinese-clip-vit-large-patch14-336px/preprocessor_config.json",
                "c52a0b3abe9bdd1c3c5a3d56797f4a03",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/models/community/OFA-Sys/chinese-clip-vit-large-patch14-336px/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/models/community/OFA-Sys/chinese-clip-vit-large-patch14-336px/tokenizer_config.json",
                "573ba0466e15cdb5bd423ff7010735ce",
            ],
        },
        "__internal_testing__/tiny-random-ernievil2": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/models/community/__internal_testing__/tiny-random-ernievil2/model_state.pdparams",
                "771c844e7b75f61123d9606c8c17b1d6",
            ],
            "config": [
                "https://paddlenlp.bj.bcebos.com/models/community/__internal_testing__/tiny-random-ernievil2/config.json",
                "ae27a68336ccec6d3ffd14b48a6d1f25",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/models/community/__internal_testing__/tiny-random-ernievil2/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8",
            ],
            "preprocessor_config": [
                "https://paddlenlp.bj.bcebos.com/models/community/__internal_testing__/tiny-random-ernievil2/preprocessor_config.json",
                "9a2e8da9f41896fedb86756b79355ee2",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/models/community/__internal_testing__/tiny-random-ernievil2/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/models/community/__internal_testing__/tiny-random-ernievil2/tokenizer_config.json",
                "2333f189cad8dd559de61bbff4d4a789",
            ],
        },
    }

    def __init__(self, task, model, batch_size=1, is_static_model=True, max_length=128, return_tensors="pd", **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._seed = None
        self.export_type = "text"
        self._batch_size = batch_size
        self.return_tensors = return_tensors
        if not self.from_hf_hub:
            self._check_task_files()
        self._max_length = max_length
        self._construct_tokenizer()
        self.is_static_model = is_static_model
        self._config_map = {}
        self.predictor_map = {}
        self.input_names_map = {}
        self.input_handles_map = {}
        self.output_handle_map = {}
        self._check_predictor_type()
        if self.is_static_model:
            self._get_inference_model()
        else:
            self._construct_model(model)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        self._model = AutoModel.from_pretrained(self._task_path)
        self._model.eval()

    def _construct_tokenizer(self):
        """
        Construct the tokenizer for the predictor.
        """
        self._processor = AutoProcessor.from_pretrained(self._task_path)

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
            if self.is_static_model:
                # The input of static model is numpy array
                tokenized_inputs = self._processor(
                    text=batch_texts,
                    images=batch_images,
                    return_tensors="np",
                    padding="max_length",
                    max_length=self._max_length,
                    truncation=True,
                )
            else:
                # The input of dygraph model is padddle.Tensor
                tokenized_inputs = self._processor(
                    text=batch_texts,
                    images=batch_images,
                    return_tensors="pd",
                    padding="max_length",
                    max_length=self._max_length,
                    truncation=True,
                )
            return tokenized_inputs

        # Separates data into some batches.
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
        if isinstance(inputs, str):
            if len(inputs) == 0:
                raise ValueError("Invalid inputs, input text should not be empty, please check your input.")
            inputs = [inputs]
        elif isinstance(inputs, Image.Image):
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
        Transform the raw inputs to the model inputs, two steps involved:
           1) Transform the raw text/image to token ids/pixel_values.
           2) Generate the other model inputs from the raw text/image and token ids/pixel_values.
        """
        inputs = self._check_input_text(inputs)
        batches = self._batchify(inputs, self._batch_size)
        outputs = {"batches": batches, "inputs": inputs}
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """
        all_feats = []
        if self.is_static_model:
            with static_mode_guard():
                for batch_inputs in inputs["batches"]:
                    if self._predictor_type == "paddle-inference":
                        if "input_ids" in batch_inputs:
                            self.input_handles_map["text"][0].copy_from_cpu(batch_inputs["input_ids"])
                            self.predictor_map["text"].run()
                            text_features = self.output_handle_map["text"][0].copy_to_cpu()
                            all_feats.append(text_features)
                        elif "pixel_values" in batch_inputs:
                            self.input_handles_map["image"][0].copy_from_cpu(batch_inputs["pixel_values"])
                            self.predictor_map["image"].run()
                            image_features = self.output_handle_map["image"][0].copy_to_cpu()
                            all_feats.append(image_features)
                    else:
                        # onnx mode
                        if "input_ids" in batch_inputs:
                            input_dict = {}
                            input_dict["input_ids"] = batch_inputs["input_ids"]
                            text_features = self.predictor_map["text"].run(None, input_dict)[0].tolist()
                            all_feats.append(text_features)
                        elif "pixel_values" in batch_inputs:
                            input_dict = {}
                            input_dict["pixel_values"] = batch_inputs["pixel_values"]
                            image_features = self.predictor_map["image"].run(None, input_dict)[0].tolist()
                            all_feats.append(image_features)
        else:
            for batch_inputs in inputs["batches"]:
                if "input_ids" in batch_inputs:
                    text_features = self._model.get_text_features(input_ids=batch_inputs["input_ids"])
                    all_feats.append(text_features.numpy())
                if "pixel_values" in batch_inputs:
                    image_features = self._model.get_image_features(pixel_values=batch_inputs["pixel_values"])
                    all_feats.append(image_features.numpy())
        inputs.update({"features": all_feats})
        return inputs

    def _postprocess(self, inputs):
        inputs["features"] = np.concatenate(inputs["features"], axis=0)
        if self.return_tensors == "pd":
            inputs["features"] = paddle.to_tensor(inputs["features"])
        return inputs

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_text_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
        ]

        self._input_image_spec = [
            paddle.static.InputSpec(shape=[None, 3, 224, 224], dtype="float32", name="pixel_values"),
        ]

    def _convert_dygraph_to_static(self):
        """
        Convert the dygraph model to static model.
        """
        assert (
            self._model is not None
        ), "The dygraph model must be created before converting the dygraph model to static model."
        assert (
            self._input_image_spec is not None or self._input_text_spec is not None
        ), "The input spec must be created before converting the dygraph model to static model."
        logger.info("Converting to the inference model cost a little time.")

        static_model = paddle.jit.to_static(self._model.get_text_features, input_spec=self._input_text_spec)
        self.inference_model_path = self.inference_text_model_path
        paddle.jit.save(static_model, self.inference_model_path)
        logger.info("The inference model save in the path:{}".format(self.inference_model_path))

        static_model = paddle.jit.to_static(self._model.get_image_features, input_spec=self._input_image_spec)
        self.inference_model_path = self.inference_image_model_path
        paddle.jit.save(static_model, self.inference_model_path)
        logger.info("The inference model save in the path:{}".format(self.inference_model_path))

    def _get_inference_model(self):
        """
        Return the inference program, inputs and outputs in static mode.
        """
        _base_path = os.path.join(self._home_path, "taskflow", self.task, self.model)
        self.inference_image_model_path = os.path.join(_base_path, "static", "get_image_features")
        self.inference_text_model_path = os.path.join(_base_path, "static", "get_text_features")
        if (
            not os.path.exists(self.inference_image_model_path + ".pdiparams")
            or self._param_updated
            or not os.path.exists(self.inference_text_model_path + ".pdiparams")
        ):
            with dygraph_mode_guard():
                self._construct_model(self.model)
                self._construct_input_spec()
                self._convert_dygraph_to_static()
        if self._predictor_type == "paddle-inference":
            # Get text inference model
            self.inference_model_path = self.inference_text_model_path
            self._static_model_file = self.inference_model_path + ".pdmodel"
            self._static_params_file = self.inference_model_path + ".pdiparams"
            self._config = paddle.inference.Config(self._static_model_file, self._static_params_file)
            self._prepare_static_mode()

            self.predictor_map["text"] = self.predictor
            self.input_names_map["text"] = self.input_names
            self.input_handles_map["text"] = self.input_handles
            self.output_handle_map["text"] = self.output_handle
            self._config_map["text"] = self._config

            # Get image inference model
            self.inference_model_path = self.inference_image_model_path
            self._static_model_file = self.inference_model_path + ".pdmodel"
            self._static_params_file = self.inference_model_path + ".pdiparams"
            self._config = paddle.inference.Config(self._static_model_file, self._static_params_file)
            self._prepare_static_mode()

            self.predictor_map["image"] = self.predictor
            self.input_names_map["image"] = self.input_names
            self.input_handles_map["image"] = self.input_handles
            self.output_handle_map["image"] = self.output_handle
            self._config_map["image"] = self._config
        else:
            # Get text onnx model
            self.export_type = "text"
            self.inference_model_path = self.inference_text_model_path
            self._static_model_file = self.inference_model_path + ".pdmodel"
            self._static_params_file = self.inference_model_path + ".pdiparams"
            self._prepare_onnx_mode()
            self.predictor_map["text"] = self.predictor

            # Get image onnx model
            self.export_type = "image"
            self.inference_model_path = self.inference_image_model_path
            self._static_model_file = self.inference_model_path + ".pdmodel"
            self._static_params_file = self.inference_model_path + ".pdiparams"
            self._prepare_onnx_mode()
            self.predictor_map["image"] = self.predictor
