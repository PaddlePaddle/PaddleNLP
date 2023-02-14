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
import numpy as np
import paddle

from paddlenlp.transformers import AutoTokenizer, ErnieDualEncoder

from ..utils.log import logger
from .task import Task
from .utils import static_mode_guard


class TextFeatureExtractionTask(Task):
    def __init__(self, task, model, batch_size=1, _static_mode=True, return_tensors=True, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._seed = None
        # we do not use batch
        self.export_type = "text"
        self._batch_size = batch_size
        self.model = model
        self._static_mode = _static_mode
        self.return_tensors = return_tensors
        # self._check_task_files()
        self._check_predictor_type()
        self._construct_tokenizer()
        if self._static_mode:
            self._get_inference_model()
        else:
            self._construct_model(model)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        self._model = ErnieDualEncoder(self.model)
        self._model.eval()

    def _construct_tokenizer(self):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
        ]

    def _batchify(self, data, batch_size):
        """
        Generate input batches.
        """

        def _parse_batch(batch_examples):
            if self._static_mode:
                tokenized_inputs = self._tokenizer(
                    text=batch_examples, return_tensors="np", padding="max_length", truncation=True
                )
            else:
                tokenized_inputs = self._tokenizer(
                    text=batch_examples, return_tensors="pd", padding="max_length", truncation=True
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
        if self._static_mode:
            with static_mode_guard():
                for batch_inputs in inputs["batches"]:
                    if self._predictor_type == "paddle-inference":
                        if "input_ids" in batch_inputs:
                            self.input_handles[0].copy_from_cpu(batch_inputs["input_ids"])
                            self.input_handles[1].copy_from_cpu(batch_inputs["token_type_ids"])
                            self.predictor.run()
                            text_features = self.output_handle[0].copy_to_cpu()
                            all_feats.append(text_features)
                    else:
                        # onnx mode
                        if "input_ids" in batch_inputs:
                            input_dict = {}
                            input_dict["input_ids"] = batch_inputs["input_ids"]
                            input_dict["token_type_ids"] = batch_inputs["token_type_ids"]
                            text_features = self.predictor.run(None, input_dict)[0].tolist()
                            all_feats.append(text_features)

        else:
            for batch_inputs in inputs["batches"]:
                text_features = self._model.get_pooled_embedding(
                    input_ids=batch_inputs["input_ids"], token_type_ids=batch_inputs["token_type_ids"]
                )
                all_feats.append(text_features.numpy())
        inputs.update({"features": all_feats})
        return inputs

    def _postprocess(self, inputs):
        inputs["features"] = np.concatenate(inputs["features"], axis=0)
        if self.return_tensors:
            inputs["features"] = paddle.to_tensor(inputs["features"])
        return inputs

    def _convert_dygraph_to_static(self):
        """
        Convert the dygraph model to static model.
        """
        assert (
            self._model is not None
        ), "The dygraph model must be created before converting the dygraph model to static model."
        assert (
            self._input_spec is not None
        ), "The input spec must be created before converting the dygraph model to static model."
        logger.info("Converting to the inference model cost a little time.")

        static_model = paddle.jit.to_static(self._model.get_pooled_embedding, input_spec=self._input_spec)
        paddle.jit.save(static_model, self.inference_model_path)
        logger.info("The inference model save in the path:{}".format(self.inference_model_path))
