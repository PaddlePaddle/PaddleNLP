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

from typing import Optional

import numpy as np
import paddle

from paddlenlp.transformers import AutoTokenizer, ErnieDualEncoder

from ..utils.log import logger
from .task import Task
from .utils import static_mode_guard


class TextFeatureExtractionTask(Task):

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "config": "config.json",
        "vocab_file": "vocab.txt",
        "special_tokens_map": "special_tokens_map.json",
        "tokenizer_config": "tokenizer_config.json",
    }

    resource_files_urls = {
        "rocketqa-zh-dureader-query-encoder": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-dureader-query-encoder/model_state.pdparams",
                "6125930530fd55ed715b0595e65789aa",
            ],
            "config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-dureader-query-encoder/config.json",
                "efc1280069bb22b5bd06dc44b780bc6a",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-dureader-query-encoder/vocab.txt",
                "062f696cad47bb62da86d8ae187b0ef4",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-dureader-query-encoder/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-dureader-query-encoder/tokenizer_config.json",
                "3a50349b8514e744fed72e59baca51b5",
            ],
        },
        "rocketqa-zh-base-query-encoder": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-base-query-encoder/model_state.pdparams",
                "3bb1a7870792146c6dd2fa47a45e15cc",
            ],
            "config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-base-query-encoder/config.json",
                "be88115dd8a00e9de6b44f8c9a055e1a",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-base-query-encoder/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-base-query-encoder/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-base-query-encoder/tokenizer_config.json",
                "be86466f6769fde498690269d099ea7c",
            ],
        },
    }

    def __init__(
        self,
        task: str = None,
        model: str = None,
        batch_size: int = 1,
        max_seq_len: int = 128,
        _static_mode: bool = True,
        return_tensors: bool = True,
        reinitialize: bool = False,
        share_parameters: bool = False,
        output_emb_size: Optional[int] = None,
        **kwargs
    ):
        super().__init__(task=task, model=model, **kwargs)
        self._seed = None
        # we do not use batch
        self.export_type = "text"
        self._batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.model = model
        self._static_mode = _static_mode
        self.return_tensors = return_tensors

        self.reinitialize = reinitialize
        self.share_parameters = share_parameters
        self.output_emb_size = output_emb_size

        # self._check_task_files()
        self._check_predictor_type()
        self._construct_tokenizer()
        self._get_inference_model()
        if self._static_mode:
            self._get_inference_model()
        else:
            self._construct_model(model)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        # self._model = ErnieDualEncoder(self._task_path)
        self._model = ErnieDualEncoder(
            query_model_name_or_path=self.model,
            output_emb_size=self.output_emb_size,
            reinitialize=self.reinitialize,
            share_parameters=self.share_parameters,
        )
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
                    text=batch_examples,
                    return_tensors="np",
                    padding="max_length",
                    truncation=True,
                    max_seq_len=self.max_seq_len,
                )
            else:
                tokenized_inputs = self._tokenizer(
                    text=batch_examples,
                    return_tensors="pd",
                    padding="max_length",
                    truncation=True,
                    max_seq_len=self.max_seq_len,
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
