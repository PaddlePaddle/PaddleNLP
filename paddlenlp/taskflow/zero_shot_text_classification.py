# coding:utf-8
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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

from typing import Any, Dict, List, Union

import numpy as np
from paddle.static import InputSpec
from scipy.special import expit as np_sigmoid
from scipy.special import softmax as np_softmax

from ..prompt import PromptDataCollatorWithPadding, UTCTemplate
from ..transformers import UTC, AutoTokenizer
from .task import Task
from .utils import static_mode_guard

usage = r"""
        from paddlenlp import Taskflow

        schema = ['这是一条差评', '这是一条好评']
        text_cls = Taskflow("zero_shot_text_classification", schema=schema)
        text_cls('房间干净明亮，非常不错')
        '''
        [{'predictions': [{'label': '这是一条好评', 'score': 0.9695149765679986}], 'text_a': '房间干净明亮，非常不错'}]
        '''
         """


class ZeroShotTextClassificationTask(Task):
    """
    Zero-shot Universial Text Classification Task.

    Args:
        task (string): The name of task.
        model (string): The model_name in the task.
        schema (list): List of candidate labels.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "config": "config.json",
        "vocab_file": "vocab.txt",
        "special_tokens_map": "special_tokens_map.json",
        "tokenizer_config": "tokenizer_config.json",
    }
    resource_files_urls = {
        "utc-xbase": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-xbase/model_state.pdparams",
                "e751c3a78d4caff923759c0d0547bfe6",
            ],
            "config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-xbase/config.json",
                "4c2b035c71ff226a14236171a1a202a4",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-xbase/vocab.txt",
                "97eb0ec5a5890c8190e10e251af2e133",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-xbase/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-xbase/tokenizer_config.json",
                "be86466f6769fde498690269d099ea7c",
            ],
        },
        "utc-base": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-base/model_state.pdparams",
                "72089351c6fb02bcf8f270fe0cc508e9",
            ],
            "config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-base/config.json",
                "79aa9a69286604436937b03f429f4d34",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-base/vocab.txt",
                "97eb0ec5a5890c8190e10e251af2e133",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-base/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-base/tokenizer_config.json",
                "be86466f6769fde498690269d099ea7c",
            ],
        },
        "utc-medium": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-medium/model_state.pdparams",
                "2802c766a8b880aad910dd5a7db809ae",
            ],
            "config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-medium/config.json",
                "2899cd7c8590dcdc4223e4b1262e2f4e",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-medium/vocab.txt",
                "97eb0ec5a5890c8190e10e251af2e133",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-medium/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-medium/tokenizer_config.json",
                "be86466f6769fde498690269d099ea7c",
            ],
        },
        "utc-micro": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-micro/model_state.pdparams",
                "d9ebdfce9a8c6ebda43630ed18b07c58",
            ],
            "config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-micro/config.json",
                "8c8da9337e09e0c3962196987dca18bd",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-micro/vocab.txt",
                "97eb0ec5a5890c8190e10e251af2e133",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-micro/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-micro/tokenizer_config.json",
                "be86466f6769fde498690269d099ea7c",
            ],
        },
        "utc-mini": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-mini/model_state.pdparams",
                "848a2870cd51bfc22174a2a38884085c",
            ],
            "config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-mini/config.json",
                "933b8ebfcf995b1f965764ac426a2ffa",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-mini/vocab.txt",
                "97eb0ec5a5890c8190e10e251af2e133",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-mini/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-mini/tokenizer_config.json",
                "be86466f6769fde498690269d099ea7c",
            ],
        },
        "utc-nano": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-nano/model_state.pdparams",
                "2bd31212d989619148eda3afebc7354d",
            ],
            "config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-nano/config.json",
                "02fe311fdcc127e56ff0975038cc4d65",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-nano/vocab.txt",
                "97eb0ec5a5890c8190e10e251af2e133",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-nano/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-nano/tokenizer_config.json",
                "be86466f6769fde498690269d099ea7c",
            ],
        },
        "utc-pico": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-pico/model_state.pdparams",
                "f7068d63ad2930de7ac850d475052946",
            ],
            "config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-pico/config.json",
                "c0c7412cdd070edb5a1ce70c7fc68ad3",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-pico/vocab.txt",
                "97eb0ec5a5890c8190e10e251af2e133",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-pico/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-pico/tokenizer_config.json",
                "be86466f6769fde498690269d099ea7c",
            ],
        },
        "utc-large": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/zero_shot_text_classification/utc-large/model_state.pdparams",
                "71eb9a732c743a513b84ca048dc4945b",
            ],
            "config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/zero_shot_text_classification/utc-large/config.json",
                "9496be2cc99f7e6adf29280320274142",
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/zero_text_classification/utc-large/vocab.txt",
                "afc01b5680a53525df5afd7518b42b48",
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/zero_text_classification/utc-large/special_tokens_map.json",
                "2458e2131219fc1f84a6e4843ae07008",
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/zero_text_classification/utc-large/tokenizer_config.json",
                "dcb0f3257830c0eb1f2de47f2d86f89a",
            ],
        },
        "__internal_testing__/tiny-random-utc": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/models/community/__internal_testing__/tiny-random-utc/model_state.pdparams",
                "d303b59447be690530c35c73f8fd03cd",
            ],
            "config": [
                "https://bj.bcebos.com/paddlenlp/models/community/__internal_testing__/tiny-random-utc/config.json",
                "3420a6638a7c73c6239eb1d7ca1bc5fe",
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/models/community/__internal_testing__/tiny-random-utc/vocab.txt",
                "97eb0ec5a5890c8190e10e251af2e133",
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/models/community/__internal_testing__/tiny-random-utc/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/models/community/__internal_testing__/tiny-random-utc/tokenizer_config.json",
                "258fc552c15cec90046066ca122899e2",
            ],
        },
    }

    def __init__(self, task: str, model: str, schema: list = None, **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        self._set_utc_schema(schema)
        self._max_seq_len = kwargs.get("max_seq_len", 512)
        self._batch_size = kwargs.get("batch_size", 1)
        self._pred_threshold = kwargs.get("pred_threshold", 0.5)
        self._num_workers = kwargs.get("num_workers", 0)
        self._single_label = kwargs.get("single_label", False)

        self._check_task_files()
        self._construct_tokenizer()
        self._check_predictor_type()
        self._get_inference_model()

    def _set_utc_schema(self, schema):
        if schema is None:
            self._choices = None
        elif isinstance(schema, list):
            self._choices = schema
        elif isinstance(schema, dict) and len(schema) == 1:
            for key in schema:
                self._choices = schema[key]
        else:
            raise ValueError(f"Invalid schema: {schema}.")

    def set_schema(self, schema):
        self._set_utc_schema(schema)

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
            InputSpec(shape=[None, None, None, None], dtype="float32", name="attention_mask"),
            InputSpec(shape=[None, None], dtype="int64", name="omask_positions"),
            InputSpec(shape=[None], dtype="int64", name="cls_positions"),
        ]

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        model_instance = UTC.from_pretrained(self._task_path, from_hf_hub=self.from_hf_hub)
        self._model = model_instance
        self._model.eval()

    def _construct_tokenizer(self):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(self._task_path, from_hf_hub=self.from_hf_hub)
        self._collator = PromptDataCollatorWithPadding(self._tokenizer, return_tensors="np")
        self._template = UTCTemplate(self._tokenizer, self._max_seq_len)

    def _check_input_text(self, inputs):
        inputs = inputs[0]
        if isinstance(inputs, str) or isinstance(inputs, dict):
            inputs = [inputs]

        if isinstance(inputs, list):
            input_list = []
            for example in inputs:
                data = {"text_a": "", "text_b": "", "choices": self._choices}
                if isinstance(example, dict):
                    for k in example:
                        if k in data:
                            data[k] = example[k]
                elif isinstance(example, str):
                    data["text_a"] = example
                    data["text_b"] = ""
                elif isinstance(example, list):
                    for x in example:
                        if not isinstance(x, str):
                            raise ValueError("Invalid inputs, input text should be strings.")
                    data["text_a"] = example[0]
                    data["text_b"] = "".join(example[1:]) if len(example) > 1 else ""
                else:
                    raise ValueError(
                        "Invalid inputs, the input should be {'text_a': a, 'text_b': b}, a text or a list of text."
                    )

                if len(data["text_a"]) < 1 and len(data["text_b"]) < 1:
                    raise ValueError("Invalid inputs, input `text_a` and `text_b` are both missing or empty.")
                if not isinstance(data["choices"], list) or len(data["choices"]) < 2:
                    raise ValueError("Invalid inputs, label candidates should be a list with length >= 2.")
                input_list.append(data)
        else:
            raise TypeError("Invalid input format!")
        return input_list

    def _preprocess(self, inputs: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        # Get the config from the kwargs
        tokenized_inputs = [self._template(i) for i in inputs]
        batches = [
            tokenized_inputs[idx : idx + self._batch_size] for idx in range(0, len(tokenized_inputs), self._batch_size)
        ]
        inputs = [inputs[idx : idx + self._batch_size] for idx in range(0, len(inputs), self._batch_size)]
        outputs = {}
        outputs["text"] = inputs
        outputs["batches"] = [self._collator(batch) for batch in batches]

        return outputs

    def _run_model(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = {}
        outputs["text"] = inputs["text"]
        outputs["batch_logits"] = []
        dtype_dict = {
            "input_ids": "int64",
            "token_type_ids": "int64",
            "position_ids": "int64",
            "attention_mask": "float32",
            "omask_positions": "int64",
            "cls_positions": "int64",
        }
        with static_mode_guard():
            for batch in inputs["batches"]:
                if self._predictor_type == "paddle-inference":
                    for i, input_name in enumerate(self.input_names):
                        self.input_handles[i].copy_from_cpu(batch[input_name].astype(dtype_dict[input_name]))
                    self.predictor.run()
                    logits = self.output_handle[0].copy_to_cpu().tolist()
                else:
                    input_dict = {}
                    for input_name in dtype_dict:
                        input_dict[input_name] = batch[input_name].astype(dtype_dict[input_name])
                    logits = self.predictor.run(None, input_dict)[0].tolist()
                outputs["batch_logits"].append(logits)

        return outputs

    def _postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        This function converts the model logits output to class score and predictions
        """
        outputs = []
        for batch_text, batch_logits in zip(inputs["text"], inputs["batch_logits"]):
            for text, logits in zip(batch_text, batch_logits):
                output = {}
                if len(text["text_a"]) > 0:
                    output["text_a"] = text["text_a"]
                if len(text["text_b"]) > 0:
                    output["text_b"] = text["text_b"]

                if self._single_label:
                    score = np_softmax(logits, axis=-1)
                    label = np.argmax(logits, axis=-1)
                    output["predictions"] = [{"label": text["choices"][label], "score": score[label]}]
                else:
                    scores = np_sigmoid(logits)
                    output["predictions"] = []
                    if scores.ndim == 2:
                        scores = scores[0]
                    for i, class_score in enumerate(scores):
                        if class_score > self._pred_threshold:
                            output["predictions"].append({"label": text["choices"][i], "score": class_score})
                outputs.append(output)

        return outputs
