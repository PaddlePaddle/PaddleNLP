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

from paddle.static import InputSpec
from scipy.special import expit as np_sigmoid

from paddlenlp.prompt import PromptDataCollatorWithPadding, UTCTemplate
from paddlenlp.transformers import UTC, AutoTokenizer

from .task import Task
from .utils import static_mode_guard

usage = r"""
        from paddlenlp import Taskflow

        question = '酒店评价情感分析'
        choices = ['这是一条差评', '这是一条好评']
        text_cls = Taskflow("zero_text_classification", question=question, choices=choices)
        text_cls('房间很整洁，相当不错')
        '''
        [{'text': '房间很整洁，相当不错', 'label': '这是一条好评', 'score': 0.80}]
        '''
         """


class ZeroTextClassificationTask(Task):
    """
    Unified Tag Classification Task.

    Args:
        task (string): The name of task.
        model (string): The model_name in the task.
        question (string): Discription of the task.
        choices (list): List of candidate labels.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "model_config": "model_config.json",
        "vocab_file": "vocab.txt",
        "special_tokens_map": "special_tokens_map.json",
        "tokenizer_config": "tokenizer_config.json",
    }
    resource_files_urls = {
        "utc-large": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/zero_text_classification/utc-large/model_state.pdparams",
                "a4b4693f0021ec94cdd32ecf3c5e168c",
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/zero_text_classification/utc-large/model_config.json",
                "21f3fa9aa4465d4d07afb0f24f57fae4",
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
            "added_tokens.json": [
                "https://bj.bcebos.com/paddlenlp/taskflow/zero_text_classification/utc-large/added_tokens.json",
                "ac8532655ddc0a1ce3b8a87dda81de5b",
            ],
        },
    }

    def __init__(self, task: str, model: str = "utc-large", question: str = None, choices: list = None, **kwargs):
        super(ZeroTextClassificationTask, self).__init__(task=task, model=model, **kwargs)

        self._question = "" if question is None else question
        self._choices = choices
        self._max_seq_len = kwargs.get("max_seq_len", 512)
        self._batch_size = kwargs.get("batch_size", 1)
        self._pred_threshold = kwargs.get("pred_threshold", 0.5)
        self._num_workers = kwargs.get("num_workers", 0)

        self._construct_tokenizer()
        self._check_predictor_type()
        self._get_inference_model()

    def set_argument(self, argument: dict):
        for k, v in argument.items():
            if k == "question" or k == "choices":
                setattr(self, f"_{k}", v)

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
        model_instance = UTC.from_pretrained(model)
        self._model = model_instance
        self._model.eval()

    def _construct_tokenizer(self):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._collator = PromptDataCollatorWithPadding(self._tokenizer, return_tensors="np")
        self._template = UTCTemplate(self._tokenizer, self._max_seq_len)

    def _check_input_text(self, inputs):
        inputs = inputs[0]
        if isinstance(inputs, str) or isinstance(inputs, dict):
            inputs = [inputs]

        if isinstance(inputs, list):
            input_list = []
            for example in inputs:
                data = {"text_a": "", "text_b": "", "choices": self._choices, "question": self._question}
                if isinstance(example, dict):
                    for k, v in example.items():
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
                if not isinstance(data["question"], str):
                    raise ValueError("Invalid inputs, prompt question should be a string.")
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
                for i, input_name in enumerate(self.predictor.get_input_names()):
                    self.input_handles[i].copy_from_cpu(batch[input_name].astype(dtype_dict[input_name]))
                self.predictor.run()
                logits = self.output_handle[0].copy_to_cpu().tolist()
                outputs["batch_logits"].append(logits)
        return outputs

    def _postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        This function converts the model logits output to class score and predictions
        """
        outputs = []
        for logits in inputs["batch_logits"]:
            scores = np_sigmoid(logits)
            output = {}
            output["predictions"] = []
            for i, class_score in enumerate(scores):
                if class_score > self._pred_threshold:
                    output["predictions"].append({"label": i, "score": class_score})
            outputs.append(output)

        for i, output in enumerate(outputs):
            if len(inputs["text"][i]["text_a"]) > 0:
                output["text_a"] = inputs["text"][i]["text_a"]
            if len(inputs["text"][i]["text_b"]) > 0:
                output["text_b"] = inputs["text"][i]["text_b"]
            for j, pred in enumerate(output["predictions"]):
                output["predictions"][j] = inputs["text"][i]["choices"][pred["label"]]

        return outputs
