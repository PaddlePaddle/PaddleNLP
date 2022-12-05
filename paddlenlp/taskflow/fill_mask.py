# coding:utf-8
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

from typing import Any, Dict, List, Optional, Union

import paddle
import paddle.nn.functional as F

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.transformers import AutoModelForMaskedLM, AutoTokenizer

from .task import Task
from .utils import dygraph_mode_guard

usage = r"""
        from paddlenlp import Taskflow
        text_cls = Taskflow(
            "fill_mask",
            task_path=<local_saved_model>,
            top_k=1
            )
        text_cls('飞桨[MASK]度学习架')
        '''
        [
            {
                'token': <token_index>,
                'token_str': '深',
                'sequence': 飞桨深度学习框架,
                'score': 0.65
            }
        ]
        '''
        text_cls(['飞桨[MASK]度学习架', '生活的真谛是[MASK]'])
        '''
        [
            {
                'token': <token_index>,
                'token_str': '深',
                'sequence': 飞桨深度学习框架,
                'score': 0.65
            },
            {
                'token': <token_index>,
                'token_str': '爱',
                'sequence': 生活的真谛是爱,
                'score': 0.65
            }
        ]
         """


class FillMaskTask(Task):
    """
    Perform cloze-style mask filling with Masked Language Modeling (MLM)
    NOTE: This task is different from all other tasks that it has no out-of-box zero-shot capabilities.
    Instead, it's used as a simple inference pipeline.
    Args:
        task (string): The name of task.
        task_path (string): The local file path to the model path or a pre-trained model
        top_k (string, optional): The number of predictions to return.. Defaults to 5.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    def __init__(self, task: str, model: Optional[str] = None, top_k: Optional[str] = 5, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self.top_k = top_k
        self._construct_tokenizer(self._task_path)
        self._construct_model(self._task_path)

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        raise NotImplementedError(f"Conversion from dygraph to static graph is not supported in {self.__name__}")

    def _construct_model(self, model: str):
        """
        Construct the inference model for the predictor.
        """
        model_instance = AutoModelForMaskedLM.from_pretrained(model, from_hf_hub=self.from_hf_hub)
        model_instance.eval()
        self._model = model_instance

    def _construct_tokenizer(self, model: str):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(model, from_hf_hub=self.from_hf_hub)

    def get_masked_index(self, input_ids):
        return paddle.nonzero(input_ids == self._tokenizer.mask_token_id)

    def ensure_exactly_one_mask_token(self, input_ids: List[int]):
        num_mask_token = input_ids.count(self._tokenizer.mask_token_id)
        if num_mask_token != 1:
            raise ValueError(f"FillMaskTask expects 1 mask token for each input but found {num_mask_token}")

    def _preprocess(self, inputs: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        # Get the config from the kwargs
        batch_size = self.kwargs["batch_size"] if "batch_size" in self.kwargs else 1

        max_length = self.kwargs["max_length"] if "max_length" in self.kwargs else 512
        collator = DataCollatorWithPadding(self._tokenizer, return_tensors="pd")
        tokenized_inputs = []
        for i in inputs:
            tokenized_input = self._tokenizer(i, max_length=max_length)
            self.ensure_exactly_one_mask_token(tokenized_input["input_ids"])
            tokenized_inputs.append(tokenized_input)

        batches = [tokenized_inputs[idx : idx + batch_size] for idx in range(0, len(tokenized_inputs), batch_size)]
        outputs = {}
        outputs["text"] = inputs
        outputs["batches"] = [collator(batch) for batch in batches]

        return outputs

    def _run_model(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the task model from the outputs of the `_tokenize` function.
        """
        model_outputs = []
        with dygraph_mode_guard():
            for batch in inputs["batches"]:
                logits = self._model(**batch)
                masked_index = self.get_masked_index(batch["input_ids"])
                mask_token_logits = paddle.gather_nd(logits, masked_index)
                mask_token_probs = F.softmax(mask_token_logits, axis=-1)
                top_probs, top_pred_indices = paddle.topk(mask_token_probs, k=self.top_k, axis=-1)
                for probs, pred_indices in zip(top_probs.tolist(), top_pred_indices.tolist()):
                    model_output = []
                    for prob, pred in zip(probs, pred_indices):
                        model_output.append({"token": pred, "score": prob})
                    model_outputs.append(model_output)
        outputs = {}
        outputs["text"] = inputs["text"]
        outputs["model_outputs"] = model_outputs
        return outputs

    def _postprocess(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """
        for i, model_output in enumerate(inputs["model_outputs"]):
            # Same API with https://huggingface.co/tasks/fill-mask
            for token_output in model_output:
                token_output["token_str"] = self._tokenizer.decode(token_output["token"])
                # Since we limit to 1 MASK per input, we can directly use .replace here
                token_output["sequence"] = inputs["text"][i].replace("[MASK]", token_output["token_str"])
        return inputs["model_outputs"]
