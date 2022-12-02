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
import paddle
import paddle.nn.functional as F
from scipy.special import expit as np_sigmoid
from scipy.special import softmax as np_softmax

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

from .task import Task
from .utils import dygraph_mode_guard, static_mode_guard

usage = r"""
        from paddlenlp import Taskflow
        text_cls = Taskflow(
            "text_classification",
            model="multi_class",
            task_path=<local_saved_model>,
            id2label={0: "negative", 1: "positive"}
            )
        text_cls('房间依然很整洁，相当不错')
        '''
        [
            {
                'text': '房间依然很整洁，相当不错',
                'predictions: [{
                    'label': 'positive',
                    'score': 0.80
                }]
            }
        ]
        '''
        text_cls = Taskflow(
            "text_classification",
            model="multi_label",
            task_path=<local_saved_model>,
            id2label={ 0: "体育", 1: "经济", 2: "娱乐"}
            )
        text_cls(['这是一条体育娱乐新闻的例子',
                        '这是一条经济新闻'])
        '''
        [
            {
                'text': '这是一条体育娱乐新闻的例子',
                'predictions: [
                    {
                        'label': '体育',
                        'score': 0.80
                    },
                    {
                        'label': '娱乐',
                        'score': 0.90
                    }
                ]
            },
            {
                'text': '这是一条经济新闻',
                'predictions: [
                    {
                    'label': '经济',
                    'score': 0.80
                    }
                ]
            }
        ]
         """


def softmax(x, axis=None):
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


class TextClassificationTask(Task):
    """
    The text classfication model to classify text.
    NOTE: This task is different from all other tasks that it has no out-of-box zero-shot capabilities.
    Instead, it's used as a simple inference pipeline.

    Args:
        task (string): The name of task.
        model (string): Mode of the classification, Supports ["multi_class", "multi_class"]
        task_path (string): The local file path to the model path or a pre-trained model
        id2label (string): The dictionary to map the predictions from class ids to class names
        is_static_model (string): Whether the model is a static model
        multilabel_threshold (float): The probability threshold used for the multi_label setup. Only effective if model = "multi_label". Defaults to 0.5
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    def __init__(
        self,
        task: str,
        model: str,
        id2label: Dict[int, str],
        is_static_model: bool = False,
        multilabel_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__(task=task, model=model, is_static_model=is_static_model, **kwargs)
        self.id2label = id2label
        self.is_static_model = is_static_model
        self._construct_tokenizer(self._task_path)
        self.multilabel_threshold = multilabel_threshold

        if self.is_static_model:
            self._get_inference_model()
        else:
            self._construct_model(self._task_path)

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        raise NotImplementedError("Conversion from dygraph to static graph is not supported in TextClassificationTask")

    def _construct_model(self, model: str):
        """
        Construct the inference model for the predictor.
        """
        model_instance = AutoModelForSequenceClassification.from_pretrained(model, num_labels=len(self.id2label))
        # Load the model parameter for the predict
        model_instance.eval()
        self._model = model_instance

    def _construct_tokenizer(self, model: str):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(model)

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
        collator = DataCollatorWithPadding(self._tokenizer, return_tensors="np" if self.is_static_model else "pd")
        tokenized_inputs = [self._tokenizer(i, max_length=max_length) for i in inputs]
        batches = [tokenized_inputs[idx : idx + batch_size] for idx in range(0, len(tokenized_inputs), batch_size)]
        outputs = {}
        outputs["text"] = inputs
        outputs["batches"] = [collator(batch) for batch in batches]

        return outputs

    def _run_model(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the task model from the outputs of the `_tokenize` function.
        """
        # TODO: support hierachical classification
        outputs = {}
        outputs["text"] = inputs["text"]
        outputs["batch_logits"] = []
        if self.is_static_model:
            with static_mode_guard():
                for batch in inputs["batches"]:
                    for i, input_name in enumerate(self.predictor.get_input_names()):
                        self.input_handles[i].copy_from_cpu(batch[input_name])
                    self.predictor.run()
                    logits = self.output_handle[0].copy_to_cpu().tolist()
                    outputs["batch_logits"].append(logits)
        else:
            with dygraph_mode_guard():
                for batch in inputs["batches"]:
                    logits = self._model(**batch)
                    outputs["batch_logits"].append(logits)
        return outputs

    def _postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        This function converts the model logits output to class score and predictions
        """
        # TODO: support hierachical classification
        postprocessed_outputs = []
        for logits in inputs["batch_logits"]:
            if self.model == "multi_class":
                if isinstance(logits, paddle.Tensor):  # dygraph
                    scores = F.softmax(logits, axis=-1).numpy()
                    labels = paddle.argmax(logits, axis=-1).numpy()
                else:  # static graph
                    scores = np_softmax(logits, axis=-1)
                    labels = np.argmax(logits, axis=-1)
                for score, label in zip(scores, labels):
                    postprocessed_output = {}
                    postprocessed_output["predictions"] = [{"label": self.id2label[label], "score": score[label]}]
                    postprocessed_outputs.append(postprocessed_output)
            else:  # multi_label
                if isinstance(logits, paddle.Tensor):  # dygraph
                    scores = F.sigmoid(logits).numpy()
                else:  # static graph
                    scores = np_sigmoid(logits)
                for score in scores:
                    postprocessed_output = {}
                    postprocessed_output["predictions"] = []
                    for i, class_score in enumerate(score):
                        if class_score > self.multilabel_threshold:
                            postprocessed_output["predictions"].append(
                                {"label": self.id2label[i], "score": class_score}
                            )
                    postprocessed_outputs.append(postprocessed_output)

        for i, postprocessed_output in enumerate(postprocessed_outputs):
            postprocessed_output["text"] = inputs["text"][i]
        return postprocessed_outputs
