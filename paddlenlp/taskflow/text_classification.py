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
import json
import os
from typing import Any, Dict, List, Union

import numpy as np
import paddle
import paddle.nn.functional as F
from scipy.special import expit as np_sigmoid
from scipy.special import softmax as np_softmax

from ..data import DataCollatorWithPadding
from ..prompt import (
    AutoTemplate,
    PromptDataCollatorWithPadding,
    PromptModelForSequenceClassification,
    SoftVerbalizer,
)
from ..transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from ..utils.env import CONFIG_NAME, LEGACY_CONFIG_NAME
from .task import Task
from .utils import static_mode_guard

usage = r"""
        from paddlenlp import Taskflow
        text_cls = Taskflow(
            "text_classification",
            model="finetune",
            problem_type="multi_class",
            task_path=<local_saved_dynamic_model>,
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
            model="prompt",
            problem_type="multi_label",
            is_static_model=True,
            task_path=<local_saved_static_model>,
            static_model_prefix=<static_model_prefix>,
            plm_model_path=<local_saved_plm_model>,
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
        task_path (string): The local file path to the model path or a pre-trained model.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    def __init__(self, task: str, model: str = "finetune", **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self.problem_type = kwargs.get("problem_type", "multi_class")
        self.multilabel_threshold = kwargs.get("multilabel_threshold", 0.5)

        self._construct_tokenizer()
        if model == "prompt":
            self._construct_plm_model()
            self._construct_template()
            self._construct_verbalizer()

        self._construct_id2label()
        self._check_predictor_type()
        self._get_inference_model()

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        if "input_spec" in self.kwargs:
            self._input_spec = self.kwargs["input_spec"]
        elif self.model == "finetune":
            if os.path.exists(os.path.join(self._task_path, LEGACY_CONFIG_NAME)):
                with open(os.path.join(self._task_path, LEGACY_CONFIG_NAME)) as fb:
                    init_class = json.load(fb)["init_class"]
                fb.close()
            elif os.path.exists(os.path.join(self._task_path, CONFIG_NAME)):
                with open(os.path.join(self._task_path, CONFIG_NAME)) as fb:
                    init_class = json.load(fb)["architectures"].pop()
                fb.close()
            else:
                raise IOError(
                    f"Model configuration file dosen't exist.[task_path] should inclue {LEGACY_CONFIG_NAME} or {CONFIG_NAME}"
                )

            if init_class in ["ErnieMForSequenceClassification"]:
                self._input_spec = [paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids")]
            else:
                self._input_spec = [
                    paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
                    paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
                ]
        elif self.model == "prompt":
            self._input_spec = self._model.get_input_spec()
        else:
            raise NotImplementedError(
                f"'{self.model}' is not a supported model_type. Please select among ['finetune', 'prompt']"
            )

    def _construct_model(self, model: str):
        """
        Construct the inference model for the predictor.
        """
        if model == "finetune":
            model_instance = AutoModelForSequenceClassification.from_pretrained(self._task_path)
        elif model == "prompt":
            model_instance = PromptModelForSequenceClassification(self._plm_model, self._template, self._verbalizer)
            # We load the model state dict on the CPU to avoid an OOM error.
            # If the model is on the GPU, it still works!
            state_dict = paddle.load(os.path.join(self._task_path, "model_state.pdparams"), return_numpy=True)
            model_instance.set_state_dict(state_dict)
            # release memory
            del state_dict
        else:
            raise NotImplementedError(
                f"'{model}' is not a supported model_type. Please select among ['finetune', 'prompt']"
            )

        # Load the model parameter for the predict
        model_instance.eval()
        self._model = model_instance

    def _construct_tokenizer(self):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(self._task_path)

    def _construct_plm_model(self):
        """
        Construct the plm model for prompt-model.
        """
        if "plm_model_name" in self.kwargs:
            self._plm_model = AutoModelForMaskedLM.from_pretrained(self.kwargs["plm_model_name"])
        else:
            if "plm_model_path" in self.kwargs:
                plm_model_path = self.kwargs["plm_model_path"]
            else:
                plm_model_path = os.path.join(self._task_path, "plm")

            if os.path.isdir(plm_model_path):
                self._plm_model = AutoModelForMaskedLM.from_pretrained(plm_model_path)
            else:
                raise IOError(
                    f"{plm_model_path} does not exist. Please specify the pretrained language model name （ex. plm_model_name='ernie-3.0-medium-zh'）or the pretrained language model parameter path(ex. plm_model_path='./checkpoints/plm')"
                )

    def _construct_template(self):
        """
        Construct the template for prompt-model.
        """
        with open(os.path.join(self._task_path, "template_config.json"), "r", encoding="utf-8") as fp:
            prompt = json.loads(fp.readline())
        fp.close()
        max_length = self.kwargs["max_length"] if "max_length" in self.kwargs else 512
        self._template = AutoTemplate.create_from(prompt, self._tokenizer, max_length, model=self._plm_model)

    def _construct_verbalizer(self):
        """
        Construct the verbalizer for prompt-model.
        """
        with open(os.path.join(self._task_path, "verbalizer_config.json"), "r", encoding="utf-8") as fp:
            self._label_words = json.load(fp)
        fp.close()
        self._verbalizer = SoftVerbalizer(self._label_words, self._tokenizer, self._plm_model)

    def _construct_id2label(self):
        if "id2label" in self.kwargs:
            self.id2label = self.kwargs["id2label"]
        elif os.path.exists(os.path.join(self._task_path, "id2label.json")):
            self.id2label = json.load(open(os.path.join(self._task_path, "id2label.json")))
        elif self.model == "prompt" and os.path.exists(os.path.join(self._task_path, "verbalizer_config.json")):
            label_list = sorted(list(self._label_words.keys()))
            self.id2label = {}
            for i, l in enumerate(label_list):
                self.id2label[i] = l
        elif self.model == "finetune" and os.path.exists(os.path.join(self._task_path, CONFIG_NAME)):
            with open(os.path.join(self._task_path, CONFIG_NAME)) as fb:
                id2label = json.load(fb)["id2label"]
            fb.close()
            self.id2label = {}
            for i in id2label:
                self.id2label[int(i)] = id2label[i]
        else:
            self.id2label = None

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
        if self.model == "finetune":
            collator = DataCollatorWithPadding(self._tokenizer, return_tensors="np")
            tokenized_inputs = [self._tokenizer(i, max_length=max_length, truncation=True) for i in inputs]
            batches = [tokenized_inputs[idx : idx + batch_size] for idx in range(0, len(tokenized_inputs), batch_size)]
        elif self.model == "prompt":
            collator = PromptDataCollatorWithPadding(
                self._tokenizer, padding=True, return_tensors="np", return_attention_mask=True
            )
            template_inputs = [self._template({"text_a": x}) for x in inputs]
            batches = [template_inputs[idx : idx + batch_size] for idx in range(0, len(template_inputs), batch_size)]

        else:
            raise NotImplementedError(
                f"'{self.model}' is not a supported model_type. Please select among ['finetune', 'prompt']"
            )
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
        dtype_dict = {
            "input_ids": "int64",
            "token_type_ids": "int64",
            "position_ids": "int64",
            "attention_mask": "float32",
            "masked_positions": "int64",
            "soft_token_ids": "int64",
            "encoder_ids": "int64",
        }
        with static_mode_guard():
            for batch in inputs["batches"]:
                if self._predictor_type == "paddle-inference":
                    for i, input_name in enumerate(self.predictor.get_input_names()):
                        self.input_handles[i].copy_from_cpu(batch[input_name].astype(dtype_dict[input_name]))
                    self.predictor.run()
                    logits = self.output_handle[0].copy_to_cpu().tolist()
                else:
                    input_dict = {}
                    for input_name in self.input_handler:
                        if input_name == "attention_mask":
                            if batch[input_name].ndim == 2:
                                batch[input_name] = (1 - batch[input_name][:, np.newaxis, np.newaxis, :]) * -1e4
                            elif batch[input_name].ndim != 4:
                                raise ValueError(
                                    "Expect attention mask with ndim=2 or 4, but get ndim={}".format(
                                        batch[input_name].ndim
                                    )
                                )
                        input_dict[input_name] = batch[input_name].astype(dtype_dict[input_name])
                    logits = self.predictor.run(None, input_dict)[0].tolist()
                outputs["batch_logits"].append(logits)
        return outputs

    def _postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        This function converts the model logits output to class score and predictions
        """
        # TODO: support hierachical classification
        postprocessed_outputs = []
        for logits in inputs["batch_logits"]:
            if self.problem_type == "multi_class":
                if isinstance(logits, paddle.Tensor):  # dygraph
                    scores = F.softmax(logits, axis=-1).numpy()
                    labels = paddle.argmax(logits, axis=-1).numpy()
                else:  # static graph
                    scores = np_softmax(logits, axis=-1)
                    labels = np.argmax(logits, axis=-1)
                for score, label in zip(scores, labels):
                    postprocessed_output = {}
                    if self.id2label is None:
                        postprocessed_output["predictions"] = [{"label": label, "score": score[label]}]
                    else:
                        postprocessed_output["predictions"] = [{"label": self.id2label[label], "score": score[label]}]
                    postprocessed_outputs.append(postprocessed_output)
            elif self.problem_type == "multi_label":  # multi_label
                if isinstance(logits, paddle.Tensor):  # dygraph
                    scores = F.sigmoid(logits).numpy()
                else:  # static graph
                    scores = np_sigmoid(logits)
                for score in scores:
                    postprocessed_output = {}
                    postprocessed_output["predictions"] = []
                    for i, class_score in enumerate(score):
                        if class_score > self.multilabel_threshold:
                            if self.id2label is None:
                                postprocessed_output["predictions"].append({"label": i, "score": class_score})
                            else:
                                postprocessed_output["predictions"].append(
                                    {"label": self.id2label[i], "score": class_score}
                                )
                    postprocessed_outputs.append(postprocessed_output)
            else:
                raise NotImplementedError(
                    f"'{self.problem_type}' is not a supported problem type. Please select among ['multi_class', 'multi_label']"
                )
        for i, postprocessed_output in enumerate(postprocessed_outputs):
            postprocessed_output["text"] = inputs["text"][i]
        return postprocessed_outputs
