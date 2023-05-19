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
from ..utils.log import logger
from .task import Task
from .utils import static_mode_guard

usage = r"""
        from paddlenlp import Taskflow
        text_cls = Taskflow(
            "text_classification",
            mode="finetune",
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
            mode="prompt",
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
        model (string): Mode of the classification, Supports ["prompt", "finetune"].
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
            task_path (string): The local file path to the model path or a pre-trained model.
            is_static_model (string): Whether the model in task path  is a static model.
            problem_type (str, optional): Select among ["multi_class", "multi_label"] based on the nature of your problem. Default to "multi_class".
            multilabel_threshold (float): The probability threshold used for the multi_label setup. Only effective if model = "multi_label". Defaults to 0.5.
            max_length (int): Maximum number of tokens for the model.
            precision (int): Select among ["fp32", "fp16"]. Default to "fp32".
            plm_model_name (str): Pretrained langugae model name for PromptModel.
            input_spec [list]: Specify the tensor information for each input parameter of the forward function.
            id2label(dict(int,string)): The dictionary to map the predictions from class ids to class names.
            batch_size(int): The sample number of a mini-batch.
    """

    def __init__(self, task: str, model: str = "finetune", **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self.problem_type = self.kwargs.get("problem_type", "multi_class")
        self.multilabel_threshold = self.kwargs.get("multilabel_threshold", 0.5)
        self._max_length = self.kwargs.get("max_length", 512)

        self._construct_tokenizer()
        if self.model == "prompt":
            self._initialize_prompt()
        self._check_predictor_type()
        self._get_inference_model()
        self._construct_id2label()

    def _initialize_prompt(self):
        if "plm_model_name" in self.kwargs:
            self._plm_model = AutoModelForMaskedLM.from_pretrained(self.kwargs["plm_model_name"])
        elif os.path.isdir(os.path.join(self._task_path, "plm")):
            self._plm_model = AutoModelForMaskedLM.from_pretrained(os.path.join(self._task_path, "plm"))
            logger.info(f"Load pretrained language model from {self._plm_model}")
        else:
            raise NotImplementedError(
                "Please specify the pretrained language model name （ex. plm_model_name='ernie-3.0-medium-zh'）."
            )
        self._template = AutoTemplate.load_from(self._task_path, self._tokenizer, self._max_length, self._plm_model)
        with open(os.path.join(self._task_path, "verbalizer_config.json"), "r", encoding="utf-8") as fp:
            self._label_words = json.load(fp)
        self._verbalizer = SoftVerbalizer(self._label_words, self._tokenizer, self._plm_model)

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
            elif os.path.exists(os.path.join(self._task_path, CONFIG_NAME)):
                with open(os.path.join(self._task_path, CONFIG_NAME)) as fb:
                    init_class = json.load(fb)["architectures"].pop()
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

    def _construct_id2label(self):
        if "id2label" in self.kwargs:
            id2label = self.kwargs["id2label"]
        elif os.path.exists(os.path.join(self._task_path, "id2label.json")):
            id2label_path = os.path.join(self._task_path, "id2label.json")
            with open(id2label_path) as fb:
                id2label = json.load(fb)
            logger.info(f"Load id2label from {id2label_path}.")
        elif self.model == "prompt" and os.path.exists(os.path.join(self._task_path, "verbalizer_config.json")):
            label_list = sorted(list(self._verbalizer.label_words.keys()))
            id2label = {}
            for i, l in enumerate(label_list):
                id2label[i] = l
            logger.info("Load id2label from verbalizer.")
        elif self.model == "finetune" and os.path.exists(os.path.join(self._task_path, CONFIG_NAME)):
            config_path = os.path.join(self._task_path, CONFIG_NAME)
            with open(config_path) as fb:
                config = json.load(fb)
                if "id2label" in config:
                    id2label = config["id2label"]
                    logger.info(f"Load id2label from {config_path}.")
                else:
                    id2label = None
        else:
            id2label = None

        if id2label is None:
            self.id2label = id2label
        else:
            self.id2label = {}
            for i in id2label:
                self.id2label[int(i)] = id2label[i]

    def _preprocess(self, inputs: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        # Get the config from the kwargs
        batch_size = self.kwargs["batch_size"] if "batch_size" in self.kwargs else 1

        if self.model == "finetune":
            collator = DataCollatorWithPadding(self._tokenizer, return_tensors="np")
            tokenized_inputs = [self._tokenizer(i, max_length=self._max_length, truncation=True) for i in inputs]
            batches = [tokenized_inputs[idx : idx + batch_size] for idx in range(0, len(tokenized_inputs), batch_size)]
        elif self.model == "prompt":
            collator = PromptDataCollatorWithPadding(
                self._tokenizer, padding=True, return_tensors="np", return_attention_mask=True
            )
            part_text = "text"
            for part in self._template.prompt:
                if "text" in part:
                    part_text = part["text"]
            template_inputs = [self._template({part_text: x}) for x in inputs]
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
                if "attention_mask" in batch:
                    input_name = "attention_mask"
                    if batch[input_name].ndim == 2:
                        batch[input_name] = (1 - batch[input_name][:, np.newaxis, np.newaxis, :]) * -1e4
                    elif batch[input_name].ndim != 4:
                        raise ValueError(
                            "Expect attention mask with ndim=2 or 4, but get ndim={}".format(batch[input_name].ndim)
                        )
                if self._predictor_type == "paddle-inference":
                    for i, input_name in enumerate(self.predictor.get_input_names()):
                        self.input_handles[i].copy_from_cpu(batch[input_name].astype(dtype_dict[input_name]))
                    self.predictor.run()
                    logits = self.output_handle[0].copy_to_cpu().tolist()
                else:
                    input_dict = {}
                    for input_name in self.input_handler:
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
