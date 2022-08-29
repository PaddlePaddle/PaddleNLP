# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from ..transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..data import Pad, Tuple
from .utils import static_mode_guard
from .task import Task

usage = r"""
        from paddlenlp import Taskflow

        classify = Taskflow("text_classification")
        classify("7月6日，据《足球报》报道，中超联赛大概率延期到8月6日前后开启，参加东亚杯的国脚预计缺席一轮中超。")
        '''
        [{'text': '7月6日，据《足球报》报道，中超联赛大概率延期到8月6日前后开启，参加东亚杯的国脚预计缺席一轮中超。', 'label': ['体育', '足球'], 'confidence': [0.9848, 0.89951]}]
        '''
         
        classify(["7月5日，周杰伦新专辑《最伟大的作品》开启预约，目前，QQ音乐上已经有超过270万人参与了预约。据悉，《最伟大的作品》7月6日迎来MV首播，7月8日会开启专辑预售，7月15日专辑将正式上线",  "7月6日，据《足球报》报道，中超联赛大概率延期到8月6日前后开启，参加东亚杯的国脚预计缺席一轮中超。"])
        '''
        [{'text': '7月5日，周杰伦新专辑《最伟大的作品》开启预约，目前，QQ音乐上已经有超过270万人参与了预约。据悉，《最伟大的作品》7月6日迎来MV首播，7月8日会开启专辑预售，7月15日专辑将正式上线', 'label': ['娱乐', '音乐'], 'confidence': [0.92587, 0.526]}, {'text': '7月6日，据《足球报》报道，中超联赛大概率延期到8月6日前后开启，参加东亚杯的国脚预计缺席一轮中超。', 'label': ['体育', '足球'], 'confidence': [0.9848, 0.89951]}]
        '''
        """


class TextClassificationTask(Task):
    """
    Text classification aims to predict the text topics.
    Args:
        task(string): The name of task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "model_config": "model_config.json",
        "label_list": "label.txt"
    }
    resource_files_urls = {
        "ernie-3.0-medium-zh": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_classification/model_state.pdparams",
                "ea9990adc67f7a809e77f095e2aac7c0"
            ],
            "model_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_classification/model_config.json",
                "b1b4edcda77fb39a0251de64a51ad7e5"
            ],
            "label_list": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_classification/label.txt",
                "3095c2d027e5c7dc22247986c866d92f"
            ]
        }
    }

    def __init__(self,
                 task,
                 model="ernie-3.0-medium-zh",
                 batch_size=1,
                 max_seq_len=128,
                 **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._static_mode = True
        self._check_task_files()
        self._label_map = self._construct_label_map()
        if self._static_mode:
            self._get_inference_model()
        else:
            self._construct_model(model)
        self._construct_tokenizer(model)
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._usage = usage
        self.model_name = model

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64",
                                    name='input_ids'),
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64",
                                    name='token_type_ids'),
        ]

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._task_path)
        self._model.eval()

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(model)

    def _construct_label_map(self):
        """
        Construct label mapping for the predictor.
        """
        label_list_path = os.path.join(self._task_path, "label.txt")
        label_map = {}
        with open(label_list_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                label_map[i] = line.strip()
        return label_map

    def _check_input_text(self, inputs):
        inputs = inputs[0]
        if isinstance(inputs, str):
            if len(inputs) == 0:
                raise ValueError(
                    "Invalid inputs, input text should not be empty text, please check your input."
                    .format(type(inputs)))
            inputs = [inputs]
        elif isinstance(inputs, list):
            if not (isinstance(inputs[0], str) and len(inputs[0].strip()) > 0):
                raise TypeError(
                    "Invalid inputs, input text should be list of str, and first element of list should not be empty text."
                    .format(type(inputs[0])))
        else:
            raise TypeError(
                "Invalid inputs, input text should be str or list of str, but type of {} found!"
                .format(type(inputs)))
        return inputs

    def _preprocess(self, inputs, padding=True, add_special_tokens=True):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        examples = []
        filter_inputs = []
        for input_data in inputs:
            if not (isinstance(input_data, str)
                    and len(input_data.strip()) > 0):
                continue
            filter_inputs.append(input_data)
            encoded_inputs = self._tokenizer(text=input_data,
                                             max_seq_len=self._max_seq_len)
            ids = encoded_inputs["input_ids"]
            segment_ids = encoded_inputs["token_type_ids"]
            examples.append((ids, segment_ids))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self._tokenizer.pad_token_id),  # input ids
            Pad(axis=0, pad_val=self._tokenizer.pad_token_type_id
                ),  # token type ids
        ): [data for data in fn(samples)]
        batches = [
            examples[idx:idx + self._batch_size]
            for idx in range(0, len(examples), self._batch_size)
        ]
        outputs = {}
        outputs['text'] = filter_inputs
        outputs['data_loader'] = batches
        self._batchify_fn = batchify_fn
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function. 
        """
        labels = []
        confidences = []
        with static_mode_guard():
            for batch in inputs['data_loader']:
                input_ids, segment_ids = self._batchify_fn(batch)
                self.input_handles[0].copy_from_cpu(input_ids)
                self.input_handles[1].copy_from_cpu(segment_ids)
                self.predictor.run()
                probs = self.output_handle[0].copy_to_cpu()

                for prob in probs:
                    label = []
                    confidence = []
                    prob = 1.0 / (1.0 + np.exp(-prob))
                    for i, p in enumerate(prob):
                        if p > 0.5:
                            label.append(self._label_map[i])
                            confidence.append(round(p, 5))
                    labels.append(label)
                    confidences.append(confidence)
        inputs['labels'] = labels
        inputs['confidence'] = confidences

        return inputs

    def _postprocess(self, inputs):
        """
        This function will convert the model output to raw text.
        """
        final_results = []
        for i in range(len(inputs['text'])):
            result = {}
            result['text'] = inputs['text'][i]
            result['label'] = inputs['labels'][i]
            result['confidence'] = inputs['confidence'][i]
            final_results.append(result)
        return final_results
