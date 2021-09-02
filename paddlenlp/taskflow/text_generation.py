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

import glob
import json
import math
import os
import copy
import itertools

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..transformers import GPTForGreedyGeneration
from ..transformers import GPTChineseTokenizer, GPTTokenizer
from ..datasets import load_dataset
from ..data import Stack, Pad, Tuple
from .utils import download_file, add_docstrings, static_mode_guard, dygraph_mode_guard
from .task import Task

usage = r"""
           from paddlenlp import Taskflow 

           question = Taskflow("text_generation")
           question("中国的国土面积有多大？")
           '''
           [{'text': '中国的国土面积有多大？', 'answer': '960万平方公里。'}]
           '''
           poetry  = Taskflow("text_generation",  generation_task="poetry")
           poetry("林密不见人")
           '''
           [{'text': '林密不见人', 'answer': ',但闻人语响。'}]
           '''
           poetry(["林密不见人", "举头邀明月"])
           '''
           [{'text': '林密不见人', 'answer': ',但闻人语响。'}, {'text': '举头邀明月', 'answer': ',低头思故乡。'}]
           '''
         """

URLS = {
    "gpt-cpm-large-cn": [
        "https://paddlenlp.bj.bcebos.com/taskflow/text_generation/gpt-cpm/gpt-cpm-large-cn_params.tar",
        None
    ],
}


class TextGenerationTask(Task):
    """
    The text generation model to predict the question or chinese  poetry. 
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._static_mode = True
        self._usage = usage
        if self._static_mode:
            download_file(self._task_path,
                          "static" + os.path.sep + "inference.pdiparams",
                          URLS[self.model][0], URLS[self.model][1])
            self._get_inference_model()
        else:
            self._construct_model(model)
        self._construct_tokenizer(model)

    def _construct_input_spec(self):
        """
       Construct the input spec for the convert dygraph model to static model.
       """
        self._input_spec = [
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name='token_ids')
        ]

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        paddle.nn.initializer.set_global_initializer(None, None)
        model_instance = GPTForGreedyGeneration.from_pretrained(
            self.model, max_predict_len=32)
        # Load the model parameter for the predict
        model_instance.eval()
        self._model = model_instance

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        if self.model == "gpt-cpm-large-cn":
            tokenizer_instance = GPTChineseTokenizer.from_pretrained(model)
        else:
            tokenizer_instance = GPTTokenizer.from_pretrained(model)

        self._tokenizer = tokenizer_instance

    def _preprocess(self, inputs, padding=True, add_special_tokens=True):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = inputs[0]
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, str) and not isinstance(inputs, list):
            raise TypeError(
                "Invalid inputs, input text should be str or list of str, {type(inputs)} found!"
            )
        # Get the config from the kwargs
        batch_size = self.kwargs[
            'batch_size'] if 'batch_size' in self.kwargs else 1
        num_workers = self.kwargs[
            'num_workers'] if 'num_workers' in self.kwargs else 0
        lazy_load = self.kwargs[
            'lazy_load'] if 'lazy_load' in self.kwargs else False
        generation_task = self.kwargs[
            'generation_task'] if 'generation_task' in self.kwargs else 'question'
        max_seq_len = 32

        def select_few_shot_input(model_name, generation_task):
            pre_input = ""
            if generation_task not in ['question', 'poetry']:
                raise ValueError(
                    "The generation task must be question or poetry")
            if model_name == "gpt-cpm-large-cn":
                if generation_task == "question":
                    pre_input = '问题：中国的首都是哪里？答案：北京。\n问题：{} 答案：'
                else:
                    pre_input = '默写古诗: 大漠孤烟直，长河落日圆。\n{}'
            return pre_input

        pre_input = select_few_shot_input(self.model, generation_task)

        infer_data = []

        def read(inputs):
            for input_text in inputs:
                few_shot_input = pre_input.format(input_text)
                ids = self._tokenizer(few_shot_input)["input_ids"]
                yield ids, len(ids)

        infer_ds = load_dataset(read, inputs=inputs, lazy=lazy_load)
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=0, dtype="int64"),
            Stack(dtype='int64'),  # seq_len
        ): fn(samples)
        infer_data_loader = paddle.io.DataLoader(
            infer_ds,
            collate_fn=batchify_fn,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=False,
            return_list=True)
        outputs = {}
        outputs['text'] = inputs
        outputs['data_loader'] = infer_data_loader
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function. 
        """
        results = []
        lens = []
        if not self._static_mode:
            with dygraph_mode_guard():
                for batch in inputs['data_loader']:
                    input_ids, seq_len = batch
                    out = self._model(input_ids)
                    out = [int(x) for x in out.numpy().reshape([-1])]
                    results.append(out)
                    lens.extend(seq_len.numpy().tolist())
        else:
            with static_mode_guard():
                for batch in inputs['data_loader']:
                    data_dict = {}
                    for name, value in zip(self._static_feed_names, batch):
                        data_dict[name] = value
                    tags_ids = self._exe.run(
                        self._static_program,
                        feed=data_dict,
                        fetch_list=self._static_fetch_targets)
                    result = tags_ids[0].tolist()
                    results.extend(result)
                    lens.extend(np.array(batch[1]).tolist())
        inputs['results'] = results
        inputs['lens'] = lens
        return inputs

    def _postprocess(self, inputs):
        """
        The model output is allways the logits and pros, this function will convert the model output to raw text.
        """
        batch_out = []
        preds = inputs['results']
        for index in range(0, len(preds)):
            seq_len = inputs['lens'][index]
            single_result = {}
            single_result['text'] = inputs['text'][index]
            single_result['answer'] = self._tokenizer.convert_ids_to_string(
                preds[index][seq_len:-1])
            batch_out.append(single_result)
        return batch_out
