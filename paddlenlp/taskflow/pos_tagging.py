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
from .utils import download_file
from .lexical_analysis import load_vocab, LacTask

URLS = {
    "pos_tagging_params": [
        "https://paddlenlp.bj.bcebos.com/taskflow/lexical_analysis/lac/lac_params.tar.gz",
        'ee9a3eaba5f74105410410e3c5b28fbc'
    ],
}

usage = r"""
           from paddlenlp import Taskflow 

           pos = Taskflow("pos_tagging")
           pos("第十四届全运会在西安举办")
           '''
           [('第十四届', 'm'), ('全运会', 'nz'), ('在', 'p'), ('西安', 'LOC'), ('举办', 'v')]
           '''

           pos(["第十四届全运会在西安举办", "三亚是一个美丽的城市"])
           '''
           [[('第十四届', 'm'), ('全运会', 'nz'), ('在', 'p'), ('西安', 'LOC'), ('举办', 'v')], [('三亚', 'LOC'), ('是', 'v'), ('一个', 'm'), ('美丽', 'a'), ('的', 'u'), ('城市', 'n')]]
           '''
         """


class POSTaggingTask(LacTask):
    """
    Part-of-speech tagging task for the raw text.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._static_mode = False
        self._usage = usage
        word_dict_path = download_file(
            self._task_path, "lac_params" + os.path.sep + "word.dic",
            URLS['pos_tagging_params'][0], URLS['pos_tagging_params'][1],
            'pos_tagging')
        tag_dict_path = download_file(
            self._task_path, "lac_params" + os.path.sep + "tag.dic",
            URLS['pos_tagging_params'][0], URLS['pos_tagging_params'][1])
        q2b_dict_path = download_file(
            self._task_path, "lac_params" + os.path.sep + "q2b.dic",
            URLS['pos_tagging_params'][0], URLS['pos_tagging_params'][1])
        self._word_vocab = load_vocab(word_dict_path)
        self._tag_vocab = load_vocab(tag_dict_path)
        self._q2b_vocab = load_vocab(q2b_dict_path)
        self._id2word_dict = dict(
            zip(self._word_vocab.values(), self._word_vocab.keys()))
        self._id2tag_dict = dict(
            zip(self._tag_vocab.values(), self._tag_vocab.keys()))
        if self._static_mode:
            self._get_inference_model()
        else:
            self._construct_model(model)

    def _postprocess(self, inputs):
        """
        The model output is the tag ids, this function will convert the model output to raw text.
        """
        batch_out = []
        lengths = inputs['lens']
        preds = inputs['result']
        sents = inputs['text']
        final_results = []
        for sent_index in range(len(lengths)):
            single_result = {}
            tags = [
                self._id2tag_dict[str(index)]
                for index in preds[sent_index][:lengths[sent_index]]
            ]
            sent = sents[sent_index]
            sent_out = []
            tags_out = []
            parital_word = ""
            for ind, tag in enumerate(tags):
                if parital_word == "":
                    parital_word = sent[ind]
                    tags_out.append(tag.split('-')[0])
                    continue
                if tag.endswith("-B") or (tag == "O" and tags[ind - 1] != "O"):
                    sent_out.append(parital_word)
                    tags_out.append(tag.split('-')[0])
                    parital_word = sent[ind]
                    continue
                parital_word += sent[ind]

            if len(sent_out) < len(tags_out):
                sent_out.append(parital_word)
            result = list(zip(sent_out, tags_out))
            final_results.append(result)
        final_results = final_results if len(
            final_results) > 1 else final_results[0]
        return final_results
