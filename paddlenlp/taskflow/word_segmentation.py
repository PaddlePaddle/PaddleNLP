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

usage = r"""
           from paddlenlp import Taskflow 

           seg = Taskflow("word_segmentation")
           seg("第十四届全运会在西安举办")
           '''
           ['第十四届', '全运会', '在', '西安', '举办']
           '''

           seg(["第十四届全运会在西安举办", "三亚是一个美丽的城市"])
           '''
           [['第十四届', '全运会', '在', '西安', '举办'], ['三亚', '是', '一个', '美丽', '的', '城市']]
           '''
         """


class WordSegmentationTask(LacTask):
    """
    Segement the sentences to the words. 
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)

    def _auto_joiner(self, results, input_mapping):
        concat_results = []
        single_results = []
        for k, vs in input_mapping.items():
            for v in vs:
                if len(single_results) == 0:
                    single_results = results[v]
                else:
                    single_results.extend(results[v])
            concat_results.append(single_results)
            single_results = []
        concat_results = concat_results if len(
            concat_results) > 1 else concat_results[0]
        return concat_results

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
            if self._custom:
                self._custom.parse_customization(sent, tags)
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
            final_results.append(sent_out)
        final_results = self._auto_joiner(final_results, self.input_mapping)
        return final_results
