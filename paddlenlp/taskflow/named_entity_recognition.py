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
import csv
import itertools

from .utils import download_file
from .utils import TermTree
from .knowledge_mining import WordTagTask

usage = r"""
          from paddlenlp import Taskflow 

          ner = Taskflow("ner")
          ner("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
          '''
          [('《', 'w'), ('孤女', '作品类_实体'), ('》', 'w'), ('是', '肯定词'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('的', '助词'), ('小说', '作品类_概念'), ('，', 'w'), ('作者', '人物类_概念'), ('是', '肯定词'), ('余兼羽', '人物类_实体')]
          '''

          ner = Taskflow("ner")
          ner(["热梅茶是一道以梅子为主要原料制作的茶饮",
               "《孤女》是2010年九州出版社出版的小说，作者是余兼羽"])
          '''
          [[('热梅茶', '饮食类_饮品'), ('是', '肯定词'), ('一道', '数量词'), ('以', '介词'), ('梅子', '饮食类'), ('为', '肯定词'), ('主要原料', '物体类'), ('制作', '场景事件'), ('的', '助词'), ('茶饮', '饮食类_饮品')], [('《', 'w'), ('孤女', '作品类_实体'), ('》', 'w'), ('是', '肯定词'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('的', '助词'), ('小说', '作品类_概念'), ('，', 'w'), ('作者', '人物类_概念'), ('是', '肯定词'), ('余兼羽', '人物类_实体')]]
          '''
          """


class NERTask(WordTagTask):
    """
    This the NER(Named Entity Recognition) task that convert the raw text to entities. And the task with the `wordtag` 
    model will link the more meesage with the entity.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 

    """

    def __init__(self, model, task, **kwargs):
        super().__init__(model=model, task=task, **kwargs)

    def _decode(self, batch_texts, batch_pred_tags):
        batch_results = []
        for i, pred_tags in enumerate(batch_pred_tags):
            pred_words, pred_word = [], []
            text = batch_texts[i]
            for j, tag in enumerate(pred_tags[self.summary_num:-1]):
                if j >= len(text):
                    break
                pred_label = self._index_to_tags[tag]
                if pred_label.find("-") != -1:
                    _, label = pred_label.split("-")
                else:
                    label = pred_label
                if pred_label.startswith("S") or pred_label.startswith("O"):
                    pred_words.append({"item": text[j], "wordtag_label": label})
                else:
                    pred_word.append(text[j])
                    if pred_label.startswith("E"):
                        pred_words.append({
                            "item": "".join(pred_word),
                            "wordtag_label": label
                        })
                        del pred_word[:]

            result = {"text": text, "items": pred_words}
            batch_results.append(result)
        return batch_results

    def _concat_short_text_reuslts(self, input_texts, results):
        """
        Concat the model output of short texts to the total result of long text.
        """
        long_text_lens = [len(text) for text in input_texts]
        concat_results = []
        single_results = {}
        count = 0
        for text in input_texts:
            text_len = len(text)
            while True:
                if len(single_results) == 0 or len(single_results[
                        "text"]) < text_len:
                    if len(single_results) == 0:
                        single_results = copy.deepcopy(results[count])
                    else:
                        single_results["text"] += results[count]["text"]
                        single_results["items"].extend(results[count]["items"])
                    count += 1
                elif len(single_results["text"]) == text_len:
                    concat_results.append(single_results)
                    single_results = {}
                    break
                else:
                    raise Exception(
                        "The length of input text and raw text is not equal.")
        simple_results = []
        for result in concat_results:
            simple_result = []
            if 'items' in result:
                for item in result['items']:
                    simple_result.append((item['item'], item['wordtag_label']))
            simple_results.append(simple_result)
        simple_results = simple_results[0] if len(
            simple_results) == 1 else simple_results
        return simple_results

    def _postprocess(self, inputs):
        """
        The model output is the tag ids, this function will convert the model output to raw text.
        """
        results = self._decode(inputs['short_input_texts'],
                               inputs['all_pred_tags'])
        results = self._concat_short_text_reuslts(inputs['inputs'], results)
        return results
