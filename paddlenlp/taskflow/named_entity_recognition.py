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
from .lexical_analysis import LacTask
from .utils import Customization

POS_LABEL_WORDTAG = [
    "介词",
    "介词_方位介词",
    "助词",
    "代词",
    "连词",
    "副词",
    "疑问词",
    "肯定词",
    "否定词",
    "数量词",
    "叹词",
    "拟声词",
    "修饰词",
    "外语单词",
    "英语单词",
    "汉语拼音",
    "词汇用语",
    "w",
]

POS_LABEL_LAC = ["n", "f", "s", "t", "v", "vd", "vn", "a", "ad", "an", "d", "m", "q", "r", "p", "c", "u", "xc", "w"]

usage = r"""
          from paddlenlp import Taskflow 

          # WordTag精确模式
          ner = Taskflow("ner")
          ner("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
          '''
          [('《', 'w'), ('孤女', '作品类_实体'), ('》', 'w'), ('是', '肯定词'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('的', '助词'), ('小说', '作品类_概念'), ('，', 'w'), ('作者', '人物类_概念'), ('是', '肯定词'), ('余兼羽', '人物类_实体')]
          '''

          ner(["热梅茶是一道以梅子为主要原料制作的茶饮", "《孤女》是2010年九州出版社出版的小说，作者是余兼羽"])
          '''
          [[('热梅茶', '饮食类_饮品'), ('是', '肯定词'), ('一道', '数量词'), ('以', '介词'), ('梅子', '饮食类'), ('为', '肯定词'), ('主要原料', '物体类'), ('制作', '场景事件'), ('的', '助词'), ('茶饮', '饮食类_饮品')], [('《', 'w'), ('孤女', '作品类_实体'), ('》', 'w'), ('是', '肯定词'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('的', '助词'), ('小说', '作品类_概念'), ('，', 'w'), ('作者', '人物类_概念'), ('是', '肯定词'), ('余兼羽', '人物类_实体')]]
          '''

          # 只返回实体/概念词
          ner = Taskflow("ner", entity_only=True)
          ner("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
          '''
          [('孤女', '作品类_实体'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('小说', '作品类_概念'), ('作者', '人物类_概念'), ('余兼羽', '人物类_实体')]
          '''

          # 使用快速模式，只返回实体词
          ner = Taskflow("ner", mode="fast", entity_only=True)
          ner("三亚是一个美丽的城市")
          '''
          [('三亚', 'LOC')]
          '''
          """


class NERWordTagTask(WordTagTask):
    """
    This the NER(Named Entity Recognition) task that convert the raw text to entities. And the task with the `wordtag`
    model will link the more meesage with the entity.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.

    """

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "model_config": "model_config.json",
        "tags": "tags.txt",
    }
    resource_files_urls = {
        "wordtag": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/knowledge_mining/wordtag_v1.3/model_state.pdparams",
                "32b4ed27e99d6b2c76e50a24d1a9fd56",
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/knowledge_mining/wordtag_v1.1/model_config.json",
                "9dcbd5d6f67792b2a2be058799a144ea",
            ],
            "tags": [
                "https://bj.bcebos.com/paddlenlp/taskflow/knowledge_mining/wordtag_v1.1/tags.txt",
                "f33feedd01d478b03bac81be19b48d00",
            ],
        }
    }

    def __init__(self, model, task, entity_only=False, **kwargs):
        super().__init__(model="wordtag", task=task, **kwargs)
        self.entity_only = entity_only
        if self._user_dict:
            self._custom = Customization()
            self._custom.load_customization(self._user_dict)
        else:
            self._custom = None

    def _decode(self, batch_texts, batch_pred_tags):
        batch_results = []
        for sent_index in range(len(batch_texts)):
            sent = batch_texts[sent_index]
            indexes = batch_pred_tags[sent_index][self.summary_num : len(sent) + self.summary_num]
            tags = [self._index_to_tags[index] for index in indexes]
            if self._custom:
                self._custom.parse_customization(sent, tags, prefix=True)
            sent_out = []
            tags_out = []
            partial_word = ""
            for ind, tag in enumerate(tags):
                if partial_word == "":
                    partial_word = sent[ind]
                    tags_out.append(tag.split("-")[-1])
                    continue
                if tag.startswith("B") or tag.startswith("S") or tag.startswith("O"):
                    sent_out.append(partial_word)
                    tags_out.append(tag.split("-")[-1])
                    partial_word = sent[ind]
                    continue
                partial_word += sent[ind]

            if len(sent_out) < len(tags_out):
                sent_out.append(partial_word)

            pred_words = []
            for s, t in zip(sent_out, tags_out):
                pred_words.append({"item": s, "wordtag_label": t})

            result = {"text": sent, "items": pred_words}
            batch_results.append(result)
        return batch_results

    def _simplify_result(self, results):
        simple_results = []
        for result in results:
            simple_result = []
            if "items" in result:
                for item in result["items"]:
                    if self.entity_only and item["wordtag_label"] in POS_LABEL_WORDTAG:
                        continue
                    simple_result.append((item["item"], item["wordtag_label"]))
            simple_results.append(simple_result)
        simple_results = simple_results[0] if len(simple_results) == 1 else simple_results
        return simple_results

    def _postprocess(self, inputs):
        """
        The model output is the tag ids, this function will convert the model output to raw text.
        """
        results = self._decode(inputs["short_input_texts"], inputs["all_pred_tags"])
        results = self._auto_joiner(results, self.input_mapping, is_dict=True)
        results = self._simplify_result(results)
        return results


class NERLACTask(LacTask):
    """
    Part-of-speech tagging task for the raw text.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    def __init__(self, model, task, entity_only=False, **kwargs):
        super().__init__(task=task, model="lac", **kwargs)
        self.entity_only = entity_only

    def _postprocess(self, inputs):
        """
        The model output is the tag ids, this function will convert the model output to raw text.
        """
        batch_out = []
        lengths = inputs["lens"]
        preds = inputs["result"]
        sents = inputs["text"]
        final_results = []
        for sent_index in range(len(lengths)):
            single_result = {}
            tags = [self._id2tag_dict[str(index)] for index in preds[sent_index][: lengths[sent_index]]]
            sent = sents[sent_index]
            if self._custom:
                self._custom.parse_customization(sent, tags)
            sent_out = []
            tags_out = []
            parital_word = ""
            for ind, tag in enumerate(tags):
                if parital_word == "":
                    parital_word = sent[ind]
                    tags_out.append(tag.split("-")[0])
                    continue
                if tag.endswith("-B") or (tag == "O" and tags[ind - 1] != "O"):
                    sent_out.append(parital_word)
                    tags_out.append(tag.split("-")[0])
                    parital_word = sent[ind]
                    continue
                parital_word += sent[ind]

            if len(sent_out) < len(tags_out):
                sent_out.append(parital_word)

            result = []
            for s, t in zip(sent_out, tags_out):
                if self.entity_only and t in POS_LABEL_LAC:
                    continue
                result.append((s, t))
            final_results.append(result)
        final_results = self._auto_joiner(final_results, self.input_mapping)
        final_results = final_results if len(final_results) > 1 else final_results[0]
        return final_results
