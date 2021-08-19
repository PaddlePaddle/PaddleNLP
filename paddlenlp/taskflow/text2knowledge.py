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

import paddle
import paddle.nn as nn
from ..datasets import MapDataset, load_dataset
from ..data import Stack, Pad, Tuple
from ..transformers import ErnieCtmWordtagModel, ErnieCtmTokenizer
from .utils import download_file, add_docstrings
from .task import Task

LABEL_TO_SCHEMA = {
    "人物类_实体": ["人物|E", "虚拟角色|E", "演艺团体|E"],
    "人物类_概念": ["人物|C", "虚拟角色|C"],
    "作品类_实体": ["作品与出版物|E"],
    "作品类_概念": ["作品与出版物|C", "文化类"],
    "组织机构类": ["组织机构"],
    "组织机构类_企事业单位": ["企事业单位", "品牌", "组织机构"],
    "组织机构类_医疗卫生机构": ["医疗卫生机构", "组织机构"],
    "组织机构类_国家机关": ["国家机关", "组织机构"],
    "组织机构类_体育组织机构": ["体育组织机构", "组织机构"],
    "组织机构类_教育组织机构": ["教育组织机构", "组织机构"],
    "组织机构类_军事组织机构": ["军事组织机构", "组织机构"],
    "物体类": ["物体与物品", "品牌", "虚拟物品", "虚拟物品"],
    "物体类_兵器": ["兵器"],
    "物体类_化学物质": ["物体与物品", "化学术语"],
    "其他角色类": ["角色"],
    "文化类": ["文化", "作品与出版物|C", "体育运动项目", "语言文字"],
    "文化类_语言文字": ["语言学术语"],
    "文化类_奖项赛事活动": ["奖项赛事活动", "特殊日", "事件"],
    "文化类_制度政策协议": ["制度政策协议", "法律法规"],
    "文化类_姓氏与人名": ["姓氏与人名"],
    "生物类": ["生物"],
    "生物类_植物": ["植物", "生物"],
    "生物类_动物": ["动物", "生物"],
    "品牌名": ["品牌", "企事业单位"],
    "场所类": ["区域场所", "居民服务机构", "医疗卫生机构"],
    "场所类_交通场所": ["交通场所", "设施"],
    "位置方位": ["位置方位"],
    "世界地区类": ["世界地区", "区域场所", "政权朝代"],
    "饮食类": ["饮食", "生物类", "药物"],
    "饮食类_菜品": ["饮食"],
    "饮食类_饮品": ["饮食"],
    "药物类": ["药物", "生物类"],
    "药物类_中药": ["药物", "生物类"],
    "医学术语类": ["医药学术语"],
    "术语类_生物体": ["生物学术语"],
    "疾病损伤类": ["疾病损伤", "动物疾病", "医药学术语"],
    "疾病损伤类_植物病虫害": ["植物病虫害", "医药学术语"],
    "宇宙类": ["天文学术语"],
    "事件类": ["事件", "奖项赛事活动"],
    "时间类": ["时间阶段", "政权朝代"],
    "术语类": ["术语"],
    "术语类_符号指标类": ["编码符号指标", "术语"],
    "信息资料": ["生活用语"],
    "链接地址": ["生活用语"],
    "个性特征": ["个性特点", "生活用语"],
    "感官特征": ["生活用语"],
    "场景事件": ["场景事件", "情绪", "态度", "个性特点"],
    "介词": ["介词"],
    "介词_方位介词": ["介词"],
    "助词": ["助词"],
    "代词": ["代词"],
    "连词": ["连词"],
    "副词": ["副词"],
    "疑问词": ["疑问词"],
    "肯定词": ["肯定否定词"],
    "否定词": ["肯定否定词"],
    "数量词": ["数量词", "量词"],
    "叹词": ["叹词"],
    "拟声词": ["拟声词"],
    "修饰词": ["修饰词", "生活用语"],
    "外语单词": ["日文假名", "词汇用语"],
    "汉语拼音": ["汉语拼音"],
}

URLS = {
    "TermTree.V1.0":
    ["https://kg-concept.bj.bcebos.com/TermTree/TermTree.V1.0.tar.gz", None],
    "termtree_type": [
        "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_ctm/termtree_type.csv",
        None
    ],
    "termtree_tags_pos": [
        "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_ctm/termtree_tags_pos.txt",
        None
    ],
}

usage = r"""
          from paddlenlp.taskflow import TaskFlow 

          task = TaskFlow("text2knowledge")
          task("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
          '''
          [{'text': '《孤女》是2010年九州出版社出版的小说，作者是余兼羽', 'items': [{'item': '《', 'offset': 0, 'wordtag_label': 'w', 'length': 1}, {'item': '孤女', 'offset': 1, 'wordtag_label': '作品类_实体', 'length': 2}, {'item': '》', 'offset': 3, 'wordtag_label': 'w', 'length': 1}, {'item': '是', 'offset': 4, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '2010年', 'offset': 5, 'wordtag_label': '时间类', 'length': 5, 'termid': '时间阶段_cb_2010年'}, {'item': '九州出版社', 'offset': 10, 'wordtag_label': '组织机构类', 'length': 5, 'termid': '组织机构_eb_九州出版社'}, {'item': '出版', 'offset': 15, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_出版'}, {'item': '的', 'offset': 17, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '小说', 'offset': 18, 'wordtag_label': '作品类_概念', 'length': 2, 'termid': '小说_cb_小说'}, {'item': '，', 'offset': 20, 'wordtag_label': 'w', 'length': 1}, {'item': '作者', 'offset': 21, 'wordtag_label': '人物类_概念', 'length': 2, 'termid': '人物_cb_作者'}, {'item': '是', 'offset': 23, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '余兼羽', 'offset': 24, 'wordtag_label': '人物类_实体', 'length': 3}]}]
          '''

          task = TaskFlow("text2knowledge", lazy_load=True)
          task("热梅茶是一道以梅子为主要原料制作的茶饮")
          '''
          [{'text': '热梅茶是一道以梅子为主要原料制作的茶饮', 'items': [{'item': '热梅茶', 'offset': 0, 'wordtag_label': '饮食类_饮品', 'length': 3}, {'item': '是', 'offset': 3, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '一道', 'offset': 4, 'wordtag_label': '数量词', 'length': 2}, {'item': '以', 'offset': 6, 'wordtag_label': '介词', 'length': 1, 'termid': '介词_cb_以'}, {'item': '梅子', 'offset': 7, 'wordtag_label': '饮食类', 'length': 2, 'termid': '饮食_cb_梅'}, {'item': '为', 'offset': 9, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_为'}, {'item': '主要原料', 'offset': 10, 'wordtag_label': '物体类', 'length': 4, 'termid': '物品_cb_主要原料'}, {'item': '制作', 'offset': 14, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_制作'}, {'item': '的', 'offset': 16, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '茶饮', 'offset': 17, 'wordtag_label': '饮食类_饮品', 'length': 2, 'termid': '饮品_cb_茶饮'}]}]
          '''

          task = TaskFlow("text2knowledge", batch_size=2)
          task(["热梅茶是一道以梅子为主要原料制作的茶饮",
                "《孤女》是2010年九州出版社出版的小说，作者是余兼羽",
                "中山中环广场，位于广东省中山市东区，地址是东区兴政路1号",
                "宫之王是一款打发休闲时光的迷宫游戏"])
          '''
          [{'text': '热梅茶是一道以梅子为主要原料制作的茶饮', 'items': [{'item': '热梅茶', 'offset': 0, 'wordtag_label': '饮食类_饮品', 'length': 3}, {'item': '是', 'offset': 3, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '一道', 'offset': 4, 'wordtag_label': '数量词', 'length': 2}, {'item': '以', 'offset': 6, 'wordtag_label': '介词', 'length': 1, 'termid': '介词_cb_以'}, {'item': '梅子', 'offset': 7, 'wordtag_label': '饮食类', 'length': 2, 'termid': '饮食_cb_梅'}, {'item': '为', 'offset': 9, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_为'}, {'item': '主要原料', 'offset': 10, 'wordtag_label': '物体类', 'length': 4, 'termid': '物品_cb_主要原料'}, {'item': '制作', 'offset': 14, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_制作'}, {'item': '的', 'offset': 16, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '茶饮', 'offset': 17, 'wordtag_label': '饮食类_饮品', 'length': 2, 'termid': '饮品_cb_茶饮'}]}, {'text': '《孤女》是2010年九州出版社出版的小说，作者是余兼羽', 'items': [{'item': '《', 'offset': 0, 'wordtag_label': 'w', 'length': 1}, {'item': '孤女', 'offset': 1, 'wordtag_label': '作品类_实体', 'length': 2}, {'item': '》', 'offset': 3, 'wordtag_label': 'w', 'length': 1}, {'item': '是', 'offset': 4, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '2010年', 'offset': 5, 'wordtag_label': '时间类', 'length': 5, 'termid': '时间阶段_cb_2010年'}, {'item': '九州出版社', 'offset': 10, 'wordtag_label': '组织机构类', 'length': 5, 'termid': '组织机构_eb_九州出版社'}, {'item': '出版', 'offset': 15, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_出版'}, {'item': '的', 'offset': 17, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '小说', 'offset': 18, 'wordtag_label': '作品类_概念', 'length': 2, 'termid': '小说_cb_小说'}, {'item': '，', 'offset': 20, 'wordtag_label': 'w', 'length': 1}, {'item': '作者', 'offset': 21, 'wordtag_label': '人物类_概念', 'length': 2, 'termid': '人物_cb_作者'}, {'item': '是', 'offset': 23, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '余兼羽', 'offset': 24, 'wordtag_label': '人物类_实体', 'length': 3}]}, {'text': '中山中环广场，位于广东省中山市东区，地址是东区兴政路1号', 'items': [{'item': '中山中环广场', 'offset': 0, 'wordtag_label': '场所类', 'length': 6}, {'item': '，', 'offset': 6, 'wordtag_label': 'w', 'length': 1}, {'item': '位于', 'offset': 7, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_位于'}, {'item': '广东省', 'offset': 9, 'wordtag_label': '世界地区类', 'length': 3, 'termid': '中国地区_cb_广东省'}, {'item': '中山市东', 'offset': 12, 'wordtag_label': '世界地区类', 'length': 4}, {'item': '区', 'offset': 16, 'wordtag_label': '词汇用语', 'length': 1}, {'item': '，', 'offset': 17, 'wordtag_label': 'w', 'length': 1}, {'item': '地址', 'offset': 18, 'wordtag_label': '场所类', 'length': 2, 'termid': '区域场所_cb_地址'}, {'item': '是', 'offset': 20, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '东区', 'offset': 21, 'wordtag_label': '位置方位', 'length': 2, 'termid': '位置方位_cb_东区'}, {'item': '兴政路1号', 'offset': 23, 'wordtag_label': '世界地区类', 'length': 5}]}, {'text': '宫之王是一款打发休闲时光的迷宫游戏', 'items': [{'item': '宫之王', 'offset': 0, 'wordtag_label': '人物类_实体', 'length': 3}, {'item': '是', 'offset': 3, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '一款', 'offset': 4, 'wordtag_label': '数量词', 'length': 2}, {'item': '打发', 'offset': 6, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_打发'}, {'item': '休闲', 'offset': 8, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_休闲'}, {'item': '时光', 'offset': 10, 'wordtag_label': '时间类', 'length': 2, 'termid': '时间阶段_cb_时光'}, {'item': '的', 'offset': 12, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '迷宫游戏', 'offset': 13, 'wordtag_label': '作品类_概念', 'length': 4}]}]
          '''
          """


@add_docstrings(usage)
class WordTagTask(Task):
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
        term_schema_path = download_file(
            self.model, "termtree_type.csv", URLS['termtree_type'][0],
            URLS['termtree_type'][1], "text2knowledge")
        term_data_path = download_file(self.model, "TermTree.V1.0",
                                       URLS['TermTree.V1.0'][0],
                                       URLS['TermTree.V1.0'][1])
        tag_path = download_file(self.model, "termtree_tags_pos.txt",
                                 URLS['termtree_tags_pos'][0],
                                 URLS['termtree_tags_pos'][1])
        self._tags_to_index, self._index_to_tags = self._load_labels(tag_path)

        if term_schema_path is not None:
            self._term_schema = self._load_schema(term_schema_path)
        if term_data_path is not None:
            self._term_dict = self._load_term_tree_data(term_data_path)
        if term_data_path is not None and term_schema_path is not None:
            self._linking = True
        else:
            self._linking = False
        self._tokenizer = self._construct_tokenizer(model)
        self._model = self._construct_model(model)
        self._summary_num = self._model.ernie_ctm.content_summary_index + 1
        self._usage = usage

    def _download_termtree(self, filename):
        default_root = os.path.join(MODEL_HOME, 'ernie-ctm')
        fullname = os.path.join(default_root, filename)
        url = URLS[filename]
        if not os.path.exists(fullname):
            get_path_from_url(url, default_root)
        return fullname

    @property
    def summary_num(self):
        """
        Number of model summary token
        """
        return self._summary_num

    @property
    def linking(self):
        """
        Whether to do term linking.
        """
        return self._linking

    @staticmethod
    def _load_labels(tag_path):
        tags_to_idx = {}
        i = 0
        with open(tag_path, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                tags_to_idx[line] = i
                i += 1
        idx_to_tags = dict(zip(*(tags_to_idx.values(), tags_to_idx.keys())))
        return tags_to_idx, idx_to_tags

    @staticmethod
    def _load_schema(schema_path):
        schema = {}
        with open(schema_path, encoding="utf8") as f:
            reader = csv.reader(f)
            first_line = True
            for line in reader:
                if first_line:
                    first_line = False
                    continue
                items = line[0].split("\t")
                if len(items[0]):
                    schema[items[0]] = "root"
                if len(items[1]):
                    schema[items[1]] = items[0]
                if len(items[2]):
                    schema[items[2]] = items[1]
        return schema

    @staticmethod
    def _load_term_tree_data(term_tree_name_or_path):
        if os.path.isdir(term_tree_name_or_path):
            fn_list = glob.glob(f"{term_tree_name_or_path}/*", recursive=True)
        else:
            fn_list = [term_tree_name_or_path]
        term_dict = {}
        for fn in fn_list:
            with open(fn, encoding="utf-8") as fp:
                for line in fp:
                    data = json.loads(line)
                    if data["term"] not in term_dict:
                        term_dict[data["term"]] = {}
                    if data["termtype"] not in term_dict[data["term"]]:
                        term_dict[data["term"]][data["termtype"]] = []
                    term_dict[data["term"]][data["termtype"]].append(data[
                        "termid"])
                    for alia in data["alias"]:
                        if alia not in term_dict:
                            term_dict[alia] = {}
                        if data["termtype"] not in term_dict[alia]:
                            term_dict[alia][data["termtype"]] = []
                        term_dict[alia][data["termtype"]].append(data["termid"])
                    for alia in data["alias_ext"]:
                        if alia not in term_dict:
                            term_dict[alia] = {}
                        if data["termtype"] not in term_dict[alia]:
                            term_dict[alia][data["termtype"]] = []
                        term_dict[alia][data["termtype"]].append(data["termid"])
        return term_dict

    def _split_long_text_input(self, input_texts, max_text_len):
        """
        Split the long text to list of short text, the max_seq_len of input text is 512,
        if the text length greater than 512, will this function that spliting the long text.
        """
        short_input_texts = []
        for text in input_texts:
            if len(text) <= max_text_len:
                short_input_texts.append(text)
            else:
                lens = len(text)
                temp_text_list = text.split("？。！")
                temp_text_list = [
                    temp_text for temp_text in temp_text_list
                    if len(temp_text) > 0
                ]
                if len(temp_text_list) <= 1:
                    temp_text_list = [
                        text[i:i + max_text_len]
                        for i in range(0, len(text), max_text_len)
                    ]
                    short_input_texts.extend(temp_text_list)
                else:
                    list_len = len(temp_text_list)
                    start = 0
                    end = 0
                    for i in range(0, list_len):
                        if len(temp_text_list[i]) + 1 >= max_text_len:
                            if start != end:
                                short_input_texts.extend(
                                    self._split_long_text_input(
                                        [text[start:end]], max_text_len))
                            short_input_texts.extend(
                                self._split_long_text_input([
                                    text[end:end + len(temp_text_list[i]) + 1]
                                ], max_text_len))
                            start = end + len(temp_text_list[i]) + 1
                            end = start
                        else:
                            if start + len(temp_text_list[
                                    i]) + 1 > max_text_len:
                                short_input_texts.extend(
                                    self._split_long_text_input(
                                        [text[start:end]], max_text_len))
                                start = end
                                end = end + len(temp_text_list[i]) + 1
                            else:
                                end = len(temp_text_list[i]) + 1
                    if start != end:
                        short_input_texts.extend(
                            self._split_long_text_input([text[start:end]],
                                                        max_text_len))
        return short_input_texts

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
        for result in concat_results:
            pred_words = result['items']
            pred_words = self._reset_offset(pred_words)
            result['items'] = pred_words
        return concat_results

    def _preprocess_text(self, input_texts):
        """
        Create the dataset and dataloader for the predict.
        """
        batch_size = 1
        batch_size = self.kwargs[
            'batch_size'] if 'batch_size' in self.kwargs else 1
        num_workers = self.kwargs[
            'num_workers'] if 'num_workers' in self.kwargs else 0
        lazy_load = self.kwargs[
            'lazy_load'] if 'lazy_load' in self.kwargs else False

        max_seq_length = 512
        if 'max_position_embedding' in self.kwargs:
            max_seq_length = self.kwargs['max_position_embedding']
        infer_data = []
        max_predict_len = max_seq_length - self.summary_num - 1
        short_input_texts = self._split_long_text_input(input_texts,
                                                        max_predict_len)

        def read(inputs):
            for text in inputs:
                tokenized_output = self._tokenizer(
                    list(text),
                    return_length=True,
                    is_split_into_words=True,
                    max_seq_len=max_seq_length)
                yield tokenized_output['input_ids'], tokenized_output[
                    'token_type_ids'], tokenized_output['seq_len']

        infer_ds = load_dataset(read, inputs=short_input_texts, lazy=lazy_load)
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self._tokenizer.pad_token_id,dtype='int64'),  # input_ids
            Pad(axis=0, pad_val=self._tokenizer.pad_token_type_id,dtype='int64'),  # token_type_ids
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
        outputs['data_loader'] = infer_data_loader
        outputs['short_input_texts'] = short_input_texts
        return outputs

    def _reset_offset(self, pred_words):
        for i in range(0, len(pred_words)):
            if i > 0:
                pred_words[i]["offset"] = pred_words[i - 1]["offset"] + len(
                    pred_words[i - 1]["item"])
            pred_words[i]["length"] = len(pred_words[i]["item"])
        return pred_words

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
                    pred_words.append({
                        "item": text[j],
                        "offset": 0,
                        "wordtag_label": label
                    })
                else:
                    pred_word.append(text[j])
                    if pred_label.startswith("E"):
                        pred_words.append({
                            "item": "".join(pred_word),
                            "offset": 0,
                            "wordtag_label": label
                        })
                        del pred_word[:]

            pred_words = self._reset_offset(pred_words)
            result = {"text": text, "items": pred_words}
            batch_results.append(result)
        return batch_results

    def _term_linking(self, wordtag_res):
        for item in wordtag_res["items"]:
            if item["wordtag_label"] not in LABEL_TO_SCHEMA:
                continue
            if item["item"] not in self._term_dict:
                continue
            target_type = LABEL_TO_SCHEMA[item["wordtag_label"]]
            matched_type = list(self._term_dict[item["item"]].keys())
            matched = False
            term_id = None
            target_idx = math.inf
            for mt in matched_type:
                tmp_type = mt
                while tmp_type != "root":
                    if tmp_type not in self._term_schema:
                        break
                    for i, target in enumerate(target_type):
                        if target.startswith(tmp_type):
                            target_src = target.split("|")
                            for can_term_id in self._term_dict[item["item"]][
                                    mt]:
                                tmp_term_id = can_term_id
                                if len(target_src) == 1:
                                    matched = True
                                    if target.startswith(mt):
                                        target_idx = -1
                                        term_id = tmp_term_id
                                    if i < target_idx:
                                        target_idx = i
                                        term_id = tmp_term_id
                                else:
                                    if target_src[
                                            1] == "C" and "_cb_" in tmp_term_id:
                                        matched = True
                                        if target.startswith(mt):
                                            target_idx = -1
                                            term_id = tmp_term_id
                                        if i < target_idx:
                                            target_idx = i
                                            term_id = tmp_term_id
                                    if target_src[
                                            1] == "E" and "_eb_" in tmp_term_id:
                                        matched = True
                                        if target.startswith(mt):
                                            target_idx = -1
                                            term_id = tmp_term_id
                                        if i < target_idx:
                                            target_idx = i
                                            term_id = tmp_term_id
                    tmp_type = self._term_schema[tmp_type]
                    if matched is True:
                        break

            if matched is True:
                item["termid"] = term_id
        pass

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        model_instance = ErnieCtmWordtagModel.from_pretrained(
            model,
            num_cls_label=4,
            num_tag=len(self._tags_to_index),
            ignore_index=self._tags_to_index["O"])
        config_keys = ErnieCtmWordtagModel.pretrained_init_configuration[
            self.model]
        self.kwargs.update(config_keys)
        model_instance.eval()
        return model_instance

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        tokenizer_instance = ErnieCtmTokenizer.from_pretrained(model)
        return tokenizer_instance

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
                f"Bad inputs, input text should be str or list of str, {type(inputs)} found!"
            )
        outputs = self._preprocess_text(inputs)
        outputs['inputs'] = inputs
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function. 
        """
        all_pred_tags = []
        with paddle.no_grad():
            for batch in inputs['data_loader']:
                input_ids, token_type_ids, seq_len = batch
                seq_logits, cls_logits = self._model(
                    input_ids, token_type_ids, lengths=seq_len)
                score, pred_tags = self._model.viterbi_decoder(seq_logits,
                                                               seq_len)
                all_pred_tags.extend(pred_tags.numpy().tolist())

        inputs['all_pred_tags'] = all_pred_tags
        return inputs

    def _postprocess(self, inputs):
        """
        The model output is allways the logits and pros, this function will convert the model output to raw text.
        """
        results = self._decode(inputs['short_input_texts'],
                               inputs['all_pred_tags'])
        results = self._concat_short_text_reuslts(inputs['inputs'], results)
        if self.linking is True:
            for res in results:
                self._term_linking(res)
        return results
