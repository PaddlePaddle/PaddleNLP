# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import glob
import json
import math
import os
import copy

import paddle
import paddle.nn as nn
import pandas as pd
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Pad, Tuple
from paddlenlp.utils.downloader import get_path_from_url
from paddlenlp.utils.env import MODEL_HOME
from paddlenlp.transformers import ErnieCtmWordtagModel, ErnieCtmTokenizer

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
    "https://kg-concept.bj.bcebos.com/TermTree/TermTree.V1.0.tar.gz",
    "termtree_type.csv":
    "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_ctm/termtree_type.csv",
    "termtree_tags_pos.txt":
    "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_ctm/termtree_tags_pos.txt",
}


class WordtagPredictor(object):
    """Predictor of wordtag model.
    """

    def __init__(self, model_name="wordtag", term_linking=True, tag_path=None):
        """Initialize method of the predictor.

        Args:
            model_name (`str`): 
                The pre-trained model name.
            tag_path (`str`): 
                The tag vocab path.
        """
        term_schema_path = self._download_termtree("termtree_type.csv")
        term_data_path = self._download_termtree("TermTree.V1.0")
        if tag_path is None:
            tag_path = self._download_termtree("termtree_tags_pos.txt")
        self._tags_to_index, self._index_to_tags = self._load_labels(tag_path)

        self._model = ErnieCtmWordtagModel.from_pretrained(
            model_name,
            num_cls_label=4,
            num_tag=len(self._tags_to_index),
            ignore_index=self._tags_to_index["O"])
        self._model.eval()

        self._tokenizer = ErnieCtmTokenizer.from_pretrained(model_name)
        self._summary_num = self._model.ernie_ctm.content_summary_index + 1
        if term_schema_path is not None:
            self._term_schema = self._load_schema(term_schema_path)
        if term_data_path is not None:
            self._term_dict = self._load_term_tree_data(term_data_path)
        if term_data_path is not None and term_schema_path is not None and term_linking:
            self._linking = True
        else:
            self._linking = False

    def _download_termtree(self, filename):
        default_root = os.path.join(MODEL_HOME, 'ernie-ctm')
        fullname = os.path.join(default_root, filename)
        url = URLS[filename]
        if not os.path.exists(fullname):
            get_path_from_url(url, default_root)
        return fullname

    @property
    def summary_num(self):
        """Number of model summary token
        """
        return self._summary_num

    @property
    def linking(self):
        """Whether to do term linking.
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
        schema_df = pd.read_csv(schema_path, sep="\t", encoding="utf8")
        schema = {}
        for idx in range(schema_df.shape[0]):
            if not isinstance(schema_df["type-1"][idx], float):
                schema[schema_df["type-1"][idx]] = "root"
            if not isinstance(schema_df["type-2"][idx], float):
                schema[schema_df["type-2"][idx]] = schema_df["type-1"][idx]
            if not isinstance(schema_df["type-3"][idx], float):
                schema[schema_df["type-3"][idx]] = schema_df["type-2"][idx]
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

    def _split_long_text2short_text_list(self, input_texts, max_text_len):
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

    def _convert_short_text2long_text_result(self, input_texts, results):
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
                    raise Exception("The len of text must same as raw text.")
        return concat_results

    def _pre_process_text(self, input_texts, max_seq_len=512, batch_size=1):
        infer_data = []
        max_predict_len = max_seq_len - self.summary_num - 1
        short_input_texts = self._split_long_text2short_text_list(
            input_texts, max_predict_len)
        for text in short_input_texts:
            tokenized_input = self._tokenizer(
                list(text),
                return_length=True,
                is_split_into_words=True,
                max_seq_len=max_seq_len)
            infer_data.append([
                tokenized_input['input_ids'], tokenized_input['token_type_ids'],
                tokenized_input['seq_len']
            ])
        infer_ds = MapDataset(infer_data)

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self._tokenizer.pad_token_id,dtype='int64'),  # input_ids
            Pad(axis=0, pad_val=self._tokenizer.pad_token_type_id,dtype='int64'),  # token_type_ids
            Stack(dtype='int64'),  # seq_len
        ): fn(samples)

        infer_data_loader = paddle.io.DataLoader(
            infer_ds,
            collate_fn=batchify_fn,
            num_workers=0,
            batch_size=batch_size,
            shuffle=False,
            return_list=True)

        return infer_data_loader, short_input_texts

    def _decode(self, batch_texts, batch_pred_tags):
        batch_results = []
        for i, pred_tags in enumerate(batch_pred_tags):
            pred_words, pred_word = [], []
            text = batch_texts[i]
            for j, tag in enumerate(pred_tags[self.summary_num:-1]):
                if j > len(text) + self.summary_num - 1:
                    continue
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
            for i in range(len(pred_words)):
                if i > 0:
                    pred_words[i]["offset"] = pred_words[i - 1]["offset"] + len(
                        pred_words[i - 1]["item"])
                pred_words[i]["length"] = len(pred_words[i]["item"])
            result = {"text": text, "items": pred_words}
            batch_results.append(result)
        return batch_results

    @paddle.no_grad()
    def run(self,
            input_texts,
            max_seq_len=512,
            batch_size=1,
            return_hidden_states=None):
        """Predict a input text by wordtag.

        Args:
            input_text: input text.
            max_seq_len: max sequence length.
            batch_size: Batch size per GPU/CPU for training.

        Returns:
            dict -- wordtag results.
        """
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        if not isinstance(input_texts, str) and not isinstance(input_texts,
                                                               list):
            raise TypeError(
                f"Bad inputs, input text should be str or list of str, {type(input_texts)} found!"
            )
        infer_data_loader, short_input_texts = self._pre_process_text(
            input_texts, max_seq_len, batch_size)
        all_pred_tags = []
        with paddle.no_grad():
            for batch in infer_data_loader:
                input_ids, token_type_ids, seq_len = batch
                seq_logits, cls_logits = self._model(
                    input_ids, token_type_ids, lengths=seq_len)
                scores, pred_tags = self._model.viterbi_decoder(seq_logits,
                                                                seq_len)
                all_pred_tags += pred_tags.numpy().tolist()
        results = self._decode(short_input_texts, all_pred_tags)
        results = self._convert_short_text2long_text_result(input_texts,
                                                            results)
        if self.linking is True:
            for res in results:
                self._term_linking(res)
        outputs = results
        if return_hidden_states is True:
            outputs = (results, ) + (seq_logits, cls_logits)
        return outputs

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
