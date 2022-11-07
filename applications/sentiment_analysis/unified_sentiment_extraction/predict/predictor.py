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

import six
import os
import math
import copy
import argparse
import numpy as np

import paddle
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.tools import get_bool_ids_greater_than, get_span


class UIEPredictor(object):

    def __init__(self, args, uie, schema):
        self.uie = uie
        self._tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)
        self._position_prob = args.position_prob
        self._max_seq_len = args.max_seq_len
        self._batch_size = args.batch_size
        self._schema_tree = None
        self.schema = schema
        self.set_schema(schema)

    def set_schema(self, schema):
        if isinstance(schema, dict) or isinstance(schema, str):
                schema = [schema]
        self._schema_tree = self._build_tree(schema)

    @classmethod
    def _build_tree(cls, schema, name='root'):
        """
        Build the schema tree.
        """
        schema_tree = SchemaTree(name)
        for s in schema:
            if isinstance(s, str):
                schema_tree.add_child(SchemaTree(s))
            elif isinstance(s, dict):
                for k, v in s.items():
                    if isinstance(v, str):
                        child = [v]
                    elif isinstance(v, list):
                        child = v
                    else:
                        raise TypeError(
                            "Invalid schema, value for each key:value pairs should be list or string"
                            "but {} received".format(type(v)))
                    schema_tree.add_child(cls._build_tree(child, name=k))
            else:
                raise TypeError(
                    "Invalid schema, element should be string or dict, "
                    "but {} received".format(type(s)))
        return schema_tree

    def _single_stage_predict(self, inputs):
        input_texts = []
        prompts = []
        for i in range(len(inputs)):
            input_texts.append(inputs[i]["text"])
            prompts.append(inputs[i]["prompt"])
        # max predict length should exclude the length of prompt and summary tokens
        max_predict_len = self._max_seq_len - len(max(prompts)) - 3
        short_input_texts, self.input_mapping = self._auto_splitter(
            input_texts, max_predict_len, split_sentence=False)
        short_texts_prompts = []
        for k, v in self.input_mapping.items():
            short_texts_prompts.extend([prompts[k] for i in range(len(v))])
        short_inputs = [{
            "text": short_input_texts[i],
            "prompt": short_texts_prompts[i]
        } for i in range(len(short_input_texts))]

        prompts = []
        texts = []
        for s in short_inputs:
            prompts.append(s['prompt'])
            texts.append(s['text'])
        encoded_inputs = self._tokenizer(text=prompts,
                                         text_pair=texts,
                                         truncation=True,
                                         max_seq_len=self._max_seq_len,
                                         pad_to_max_seq_len=True,
                                         return_attention_mask=True,
                                         return_position_ids=True,
                                         return_tensors='pd',
                                         return_offsets_mapping=True)
        offset_maps = encoded_inputs["offset_mapping"]

        start_probs = []
        end_probs = []
        for idx in range(0, len(texts), self._batch_size):
            l, r = idx, idx + self._batch_size
            input_dict = {
                "input_ids":
                encoded_inputs['input_ids'][l:r].astype('int64'),
                "token_type_ids":
                encoded_inputs['token_type_ids'][l:r].astype('int64'),
                "pos_ids":
                encoded_inputs['position_ids'][l:r].astype('int64'),
                "att_mask":
                encoded_inputs["attention_mask"][l:r].astype('int64')
            }
            start_prob, end_prob = self._infer(input_dict)
            start_prob = start_prob.tolist()
            end_prob = end_prob.tolist()
            start_probs.extend(start_prob)
            end_probs.extend(end_prob)
        start_ids_list = get_bool_ids_greater_than(start_probs,
                                                   limit=self._position_prob,
                                                   return_prob=True)
        end_ids_list = get_bool_ids_greater_than(end_probs,
                                                 limit=self._position_prob,
                                                 return_prob=True)

        sentence_ids = []
        probs = []
        for start_ids, end_ids, offset_map in zip(start_ids_list, end_ids_list,
                                                  offset_maps.tolist()):
            span_list = get_span(start_ids, end_ids, with_prob=True)
            sentence_id, prob = get_id_and_prob(span_list, offset_map)
            sentence_ids.append(sentence_id)
            probs.append(prob)

        results = self._convert_ids_to_results(short_inputs, sentence_ids,
                                               probs)
        results = self._auto_joiner(results, short_input_texts,
                                    self.input_mapping)
        return results

    def _auto_splitter(self, input_texts, max_text_len, split_sentence=False):
        '''
        Split the raw texts automatically for model inference.
        Args:
            input_texts (List[str]): input raw texts.
            max_text_len (int): cutting length.
            split_sentence (bool): If True, sentence-level split will be performed.
        return:
            short_input_texts (List[str]): the short input texts for model inference.
            input_mapping (dict): mapping between raw text and short input texts.
        '''
        input_mapping = {}
        short_input_texts = []
        cnt_org = 0
        cnt_short = 0
        for text in input_texts:
            if not split_sentence:
                sens = [text]
            else:
                sens = cut_chinese_sent(text)
            for sen in sens:
                lens = len(sen)
                if lens <= max_text_len:
                    short_input_texts.append(sen)
                    if cnt_org not in input_mapping.keys():
                        input_mapping[cnt_org] = [cnt_short]
                    else:
                        input_mapping[cnt_org].append(cnt_short)
                    cnt_short += 1
                else:
                    temp_text_list = [
                        sen[i:i + max_text_len]
                        for i in range(0, lens, max_text_len)
                    ]
                    short_input_texts.extend(temp_text_list)
                    short_idx = cnt_short
                    cnt_short += math.ceil(lens / max_text_len)
                    temp_text_id = [
                        short_idx + i for i in range(cnt_short - short_idx)
                    ]
                    if cnt_org not in input_mapping.keys():
                        input_mapping[cnt_org] = temp_text_id
                    else:
                        input_mapping[cnt_org].extend(temp_text_id)
            cnt_org += 1
        return short_input_texts, input_mapping

    def _auto_joiner(self, short_results, short_inputs, input_mapping):
        concat_results = []
        is_cls_task = False
        for short_result in short_results:
            if short_result == []:
                continue
            elif 'start' not in short_result[0].keys(
            ) and 'end' not in short_result[0].keys():
                is_cls_task = True
                break
            else:
                break
        for k, vs in input_mapping.items():
            if is_cls_task:
                cls_options = {}
                single_results = []
                for v in vs:
                    if len(short_results[v]) == 0:
                        continue
                    if short_results[v][0]['text'] not in cls_options.keys():
                        cls_options[short_results[v][0]['text']] = [
                            1, short_results[v][0]['probability']
                        ]
                    else:
                        cls_options[short_results[v][0]['text']][0] += 1
                        cls_options[short_results[v][0]['text']][
                            1] += short_results[v][0]['probability']
                if len(cls_options) != 0:
                    cls_res, cls_info = max(cls_options.items(),
                                            key=lambda x: x[1])
                    concat_results.append([{
                        'text':
                        cls_res,
                        'probability':
                        cls_info[1] / cls_info[0]
                    }])
                else:
                    concat_results.append([])
            else:
                offset = 0
                single_results = []
                for v in vs:
                    if v == 0:
                        single_results = short_results[v]
                        offset += len(short_inputs[v])
                    else:
                        for i in range(len(short_results[v])):
                            if 'start' not in short_results[v][
                                    i] or 'end' not in short_results[v][i]:
                                continue
                            short_results[v][i]['start'] += offset
                            short_results[v][i]['end'] += offset
                        offset += len(short_inputs[v])
                        single_results.extend(short_results[v])
                concat_results.append(single_results)
        return concat_results

    def _convert_ids_to_results(self, examples, sentence_ids, probs):
        """
        Convert ids to raw text in a single stage.
        """
        results = []
        for example, sentence_id, prob in zip(examples, sentence_ids, probs):
            if len(sentence_id) == 0:
                results.append([])
                continue
            result_list = []
            text = example["text"]
            prompt = example["prompt"]
            for i in range(len(sentence_id)):
                start, end = sentence_id[i]
                if start < 0 and end >= 0:
                    continue
                if end < 0:
                    start += (len(prompt) + 1)
                    end += (len(prompt) + 1)
                    result = {"text": prompt[start:end], "probability": prob[i]}
                    result_list.append(result)
                else:
                    result = {
                        "text": text[start:end],
                        "start": start,
                        "end": end,
                        "probability": prob[i]
                    }
                    result_list.append(result)
            results.append(result_list)
        return results

    def _multi_stage_predict(self, data, aspects=None):
        """
        Traversal the schema tree and do multi-stage prediction.
        Args:
            data (list): a list of strings
            aspects (list[string]):  a list of pre-given aspects
        Returns:
            list: a list of predictions, where the list's length
                equals to the length of `data`
        """ 

        if aspects is not None:
            # predict with pre-give aspects
            results = []
            prefixs = []
            relations = []
            result = {"评价维度": [{"text":aspect} for aspect in aspects]}
            prefix = [aspect + "的" for aspect in aspects]
            for i in range(len(data)):
                results.append(copy.deepcopy(result))
                prefixs.append(copy.deepcopy(prefix))
                relations.append(results[-1]["评价维度"])
            # copy to stay `self._schema_tree` unchanged
            schema_list = self._schema_tree.children[:]
            for node in schema_list:
                node.prefix = prefixs
                node.parent_relations = relations

        else:
            results = [{} for _ in range(len(data))]
            # input check to early return
            if len(data) < 1 or self._schema_tree is None:
                return results
            # copy to stay `self._schema_tree` unchanged
            schema_list = self._schema_tree.children[:]

        while len(schema_list) > 0:
            node = schema_list.pop(0)
            examples = []
            input_map = {}
            cnt = 0
            idx = 0
            if not node.prefix:
                for one_data in data:
                    examples.append({
                        "text": one_data,
                        "prompt": dbc2sbc(node.name)
                    })
                    input_map[cnt] = [idx]
                    idx += 1
                    cnt += 1
            else:
                for pre, one_data in zip(node.prefix, data):
                    if len(pre) == 0:
                        input_map[cnt] = []
                    else:
                        for p in pre:
                            examples.append({
                                "text": one_data,
                                "prompt": dbc2sbc(p + node.name)
                            })
                        input_map[cnt] = [i + idx for i in range(len(pre))]
                        idx += len(pre)
                    cnt += 1
            if len(examples) == 0:
                result_list = []
            else:
                result_list = self._single_stage_predict(examples)

            if not node.parent_relations:
                relations = [[] for i in range(len(data))]
                for k, v in input_map.items():
                    for idx in v:
                        if len(result_list[idx]) == 0:
                            continue
                        if node.name not in results[k].keys():
                            results[k][node.name] = result_list[idx]
                        else:
                            results[k][node.name].extend(result_list[idx])
                    if node.name in results[k].keys():
                        relations[k].extend(results[k][node.name])
            else:
                relations = node.parent_relations
                for k, v in input_map.items():
                    for i in range(len(v)):
                        if len(result_list[v[i]]) == 0:
                            continue
                        if "relations" not in relations[k][i].keys():
                            relations[k][i]["relations"] = {
                                node.name: result_list[v[i]]
                            }
                        elif node.name not in relations[k][i]["relations"].keys(
                        ):
                            relations[k][i]["relations"][
                                node.name] = result_list[v[i]]
                        else:
                            relations[k][i]["relations"][node.name].extend(
                                result_list[v[i]])
                new_relations = [[] for i in range(len(data))]
                for i in range(len(relations)):
                    for j in range(len(relations[i])):
                        if "relations" in relations[i][j].keys(
                        ) and node.name in relations[i][j]["relations"].keys():
                            for k in range(
                                    len(relations[i][j]["relations"][
                                        node.name])):
                                new_relations[i].append(
                                    relations[i][j]["relations"][node.name][k])
                relations = new_relations

            prefix = [[] for _ in range(len(data))]
            for k, v in input_map.items():
                for idx in v:
                    for i in range(len(result_list[idx])):
                        prefix[k].append(result_list[idx][i]["text"] + "的")

            for child in node.children:
                child.prefix = prefix
                child.parent_relations = relations
                schema_list.append(child)
        return results

    def _infer(self, input_dict):
        return self.uie(input_ids=input_dict["input_ids"], token_type_ids=input_dict["token_type_ids"], pos_ids=input_dict["pos_ids"], att_mask=input_dict["att_mask"])

    def predict(self, input_data, aspects=None):
        results = self._multi_stage_predict(input_data, aspects=aspects)
        return results


class SchemaTree(object):
    """
    Implementataion of SchemaTree
    """

    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        self.prefix = None
        self.parent_relations = None
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(
            node, SchemaTree
        ), "The children of a node should be an instacne of SchemaTree."
        self.children.append(node)


def dbc2sbc(s):
    rs = ""
    for char in s:
        code = ord(char)
        if code == 0x3000:
            code = 0x0020
        else:
            code -= 0xfee0
        if not (0x0021 <= code and code <= 0x7e):
            rs += char
            continue
        rs += chr(code)
    return rs


def cut_chinese_sent(para):
    """
    Cut the Chinese sentences more precisely, reference to 
    "https://blog.csdn.net/blmoistawinde/article/details/82379256".
    """
    para = re.sub(r'([。！？\?])([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\.{6})([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\…{2})([^”’])', r'\1\n\2', para)
    para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")


def get_id_and_prob(span_set, offset_mapping):
    """
    Return text id and probability of predicted spans

    Args: 
        span_set (set): set of predicted spans.
        offset_mapping (list[int]): list of pair preserving the
                index of start and end char in original text pair (prompt + text) for each token.
    Returns: 
        sentence_id (list[tuple]): index of start and end char in original text.
        prob (list[float]): probabilities of predicted spans.
    """
    prompt_end_token_id = offset_mapping[1:].index([0, 0])
    bias = offset_mapping[prompt_end_token_id][1] + 1
    for index in range(1, prompt_end_token_id + 1):
        offset_mapping[index][0] -= bias
        offset_mapping[index][1] -= bias

    sentence_id = []
    prob = []
    for start, end in span_set:
        prob.append(start[1] * end[1])
        start_id = offset_mapping[start[0]][0]
        end_id = offset_mapping[end[0]][1]
        sentence_id.append((start_id, end_id))
    return sentence_id, prob


