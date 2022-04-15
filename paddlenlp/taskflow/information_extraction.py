# coding:utf-8
# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

import os

import numpy as np
import paddle
from ..datasets import load_dataset
from ..transformers import AutoTokenizer
from .models import UIE
from .task import Task
from .utils import SchemaTree, get_span, get_id_and_prob, get_bool_ids_greater_than

usage = r"""
            from paddlenlp import Taskflow

            # 定义schema（抽取目标）
            schema = ["寺庙", {"丈夫": ["妻子"]}]
            ie = Taskflow("information_extraction", schema=schema) 
            ie("李治即位后，让身在感业寺的武则天续起头发，重新纳入后宫。")           
            '''
            [{'寺庙': [{'text': '感业寺', 'start': 9, 'end': 12, 'probability': 0.8899254648933592}], '丈夫': [{'text': '李治', 'start': 0, 'end': 2, 'probability': 0.9852263795480809, 'relation': {'妻子': [{'text': '武则天', 'start': 13, 'end': 16, 'probability': 0.9968914045166457}]}}]}]
            '''

            schema_senta = [{"水果": ["情感倾向[正向，负向]"]}]
            # 使用新的schema进行预测
            ie.set_schema(schema_senta)
            ie("今天去超市买了葡萄、苹果，都很好吃")
            '''
            [{'水果': [{'text': '苹果', 'start': 10, 'end': 12, 'probability': 0.9369744072599353, 'relation': {'情感倾向[正向，负向]': [{'text': '正向', 'start': -7, 'end': -5, 'probability': 0.9733151286049946}]}}, {'text': '葡萄', 'start': 7, 'end': 9, 'probability': 0.5165698195669179, 'relation': {'情感倾向[正向，负向]': [{'text': '正向', 'start': -7, 'end': -5, 'probability': 0.976208180045397}]}}]}]
            '''
         """


class UIETask(Task):
    """
    Universal Information Extraction Task. 
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    resource_files_names = {"model_state": "model_state.pdparams", }
    resource_files_urls = {
        "uie": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie/model_state.pdparams",
                "af76619dda776ab9432de06db3a002c0"
            ],
        }
    }

    def __init__(self, task, model, schema, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._static_mode = True
        self._encoding_model = "ernie-1.0"
        self._schema = schema
        self._check_task_files()
        self._construct_tokenizer()
        if self._static_mode:
            self._get_inference_model()
        else:
            self._construct_model(model)
        self._usage = usage
        self._batch_size = self.kwargs[
            'batch_size'] if 'batch_size' in self.kwargs else 1

    def set_schema(self, schema):
        self._schema = schema

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name='input_ids'),
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name='token_type_ids'),
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name='pos_ids'),
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name='att_mask'),
        ]

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        model_instance = UIE()
        model_path = os.path.join(self._task_path, "model_state.pdparams")

        # Load the model parameter for the predict
        state_dict = paddle.load(model_path)
        model_instance.set_dict(state_dict)
        self._model = model_instance
        self._model.eval()

    def _construct_tokenizer(self):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(self._encoding_model)

    def _preprocess(self, inputs):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        outputs = {}
        outputs['text'] = inputs
        return outputs

    def _single_stage_predict(self, inputs):
        def read(inputs):
            max_length = 512
            for example in inputs:
                encoded_inputs = self._tokenizer(
                    text=[example["prompt"]],
                    text_pair=[example["text"]],
                    stride=len(example["prompt"]),
                    max_seq_len=max_length,
                    pad_to_max_seq_len=True,
                    return_attention_mask=True,
                    return_position_ids=True,
                    return_dict=False)
                encoded_inputs = encoded_inputs[0]
                offset_mapping = [
                    list(x) for x in encoded_inputs["offset_mapping"]
                ]
                bias = 0
                for index in range(len(offset_mapping)):
                    if index == 0:
                        continue
                    mapping = offset_mapping[index]
                    if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
                        bias = index
                    if mapping[0] == 0 and mapping[1] == 0:
                        continue
                    offset_mapping[index][0] += bias
                    offset_mapping[index][1] += bias

                yield tuple([
                    np.array(
                        x, dtype="int64") for x in [
                            encoded_inputs["input_ids"],
                            encoded_inputs["token_type_ids"],
                            encoded_inputs["position_ids"],
                            encoded_inputs["attention_mask"]
                        ]
                ])

        infer_ds = load_dataset(read, inputs=inputs, lazy=False)
        batch_sampler = paddle.io.BatchSampler(
            dataset=infer_ds, batch_size=self._batch_size, shuffle=False)

        infer_data_loader = paddle.io.DataLoader(
            dataset=infer_ds, batch_sampler=batch_sampler, return_list=True)

        sentence_ids = []
        probs = []
        for [input_ids, token_type_ids, pos_ids, att_mask] in infer_data_loader(
        ):
            #start_prob, end_prob = self._model(input_ids, token_type_ids, pos_ids, att_mask)

            self.input_handles[0].copy_from_cpu(input_ids.numpy())
            self.input_handles[1].copy_from_cpu(token_type_ids.numpy())
            self.input_handles[2].copy_from_cpu(pos_ids.numpy())
            self.input_handles[3].copy_from_cpu(att_mask.numpy())
            self.predictor.run()
            start_prob = self.output_handle[0].copy_to_cpu().tolist()
            end_prob = self.output_handle[1].copy_to_cpu().tolist()

            start_ids_list = get_bool_ids_greater_than(
                start_prob, return_prob=True)
            end_ids_list = get_bool_ids_greater_than(end_prob, return_prob=True)

            for start_ids, end_ids, ids in zip(start_ids_list, end_ids_list,
                                               input_ids.tolist()):
                for i in reversed(range(len(ids))):
                    if ids[i] != 0:
                        ids = ids[:i]
                        break
                span_list = get_span(start_ids, end_ids, with_prob=True)
                sentence_id, prob = get_id_and_prob(span_list, ids)
                sentence_ids.append(sentence_id)
                probs.append(prob)
        results = self._convert_ids_to_results(inputs, sentence_ids, probs)
        return results

    def _run_model(self, inputs):
        raw_inputs = inputs['text']
        schema_tree = self._build_tree(self._schema)
        results = self._multi_stage_predict(raw_inputs, schema_tree)
        inputs['result'] = results
        return inputs

    def _multi_stage_predict(self, datas, schema_tree):
        """
        Traversal the schema tree and do multi-stage prediction.
        """
        results = [{} for i in range(len(datas))]
        schema_list = schema_tree.children
        while len(schema_list) > 0:
            node = schema_list.pop(0)
            examples = []
            input_map = {}
            cnt = 0
            id = 0
            if not node.prefix:
                for data in datas:
                    examples.append({"text": data, "prompt": node.name})
                    input_map[cnt] = [id]
                    id += 1
                    cnt += 1
            else:
                for pre, data in zip(node.prefix, datas):
                    if len(pre) == 0:
                        input_map[cnt] = []
                    else:
                        for p in pre:
                            examples.append({
                                "text": data,
                                "prompt": p + node.name
                            })
                        input_map[cnt] = [i + id for i in range(len(pre))]
                        id += len(pre)
                    cnt += 1
            result_list = self._single_stage_predict(examples)

            if not node.parent_relations:
                relations = [[] for i in range(len(datas))]
                for k, v in input_map.items():
                    for id in v:
                        if len(result_list[id]) == 0:
                            continue
                        if node.name not in results[k].keys():
                            results[k][node.name] = result_list[id]
                        else:
                            results[k][node.name].extend(result_list[id])
                    if node.name in results[k].keys():
                        relations[k].extend(results[k][node.name])
            else:
                relations = node.parent_relations
                for k, v in input_map.items():
                    for i in range(len(v)):
                        if len(result_list[v[i]]) == 0:
                            continue
                        if "relation" not in relations[k][i].keys():
                            relations[k][i]["relation"] = {
                                node.name: result_list[v[i]]
                            }
                        elif node.name not in relations[k][i]["relation"].keys(
                        ):
                            relations[k][i]["relation"][
                                node.name] = result_list[v[i]]
                        else:
                            relations[k][i]["relation"][node.name].extend(
                                result_list[v[i]])
                new_relations = [[] for i in range(len(datas))]
                for i in range(len(relations)):
                    for j in range(len(relations[i])):
                        if "relation" in relations[i][j].keys():
                            for k in range(
                                    len(relations[i][j]["relation"][
                                        node.name])):
                                new_relations[i].append(relations[i][j][
                                    "relation"][node.name][k])
                relations = new_relations

            prefix = [[] for i in range(len(datas))]
            for k, v in input_map.items():
                for id in v:
                    for i in range(len(result_list[id])):
                        prefix[k].append(result_list[id][i]["text"] + "的")

            for child in node.children:
                child.prefix = prefix
                child.parent_relations = relations
                schema_list.append(child)
        return results

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
                if end < 0:
                    # ignore [SEP]
                    result = {
                        "text": prompt[start + 1:end + 1],
                        "start": start,
                        "end": end,
                        "probability": prob[i]
                    }
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

    def _build_tree(self, schema, name='root'):
        """
        Build the schema tree.
        """
        schema_tree = SchemaTree(name)
        for s in schema:
            if isinstance(s, str):
                schema_tree.add_child(SchemaTree(s))
            elif isinstance(s, dict):
                for k, v in s.items():
                    schema_tree.add_child(self._build_tree(v, name=k))
        return schema_tree

    def _postprocess(self, inputs):
        """
        This function will convert the model output to raw text.
        """
        return inputs['result']
