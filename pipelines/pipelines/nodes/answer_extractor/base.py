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

import re
import os
import json
import numpy as np
import math
from multiprocessing import cpu_count

import paddle
import paddle.nn as nn
from paddlenlp.transformers import ErniePretrainedModel, ErnieMPretrainedModel
from paddlenlp.datasets import load_dataset
from paddle.dataset.common import md5file
from paddlenlp.transformers import AutoTokenizer, AutoModel
from paddlenlp.taskflow.utils import SchemaTree, get_span, get_id_and_prob, get_bool_ids_greater_than, dbc2sbc, gp_decode, DataCollatorGP
from paddlenlp.taskflow.utils import download_check, static_mode_guard, dygraph_mode_guard, download_file, cut_chinese_sent

from paddlenlp.utils.env import PPNLP_HOME

from pipelines.nodes.base import BaseComponent
from .task import Task


class UIE(ErniePretrainedModel):

    def __init__(self, encoding_model):
        super(UIE, self).__init__()
        self.encoder = encoding_model
        hidden_size = self.encoder.config["hidden_size"]
        self.linear_start = paddle.nn.Linear(hidden_size, 1)
        self.linear_end = paddle.nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, pos_ids, att_mask):
        sequence_output, _ = self.encoder(input_ids=input_ids,
                                          token_type_ids=token_type_ids,
                                          position_ids=pos_ids,
                                          attention_mask=att_mask)
        start_logits = self.linear_start(sequence_output)
        start_logits = paddle.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = paddle.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)
        return start_prob, end_prob


class UIEM(ErnieMPretrainedModel):

    def __init__(self, encoding_model):
        super(UIEM, self).__init__()
        self.encoder = encoding_model
        hidden_size = self.encoder.config["hidden_size"]
        self.linear_start = paddle.nn.Linear(hidden_size, 1)
        self.linear_end = paddle.nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, pos_ids):
        sequence_output, _ = self.encoder(input_ids=input_ids,
                                          position_ids=pos_ids)
        start_logits = self.linear_start(sequence_output)
        start_logits = paddle.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = paddle.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)
        return start_prob,


class UIEComponent(BaseComponent, Task):
    """
    Universal Information Extraction Task. 
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "model_config": "model_config.json",
        "vocab_file": "vocab.txt",
        "special_tokens_map": "special_tokens_map.json",
        "tokenizer_config": "tokenizer_config.json"
    }

    resource_files_urls = {}

    return_no_answers: bool
    outgoing_edges = 1
    query_count = 0
    query_time = 0

    def __init__(self,
                 model=None,
                 schema=None,
                 task_path=None,
                 device="gpu",
                 schema_lang="zh",
                 max_seq_len=512,
                 batch_size=64,
                 split_sentence=False,
                 position_prob=0.5,
                 lazy_load=False,
                 num_workers=0,
                 use_faster=False):
        paddle.set_device(device)
        if model in ['uie-m-base', 'uie-m-large']:
            self._multilingual = True
            self.resource_files_names[
                'sentencepiece_model_file'] = "sentencepiece.bpe.model"
        else:
            self._multilingual = False
            if 'sentencepiece_model_file' in self.resource_files_names.keys():
                del self.resource_files_names['sentencepiece_model_file']
        self._schema_tree = None
        self.set_schema(schema)

        self._is_en = True if model in ['uie-base-en'
                                        ] or schema_lang == 'en' else False
        ####
        self.model = model
        self.task = None
        self._priority_path = None
        self._usage = ""
        self._model = None
        self._input_spec = None
        self._config = None
        self._custom_model = False
        self._param_updated = False
        self._num_threads = math.ceil(cpu_count() / 2)
        self._infer_precision = 'fp32'
        self._predictor_type = 'paddle-inference'
        self._home_path = PPNLP_HOME
        self._task_flag = self.model
        if task_path:
            self._task_path = task_path
            self._custom_model = True
        else:
            self._task_path = os.path.join(PPNLP_HOME, "pipelines",
                                           "unsupervised_question_answering",
                                           self.model)

        self._check_task_files()
        self._check_predictor_type()
        self._get_inference_model()
        self._construct_tokenizer(model)

        self._schema = schema
        self._max_seq_len = max_seq_len
        self._batch_size = batch_size
        self._split_sentence = split_sentence
        self._position_prob = position_prob
        self._lazy_load = lazy_load
        self._num_workers = num_workers
        self.use_faster = use_faster

    def set_schema(self, schema):
        if isinstance(schema, dict) or isinstance(schema, str):
            schema = [schema]
        self._schema_tree = self._build_tree(schema)

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        if self._multilingual:
            self._input_spec = [
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='input_ids'),
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='pos_ids'),
            ]
        else:
            self._input_spec = [
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='input_ids'),
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='token_type_ids'),
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='pos_ids'),
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='att_mask'),
            ]

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        if self._multilingual:
            model_instance = UIEM.from_pretrained(self._task_path)
        else:
            model_instance = UIE.from_pretrained(self._task_path)
        self._model = model_instance
        self._model.eval()

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(self._task_path)

    def _preprocess(self, inputs):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        outputs = {}
        outputs['text'] = inputs
        return outputs

    def _single_stage_predict(self, inputs):
        input_texts = []
        prompts = []
        for i in range(len(inputs)):
            input_texts.append(inputs[i]["text"])
            prompts.append(inputs[i]["prompt"])
        # max predict length should exclude the length of prompt and summary tokens
        max_predict_len = self._max_seq_len - len(max(prompts)) - 3

        short_input_texts, self.input_mapping = self._auto_splitter(
            input_texts, max_predict_len, split_sentence=self._split_sentence)

        short_texts_prompts = []
        for k, v in self.input_mapping.items():
            short_texts_prompts.extend([prompts[k] for i in range(len(v))])
        short_inputs = [{
            "text": short_input_texts[i],
            "prompt": short_texts_prompts[i]
        } for i in range(len(short_input_texts))]

        def read(inputs):
            for example in inputs:
                encoded_inputs = self._tokenizer(text=[example["prompt"]],
                                                 text_pair=[example["text"]],
                                                 truncation=True,
                                                 max_seq_len=self._max_seq_len,
                                                 pad_to_max_seq_len=True,
                                                 return_attention_mask=True,
                                                 return_position_ids=True,
                                                 return_offsets_mapping=True)
                if self._multilingual:
                    tokenized_output = [
                        encoded_inputs["input_ids"][0],
                        encoded_inputs["position_ids"][0],
                        encoded_inputs["offset_mapping"][0]
                    ]
                else:
                    tokenized_output = [
                        encoded_inputs["input_ids"][0],
                        encoded_inputs["token_type_ids"][0],
                        encoded_inputs["position_ids"][0],
                        encoded_inputs["attention_mask"][0],
                        encoded_inputs["offset_mapping"][0]
                    ]
                tokenized_output = [
                    np.array(x, dtype="int64") for x in tokenized_output
                ]
                yield tuple(tokenized_output)

        infer_ds = load_dataset(read, inputs=short_inputs, lazy=self._lazy_load)
        batch_sampler = paddle.io.BatchSampler(dataset=infer_ds,
                                               batch_size=self._batch_size,
                                               shuffle=False)

        infer_data_loader = paddle.io.DataLoader(dataset=infer_ds,
                                                 batch_sampler=batch_sampler,
                                                 num_workers=self._num_workers,
                                                 return_list=True)

        sentence_ids = []
        probs = []
        for batch in infer_data_loader:
            if self._multilingual:
                input_ids, pos_ids, offset_maps = batch
            else:
                input_ids, token_type_ids, pos_ids, att_mask, offset_maps = batch
            if self._predictor_type == "paddle-inference":
                if self._multilingual:
                    self.input_handles[0].copy_from_cpu(input_ids.numpy())
                    self.input_handles[1].copy_from_cpu(pos_ids.numpy())
                else:
                    self.input_handles[0].copy_from_cpu(input_ids.numpy())
                    self.input_handles[1].copy_from_cpu(token_type_ids.numpy())
                    self.input_handles[2].copy_from_cpu(pos_ids.numpy())
                    self.input_handles[3].copy_from_cpu(att_mask.numpy())
                self.predictor.run()
                start_prob = self.output_handle[0].copy_to_cpu().tolist()
                end_prob = self.output_handle[1].copy_to_cpu().tolist()
            else:
                if self._multilingual:
                    input_dict = {
                        "input_ids": input_ids.numpy(),
                        "pos_ids": pos_ids.numpy(),
                    }
                else:
                    input_dict = {
                        "input_ids": input_ids.numpy(),
                        "token_type_ids": token_type_ids.numpy(),
                        "pos_ids": pos_ids.numpy(),
                        "att_mask": att_mask.numpy()
                    }
                start_prob, end_prob = self.predictor.run(None, input_dict)
                start_prob = start_prob.tolist()
                end_prob = end_prob.tolist()

            start_ids_list = get_bool_ids_greater_than(
                start_prob, limit=self._position_prob, return_prob=True)
            end_ids_list = get_bool_ids_greater_than(end_prob,
                                                     limit=self._position_prob,
                                                     return_prob=True)

            for start_ids, end_ids, offset_map in zip(start_ids_list,
                                                      end_ids_list,
                                                      offset_maps.tolist()):
                span_set = get_span(start_ids, end_ids, with_prob=True)
                sentence_id, prob = get_id_and_prob(span_set, offset_map)
                sentence_ids.append(sentence_id)
                probs.append(prob)
        results = self._convert_ids_to_results(short_inputs, sentence_ids,
                                               probs)
        results = self._auto_joiner(results, short_input_texts,
                                    self.input_mapping)
        return results

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

    def _run_model(self, inputs):
        raw_inputs = inputs['text']
        results = self._multi_stage_predict(raw_inputs)
        inputs['result'] = results
        return inputs

    def _multi_stage_predict(self, data):
        """
        Traversal the schema tree and do multi-stage prediction.
        Args:
            data (list): a list of strings
        Returns:
            list: a list of predictions, where the list's length
                equals to the length of `data`
        """
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
                            if self._is_en:
                                if re.search(r'\[.*?\]$', node.name):
                                    prompt_prefix = node.name[:node.name.find(
                                        "[", 1)].strip()
                                    cls_options = re.search(
                                        r'\[.*?\]$', node.name).group()
                                    # Sentiment classification of xxx [positive, negative]
                                    prompt = prompt_prefix + p + " " + cls_options
                                else:
                                    prompt = node.name + p
                            else:
                                prompt = p + node.name
                            examples.append({
                                "text": one_data,
                                "prompt": dbc2sbc(prompt)
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
                        if self._is_en:
                            prefix[k].append(" of " +
                                             result_list[idx][i]["text"])
                        else:
                            prefix[k].append(result_list[idx][i]["text"] + "的")

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

    def _postprocess(self, inputs):
        """
        This function will convert the model output to raw text.
        """
        return inputs['result']

    def run(self, meta):
        preprocessed = self._preprocess(meta)
        generated = self._run_model(preprocessed)
        postprocessed = self._postprocess(generated)
        results = {"results": postprocessed}
        return results, "output_1"
