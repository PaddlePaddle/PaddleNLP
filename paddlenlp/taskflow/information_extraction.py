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

import re
import numpy as np
import paddle
from ..datasets import load_dataset
from ..transformers import AutoTokenizer
from .models import UIE
from .task import Task
from .utils import SchemaTree, get_span, get_id_and_prob, get_bool_ids_greater_than, dbc2sbc

usage = r"""
            from paddlenlp import Taskflow

            # Entity Extraction
            schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
            ie = Taskflow('information_extraction', schema=schema)
            ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")
            '''
            [{'时间': [{'text': '2月8日上午', 'start': 0, 'end': 6, 'probability': 0.9857378532924486}], '选手': [{'text': '谷爱凌', 'start': 28, 'end': 31, 'probability': 0.8981548639781138}], '赛事名称': [{'text': '北京冬奥会自由式滑雪女子大跳台决赛', 'start': 6, 'end': 23, 'probability': 0.8503089953268272}]}]
            '''

            # Relation Extraction
            schema = [{"歌曲名称":["歌手", "所属专辑"]}] # Define the schema for relation extraction
            ie.set_schema(schema) # Reset schema
            ie("《告别了》是孙耀威在专辑爱的故事里面的歌曲")
            '''
            [{'歌曲名称': [{'text': '告别了', 'start': 1, 'end': 4, 'probability': 0.6296155977145546, 'relations': {'歌手': [{'text': '孙耀威', 'start': 6, 'end': 9, 'probability': 0.9988381005599081}], '所属专辑': [{'text': '爱的故事', 'start': 12, 'end': 16, 'probability': 0.9968462078543183}]}}, {'text': '爱的故事', 'start': 12, 'end': 16, 'probability': 0.2816869478191606, 'relations': {'歌手': [{'text': '孙耀威', 'start': 6, 'end': 9, 'probability': 0.9951415104192272}]}}]}]
            '''

            # Event Extraction
            schema = [{'地震触发词': ['地震强度', '时间', '震中位置', '震源深度']}] # Define the schema for event extraction
            ie.set_schema(schema) # Reset schema
            ie('中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。')
            '''
            [{'地震触发词': [{'text': '地震', 'start': 56, 'end': 58, 'probability': 0.9977425555988333, 'relations': {'地震强度': [{'text': '3.5级', 'start': 52, 'end': 56, 'probability': 0.998080217831891}], '时间': [{'text': '5月16日06时08分', 'start': 11, 'end': 22, 'probability': 0.9853299772936026}], '震中位置': [{'text': '云南临沧市凤庆县(北纬24.34度，东经99.98度)', 'start': 23, 'end': 50, 'probability': 0.7874012889740385}], '震源深度': [{'text': '10千米', 'start': 63, 'end': 67, 'probability': 0.9937974422968665}]}}]}]
            '''

            # Opinion Extraction
            schema = [{'评价维度': ['观点词', '情感倾向[正向，负向]']}] # Define the schema for opinion extraction
            ie.set_schema(schema) # Reset schema
            ie("地址不错，服务一般，设施陈旧")
            '''
            [{'评价维度': [{'text': '地址', 'start': 0, 'end': 2, 'probability': 0.9888139270606509, 'relations': {'观点词': [{'text': '不错', 'start': 2, 'end': 4, 'probability': 0.9927847072459528}], '情感倾向[正向，负向]': [{'text': '正向', 'probability': 0.998228967796706}]}}, {'text': '设施', 'start': 10, 'end': 12, 'probability': 0.9588297379365116, 'relations': {'观点词': [{'text': '陈旧', 'start': 12, 'end': 14, 'probability': 0.9286753967902683}], '情感倾向[正向，负向]': [{'text': '负向', 'probability': 0.9949389795770394}]}}, {'text': '服务', 'start': 5, 'end': 7, 'probability': 0.9592857070501211, 'relations': {'观点词': [{'text': '一般', 'start': 7, 'end': 9, 'probability': 0.9949359182521675}], '情感倾向[正向，负向]': [{'text': '负向', 'probability': 0.9952498258302498}]}}]}]
            '''

            # Sentence-level Sentiment Classification
            schema = ['情感倾向[正向，负向]'] # Define the schema for sentence-level sentiment classification
            ie.set_schema(schema) # Reset schema
            ie('这个产品用起来真的很流畅，我非常喜欢')
            '''
            [{'情感倾向[正向，负向]': [{'text': '正向', 'probability': 0.9990024058203417}]}]
            '''

            # English Model
            schema = [{'Person': ['Company', 'Position']}]
            ie_en = Taskflow('information_extraction', schema=schema, model='uie-base-en')
            ie_en('In 1997, Steve was excited to become the CEO of Apple.')
            '''
            [{'Person': [{'text': 'Steve', 'start': 9, 'end': 14, 'probability': 0.999631971804547, 'relations': {'Company': [{'text': 'Apple', 'start': 48, 'end': 53, 'probability': 0.9960158209451642}], 'Position': [{'text': 'CEO', 'start': 41, 'end': 44, 'probability': 0.8871063806420736}]}}]}]
            '''

            schema = ['Sentiment classification [negative, positive]']
            ie_en.set_schema(schema)
            ie_en('I am sorry but this is the worst film I have ever seen in my life.')
            '''
            [{'Sentiment classification [negative, positive]': [{'text': 'negative', 'probability': 0.9998415771287057}]}]
            '''

            schema = [{'Comment object': ['Opinion', 'Sentiment classification [negative, positive]']}]
            ie_en.set_schema(schema)
            ie_en("overall i 'm happy with my toy.")
            '''
            
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

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "model_config": "model_config.json",
        "vocab_file": "vocab.txt",
        "special_tokens_map": "special_tokens_map.json",
        "tokenizer_config": "tokenizer_config.json"
    }
    # vocab.txt/special_tokens_map.json/tokenizer_config.json are common to the default model.
    resource_files_urls = {
        "uie-base": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_v1.0/model_state.pdparams",
                "aeca0ed2ccf003f4e9c6160363327c9b"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/model_config.json",
                "a36c185bfc17a83b6cfef6f98b29c909"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
                "59acb0ce78e79180a2491dfd8382b28c"
            ]
        },
        "uie-medium": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medium_v1.0/model_state.pdparams",
                "15874e4e76d05bc6de64cc69717f172e"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medium/model_config.json",
                "6f1ee399398d4f218450fbbf5f212b15"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
                "59acb0ce78e79180a2491dfd8382b28c"
            ]
        },
        "uie-mini": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini_v1.0/model_state.pdparams",
                "f7b493aae84be3c107a6b4ada660ce2e"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini/model_config.json",
                "9229ce0a9d599de4602c97324747682f"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
                "59acb0ce78e79180a2491dfd8382b28c"
            ]
        },
        "uie-micro": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_micro_v1.0/model_state.pdparams",
                "80baf49c7f853ab31ac67802104f3f15"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_micro/model_config.json",
                "07ef444420c3ab474f9270a1027f6da5"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
                "59acb0ce78e79180a2491dfd8382b28c"
            ]
        },
        "uie-nano": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_nano_v1.0/model_state.pdparams",
                "ba934463c5cd801f46571f2588543700"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_nano/model_config.json",
                "e3a9842edf8329ccdd0cf6039cf0a8f8"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
                "59acb0ce78e79180a2491dfd8382b28c"
            ]
        },
        # Rename to `uie-medium` and the name of `uie-tiny` will be deprecated in future.
        "uie-tiny": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny_v0.1/model_state.pdparams",
                "15874e4e76d05bc6de64cc69717f172e"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/model_config.json",
                "6f1ee399398d4f218450fbbf5f212b15"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
                "59acb0ce78e79180a2491dfd8382b28c"
            ]
        },
        "uie-medical-base": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medical_base_v0.1/model_state.pdparams",
                "569b4bc1abf80eedcdad5a6e774d46bf"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/model_config.json",
                "a36c185bfc17a83b6cfef6f98b29c909"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
                "59acb0ce78e79180a2491dfd8382b28c"
            ]
        },
        "uie-base-en": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en_v1.0/model_state.pdparams",
                "d12e03c2bfe2824c876883b4b836d79d"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/model_config.json",
                "2ca9fe0eea8ff9418725d1a24fcf5c36"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/vocab.txt",
                "64800d5d8528ce344256daf115d4965e"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/tokenizer_config.json",
                "59acb0ce78e79180a2491dfd8382b28c"
            ]
        },
    }

    def __init__(self, task, model, schema, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._schema_tree = None
        self.set_schema(schema)
        self._check_task_files()
        self._construct_tokenizer()
        self._check_predictor_type()
        self._get_inference_model()
        self._usage = usage
        self._is_en = False if model not in [
            "uie-base-en",
        ] else True
        self._max_seq_len = self.kwargs[
            'max_seq_len'] if 'max_seq_len' in self.kwargs else 512
        self._batch_size = self.kwargs[
            'batch_size'] if 'batch_size' in self.kwargs else 64
        self._split_sentence = self.kwargs[
            'split_sentence'] if 'split_sentence' in self.kwargs else False
        self._position_prob = self.kwargs[
            'position_prob'] if 'position_prob' in self.kwargs else 0.5
        self._lazy_load = self.kwargs[
            'lazy_load'] if 'lazy_load' in self.kwargs else False
        self._num_workers = self.kwargs[
            'num_workers'] if 'num_workers' in self.kwargs else 0

    def set_schema(self, schema):
        if isinstance(schema, dict) or isinstance(schema, str):
            schema = [schema]
        self._schema_tree = self._build_tree(schema)

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
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
        model_instance = UIE.from_pretrained(self._task_path)
        self._model = model_instance
        self._model.eval()

    def _construct_tokenizer(self):
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
        inputs = self._check_input_text(inputs)
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
                                                 return_dict=False,
                                                 return_offsets_mapping=True)
                encoded_inputs = encoded_inputs[0]
                tokenized_output = [
                    encoded_inputs["input_ids"],
                    encoded_inputs["token_type_ids"],
                    encoded_inputs["position_ids"],
                    encoded_inputs["attention_mask"],
                    encoded_inputs["offset_mapping"]
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
            input_ids, token_type_ids, pos_ids, att_mask, offset_maps = batch
            if self._predictor_type == "paddle-inference":
                self.input_handles[0].copy_from_cpu(input_ids.numpy())
                self.input_handles[1].copy_from_cpu(token_type_ids.numpy())
                self.input_handles[2].copy_from_cpu(pos_ids.numpy())
                self.input_handles[3].copy_from_cpu(att_mask.numpy())
                self.predictor.run()
                start_prob = self.output_handle[0].copy_to_cpu().tolist()
                end_prob = self.output_handle[1].copy_to_cpu().tolist()
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

            for start_ids, end_ids, ids, offset_map in zip(
                    start_ids_list, end_ids_list, input_ids.tolist(),
                    offset_maps.tolist()):
                for i in reversed(range(len(ids))):
                    if ids[i] != 0:
                        ids = ids[:i]
                        break
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
