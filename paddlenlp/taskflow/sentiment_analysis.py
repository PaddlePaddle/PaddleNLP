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

import copy
import os

import numpy as np
import paddle

from ..data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
from ..datasets import load_dataset
from ..transformers import UIE, AutoTokenizer, SkepTokenizer
from ..utils.tools import get_bool_ids_greater_than, get_span
from .models import LSTMModel, SkepSequenceModel
from .task import Task
from .utils import SchemaTree, dbc2sbc, get_id_and_prob, static_mode_guard

usage = r"""
            from paddlenlp import Taskflow

            # sentiment analysis with bilstm
            senta = Taskflow("sentiment_analysis")
            senta("怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片")
            '''
            [{'text': '怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片', 'label': 'negative', 'score': 0.6691398620605469}]
            '''

            senta(["怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片",
                   "作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间"])
            '''
            [{'text': '怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片', 'label': 'negative', 'score': 0.6691398620605469},
             {'text': '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间', 'label': 'positive', 'score': 0.9857505559921265}
            ]
            '''

            # sentiment analysis with skep
            senta = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch")
            senta("作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。")
            '''
            [{'text': '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。', 'label': 'positive', 'score': 0.984320878982544}]
            '''

            # sentiment analysis with UIE
            # aspect, opinion and sentiment extraction
            schema = [{'评价维度': ['观点词', '情感倾向[正向,负向,未提及]']}]
            ie = Taskflow('information_extraction', schema=schema,  model="uie-base")
            ie("地址不错，服务一般，设施陈旧")
            '''
            [{'评价维度': [{'text': '地址', 'start': 0, 'end': 2, 'probability': 0.9888139270606509, 'relations': {'观点词': [{'text': '不错', 'start': 2, 'end': 4, 'probability': 0.9927847072459528}], '情感倾向[正向，负向]': [{'text': '正向', 'probability': 0.998228967796706}]}}, {'text': '设施', 'start': 10, 'end': 12, 'probability': 0.9588297379365116, 'relations': {'观点词': [{'text': '陈旧', 'start': 12, 'end': 14, 'probability': 0.9286753967902683}], '情感倾向[正向，负向]': [{'text': '负向', 'probability': 0.9949389795770394}]}}, {'text': '服务', 'start': 5, 'end': 7, 'probability': 0.9592857070501211, 'relations': {'观点词': [{'text': '一般', 'start': 7, 'end': 9, 'probability': 0.9949359182521675}], '情感倾向[正向，负向]': [{'text': '负向', 'probability': 0.9952498258302498}]}}]}]
            '''
            # opinion and sentiment extraction according to pre-given aspects
            schema = [{'评价维度': ['观点词', '情感倾向[正向,负向,未提及]']}]
            aspects = ['服务', '价格']
            ie = Taskflow("sentiment_analysis", model="uie-base", schema=schema, aspects=aspects)
            ie("蛋糕味道不错，很好吃，店家服务也很好")
            '''
            [{'评价维度': [{'text': '服务', 'relations': {'观点词': [{'text': '好', 'start': 17, 'end': 18, 'probability': 0.9998383583299955}], '情感倾向[正向,负向,未提及]': [{'text': '正向', 'probability': 0.9999240650320473}]}}, {'text': '价格', 'relations': {'情感倾向[正向,负向,未提及]': [{'text': '未提及', 'probability': 0.9999845028521719}]}}]}]
            '''
         """


class SentaTask(Task):
    """
    Sentiment analysis task using RNN or BOW model to predict sentiment opinion on Chinese text.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    resource_files_names = {"model_state": "model_state.pdparams", "vocab": "vocab.txt"}
    resource_files_urls = {
        "bilstm": {
            "vocab": [
                "https://bj.bcebos.com/paddlenlp/taskflow/sentiment_analysis/bilstm/vocab.txt",
                "df714f0bfd6d749f88064679b4c97fd5",
            ],
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/sentiment_analysis/bilstm/model_state.pdparams",
                "609fc068aa35339e20f8310b5c20887c",
            ],
        }
    }

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._static_mode = True
        self._label_map = {0: "negative", 1: "positive"}
        self._check_task_files()
        self._construct_tokenizer(model)
        if self._static_mode:
            self._get_inference_model()
        else:
            self._construct_model(model)
        self._usage = usage

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_ids"),
            paddle.static.InputSpec(shape=[None], dtype="int64", name="length"),
        ]

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        vocab_size = self.kwargs["vocab_size"]
        pad_token_id = self.kwargs["pad_token_id"]
        num_classes = 2

        # Select the senta network for the inference
        model_instance = LSTMModel(
            vocab_size, num_classes, direction="bidirect", padding_idx=pad_token_id, pooling_type="max"
        )
        model_path = os.path.join(self._task_path, "model_state.pdparams")

        # Load the model parameter for the predict
        state_dict = paddle.load(model_path)
        model_instance.set_dict(state_dict)
        self._model = model_instance
        self._model.eval()

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        vocab_path = os.path.join(self._task_path, "vocab.txt")
        vocab = Vocab.load_vocabulary(vocab_path, unk_token="[UNK]", pad_token="[PAD]")

        vocab_size = len(vocab)
        pad_token_id = vocab.to_indices("[PAD]")
        # Construct the tokenizer form the JiebaToeknizer
        self.kwargs["pad_token_id"] = pad_token_id
        self.kwargs["vocab_size"] = vocab_size
        tokenizer = JiebaTokenizer(vocab)
        self._tokenizer = tokenizer

    def _preprocess(self, inputs, padding=True, add_special_tokens=True):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        # Get the config from the kwargs
        batch_size = self.kwargs["batch_size"] if "batch_size" in self.kwargs else 1
        examples = []
        filter_inputs = []
        for input_data in inputs:
            if not (isinstance(input_data, str) and len(input_data) > 0):
                continue
            filter_inputs.append(input_data)
            ids = self._tokenizer.encode(input_data)
            lens = len(ids)
            examples.append((ids, lens))

        batches = [examples[idx : idx + batch_size] for idx in range(0, len(examples), batch_size)]
        outputs = {}
        outputs["data_loader"] = batches
        outputs["text"] = filter_inputs
        return outputs

    def _batchify_fn(self, samples):
        fn = Tuple(
            Pad(axis=0, pad_val=self._tokenizer.vocab.token_to_idx.get("[PAD]", 0)),  # input_ids
            Stack(dtype="int64"),  # seq_len
        )
        return fn(samples)

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function.
        """
        results = []
        scores = []
        with static_mode_guard():
            for batch in inputs["data_loader"]:
                ids, lens = self._batchify_fn(batch)
                self.input_handles[0].copy_from_cpu(ids)
                self.input_handles[1].copy_from_cpu(lens)
                self.predictor.run()
                idx = self.output_handle[0].copy_to_cpu().tolist()
                probs = self.output_handle[1].copy_to_cpu().tolist()
                labels = [self._label_map[i] for i in idx]
                score = [max(prob) for prob in probs]
                results.extend(labels)
                scores.extend(score)

        inputs["result"] = results
        inputs["score"] = scores
        return inputs

    def _postprocess(self, inputs):
        """
        This function will convert the model output to raw text.
        """
        final_results = []
        for text, label, score in zip(inputs["text"], inputs["result"], inputs["score"]):
            result = {}
            result["text"] = text
            result["label"] = label
            result["score"] = score
            final_results.append(result)
        return final_results


class SkepTask(Task):
    """
    Sentiment analysis task using ERNIE-Gram model to predict sentiment opinion on Chinese text.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "model_config": "model_config.json",
    }
    resource_files_urls = {
        "skep_ernie_1.0_large_ch": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/sentiment_analysis/skep_ernie_1.0_large_ch/model_state.pdparams",
                "cf7aa5f5ffa834b329bbcb1dca54e9fc",
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/sentiment_analysis/skep_ernie_1.0_large_ch/model_config.json",
                "847b84ab08611a2f5a01a22c18b0be23",
            ],
        },
        "__internal_testing__/tiny-random-skep": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/models/community/__internal_testing__/tiny-random-skep/model_state.pdparams",
                "3bedff32b4de186252094499d1c8ede3",
            ],
            "model_config": [
                "https://paddlenlp.bj.bcebos.com/models/community/__internal_testing__/tiny-random-skep/model_config.json",
                "f891e4a927f946c23bc32653f535510b",
            ],
        },
    }

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._static_mode = True
        self._label_map = {0: "negative", 1: "positive"}
        if not self._custom_model:
            self._check_task_files()
        self._construct_tokenizer(self._task_path if self._custom_model else model)
        if self._static_mode:
            self._get_inference_model()
        else:
            self._construct_model(self._task_path if self._custom_model else model)
        self._usage = usage

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        model_instance = SkepSequenceModel.from_pretrained(self._task_path, num_labels=len(self._label_map))
        self._model = model_instance
        self._model.eval()

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # segment_ids
        ]

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        tokenizer = SkepTokenizer.from_pretrained(model)
        self._tokenizer = tokenizer

    def _preprocess(self, inputs, padding=True, add_special_tokens=True):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        # Get the config from the kwargs
        batch_size = self.kwargs["batch_size"] if "batch_size" in self.kwargs else 1

        examples = []
        filter_inputs = []
        for input_data in inputs:
            if not (isinstance(input_data, str) and len(input_data.strip()) > 0):
                continue
            filter_inputs.append(input_data)
            encoded_inputs = self._tokenizer(text=input_data, max_seq_len=128)
            ids = encoded_inputs["input_ids"]
            segment_ids = encoded_inputs["token_type_ids"]
            examples.append((ids, segment_ids))

        batches = [examples[idx : idx + batch_size] for idx in range(0, len(examples), batch_size)]
        outputs = {}
        outputs["text"] = filter_inputs
        outputs["data_loader"] = batches
        return outputs

    def _batchify_fn(self, samples):
        fn = Tuple(
            Pad(axis=0, pad_val=self._tokenizer.pad_token_id),  # input ids
            Pad(axis=0, pad_val=self._tokenizer.pad_token_type_id),  # token type ids
        )
        return fn(samples)

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function.
        """
        results = []
        scores = []
        with static_mode_guard():
            for batch in inputs["data_loader"]:
                ids, segment_ids = self._batchify_fn(batch)
                self.input_handles[0].copy_from_cpu(ids)
                self.input_handles[1].copy_from_cpu(segment_ids)
                self.predictor.run()
                idx = self.output_handle[0].copy_to_cpu().tolist()
                probs = self.output_handle[1].copy_to_cpu().tolist()
                labels = [self._label_map[i] for i in idx]
                score = [max(prob) for prob in probs]
                results.extend(labels)
                scores.extend(score)

        inputs["result"] = results
        inputs["score"] = scores
        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """
        final_results = []
        for text, label, score in zip(inputs["text"], inputs["result"], inputs["score"]):
            result = {}
            result["text"] = text
            result["label"] = label
            result["score"] = score
            final_results.append(result)
        return final_results


class UIESentaTask(Task):
    """
    Universal Information Extraction Task.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        aspects (list[string]):  a list of pre-given aspects
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "model_config": "model_config.json",
        "vocab_file": "vocab.txt",
        "special_tokens_map": "special_tokens_map.json",
        "tokenizer_config": "tokenizer_config.json",
    }
    # vocab.txt/special_tokens_map.json/tokenizer_config.json are common to the default model.
    resource_files_urls = {
        "uie-senta-base": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-base/model_state.pdparams",
                "88fcf3aa5afee16ddb61b4ecdf53f572",
            ],
            "model_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-base/model_config.json",
                "74f033ab874a1acddb3aec9b9c4d9cde",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-base/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-base/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-base/tokenizer_config.json",
                "3e623b57084882fd73e17f544bdda47d",
            ],
        },
        "uie-senta-medium": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-medium/model_state.pdparams",
                "afc11ed983a0075f4bb13cf203ccd841",
            ],
            "model_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-medium/model_config.json",
                "4c98a7bc547d60ac94e44e17c47a3488",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-medium/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-medium/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-medium/tokenizer_config.json",
                "3e623b57084882fd73e17f544bdda47d",
            ],
        },
        "uie-senta-mini": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-mini/model_state.pdparams",
                "83d5082596cfd95b9548aefc248c7ad1",
            ],
            "model_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-mini/model_config.json",
                "9628a5c64a1e6ed8278c0344c8ef874a",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-mini/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-mini/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-mini/tokenizer_config.json",
                "3e623b57084882fd73e17f544bdda47d",
            ],
        },
        "uie-senta-micro": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-micro/model_state.pdparams",
                "047b5549dc182cfca036c3fce1e7f6f7",
            ],
            "model_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-micro/model_config.json",
                "058a28845781dbe89a3827bc11355bc8",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-micro/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-micro/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-micro/tokenizer_config.json",
                "3e623b57084882fd73e17f544bdda47d",
            ],
        },
        "uie-senta-nano": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-nano/model_state.pdparams",
                "27afd8946f47a2b8618ffae9ac0f5922",
            ],
            "model_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-nano/model_config.json",
                "b9f74bdf02f5fb2d208e1535c8a13649",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-nano/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-nano/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/uie-senta-nano/tokenizer_config.json",
                "3e623b57084882fd73e17f544bdda47d",
            ],
        },
    }

    def __init__(self, task, model, schema, aspects=None, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._schema_tree = None
        self.set_schema(schema)
        self._check_task_files()
        self._check_predictor_type()
        self._get_inference_model()
        self._usage = usage
        self._max_seq_len = self.kwargs["max_seq_len"] if "max_seq_len" in self.kwargs else 512
        self._batch_size = self.kwargs["batch_size"] if "batch_size" in self.kwargs else 64
        self._split_sentence = self.kwargs["split_sentence"] if "split_sentence" in self.kwargs else False
        self._position_prob = self.kwargs["position_prob"] if "position_prob" in self.kwargs else 0.5
        self._lazy_load = self.kwargs["lazy_load"] if "lazy_load" in self.kwargs else False
        self._num_workers = self.kwargs["num_workers"] if "num_workers" in self.kwargs else 0
        self.use_fast = self.kwargs["use_fast"] if "use_fast" in self.kwargs else False
        self._construct_tokenizer()
        self.aspects = self._check_aspects(aspects)

    def set_schema(self, schema):
        """
        Set schema for UIE Model.
        """
        if isinstance(schema, dict) or isinstance(schema, str):
            schema = [schema]
        self._schema_tree = self._build_tree(schema)

    def _check_aspects(self, aspects):
        """
        Check aspects whether to be valid.
        """
        if aspects is None:
            return aspects
        elif not isinstance(aspects, list):
            raise TypeError(
                "Invalid aspects, input aspects should be list of str, but type of {} found!".format(type(aspects))
            )
        elif not aspects:
            raise ValueError("Invalid aspects, input aspects should not be empty, but {} found!".format(aspects))
        else:
            for i, aspect in enumerate(aspects):
                if not isinstance(aspect, str):
                    raise TypeError(
                        "Invalid aspect, the aspect at index {} should be str, but type of {} found!".format(
                            i, type(aspect)
                        )
                    )
                if not aspect.strip():
                    raise ValueError(
                        "Invalid aspect, the aspect at index {} should not be empty, but {} found!".format(i, aspect)
                    )
        return aspects

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="pos_ids"),
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="att_mask"),
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
        self._tokenizer = AutoTokenizer.from_pretrained(self._task_path, use_fast=self.use_fast)

    def _preprocess(self, inputs):
        """
        Read and analyze inputs.
        """
        examples = self._check_input_text(inputs)

        outputs = {}
        outputs["text"] = examples
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
            input_texts, max_predict_len, split_sentence=self._split_sentence
        )

        short_texts_prompts = []
        for k, v in self.input_mapping.items():
            short_texts_prompts.extend([prompts[k] for i in range(len(v))])
        short_inputs = [
            {"text": short_input_texts[i], "prompt": short_texts_prompts[i]} for i in range(len(short_input_texts))
        ]

        def read(inputs):
            for example in inputs:
                encoded_inputs = self._tokenizer(
                    text=[example["prompt"]],
                    text_pair=[example["text"]],
                    truncation=True,
                    max_seq_len=self._max_seq_len,
                    pad_to_max_seq_len=True,
                    return_attention_mask=True,
                    return_position_ids=True,
                    return_offsets_mapping=True,
                )
                tokenized_output = [
                    encoded_inputs["input_ids"][0],
                    encoded_inputs["token_type_ids"][0],
                    encoded_inputs["position_ids"][0],
                    encoded_inputs["attention_mask"][0],
                    encoded_inputs["offset_mapping"][0],
                ]
                tokenized_output = [np.array(x, dtype="int64") for x in tokenized_output]
                yield tuple(tokenized_output)

        infer_ds = load_dataset(read, inputs=short_inputs, lazy=self._lazy_load)
        batch_sampler = paddle.io.BatchSampler(dataset=infer_ds, batch_size=self._batch_size, shuffle=False)

        infer_data_loader = paddle.io.DataLoader(
            dataset=infer_ds, batch_sampler=batch_sampler, num_workers=self._num_workers, return_list=True
        )

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
                    "att_mask": att_mask.numpy(),
                }
                start_prob, end_prob = self.predictor.run(None, input_dict)
                start_prob = start_prob.tolist()
                end_prob = end_prob.tolist()

            start_ids_list = get_bool_ids_greater_than(start_prob, limit=self._position_prob, return_prob=True)
            end_ids_list = get_bool_ids_greater_than(end_prob, limit=self._position_prob, return_prob=True)

            for start_ids, end_ids, offset_map in zip(start_ids_list, end_ids_list, offset_maps.tolist()):
                span_set = get_span(start_ids, end_ids, with_prob=True)
                sentence_id, prob = get_id_and_prob(span_set, offset_map)
                sentence_ids.append(sentence_id)
                probs.append(prob)
        results = self._convert_ids_to_results(short_inputs, sentence_ids, probs)
        results = self._auto_joiner(results, short_input_texts, self.input_mapping)
        return results

    def _auto_joiner(self, short_results, short_inputs, input_mapping):
        concat_results = []
        is_cls_task = False
        for short_result in short_results:
            if short_result == []:
                continue
            elif "start" not in short_result[0].keys() and "end" not in short_result[0].keys():
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
                    if short_results[v][0]["text"] not in cls_options.keys():
                        cls_options[short_results[v][0]["text"]] = [1, short_results[v][0]["probability"]]
                    else:
                        cls_options[short_results[v][0]["text"]][0] += 1
                        cls_options[short_results[v][0]["text"]][1] += short_results[v][0]["probability"]
                if len(cls_options) != 0:
                    cls_res, cls_info = max(cls_options.items(), key=lambda x: x[1])
                    concat_results.append([{"text": cls_res, "probability": cls_info[1] / cls_info[0]}])
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
                            if "start" not in short_results[v][i] or "end" not in short_results[v][i]:
                                continue
                            short_results[v][i]["start"] += offset
                            short_results[v][i]["end"] += offset
                        offset += len(short_inputs[v])
                        single_results.extend(short_results[v])
                concat_results.append(single_results)
        return concat_results

    def _run_model(self, inputs):
        raw_inputs = inputs["text"]
        results = self._multi_stage_predict(raw_inputs)
        inputs["result"] = results
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
        if self.aspects is not None:
            # predict with pre-give aspects
            results = []
            prefixs = []
            relations = []
            result = {"评价维度": [{"text": aspect} for aspect in self.aspects]}
            prefix = [aspect + "的" for aspect in self.aspects]
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
                    examples.append({"text": one_data, "prompt": dbc2sbc(node.name)})
                    input_map[cnt] = [idx]
                    idx += 1
                    cnt += 1
            else:
                for pre, one_data in zip(node.prefix, data):
                    if len(pre) == 0:
                        input_map[cnt] = []
                    else:
                        for p in pre:
                            examples.append({"text": one_data, "prompt": dbc2sbc(p + node.name)})
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
                            relations[k][i]["relations"] = {node.name: result_list[v[i]]}
                        elif node.name not in relations[k][i]["relations"].keys():
                            relations[k][i]["relations"][node.name] = result_list[v[i]]
                        else:
                            relations[k][i]["relations"][node.name].extend(result_list[v[i]])
                new_relations = [[] for i in range(len(data))]
                for i in range(len(relations)):
                    for j in range(len(relations[i])):
                        if "relations" in relations[i][j].keys() and node.name in relations[i][j]["relations"].keys():
                            for k in range(len(relations[i][j]["relations"][node.name])):
                                new_relations[i].append(relations[i][j]["relations"][node.name][k])
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
                    start += len(prompt) + 1
                    end += len(prompt) + 1
                    result = {"text": prompt[start:end], "probability": prob[i]}
                    result_list.append(result)
                else:
                    result = {"text": text[start:end], "start": start, "end": end, "probability": prob[i]}
                    result_list.append(result)
            results.append(result_list)
        return results

    @classmethod
    def _build_tree(cls, schema, name="root"):
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
                            "but {} received".format(type(v))
                        )
                    schema_tree.add_child(cls._build_tree(child, name=k))
            else:
                raise TypeError("Invalid schema, element should be string or dict, " "but {} received".format(type(s)))
        return schema_tree

    def _postprocess(self, inputs):
        """
        This function will convert the model output to raw text.
        """
        return inputs["result"]
