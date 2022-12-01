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

import glob
import json
import math
import os
import copy
import itertools
import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..transformers import UNIMOLMHeadModel
from ..transformers import UNIMOTokenizer

from ..datasets import load_dataset
from ..data import Stack, Pad, Tuple
from .utils import download_file, add_docstrings, static_mode_guard, dygraph_mode_guard
from .task import Task

usage = r"""
           from paddlenlp import Taskflow 

           question_generation = Taskflow("question_generation")
           question_generation([{"context": "奇峰黄山千米以上的山峰有77座，整座黄山就是一座花岗岩的峰林，自古有36大峰，36小峰，最高峰莲花峰、最险峰天都峰和观日出的最佳点光明顶构成黄山的三大主峰。", "answer": "莲花峰"}]])
           '''
              ['黄山最高峰是什么']
           '''
         """


class QuestionGenerationTask(Task):
    """
    The text summarization model to predict the summary of an input text.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        paddle.set_device(kwargs.get("device", "gpu"))
        self._batch_size = kwargs.get("batch_size", 16)
        self._output_scores = kwargs.get("output_scores", False)
        self._is_select_from_num_return_sequences = kwargs.get("is_select_from_num_return_sequences", True)
        self._construct_tokenizer(model)
        self._construct_model(model)
        # Hypter-parameter during generating.
        self._max_length = kwargs.get("max_length", 50)
        self._min_length = kwargs.get("min_length", 3)
        self._decode_strategy = kwargs.get("decode_strategy", "beam_search")
        self._temperature = kwargs.get("temperature", 1.0)
        self._top_k = kwargs.get("top_k", 0)
        self._top_p = kwargs.get("top_p", 1.0)
        self._num_beams = kwargs.get("num_beams", 6)
        self._num_beam_groups = kwargs.get("num_beam_groups", 1)
        self._diversity_rate = kwargs.get("diversity_rate", 0.0)
        self._length_penalty = kwargs.get("length_penalty", 1.2)
        self._num_return_sequences = kwargs.get("num_return_sequences", 1)
        self._repetition_penalty = kwargs.get("repetition_penalty", 1)
        self._use_faster = kwargs.get("use_faster", False)
        self._use_fp16_decoding = kwargs.get("use_fp16_decoding", False)
        self._template = kwargs.get("template", 1)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        if self._custom_model:
            self._model = UNIMOLMHeadModel.from_pretrained(self._task_path)
        else:
            self._model = UNIMOLMHeadModel.from_pretrained(model)
        self._model.eval()

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        if self._custom_model:
            self._tokenizer = UNIMOTokenizer.from_pretrained(self._task_path)
        else:
            self._tokenizer = UNIMOTokenizer.from_pretrained(model)

    def _preprocess(self, inputs):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        batches = self._batchify(inputs, self._batch_size)
        outputs = {"batches": batches, "text": inputs}
        return outputs

    def _batchify(self, data, batch_size):
        """
        Generate input batches.
        """
        examples = [self._convert_example(i) for i in data]
        # Seperates data into some batches.
        one_batch = []
        for example in examples:
            one_batch.append(example)
            if len(one_batch) == batch_size:
                yield self._parse_batch(one_batch, self._tokenizer.pad_token_id)
                one_batch = []
        if one_batch:
            yield self._parse_batch(one_batch, self._tokenizer.pad_token_id)

    def _check_input_text(self, inputs):
        inputs = inputs[0]
        if isinstance(inputs, str):
            if len(inputs) == 0:
                raise ValueError(
                    "Invalid inputs, input text should not be empty text, please check your input.".format(
                        type(inputs)
                    )
                )
            inputs = [inputs]
        elif isinstance(inputs, dict):
            if not ("source" in inputs and "title" in inputs) and not ("context" in inputs and "answer" in inputs):
                raise TypeError(
                    "Invalid inputs, source and title are not in the input dictionary, nor are context and answer."
                )
        elif isinstance(inputs, list):
            if not (isinstance(inputs[0], dict)):
                raise TypeError("Invalid inputs, input text should be list of dict.".format(type(inputs[0])))
        else:
            raise TypeError(
                "Invalid inputs, input text should be str or list of str, but type of {} found!".format(type(inputs))
            )
        return inputs

    def _convert_example(self, example, max_seq_len=512, return_length=True, template=1):
        """
        Convert all examples into necessary features.
        """
        if isinstance(example, dict):
            target = None
            if "source" in example and "title" in example:
                source = example["source"]
                title = None
                if "title" in example.keys():
                    title = example["title"]
            elif "context" in example and "answer" in example:
                source = example["context"]
                title = None
                if "answer" in example.keys():
                    title = example["answer"]
            else:
                assert False, "Source and title are not in the input dictionary, nor are context and answer."
            if "target" in example.keys():
                target = example["target"]
        elif isinstance(example, list):
            source = example[0]
            title = example[1]

        if self._template == 1:
            ### use template 1
            source = "答案：" + title + self._tokenizer.sep_token + "上下文：" + source
            title = None
            if target:
                target = "问题：" + target
        elif self._template == 2:
            ### use template 2
            source = "答案：" + title + self._tokenizer.sep_token + "上下文：" + source
            title = None
            if target:
                target = "在已知答案的前提下，问题：" + target
        elif self._template == 3:
            ### use template 3
            source = "这是一个问题生成任务，根据提供的答案和上下文，来生成问题。" + title + self._tokenizer.sep_token + "上下文：" + source
            title = None
            if target:
                target = "问题：" + target

        tokenized_example = self._tokenizer.gen_encode(
            source,
            title=title,
            max_seq_len=max_seq_len,
            max_title_len=30,
            add_start_token_for_decoding=True,
            return_position_ids=True,
        )

        if "target" in example and example["target"]:
            tokenized_example["target"] = example["target"]
        # Use to gather the logits corresponding to the labels during training
        return tokenized_example

    def _parse_batch(self, batch_examples, pad_val, pad_right=False):
        """
        Batchify a batch of examples.
        """

        def pad_mask(batch_attention_mask):
            """Pad attention_mask."""
            batch_size = len(batch_attention_mask)
            max_len = max(map(len, batch_attention_mask))
            attention_mask = np.ones((batch_size, max_len, max_len), dtype="float32") * -1e9
            for i, mask_data in enumerate(attention_mask):
                seq_len = len(batch_attention_mask[i])
                if pad_right:
                    mask_data[:seq_len:, :seq_len] = np.array(batch_attention_mask[i], dtype="float32")
                else:
                    mask_data[-seq_len:, -seq_len:] = np.array(batch_attention_mask[i], dtype="float32")
            # In order to ensure the correct broadcasting mechanism, expand one
            # dimension to the second dimension (n_head of Transformer).
            attention_mask = np.expand_dims(attention_mask, axis=1)
            return attention_mask

        pad_func = Pad(pad_val=pad_val, pad_right=pad_right, dtype="int64")
        input_ids = pad_func([example["input_ids"] for example in batch_examples])
        token_type_ids = pad_func([example["token_type_ids"] for example in batch_examples])
        position_ids = pad_func([example["position_ids"] for example in batch_examples])
        attention_mask = pad_mask([example["attention_mask"] for example in batch_examples])
        # seq_len = np.asarray([example['seq_len'] for example in batch_examples],
        #                      dtype='int32')
        batch_dict = {}
        batch_dict["input_ids"] = input_ids
        batch_dict["token_type_ids"] = token_type_ids
        batch_dict["position_ids"] = position_ids
        batch_dict["attention_mask"] = attention_mask
        # batch_dict['seq_len'] = seq_len
        return batch_dict

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """
        all_ids = []
        all_scores = []

        for batch in inputs["batches"]:
            input_ids = paddle.to_tensor(batch["input_ids"], dtype="int64")
            token_type_ids = paddle.to_tensor(batch["token_type_ids"], dtype="int64")
            position_ids = paddle.to_tensor(batch["position_ids"], dtype="int64")
            attention_mask = paddle.to_tensor(batch["attention_mask"], dtype="float32")
            # seq_len = paddle.to_tensor(batch['seq_len'], dtype='int64')
            ids, scores = self._model.generate(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                max_length=self._max_length,
                min_length=self._min_length,
                decode_strategy=self._decode_strategy,
                temperature=self._temperature,
                top_k=self._top_k,
                top_p=self._top_p,
                num_beams=self._num_beams,
                num_beam_groups=self._num_beam_groups,
                diversity_rate=self._diversity_rate,
                length_penalty=self._length_penalty,
                num_return_sequences=self._num_return_sequences,
                repetition_penalty=self._repetition_penalty,
                bos_token_id=self._tokenizer.cls_token_id,
                eos_token_id=self._tokenizer.mask_token_id,
                use_faster=self._use_faster,
                use_fp16_decoding=self._use_fp16_decoding,
            )
            all_ids.extend(ids)
            all_scores.extend(scores)
        inputs["ids"] = all_ids
        inputs["scores"] = all_scores
        return inputs

    def out_run_model(self, input_ids, token_type_ids, position_ids, attention_mask):
        """
        Debug used.
        """
        all_ids = []
        all_scores = []
        # seq_len = paddle.to_tensor(batch['seq_len'], dtype='int64')
        ids, scores = self._model.generate(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            max_length=self._max_length,
            min_length=self._min_length,
            decode_strategy=self._decode_strategy,
            temperature=self._temperature,
            top_k=self._top_k,
            top_p=self._top_p,
            num_beams=self._num_beams,
            length_penalty=self._length_penalty,
            num_return_sequences=self._num_return_sequences,
            bos_token_id=self._tokenizer.cls_token_id,
            eos_token_id=self._tokenizer.mask_token_id,
        )
        all_ids.extend(ids)
        all_scores.extend(scores)

        inputs = {}
        inputs["ids"] = all_ids
        inputs["scores"] = all_scores
        return all_ids, all_scores

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """
        ids_list = inputs["ids"]
        scores_list = inputs["scores"]
        if self._is_select_from_num_return_sequences:
            results = self._select_from_num_return_sequences(
                ids_list, scores_list, self._max_length, self._num_return_sequences
            )
        else:
            results = self._return_num_return_sequences(
                ids_list, scores_list, self._max_length, self._num_return_sequences
            )
        output_tokens = [result[0] for result in results]
        output_scores = [math.exp(result[1]) for result in results]
        # output_scores = [[math.exp(s) for s in result[1]] if isinstance(result[1], list) else math.exp(result[1]) for result in results]

        if self._output_scores:
            return output_tokens, output_scores
        return output_tokens

    def _return_num_return_sequences(self, ids, scores, max_dec_len=None, num_return_sequences=1):
        """
        Select generated sequence form several return sequences.
        """
        results = []
        group = []
        tmp = []
        if scores is not None:
            ids = [i.numpy() for i in ids]
            scores = [i.numpy() for i in scores]

            if len(ids) != len(scores) or (len(ids) % num_return_sequences) != 0:
                raise ValueError(
                    "the length of `ids` is {}, but the `num_return_sequences` is {}".format(
                        len(ids), num_return_sequences
                    )
                )

            for pred, score in zip(ids, scores):
                pred_token_ids, pred_tokens = self._post_process_decoded_sequence(pred)
                num_token = len(pred_token_ids)
                target = "".join(pred_tokens)
                target = self._remove_template(target)
                # not ending
                if max_dec_len is not None and num_token >= max_dec_len:
                    score -= 1e3
                tmp.append([target, score])
                if len(tmp) == num_return_sequences:
                    group.append(tmp)
                    tmp = []
            for preds in group:
                preds = sorted(preds, key=lambda x: -x[1])
                for pred in preds:
                    results.append(pred)
        else:
            ids = ids.numpy()
            for pred in ids:
                pred_token_ids, pred_tokens = self._post_process_decoded_sequence(pred)
                num_token = len(pred_token_ids)
                response = "".join(pred_tokens)
                response = self._remove_template(response)
                # TODO: Support return scores in FT.
                tmp.append([response])
                if len(tmp) == num_return_sequences:
                    group.append(tmp)
                    tmp = []

            for preds in group:
                for pred in preds:
                    results.append(pred)
        return results

    def _select_from_num_return_sequences(self, ids, scores, max_dec_len=None, num_return_sequences=1):
        """
        Select generated sequence form several return sequences.
        """
        results = []
        group = []
        tmp = []
        if scores is not None:
            ids = [i.numpy() for i in ids]
            scores = [i.numpy() for i in scores]

            if len(ids) != len(scores) or (len(ids) % num_return_sequences) != 0:
                raise ValueError(
                    "the length of `ids` is {}, but the `num_return_sequences` is {}".format(
                        len(ids), num_return_sequences
                    )
                )

            for pred, score in zip(ids, scores):
                pred_token_ids, pred_tokens = self._post_process_decoded_sequence(pred)
                num_token = len(pred_token_ids)
                target = "".join(pred_tokens)
                target = self._remove_template(target)
                # not ending
                if max_dec_len is not None and num_token >= max_dec_len:
                    score -= 1e3
                tmp.append([target, score])
                if len(tmp) == num_return_sequences:
                    group.append(tmp)
                    tmp = []
            for preds in group:
                preds = sorted(preds, key=lambda x: -x[1])
                results.append(preds[0])
        else:
            ids = ids.numpy()
            for pred in ids:
                pred_token_ids, pred_tokens = self._post_process_decoded_sequence(pred)
                num_token = len(pred_token_ids)
                response = "".join(pred_tokens)
                response = self._remove_template(response)
                # TODO: Support return scores in FT.
                tmp.append([response])
                if len(tmp) == num_return_sequences:
                    group.append(tmp)
                    tmp = []

            for preds in group:
                results.append(preds[0])
        return results

    def _post_process_decoded_sequence(self, token_ids):
        """Post-process the decoded sequence. Truncate from the first <eos>."""
        eos_pos = len(token_ids)
        for i, tok_id in enumerate(token_ids):
            if tok_id == self._tokenizer.mask_token_id:
                eos_pos = i
                break
        token_ids = token_ids[:eos_pos]
        tokens = self._tokenizer.convert_ids_to_tokens(token_ids)
        tokens = self._tokenizer.merge_subword(tokens)
        special_tokens = ["[UNK]"]
        tokens = [token for token in tokens if token not in special_tokens]
        return token_ids, tokens

    def _remove_template(self, instr):
        """Remove template prefix of decoded sequence."""
        outstr = instr.strip("问题：")
        outstr = instr.strip("在已知答案的前提下，问题：")
        return outstr

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
        ]
