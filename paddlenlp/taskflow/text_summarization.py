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

import numpy as np
import paddle

from ..data import Pad
from ..transformers import (
    AutoModelForConditionalGeneration,
    AutoTokenizer,
    UNIMOForConditionalGeneration,
)
from .task import Task

usage = r"""
           from paddlenlp import Taskflow

           text_summarization = Taskflow("text_summarization")
           text_summarization(2022年，中国房地产进入转型阵痛期，传统“高杠杆、快周转”的模式难以为继，万科甚至直接喊话，中国房地产进入“黑铁时代”)
           '''
            ['万科喊话中国房地产进入“黑铁时代”']
           '''

           text_summarization(['据悉，2022年教育部将围绕“巩固提高、深化落实、创新突破”三个关键词展开工作。要进一步强化学校教育主阵地作用，继续把落实“双减”作为学校工作的重中之重，重点从提高作业设计水平、提高课后服务水平、提高课堂教学水平、提高均衡发展水平四个方面持续巩固提高学校“双减”工作水平。',
          '党参有降血脂，降血压的作用，可以彻底消除血液中的垃圾，从而对冠心病以及心血管疾病的患者都有一定的稳定预防工作作用，因此平时口服党参能远离三高的危害。另外党参除了益气养血，降低中枢神经作用，调整消化系统功能，健脾补肺的功能。'])
           '''
            ['教育部：将从四个方面持续巩固提高学校“双减”工作水平', '党参能降低三高的危害']
           '''
         """


class TextSummarizationTask(Task):
    """
    The text summarization model to predict the summary of an input text.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._batch_size = kwargs.get("batch_size", 1)
        self._output_scores = kwargs.get("output_scores", False)
        self._model_type = None
        self._construct_tokenizer(model)
        self._construct_model(model)
        # Hypter-parameter during generating.
        self._max_length = kwargs.get("max_length", 128)
        self._min_length = kwargs.get("min_length", 0)
        self._decode_strategy = kwargs.get("decode_strategy", "beam_search")
        self._temperature = kwargs.get("temperature", 1.0)
        self._top_k = kwargs.get("top_k", 5)
        self._top_p = kwargs.get("top_p", 1.0)
        self._num_beams = kwargs.get("num_beams", 4)
        self._length_penalty = kwargs.get("length_penalty", 0.0)
        self._num_return_sequences = kwargs.get("num_return_sequences", 1)
        self._repetition_penalty = kwargs.get("repetition_penalty", 1)
        self._use_faster = kwargs.get("use_faster", False)
        self._use_fp16_decoding = kwargs.get("use_fp16_decoding", False)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        if self._custom_model:
            self._model = AutoModelForConditionalGeneration.from_pretrained(
                self._task_path, from_hf_hub=self.from_hf_hub
            )
        else:
            self._model = AutoModelForConditionalGeneration.from_pretrained(model)
        self._model.eval()
        if isinstance(self._model, UNIMOForConditionalGeneration):
            self._model_type = "unimo-text"

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        if self._custom_model:
            self._tokenizer = AutoTokenizer.from_pretrained(self._task_path, from_hf_hub=self.from_hf_hub)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(model)

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
        pad_right = False
        if self._model_type != "unimo-text":
            pad_right = True
        examples = [self._convert_example(i) for i in data]
        # Seperates data into some batches.
        one_batch = []
        for example in examples:
            one_batch.append(example)
            if len(one_batch) == batch_size:
                yield self._parse_batch(one_batch, self._tokenizer.pad_token_id, pad_right)
                one_batch = []
        if one_batch:
            yield self._parse_batch(one_batch, self._tokenizer.pad_token_id, pad_right)

    def _convert_example(self, example, max_seq_len=512, return_length=True):
        """
        Convert all examples into necessary features.
        """
        if self._model_type != "unimo-text":
            tokenized_example = self._tokenizer(
                example, max_length=max_seq_len, padding=False, truncation=True, return_attention_mask=True
            )
        else:
            tokenized_example = self._tokenizer.gen_encode(
                example,
                max_seq_len=max_seq_len,
                add_start_token_for_decoding=True,
                return_length=True,
                is_split_into_words=False,
            )
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

        pad_func = Pad(pad_val=pad_val, pad_right=pad_right, dtype="int32")
        batch_dict = {}
        input_ids = pad_func([example["input_ids"] for example in batch_examples])
        if self._model_type != "unimo-text":
            attention_mask = (input_ids != pad_val).astype("float32")
            batch_dict["input_ids"] = input_ids
            batch_dict["attention_mask"] = attention_mask
        else:
            token_type_ids = pad_func([example["token_type_ids"] for example in batch_examples])
            position_ids = pad_func([example["position_ids"] for example in batch_examples])
            attention_mask = pad_mask([example["attention_mask"] for example in batch_examples])
            seq_len = np.asarray([example["seq_len"] for example in batch_examples], dtype="int32")
            batch_dict["input_ids"] = input_ids
            batch_dict["token_type_ids"] = token_type_ids
            batch_dict["position_ids"] = position_ids
            batch_dict["attention_mask"] = attention_mask
            batch_dict["seq_len"] = seq_len
        return batch_dict

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """
        all_ids = []
        all_scores = []

        for batch in inputs["batches"]:
            input_ids = paddle.to_tensor(batch["input_ids"], dtype="int64")
            token_type_ids = (
                paddle.to_tensor(batch["token_type_ids"], dtype="int64") if "token_type_ids" in batch else None
            )
            position_ids = paddle.to_tensor(batch["position_ids"], dtype="int64") if "position_ids" in batch else None
            attention_mask = paddle.to_tensor(batch["attention_mask"], dtype="float32")
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
                repetition_penalty=self._repetition_penalty,
                bos_token_id=None if self._model_type != "unimo-text" else self._tokenizer.cls_token_id,
                eos_token_id=None if self._model_type != "unimo-text" else self._tokenizer.mask_token_id,
                use_faster=self._use_faster,
                use_fp16_decoding=self._use_fp16_decoding,
            )
            all_ids.extend(ids)
            all_scores.extend(scores)
        inputs["ids"] = all_ids
        inputs["scores"] = all_scores
        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """
        ids_list = inputs["ids"]
        scores_list = inputs["scores"]
        if self._model_type != "unimo-text":
            output_tokens = self._tokenizer.batch_decode(
                ids_list, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            output_scores = [i.numpy() for i in scores_list]
        else:
            results = self._select_from_num_return_sequences(
                ids_list, scores_list, self._max_length, self._num_return_sequences
            )
            output_tokens = [result[0] for result in results]
            output_scores = [result[1] for result in results]

        if self._output_scores:
            return output_tokens, output_scores
        return output_tokens

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

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
        ]
