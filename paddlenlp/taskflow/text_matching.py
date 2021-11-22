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

import paddle
from paddlenlp.transformers import BertModel, BertTokenizer

from ..data import Pad, Tuple
from .utils import static_mode_guard
from .task import Task

usage = r"""
         from paddlenlp import Taskflow

         matcher = Taskflow("text_matching")
         matcher(["世界上什么东西最小", "世界上什么东西最小？"])
         '''
         [{'query': '世界上什么东西最小', 'title': '世界上什么东西最小？', 'similarity': 0.992725}]
         '''

         matcher = Taskflow("text_matching", batch_size=2)
         matcher([["光眼睛大就好看吗", "眼睛好看吗？"], ["小蝌蚪找妈妈怎么样", "小蝌蚪找妈妈是谁画的"]])
         '''
         [{'query': '光眼睛大就好看吗', 'title': '眼睛好看吗？', 'similarity': 0.7450271}, {'query': '小蝌蚪找妈妈怎么样', 'title': '小蝌蚪找妈妈是谁画的', 'similarity': 0.8192149}]
         '''
         """

class SimBERTTask(Task):
    """
    Text matching task using SimBERT to predict the similarity of sentence pair.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    def __init__(self, 
                 task, 
                 model, 
                 batch_size=1,
                 max_seq_len=128,
                 **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._static_mode = True
        self._construct_tokenizer(model)
        self._get_inference_model()
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._usage = usage

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name='input_ids'),
            paddle.static.InputSpec(
                shape=[None], dtype="int64", name='token_type_ids'),
        ]

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        self._model = BertModel.from_pretrained(model, pool_act='linear')
        self._model.eval()

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = BertTokenizer.from_pretrained(model)
    
    def _check_input_text(self, inputs):
        inputs = inputs[0]
        if isinstance(inputs[0], str):
            if not len(inputs) == 2:
                raise TypeError(
                    "Invalid inputs, input text should be list[str, str] or list[list[str, str]].")
            inputs = [inputs]
        elif isinstance(inputs[0], list):
            if not (len(inputs[0]) == 2 and isinstance(inputs[0][0], str)):
                    raise TypeError(
                        "Invalid inputs, input text should be list[str, str] or list[list[str, str]].") 
        else:
            raise TypeError(
                "Invalid inputs, input text should be list[str, str] or list[list[str, str]].")
        return inputs

    def _preprocess(self, inputs):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        num_workers = self.kwargs[
            'num_workers'] if 'num_workers' in self.kwargs else 0
        lazy_load = self.kwargs[
            'lazy_load'] if 'lazy_load' in self.kwargs else False

        examples = []

        for data in inputs:
            query, title = data[0], data[1]
            
            query_encoded_inputs = self._tokenizer(
                text=query, max_seq_len=self._max_seq_len)
            query_input_ids = query_encoded_inputs["input_ids"]
            query_token_type_ids = query_encoded_inputs["token_type_ids"]

            title_encoded_inputs = self._tokenizer(
                text=title, max_seq_len=self._max_seq_len)
            title_input_ids = title_encoded_inputs["input_ids"]
            title_token_type_ids = title_encoded_inputs["token_type_ids"]
            
            examples.append((query_input_ids, query_token_type_ids, 
                title_input_ids, title_token_type_ids))

        batches = [
            examples[idx:idx + self._batch_size]
            for idx in range(0, len(examples), self._batch_size)
        ]

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self._tokenizer.pad_token_id),  # query_input
            Pad(axis=0, pad_val=self._tokenizer.pad_token_type_id),  # query_segment
            Pad(axis=0, pad_val=self._tokenizer.pad_token_id),  # title_input
            Pad(axis=0, pad_val=self._tokenizer.pad_token_type_id),  # tilte_segment
        ): [data for data in fn(samples)]

        outputs = {}
        outputs['data_loader'] = batches
        outputs['text'] = inputs
        self._batchify_fn = batchify_fn
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function.
        """
        results = []
        with static_mode_guard():
            for batch in inputs['data_loader']:
                q_ids, q_segment_ids, t_ids, t_segment_ids = self._batchify_fn(batch)
                self.input_handles[0].copy_from_cpu(q_ids)
                self.input_handles[1].copy_from_cpu(q_segment_ids)
                self.predictor.run()
                vecs_query = self.output_handle[1].copy_to_cpu()

                self.input_handles[0].copy_from_cpu(t_ids)
                self.input_handles[1].copy_from_cpu(t_segment_ids)
                self.predictor.run()
                vecs_title = self.output_handle[1].copy_to_cpu()

                vecs_query = vecs_query / (vecs_query**2).sum(axis=1,
                                                            keepdims=True)**0.5
                vecs_title = vecs_title / (vecs_title**2).sum(axis=1,
                                                            keepdims=True)**0.5
                similarity = (vecs_query * vecs_title).sum(axis=1)
                results.extend(similarity)
        inputs['result'] = results
        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """
        final_results = []
        for text, similarity in zip(inputs['text'], inputs['result']):
            result = {}
            result['query'] = text[0]
            result['title'] = text[1]
            result['similarity'] = similarity
            final_results.append(result)
        return final_results