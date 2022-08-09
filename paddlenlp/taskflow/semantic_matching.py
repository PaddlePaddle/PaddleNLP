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

import paddle
from ..data import Pad, Tuple
from ..transformers import ErnieCrossEncoder, ErnieTokenizer
from .task import Task

usage = r"""
         from paddlenlp import Taskflow

         semantic_match = Taskflow("semantic_matching", model="rocketqa-zh-dureader-cross-encoder")
         semantic_match([['春天适合种什么花？','春天适合种什么菜？']])
         '''
         [{'text1': '春天适合种什么花？', 'text2': '春天适合种什么菜？', 'scores': 0.004866087343543768}]
         '''

         semantic_match = Taskflow("semantic_matching", model="rocketqa-zh-dureader-cross-encoder", batch_size=2)
         semantic_match([['春天适合种什么花？','春天适合种什么菜？'],['谁有狂三这张高清的','这张高清图，谁有']])
         '''
         [{'text1': '春天适合种什么花？', 'text2': '春天适合种什么菜？', 'scores': 0.004866101313382387}, {'text1': '谁有狂三这张高清的', 'text2': '这张高清图，谁有', 'scores': 0.7051035761833191}]
         '''
         """


class SemanticMatchingTask(Task):
    """
    The text semantic matching model to predict the code. 
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    def __init__(self, task, model, **kwargs):
        self._batch_size = kwargs.get("batch_size", 1)
        self._max_seq_len = kwargs.get("max_seq_len", 384)
        self._construct_tokenizer(model)
        self._construct_model(model)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        self._model = ErnieCrossEncoder(model)
        self._model.eval()

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = ErnieTokenizer.from_pretrained(model)

    def _check_input_text(self, inputs):
        inputs = inputs[0]
        if not all([isinstance(i, list) and i \
            and all(i) and len(i) == 2 for i in inputs]):
            raise TypeError("Invalid input format.")
        return inputs

    def _preprocess(self, inputs):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        examples = []
        filter_inputs = []
        for input_data in inputs:
            filter_inputs.append(input_data)
            encoded_inputs = self._tokenizer(text=input_data[0],
                                             text_pair=input_data[1],
                                             max_seq_len=self._max_seq_len)
            ids = encoded_inputs["input_ids"]
            segment_ids = encoded_inputs["token_type_ids"]
            examples.append((ids, segment_ids))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self._tokenizer.pad_token_id),  # input ids
            Pad(axis=0, pad_val=self._tokenizer.pad_token_type_id
                ),  # token type ids
        ): [data for data in fn(samples)]

        batches = [
            examples[idx:idx + self._batch_size]
            for idx in range(0, len(examples), self._batch_size)
        ]
        outputs = {}
        outputs['text'] = filter_inputs
        outputs['data_loader'] = batches
        self._batchify_fn = batchify_fn
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function.
        """
        all_scores = []
        for batch in inputs['data_loader']:
            input_ids, segment_ids = self._batchify_fn(batch)
            input_ids = paddle.to_tensor(input_ids, dtype='int64')
            segment_ids = paddle.to_tensor(segment_ids, dtype='int64')
            scores = self._model.matching(input_ids=input_ids,
                                          token_type_ids=segment_ids)
            all_scores.extend(scores.numpy().tolist())
        inputs['scores'] = all_scores
        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """
        final_results = []
        for text, score in zip(inputs['text'], inputs['scores']):
            result = {}
            result['text1'] = text[0]
            result['text2'] = text[1]
            result['scores'] = score
            final_results.append(result)
        return final_results

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64",
                                    name='input_ids'),
            paddle.static.InputSpec(shape=[None],
                                    dtype="int64",
                                    name='token_type_ids'),
        ]
