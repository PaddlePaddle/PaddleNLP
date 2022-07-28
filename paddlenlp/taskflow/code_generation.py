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
from ..data import Pad
from ..transformers import CodeGenForCausalLM, CodeGenTokenizer
from .task import Task

usage = r"""
           from paddlenlp import Taskflow 

           codegen = Taskflow("code_generation")
           codegen("def hello_world():")
           '''
           ['\n    print("Hello world")']
           '''
         """


class CodeGenerationTask(Task):
    """
    The text generation model to predict the code. 
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._batch_size = kwargs.get("batch_size", 1)
        self._max_length = kwargs.get("max_length", 128)
        self._min_length = kwargs.get("min_length", 0)
        self._decode_strategy = kwargs.get("decode_strategy", 'sampling')
        self._temperature = kwargs.get("temperature", 0.6)
        self._top_k = kwargs.get("top_k", 5)
        self._top_p = kwargs.get("top_p", 1.)
        self._num_beams = kwargs.get("num_beams", 4)
        self._length_penalty = kwargs.get("length_penalty", 1.0)
        self._repetition_penalty = kwargs.get("repetition_penalty", 1.1)
        self._output_scores = kwargs.get("output_scores", False)
        self._use_faster = kwargs.get("use_faster", False)
        self._construct_tokenizer(model)
        self._construct_model(model)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        self._model = CodeGenForCausalLM.from_pretrained(model)
        self._model.eval()

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = CodeGenTokenizer.from_pretrained(model)

    def _batchify(self, data, batch_size):
        """
        Generate input batches.
        """
        padding = False if batch_size == 1 else True
        pad_func = Pad(pad_val=self._model.pad_token_id,
                       pad_right=False,
                       dtype=np.int64)

        def _parse_batch(batch_examples):
            if padding:
                input_ids = pad_func([example for example in batch_examples])
            else:
                input_ids = np.asarray([example for example in batch_examples],
                                       dtype=np.int64)
            return input_ids

        examples = self._convert_text_to_input(data)['input_ids']

        # Seperates data into some batches.
        one_batch = []
        for example in examples:
            one_batch.append(example)
            if len(one_batch) == batch_size:
                yield _parse_batch(one_batch)
                one_batch = []
        if one_batch:
            yield _parse_batch(one_batch)

    def _convert_text_to_input(self, texts):
        """
        Convert input strings to ids.
        """
        return self._tokenizer(texts)

    def _preprocess(self, inputs):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        batches = self._batchify(inputs, self._batch_size)
        outputs = {}
        outputs['batches'] = batches
        outputs['text'] = inputs
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function.
        """
        all_ids = []
        all_scores = []

        for batch in inputs["batches"]:
            input_ids = paddle.to_tensor(batch)
            ids, scores = self._model.generate(
                input_ids=input_ids,
                max_length=self._max_length,
                min_length=self._min_length,
                decode_strategy=self._decode_strategy,
                temperature=self._temperature,
                top_k=self._top_k,
                top_p=self._top_p,
                num_beams=self._num_beams,
                length_penalty=self._length_penalty,
                repetition_penalty=self._repetition_penalty,
                use_faster=self._use_faster)
            all_ids.extend(ids.numpy().tolist())
            all_scores.extend(scores.numpy().tolist())
        inputs['ids'] = all_ids
        inputs['scores'] = all_scores
        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """
        batch_out = []
        generated_ids = inputs['ids']
        for generated_id in generated_ids:
            text = self._tokenizer.decode(generated_id,
                                          skip_special_tokens=True,
                                          spaces_between_special_tokens=False)
            text = re.split("\nclass|\ndef|\n#|\n@|\nprint|\nif",
                            text)[0].rstrip()
            batch_out.append(text)
        if self._output_scores:
            return batch_out, inputs['scores']
        return batch_out

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64",
                                    name='input_ids'),
        ]
