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

import os
import abc
import math
from abc import abstractmethod
import paddle
from ..utils.env import PPNLP_HOME
from ..utils.log import logger
from .utils import download_check, static_mode_guard, dygraph_mode_guard, download_file, cut_chinese_sent


class Task(metaclass=abc.ABCMeta):
    """
    The meta classs of task in Taskflow. The meta class has the five abstract function,
        the subclass need to inherit from the meta class.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    def __init__(self, model, task, priority_path=None, **kwargs):
        self.model = model
        self.task = task
        self.priority_path = priority_path
        self.kwargs = kwargs
        self._usage = ""
        # The dygraph model instantce 
        self._model = None
        # The static model instantce
        self._input_spec = None
        self._config = None
        # The root directory for storing Taskflow related files, default to ~/.paddlenlp.
        self._home_path = self.kwargs[
            'home_path'] if 'home_path' in self.kwargs else PPNLP_HOME
        self._task_flag = self.kwargs[
            'task_flag'] if 'task_flag' in self.kwargs else self.model
        if 'task_path' in self.kwargs:
            self._task_path = self.kwargs['task_path']
        elif self.priority_path:
            self._task_path = os.path.join(self._home_path, "taskflow",
                                           self.priority_path)
        else:
            self._task_path = os.path.join(self._home_path, "taskflow",
                                           self.task, self.model)
        download_check(self._task_flag)

    @abstractmethod
    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """

    @abstractmethod
    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """

    @abstractmethod
    def _preprocess(self, inputs, padding=True, add_special_tokens=True):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """

    @abstractmethod
    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function. 
        """

    @abstractmethod
    def _postprocess(self, inputs):
        """
        The model output is the logits and pros, this function will convert the model output to raw text.
        """

    @abstractmethod
    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """

    def _check_task_files(self):
        """
        Check files required by the task.
        """
        for file_id, file_name in self.resource_files_names.items():
            path = os.path.join(self._task_path, file_name)
            if not os.path.exists(path):
                url = self.resource_files_urls[self.model][file_id]
                download_file(self._task_path, file_name, url[0], url[1])

    def _prepare_static_mode(self):
        """
        Construct the input data and predictor in the PaddlePaddele static mode. 
        """
        place = paddle.get_device()
        if place == 'cpu':
            self._config.disable_gpu()
        else:
            self._config.enable_use_gpu(100, self.kwargs['device_id'])
            # TODO(linjieccc): enable embedding_eltwise_layernorm_fuse_pass after fixed
            self._config.delete_pass("embedding_eltwise_layernorm_fuse_pass")
        self._config.switch_use_feed_fetch_ops(False)
        self._config.disable_glog_info()
        self._config.enable_memory_optim()
        self.predictor = paddle.inference.create_predictor(self._config)
        self.input_handles = [
            self.predictor.get_input_handle(name)
            for name in self.predictor.get_input_names()
        ]
        self.output_handle = [
            self.predictor.get_output_handle(name)
            for name in self.predictor.get_output_names()
        ]

    def _get_inference_model(self):
        """
        Return the inference program, inputs and outputs in static mode. 
        """
        inference_model_path = os.path.join(self._task_path, "static",
                                            "inference")
        if not os.path.exists(inference_model_path + ".pdiparams"):
            with dygraph_mode_guard():
                self._construct_model(self.model)
                self._construct_input_spec()
                self._convert_dygraph_to_static()

        model_file = inference_model_path + ".pdmodel"
        params_file = inference_model_path + ".pdiparams"
        self._config = paddle.inference.Config(model_file, params_file)
        self._prepare_static_mode()

    def _convert_dygraph_to_static(self):
        """
        Convert the dygraph model to static model.
        """
        assert self._model is not None, 'The dygraph model must be created before converting the dygraph model to static model.'
        assert self._input_spec is not None, 'The input spec must be created before converting the dygraph model to static model.'
        logger.info("Converting to the inference model cost a little time.")
        static_model = paddle.jit.to_static(
            self._model, input_spec=self._input_spec)
        save_path = os.path.join(self._task_path, "static", "inference")
        paddle.jit.save(static_model, save_path)
        logger.info("The inference model save in the path:{}".format(save_path))

    def _check_input_text(self, inputs):
        inputs = inputs[0]
        if isinstance(inputs, str):
            if len(inputs) == 0:
                raise ValueError(
                    "Invalid inputs, input text should not be empty text, please check your input.".
                    format(type(inputs)))
            inputs = [inputs]
        elif isinstance(inputs, list):
            if not (isinstance(inputs[0], str) and len(inputs[0].strip()) > 0):
                raise TypeError(
                    "Invalid inputs, input text should be list of str, and first element of list should not be empty text.".
                    format(type(inputs[0])))
        else:
            raise TypeError(
                "Invalid inputs, input text should be str or list of str, but type of {} found!".
                format(type(inputs)))
        return inputs

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

    def _auto_joiner(self, short_results, input_mapping, is_dict=False):
        '''
        Join the short results automatically and generate the final results to match with the user inputs.
        Args:
            short_results (List[dict] / List[List[str]] / List[str]): input raw texts.
            input_mapping (dict): cutting length.
            is_dict (bool): whether the element type is dict, default to False.
        return:
            short_input_texts (List[str]): the short input texts for model inference.
        '''
        concat_results = []
        elem_type = {} if is_dict else []
        for k, vs in input_mapping.items():
            single_results = elem_type
            for v in vs:
                if len(single_results) == 0:
                    single_results = short_results[v]
                elif isinstance(elem_type, list):
                    single_results.extend(short_results[v])
                elif isinstance(elem_type, dict):
                    for sk in single_results.keys():
                        if isinstance(single_results[sk], str):
                            single_results[sk] += short_results[v][sk]
                        else:
                            single_results[sk].extend(short_results[v][sk])
                else:
                    raise ValueError(
                        "Invalid element type, the type of results "
                        "for each element should be list of dict, "
                        "but {} received.".format(type(single_results)))
            concat_results.append(single_results)
        return concat_results

    def help(self):
        """
        Return the usage message of the current task.
        """
        print("Examples:\n{}".format(self._usage))

    def __call__(self, *args):
        inputs = self._preprocess(*args)
        outputs = self._run_model(inputs)
        results = self._postprocess(outputs)
        return results
