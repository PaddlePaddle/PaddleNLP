# coding:utf-8
# copyright (c) 2022  paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license"
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import os
import time
import json
import paddlenlp
from .predictor import Predictor
from ..utils.tools import get_env_device
from ..transformers import AutoTokenizer
from .handlers import BaseModelHandler, BasePostHandler
from .utils import lock_predictor
from ..utils.log import logger


class ModelManager:
    def __init__(self, task_name, model_path, tokenizer_name, model_handler, post_handler, precision, device_id):
        self._task_name = task_name
        self._model_path = model_path
        self._tokenizer_name = tokenizer_name
        self._model_handler = model_handler
        self._post_handler = post_handler
        self._precision = precision
        self._device_id = device_id
        self._tokenizer = None
        self._register()

    def _register(self):
        # Get the model handler
        if not issubclass(self._model_handler, BaseModelHandler):
            raise TypeError(
                "The model_handler must be subclass of paddlenlp.server.handlers.BaseModelHandler, please check the type."
            )
        self._model_handler = self._model_handler.process

        if not issubclass(self._post_handler, BasePostHandler):
            raise TypeError(
                "The post_handler must be subclass of paddlenlp.server.handlers.BasePostHandler, please check the type."
            )
        self._post_handler = self._post_handler.process

        # Create the model predictor
        device = get_env_device()
        predictor_list = []
        if device == "cpu" or self._device_id == -1:
            predictor = Predictor(self._model_path, self._precision, "cpu")
            predictor_list.append(predictor)
        elif isinstance(self._device_id, int):
            predictor = Predictor(self._model_path, self._precision, "gpu:" + str(self._device_id))
            predictor_list.append(predictor)
        elif isinstance(self._device_id, list):
            for device in device_id:
                predictor = Predictor(
                    self._model_path,
                    self._model_class_or_name,
                    self._input_spec,
                    self._precision,
                    "gpu:" + str(device),
                )
                predictor_list.append(predictor)
        self._predictor_list = predictor_list

        # Get the tokenize of model
        self._get_tokenizer()

    def _get_tokenizer(self):
        if self._tokenizer_name is not None:
            if isinstance(self._tokenizer_name, str):
                self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
            else:
                logger.error("The argrument of `tokenizer_name`  must be the name of tokenizer.")
        assert self._tokenizer is not None, "The tokenizer must be not register, you could set the class of Tokenizer"

    def _get_predict_id(self):
        t = time.time()
        t = int(round(t * 1000))
        predictor_id = t % len(self._predictor_list)
        logger.info("The predictor id: {} is selected by running the model.".format(predictor_id))
        return predictor_id

    def predict(self, data, parameters):
        predictor_id = self._get_predict_id()
        with lock_predictor(self._predictor_list[predictor_id]._lock):
            model_output = self._model_handler(self._predictor_list[predictor_id], self._tokenizer, data, parameters)
            final_output = self._post_handler(model_output, parameters)
            return final_output
