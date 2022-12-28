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

import time
from .handlers import TaskflowHandler
from .utils import lock_predictor
from ..utils.log import logger


class TaskflowManager:
    """
    The TaskflowManager could predict the raw text.
    """

    def __init__(self, task, taskflow_handler=None):
        self._task = task
        if taskflow_handler is None:
            self._handler_func = TaskflowHandler.process
        else:
            self._handler_func = taskflow_handler.process

    def predict(self, data, parameters):
        t = time.time()
        t = int(round(t * 1000))
        task_index = t % len(self._task)
        logger.info("The predictor id: {} is selected by running the taskflow.".format(task_index))
        with lock_predictor(self._task[task_index]._lock):
            return self._handler_func(self._task[task_index], data, parameters)
