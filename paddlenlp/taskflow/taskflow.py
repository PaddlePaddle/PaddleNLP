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

import warnings
import paddle
from ..utils.tools import get_env_device
from ..transformers import ErnieCtmWordtagModel, ErnieCtmTokenizer
from .text2knowledge import WordTagTask
from .sentiment_analysis import SentaTask

warnings.simplefilter(action='ignore', category=Warning, lineno=0, append=False)

TASKS = {
    "text2knowledge": {
        "models": {
            "wordtag": {
                "task_class": WordTagTask,
            }
        },
        "default": {
            "model": "wordtag"
        }
    },
    'sentiment_analysis': {
        "models": {
            "senta": {
                "task_class": SentaTask
            },
        },
        "default": {
            "model": "senta"
        }
    }
}


class TaskFlow(object):
    """
    The TaskFlow is the end2end inferface that could convert the raw text to model result, and decode the model result to task result. The main functions as follows:
        1) Convert the raw text to task result.
        2) Convert the model to the inference model.
        3) Offer the usage and help message.
    Args:
        task (str): The task name for the TaskFlow, and get the task class from the name.
        model (str, optional): The model name in the task, if set None, will use the default model.  
        device_id (int, optional): The device id for the gpu, xpu and other devices, the defalut value is 0.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 

    """

    def __init__(self, task, model=None, device_id=0, **kwargs):
        assert task in TASKS, "The task name:{} is not in TaskFlow list, please check your task name.".format(
            task)
        self.task = task
        if model is not None:
            assert model in set(TASKS[task]['models'].keys(
            )), "The model name:{} is not in task:[{}]".format(model)
        else:
            model = TASKS[task]['default']['model']
        # Set the device for the task
        device = get_env_device()
        if device == 'cpu' or device_id == -1:
            paddle.set_device('cpu')
        else:
            paddle.set_device(device + ":" + str(device_id))

        self.model = model
        # Update the task config to kwargs
        config_kwargs = TASKS[self.task]['models'][self.model]
        kwargs.update(config_kwargs)
        self.kwargs = kwargs
        task_class = TASKS[self.task]['models'][self.model]['task_class']
        self.task_instance = task_class(
            model=self.model, task=self.task, **self.kwargs)
        task_list = TASKS.keys()
        TaskFlow.task_list = task_list

    def __call__(self, *inputs):
        """
        The main work function in the taskflow.
        """
        results = self.task_instance(inputs)
        return results

    def help(self):
        """
        Return the task usage message.
        """
        return self.task_instance.help()

    @staticmethod
    def tasks():
        """
        Return the available task list.
        """
        task_list = list(TASKS.keys())
        return task_list
