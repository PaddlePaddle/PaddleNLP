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

from fastapi import FastAPI
from .http_router import HttpRouterManager
from .model_manager import ModelManager
from .taskflow_manager import TaskflowManager
from ..taskflow import Taskflow


class SimpleServer(FastAPI):
    def __init__(self, **kwargs):
        """
        Initial function for the PaddleNLP SimpleServer.
        """
        super().__init__(**kwargs)
        self._router_manager = HttpRouterManager(self)
        self._taskflow_manager = None
        self._model_manager = None
        self._service_name = "paddlenlp"
        self._service_type = None

    def register(
        self, task_name, model_path, tokenizer_name, model_handler, post_handler, precision="fp32", device_id=0
    ):
        """
        The register function for the SimpleServer, the main register argrument as follows:

        Args:
            name(str): The server name for the route.
            model_path (str):
            handler(str):
            device (int|list|str, optional):
        """
        self._server_type = "models"
        model_manager = ModelManager(
            task_name, model_path, tokenizer_name, model_handler, post_handler, precision, device_id
        )
        self._model_manager = model_manager
        # Register transformers model server router
        self._router_manager.register_models_router(task_name)

    def register_taskflow(self, task_name, task, taskflow_handler=None):
        """
        The register function for the SimpleServer, the main register argrument as follows:

        Args:
            name(str): The server name for the route.
            model_or_path (str):
            handler(str):
            device (int|list|str, optional):
        """
        self._server_type = "server"
        check_flag = True

        # Check the task type, it must be the instance of Taskflow or List[Taskflow]
        if isinstance(task, Taskflow):
            task = [task]
        for t in task:
            if not isinstance(t, Taskflow):
                check_flag = False
                break
        if not check_flag:
            raise TypeError(
                "Unsupport task type {}, it must be instance of Taskflow or List[Taskflow]".format(type(task))
            )

        # Register Taskflow server router
        taskflow_manager = TaskflowManager(task, taskflow_handler)
        self._taskflow_manager = taskflow_manager
        self._router_manager.register_taskflow_router(task_name)
