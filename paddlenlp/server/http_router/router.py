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
import hashlib
import typing
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Extra, create_model

from ...utils.log import logger
from ..base_router import BaseRouterManager


class ResponseBase(BaseModel):
    text: Optional[str] = None


class RequestBase(BaseModel, extra=Extra.forbid):
    parameters: Optional[dict] = {}


class HttpRouterManager(BaseRouterManager):
    def register_models_router(self, task_name):

        # Url path to register the model
        paths = [f"/{task_name}"]
        for path in paths:
            logger.info("   Transformer model request [path]={} is genereated.".format(path))

        # Unique name to create the pydantic model
        unique_name = hashlib.md5(task_name.encode()).hexdigest()

        # Create request model
        req_model = create_model(
            "RequestModel" + unique_name,
            data=(typing.Any, ...),
            __base__=RequestBase,
        )

        # Create response model
        resp_model = create_model(
            "ResponseModel" + unique_name,
            result=(typing.Any, ...),
            __base__=ResponseBase,
        )

        # Template predict endpoint function to dynamically serve different models
        def predict(request: Request, inference_request: req_model):
            result = self._app._model_manager.predict(inference_request.data, inference_request.parameters)
            return {"result": result}

        # Register the route and add to the app
        router = APIRouter()
        for path in paths:
            router.add_api_route(
                path,
                predict,
                methods=["post"],
                summary=f"{task_name.title()}",
                response_model=resp_model,
                response_model_exclude_unset=True,
                response_model_exclude_none=True,
            )
        self._app.include_router(router)

    def register_taskflow_router(self, task_name):

        # Url path to register the model
        paths = [f"/{task_name}"]
        for path in paths:
            logger.info("   Taskflow  request [path]={} is genereated.".format(path))

        # Unique name to create the pydantic model
        unique_name = hashlib.md5(task_name.encode()).hexdigest()

        # Create request model
        req_model = create_model(
            "RequestModel" + unique_name,
            data=(typing.Any, ...),
            __base__=RequestBase,
        )

        # Create response model
        resp_model = create_model(
            "ResponseModel" + unique_name,
            result=(typing.Any, ...),
            __base__=ResponseBase,
        )

        # Template predict endpoint function to dynamically serve different models
        def predict(request: Request, inference_request: req_model):
            result = self._app._taskflow_manager.predict(inference_request.data, inference_request.parameters)
            return {"result": result}

        # Register the route and add to the app
        router = APIRouter()
        for path in paths:
            router.add_api_route(
                path,
                predict,
                methods=["post"],
                summary=f"{task_name.title()}",
                response_model=resp_model,
                response_model_exclude_unset=True,
                response_model_exclude_none=True,
            )
        self._app.include_router(router)
