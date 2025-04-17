# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import json
import queue
import time
import uuid
import shortuuid
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import tritonclient.grpc as grpcclient
from pydantic import BaseModel, Field
from tritonclient import utils as triton_utils


class Req(BaseModel):
    req_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input_ids: Optional[List[int]] = None
    text: Optional[str] = None
    stop_sequences: Optional[List] = None
    messages: Optional[List] = None
    max_dec_len: Optional[int] = None
    seq_len: Optional[int] = None
    min_dec_len: Optional[int] = None
    temperature: Optional[float] = None
    topp: Optional[float] = None
    penalty_score: Optional[float] = None
    frequency_score: Optional[float] = None
    presence_score: Optional[float] = None
    system: Optional[str] = None
    return_all_tokens: Optional[bool] = None
    eos_token_ids: Optional[List[int]] = None
    benchmark: bool = False
    return_usage: Optional[bool] = False
    stream: bool = False
    timeout: int = 300
    model: str = None

    def to_dict_for_infer(self):
        """
        Convert the request parameters into a dictionary

        Returns:
            dict: request parameters in dict format
        """
        req_dict = {}
        for key, value in self.dict().items():
            if value is not None:
                req_dict[key] = value
        return req_dict

    def load_openai_request(self, request_dict: dict):
        """
        Convert openai request to Req
        official OpenAI API documentation: https://platform.openai.com/docs/api-reference/completions/create
        """
        convert_dict = {
            "text": "prompt",
            "frequency_score": "frequency_penalty",
            "max_dec_len": "max_tokens",
            "stream": "stream",
            "return_all_tokens": "best_of",
            "temperature": "temperature",
            "topp": "top_p",
            "presence_score": "presence_penalty",
            "eos_token_ids": "stop",
            "req_id": "id",
            "model": "model",
            "messages": "messages",
        }

        self.__setattr__("req_id", f"chatcmpl-{shortuuid.random()}")
        for key, value in convert_dict.items():
            if request_dict.get(value, None):
                self.__setattr__(key, request_dict.get(value))


def chat_completion_generator(infer_grpc_url: str, req: Req, yield_json: bool) -> Dict:
    """
    Chat completion generator based on Triton inference service.

    Args:
        infer_grpc_url (str): Triton gRPC URL。
        req (Request): request parameters
        yield_json (bool): Whether to return the result in json format

    Returns:
        dict: chat completion result.
            Normal, return {'token': xxx, 'is_end': xxx, 'send_idx': xxx, ..., 'error_msg': '', 'error_code': 0}
            Others, return {'error_msg': xxx, 'error_code': xxx}, error_msg not None, error_code != 0
    """
    class _TritonOutputData:
        def __init__(self):
            self._completed_requests = queue.Queue()

    def _triton_callback(output_data, result, error):
        """Triton callback function"""
        if error:
            output_data._completed_requests.put(error)
        else:
            output_data._completed_requests.put(result)

    def _format_resp(resp_dict):
        if yield_json:
            return json.dumps(resp_dict, ensure_ascii=False) + "\n"
        else:
            return resp_dict

    timeout = req.timeout
    req_id = req.req_id
    req_dict = req.to_dict_for_infer()
    http_received_time = datetime.now()

    inputs = [grpcclient.InferInput("IN", [1], triton_utils.np_to_triton_dtype(np.object_))]
    inputs[0].set_data_from_numpy(np.array([json.dumps([req_dict])], dtype=np.object_))
    outputs = [grpcclient.InferRequestedOutput("OUT")]
    output_data = _TritonOutputData()

    with grpcclient.InferenceServerClient(url=infer_grpc_url, verbose=False) as triton_client:
        triton_client.start_stream(callback=partial(_triton_callback, output_data))

        triton_client.async_stream_infer(model_name="model",
                                            inputs=inputs,
                                            request_id=req_dict['req_id'],
                                            outputs=outputs)
        while True:
            output_item = output_data._completed_requests.get(timeout=timeout)
            if type(output_item) == triton_utils.InferenceServerException:
                error_msg = f"status is {output_item.status()}, msg is {output_item.message()}"
                yield _format_resp({"error_msg": error_msg, "error_code": 500})
                break
            else:
                result = json.loads(output_item.as_numpy("OUT")[0])
                result = result[0] if isinstance(result, list) else result
                result["error_msg"] = result.get("error_msg", "")
                result["error_code"] = result.get("error_code", 0)
                if req.benchmark:
                    result["http_received_time"] = str(http_received_time)
                yield _format_resp(result)
                if (result.get("error_msg") or result.get("error_code")) or result.get("is_end") == 1:
                    break

        triton_client.stop_stream()
        triton_client.close()

def chat_completion_result(infer_grpc_url: str, req: Req) -> Dict:
    """
    Chat completion result with not streaming mode

    Args:
        infer_grpc_url (str): Triton gRPC URL
        req (Req): request parameters

    Returns:
        dict: chat completion result.
            Normal, return {'tokens_all': xxx, ..., 'error_msg': '', 'error_code': 0}
            Others, return {'error_msg': xxx, 'error_code': xxx}, error_msg not None, error_code != 0
    """
    result = ""
    error_resp = None
    for resp in chat_completion_generator(infer_grpc_url, req, yield_json=False):
        if resp.get("error_msg") or resp.get("error_code"):
            error_resp = resp
            error_resp["result"] = ""
        else:
            result += resp.get("token")
        usage = resp.get("usage", None)

    if error_resp:
        return error_resp
    response = {'result': result, 'error_msg': '', 'error_code': 0}
    if req.return_usage:
        response["usage"] = usage
    return response
