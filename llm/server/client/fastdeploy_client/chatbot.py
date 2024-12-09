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
import logging
import queue
import traceback
import uuid
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from fastdeploy_client.message import ChatMessage
from fastdeploy_client.utils import is_enable_benchmark
from tritonclient import utils as triton_utils


class ChatBotClass(object):
    """
    initiating conversations through the tritonclient interface of the model service.
    """
    def __init__(self, hostname, port, timeout=120):
        """
        Initialization function

        Args:
            hostname (str): gRPC hostname
            port (int): gRPC port
            timeout (int): Request timeout, default is 120 seconds.

        Returns:
            None
        """
        self.url = f"{hostname}:{port}"
        self.timeout = timeout

    def stream_generate(self,
                        message,
                        max_dec_len=1024,
                        min_dec_len=1,
                        topp=0.7,
                        temperature=0.95,
                        frequency_score=0.0,
                        penalty_score=1.0,
                        presence_score=0.0,
                        system=None,
                        **kwargs):
        """
        Streaming interface

        Args:
            message (Union[str, List[str], ChatMessage]):  message or ChatMessage object
            max_dec_len (int, optional): max decoding length. Defaults to 1024.
            min_dec_len (int, optional): min decoding length. Defaults to 1.
            topp (float, optional): randomness of the generated tokens. Defaults to 0.7.
            temperature (float, optional): temperature. Defaults to 0.95.
            frequency_score (float, optional): frequency score. Defaults to 0.0.
            penalty_score (float, optional): penalty score. Defaults to 1.0.
            presence_score (float, optional): presence score. Defaults to 0.0.
            system (str, optional): system settings. Defaults to None.
            **kwargs: others

            For more details, please refer to https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/deploy_usage_tutorial.md#%E8%AF%B7%E6%B1%82%E5%8F%82%E6%95%B0%E4%BB%8B%E7%BB%8D

        Returns:
            return a generator object, which yields a dict.
            Normal, return {'token': xxx, 'is_end': xxx, 'send_idx': xxx, ..., 'error_msg': '', 'error_code': 0}
            Others, return {'error_msg': xxx, 'error_code': xxx}, error_msg not None, error_code != 0
        """
        try:
            model_name = "model"
            inputs = [grpcclient.InferInput("IN", [1], triton_utils.np_to_triton_dtype(np.object_))]
            outputs = [grpcclient.InferRequestedOutput("OUT")]
            output_data = OutputData()

            msg = message.message if isinstance(message, ChatMessage) else message
            input_data = self._prepare_input_data(msg, max_dec_len, min_dec_len,
                                        topp, temperature, frequency_score,
                                        penalty_score, presence_score, **kwargs)
            req_id = input_data["req_id"]
            inputs[0].set_data_from_numpy(np.array([json.dumps([input_data])], dtype=np.object_))
            timeout = kwargs.get("timeout", self.timeout)

            with grpcclient.InferenceServerClient(url=self.url, verbose=False) as triton_client:
                triton_client.start_stream(callback=partial(triton_callback, output_data))
                triton_client.async_stream_infer(model_name=model_name,
                                                    inputs=inputs,
                                                    request_id=req_id,
                                                    outputs=outputs)
                answer_str = ""
                enable_benchmark = is_enable_benchmark(**kwargs)
                while True:
                    try:
                        response = output_data._completed_requests.get(timeout=timeout)
                    except queue.Empty:
                        yield {"req_id": req_id, "error_msg": f"Fetch response from server timeout ({timeout}s)"}
                        break
                    if type(response) == triton_utils.InferenceServerException:
                        yield {"req_id": req_id, "error_msg": f"InferenceServerException raised by inference: {response.message()}"}
                        break
                    else:
                        if enable_benchmark:
                            response = json.loads(response.as_numpy("OUT")[0])
                            if isinstance(response, (list, tuple)):
                                response = response[0]
                        else:
                            response = self._format_response(response, req_id)
                            token = response.get("token", "")
                            if isinstance(token, list):
                                token = token[0]
                            answer_str += token
                        yield response
                        if response.get("is_end") == 1 or response.get("error_msg") is not None:
                            break
            triton_client.stop_stream(cancel_requests=True)
            triton_client.close()

            if isinstance(message, ChatMessage):
                message.message.append({"role": "assistant", "content": answer_str})
        except Exception as e:
            yield {"error_msg": f"{e}, details={str(traceback.format_exc())}"}

    def generate(self,
                 message,
                 max_dec_len=1024,
                 min_dec_len=1,
                 topp=0.7,
                 temperature=0.95,
                 frequency_score=0.0,
                 penalty_score=1.0,
                 presence_score=0.0,
                 system=None,
                 **kwargs):
        """
        Return the entire sentence using the streaming interface.

        Args:
            message (Union[str, List[str], ChatMessage]):  message or ChatMessage object
            max_dec_len (int, optional): max decoding length. Defaults to 1024.
            min_dec_len (int, optional): min decoding length. Defaults to 1.
            topp (float, optional): randomness of the generated tokens. Defaults to 0.7.
            temperature (float, optional): temperature. Defaults to 0.95.
            frequency_score (float, optional): frequency score. Defaults to 0.0.
            penalty_score (float, optional): penalty score. Defaults to 1.0.
            presence_score (float, optional): presence score. Defaults to 0.0.
            system (str, optional): system settings. Defaults to None.
            **kwargs: others

            For more details, please refer to https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/deploy_usage_tutorial.md#%E8%AF%B7%E6%B1%82%E5%8F%82%E6%95%B0%E4%BB%8B%E7%BB%8D

        Returns:
            return the entire sentence or error message.
            Normal, return {'tokens_all': xxx, ..., 'error_msg': '', 'error_code': 0}
            Others, return {'error_msg': xxx, 'error_code': xxx}, error_msg not None, error_code != 0
        """
        stream_response = self.stream_generate(message, max_dec_len,
                                               min_dec_len, topp, temperature,
                                               frequency_score, penalty_score,
                                               presence_score, system, **kwargs)
        results = ""
        token_ids = list()
        error_msg = None
        for res in stream_response:
            if "token" not in res or "error_msg" in res:
                error_msg = {"error_msg": f"response error, please check the info: {res}"}
            elif isinstance(res["token"], list):
                results = res["token"]
                token_ids = res["token_ids"]
            else:
                results += res["token"]
                token_ids += res["token_ids"]
        if error_msg:
            return {"req_id": res["req_id"], "error_msg": error_msg}
        else:
            return {"req_id": res["req_id"], "results": results, "token_ids": token_ids}

    def _prepare_input_data(self,
                        message,
                        max_dec_len=1024,
                        min_dec_len=2,
                        topp=0.0,
                        temperature=1.0,
                        frequency_score=0.0,
                        penalty_score=1.0,
                        presence_score=0.0,
                        system=None,
                        **kwargs):
        """
        Prepare to input data
        """
        inputs = {
            "max_dec_len": max_dec_len,
            "min_dec_len": min_dec_len,
            "topp": topp,
            "temperature": temperature,
            "frequency_score": frequency_score,
            "penalty_score": penalty_score,
            "presence_score": presence_score,
        }

        if system is not None:
            inputs["system"] = system

        inputs["req_id"] = kwargs.get("req_id", str(uuid.uuid4()))
        if "eos_token_ids" in kwargs and kwargs["eos_token_ids"] is not None:
            inputs["eos_token_ids"] = kwargs["eos_token_ids"]
        inputs["response_timeout"] = kwargs.get("timeout", self.timeout)

        if isinstance(message, str):
            inputs["text"] = message
        elif isinstance(message, list):
            assert len(message) % 2 == 1, \
                "The length of message should be odd while it's a list."
            assert message[-1]["role"] == "user", \
                "The {}-th element key should be 'user'".format(len(message) - 1)
            for i in range(0, len(message) - 1, 2):
                assert message[i]["role"] == "user", \
                    "The {}-th element key should be 'user'".format(i)
                assert message[i + 1]["role"] == "assistant", \
                    "The {}-th element key should be 'assistant'".format(i + 1)
            inputs["messages"] = message
        else:
            raise Exception(
                "The message should be string or list of dict like [{'role': "
                "'user', 'content': 'Hello, what's your name?''}]"
            )

        return inputs

    def _format_response(self, response, req_id):
        """
        Format the service return fields
        """
        response = json.loads(response.as_numpy("OUT")[0])
        if isinstance(response, (list, tuple)):
            response = response[0]
        is_end = response.get("is_end", False)

        if "error_msg" in response:
            return {"req_id": req_id, "error_msg": response["error_msg"]}
        elif "choices" in response:
            token = [x["token"] for x in response["choices"]]
            token_ids = [x["token_ids"] for x in response["choices"]]
            return {"req_id": req_id, "token": token, "token_ids": token_ids, "is_end": 1}
        elif "token" not in response and "result" not in response:
            return {"req_id": req_id, "error_msg": f"The response should contain 'token' or 'result', but got {response}"}
        else:
            token_ids = response.get("token_ids", [])
            if "result" in response:
                token = response["result"]
            elif "token" in response:
                token = response["token"]
            return {"req_id": req_id, "token": token, "token_ids": token_ids, "is_end": is_end}


class OutputData:
    """
    Receive data returned by Triton service
    """
    def __init__(self):
        self._completed_requests = queue.Queue()


def triton_callback(output_data, result, error):
    """
    callback function for Triton server
    """
    if error:
        output_data._completed_requests.put(error)
    else:
        output_data._completed_requests.put(result)


class ChatBot(object):
    """
    External interface, create a client object ChatBotForPushMode
    """
    def __new__(cls, hostname, port, timeout=120):
        """
        initialize a GRPCInferenceService client
        Args:
            hostname (str): server hostname
            port (int): GRPC port
            timeout (int): timeout(s), default 120 seconds
        Returns:
            ChatBotClass: BaseChatBot object
        """
        if not isinstance(hostname, str) or not hostname:
            raise ValueError("Invalid hostname")
        if not isinstance(port, int) or port <= 0 or port > 65535:
            raise ValueError("Invalid port number")

        return ChatBotClass(hostname, port, timeout)
