# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import struct
import logging
import numpy as np
import time
import random
from typing import Optional

from tritonclient import utils as client_utils
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput, service_pb2_grpc, service_pb2

LOGGER = logging.getLogger("run_inference_on_triton")


class SyncGRPCTritonRunner:
    DEFAULT_MAX_RESP_WAIT_S = 120

    def __init__(
        self,
        server_url: str,
        model_name: str,
        model_version: str,
        *,
        verbose=False,
        resp_wait_s: Optional[float] = None,
    ):
        self._server_url = server_url
        self._model_name = model_name
        self._model_version = model_version
        self._verbose = verbose
        self._response_wait_t = self.DEFAULT_MAX_RESP_WAIT_S if resp_wait_s is None else resp_wait_s

        self._client = InferenceServerClient(self._server_url,
                                             verbose=self._verbose)
        error = self._verify_triton_state(self._client)
        if error:
            raise RuntimeError(
                f"Could not communicate to Triton Server: {error}")

        LOGGER.debug(
            f"Triton server {self._server_url} and model {self._model_name}:{self._model_version} "
            f"are up and ready!")

        model_config = self._client.get_model_config(self._model_name,
                                                     self._model_version)
        model_metadata = self._client.get_model_metadata(
            self._model_name, self._model_version)
        LOGGER.info(f"Model config {model_config}")
        LOGGER.info(f"Model metadata {model_metadata}")

        self._inputs = {tm.name: tm for tm in model_metadata.inputs}
        self._input_names = list(self._inputs)
        self._outputs = {tm.name: tm for tm in model_metadata.outputs}
        self._output_names = list(self._outputs)
        self._outputs_req = [
            InferRequestedOutput(name) for name in self._outputs
        ]

    def Run(self, inputs):
        """
        Args:
            inputs: list, Each value corresponds to an input name of self._input_names
        Returns: 
            results: dict, {name : numpy.array}
        """
        infer_inputs = []
        for idx, data in enumerate(inputs):
            data = np.array([[x.encode('utf-8')] for x in data],
                            dtype=np.object_)
            infer_input = InferInput(self._input_names[idx], [len(data), 1],
                                     "BYTES")
            infer_input.set_data_from_numpy(data)
            infer_inputs.append(infer_input)

        results = self._client.infer(
            model_name=self._model_name,
            model_version=self._model_version,
            inputs=infer_inputs,
            outputs=self._outputs_req,
            client_timeout=self._response_wait_t,
        )
        results = {name: results.as_numpy(name) for name in self._output_names}
        return results

    def _verify_triton_state(self, triton_client):
        if not triton_client.is_server_live():
            return f"Triton server {self._server_url} is not live"
        elif not triton_client.is_server_ready():
            return f"Triton server {self._server_url} is not ready"
        elif not triton_client.is_model_ready(self._model_name,
                                              self._model_version):
            return f"Model {self._model_name}:{self._model_version} is not ready"
        return None


if __name__ == "__main__":
    model_name = "seqcls"
    model_version = "1"
    url = "localhost:8001"
    runner = SyncGRPCTritonRunner(url, model_name, model_version)

    texts = [[
        "五松新村房屋是被告婚前购买的；"
    ], ["被告于2016年3月将车牌号为皖B×××××出售了2.7万元，被告通过原告偿还了齐荷花人民币2.6万元，原、被告尚欠齐荷花2万元。"],
             ["一、判决原告于某某与被告杨某某离婚；"]]
    for text in texts:
        # input format:[input1, input2 ... inputn], n = len(self._input_names)
        result = runner.Run([text])
        print("text: ", text)
        print("label: ", ','.join([str(r) for r in result['label']]))
        print("confidence: ",
              ','.join([str('%.3f' % c) for c in result['confidence']]))
        print('--------------------')
