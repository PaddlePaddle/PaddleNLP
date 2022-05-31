import argparse
import struct
import logging
import numpy as np
import time
import random
from typing import Optional

from paddlenlp.datasets import load_dataset

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
            resp_wait_s: Optional[float]=None, ):
        self._server_url = server_url
        self._model_name = model_name
        self._model_version = model_version
        self._verbose = verbose
        self._response_wait_t = self.DEFAULT_MAX_RESP_WAIT_S if resp_wait_s is None else resp_wait_s

        self._client = InferenceServerClient(
            self._server_url, verbose=self._verbose)
        error = self._verify_triton_state(self._client)
        if error:
            raise RuntimeError(
                f"Could not communicate to Triton Server: {error}")

        LOGGER.debug(
            f"Triton server {self._server_url} and model {self._model_name}:{self._model_version} "
            f"are up and ready!")

        model_config = self._client.get_model_config(self._model_name,
                                                     self._model_version)
        model_metadata = self._client.get_model_metadata(self._model_name,
                                                         self._model_version)
        LOGGER.info(f"Model config {model_config}")
        LOGGER.info(f"Model metadata {model_metadata}")

        # self._inputs = {tm.name: tm for tm in model_metadata.inputs}
        self._input_names = ["INPUT", ]
        self._outputs = {tm.name: tm for tm in model_metadata.outputs}
        self._output_names = list(self._outputs)
        self._outputs_req = [
            InferRequestedOutput(name) for name in self._outputs
        ]

    def Run(self, inputs):
        infer_inputs = []
        for idx, data in enumerate(inputs):
            data = np.array(
                [[x.encode('utf-8')] for x in data], dtype=np.object_)
            infer_input = InferInput(self._input_names[idx], [len(data), 1],
                                     "BYTES")
            infer_input.set_data_from_numpy(data)
            infer_inputs.append(infer_input)

        results = self._client.infer(
            model_name=self._model_name,
            model_version=self._model_version,
            inputs=infer_inputs,
            outputs=self._outputs_req,
            client_timeout=self._response_wait_t, )
        y_pred = {name: results.as_numpy(name) for name in self._output_names}
        # print("output:", y_pred)
        return y_pred

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
    model_name = "pipeline_tnews"
    model_version = "1"
    batch_size = 2
    url = "localhost:8001"

    runner = SyncGRPCTritonRunner(url, model_name, model_version)
    datas = [["你家拆迁，要钱还是要房？答案一目了然", "军嫂探亲拧包入住，部队家属临时来队房标准有了规定，全面落实！"],
             ["区块链投资心得，能做到就不会亏钱", ]]

    result = runner.Run([datas[1]])
    print(result)
    result = runner.Run([datas[1]])
    result = runner.Run([datas[1]])

    import time
    s = time.time()
    for i in range(1000):
        result = runner.Run([datas[1]])
    e = time.time()
    print("cost time:", (e - s) * 1.0)

    # for data in datas:
    #     print("data:", data)
    #     result = runner.Run([data])
    #     print("result:", result)
    #     print(result["POST_OUTPUT1"], result["POST_OUTPUT2"])
