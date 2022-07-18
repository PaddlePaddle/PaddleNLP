import logging
import numpy as np
import time
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


def test_tnews_dataset(runner):
    from paddlenlp.datasets import load_dataset
    dev_ds = load_dataset('clue', "tnews", splits='dev')

    batches = []
    labels = []
    idx = 0
    batch_size = 32
    while idx < len(dev_ds):
        data = []
        label = []
        for i in range(batch_size):
            if idx + i >= len(dev_ds):
                break
            data.append(dev_ds[idx + i]["sentence"])
            label.append(dev_ds[idx + i]["label"])
        batches.append(data)
        labels.append(np.array(label))
        idx += batch_size

    accuracy = 0
    for i, data in enumerate(batches):
        ret = runner.Run([data])
        # print("ret:", ret)
        accuracy += np.sum(labels[i] == ret["label"])
    print("acc:", 1.0 * accuracy / len(dev_ds))


if __name__ == "__main__":
    model_name = "ernie_seqcls"
    model_version = "1"
    url = "localhost:8001"
    runner = SyncGRPCTritonRunner(url, model_name, model_version)
    texts = [["你家拆迁，要钱还是要房？答案一目了然", "军嫂探亲拧包入住，部队家属临时来队房标准有了规定，全面落实！"],
             [
                 "区块链投资心得，能做到就不会亏钱",
             ]]

    for text in texts:
        # input format:[input1, input2 ... inputn], n = len(self._input_names)
        result = runner.Run([text])
        print(result)

    test_tnews_dataset(runner)
