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

    texts = [
        [
            "经审理查明，2012年4月5日19时许，被告人王某在杭州市下城区朝晖路农贸市场门口贩卖盗版光碟、淫秽光碟时被民警当场抓获，并当场查获其贩卖的各类光碟5515张，其中5280张某属非法出版物、235张某属淫秽物品。上述事实，被告人王某在庭审中亦无异议，且有经庭审举证、质证的扣押物品清单、赃物照片、公安行政处罚决定书、抓获经过及户籍证明等书证；证人胡某、徐某的证言；出版物鉴定书、淫秽物品审查鉴定书及检查笔录等证据证实，足以认定。"
        ],
        [
            "静乐县人民检察院指控，2014年8月30日15时许，静乐县苏坊村村民张某某因占地问题去苏坊村半切沟静静铁路第五标施工地点阻拦施工时，遭被告人王某某阻止，张某某打电话叫来儿子李某某，李某某看到张某某躺在地上，打了王某某一耳光。于是王某某指使工人殴打李某某，致李某某受伤。经忻州市公安司法鉴定中心鉴定，李某某的损伤评定为轻伤一级。李某某被打伤后，被告人王某某为逃避法律追究，找到任某某，指使任某某作实施××的伪证，并承诺每月给1万元。同时王某某指使工人王某甲、韩某某去丰润派出所作由任某某打伤李某某的伪证，导致任某某被静乐县公安局以涉嫌××罪刑事拘留。公诉机关认为，被告人王某某的行为触犯了《中华人民共和国刑法》××、《中华人民共和国刑法》××××之规定，应以××罪和××罪追究其刑事责任，数罪并罚。"
        ],
        [
            "榆林市榆阳区人民检察院指控：2015年11月22日2时许，被告人王某某在自己经营的榆阳区长城福源招待所内，介绍并容留杨某向刘某某、白某向乔某某提供性服务各一次。"
        ]
    ]

    for text in texts:
        # input format:[input1, input2 ... inputn], n = len(self._input_names)
        result = runner.Run([text])
        print("text: ", text)
        print("label: ", ','.join([str(r) for r in result['label']]))
        print("confidence: ",
              ','.join([str('%.3f' % c) for c in result['confidence']]))
        print('--------------------')
