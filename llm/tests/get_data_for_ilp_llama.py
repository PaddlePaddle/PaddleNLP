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

import argparse
import os
from typing import List, Optional, Tuple

import paddle

from paddlenlp.peft.lora.lqlora_utils import lowrand_quantized_sparse_decomposition
from paddlenlp.transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, help="start layer index")
parser.add_argument("--end", type=int, help="end layer index")
parser.add_argument("--gpu", type=str, help="gpu index")

args = parser.parse_args()

paddle.set_device("gpu:" + args.gpu)


def estimate_storage_from_config(W: paddle.Tensor, quant_algo: str):
    if quant_algo in ["weight_only_int8", "llm.int8"]:
        return W.numel() * 8.0
    elif quant_algo in ["weight_only_int4", "fp4", "nf4"]:
        return W.numel() * 4.0
    else:
        raise NotImplementedError(f"{quant_algo} is not support.")


def prepare_data_for_qconfig(
    names: List[str],
    parameters: List[paddle.Tensor],
    qconfigs: List[str],
    num_ranks: Optional[int] = 64,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    costs = paddle.zeros(shape=[len(parameters), len(qconfigs)])
    weights = paddle.zeros(shape=[len(parameters), len(qconfigs)])

    for i0, param in enumerate(parameters):
        layer_index = int(names[i0].split(".")[2])
        if layer_index < args.start or layer_index > args.end:
            continue
        for i1, qconfig in enumerate(qconfigs):
            print(f"process {names[i0]}. quant_algo: {qconfig}")
            Q, L1, L2 = lowrand_quantized_sparse_decomposition(param, num_ranks, quant_algo=qconfig)
            param_ = L1 @ L2 + Q
            error = paddle.linalg.norm(param - param_, p="fro") ** 2

            costs[i0, i1] = error
            weights[i0, i1] = estimate_storage_from_config(param, quant_algo=qconfig)
            del param_
    return costs, weights


names = []
params = []
qconfigs = ["weight_only_int8", "fp4", "nf4"]
num_ranks = 64
save_dir = "../ilp_data"
save_path = os.path.join(save_dir, "llama2-7b.ilp4.ranks-64.layer." + str(args.start) + "-" + str(args.end) + ".pth")
model = AutoModelForCausalLM.from_pretrained("facebook/llama-7b")
for name, submodule in model.named_sublayers():
    if "_proj" in name:
        names.append(name)
        params.append(submodule.weight)

print(save_path)
costs, weights = prepare_data_for_qconfig(names=names, parameters=params, qconfigs=qconfigs, num_ranks=num_ranks)
save_dict = {
    "names": names,
    "shapes": [param.shape for param in params],
    "nparams": sum([param.numel() for param in params]),
    "costs": costs,
    "weights": weights,
    "qconfigs": qconfigs,
}
paddle.save(save_dict, save_path)
