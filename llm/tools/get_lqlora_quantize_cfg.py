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

import gurobipy as gp
import numpy as np
import paddle
import scipy.optimize._optimize as scipy_optimize

from paddlenlp.peft.lora.lqlora_utils import lowrand_quantized_sparse_decomposition
from paddlenlp.transformers import AutoModelForCausalLM


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, required=True, type=str, help="The directory of model.")
    parser.add_argument("--qconfigs", default=None, type=str, required=True, help="Quantize methods, use ',' split")
    parser.add_argument("--budget", default=None, type=float, required=True, help="Budget")
    parser.add_argument("--ranks", default=64, type=int, help="SVD rank")
    parser.add_argument("--output_path", default=None, type=str, required=True, help="The directory of saved model ")
    return parser.parse_args()


def estimate_storage_from_config(W: paddle.Tensor, quant_algo: str):
    if quant_algo in ["weight_only_int8", "llm.int8"]:
        return W.numel() * 8.0
    elif quant_algo in ["weight_only_int4", "fp4"]:
        return W.numel() * 4.0
    elif quant_algo in ["nf4"]:
        return W.numel() * 4.127
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
        for i1, qconfig in enumerate(qconfigs):
            print(f"process {names[i0]}. quant_algo: {qconfig}")
            Q, L1, L2 = lowrand_quantized_sparse_decomposition(param, num_ranks, quant_algo=qconfig)
            param_ = L1 @ L2 + Q
            error = paddle.linalg.norm(param - param_, p="fro") ** 2

            costs[i0, i1] = error
            weights[i0, i1] = estimate_storage_from_config(param, quant_algo=qconfig)
            del param_
    return costs, weights


def compute_qconfig_assignments(
    budget: float,
    costs: paddle.Tensor,
    weights: paddle.Tensor,
    num_chunks: int,
) -> Tuple[float, paddle.Tensor]:
    costs_np = costs.numpy(force=True).reshape(costs.shape[0], -1)
    weights_np = weights.numpy(force=True).reshape(weights.shape[0], -1)
    costs_list = np.split(costs_np, indices_or_sections=num_chunks, axis=0)
    weights_list = np.split(weights_np, indices_or_sections=num_chunks, axis=0)

    results = []
    for _costs, _weights in zip(costs_list, weights_list):
        result = mip_solve(budget=budget / float(num_chunks), costs=_costs, weights=_weights, backend="grurobi")
        results.append(result)

    assignments_cost = sum([r.fun for r in results])
    assignments = np.concatenate([r.x for r in results], axis=0)
    assignments = assignments.reshape(costs.shape)
    return assignments_cost, paddle.to_tensor(assignments)


def mip_solve(
    budget: float,
    costs: np.ndarray,
    weights: np.ndarray,
    backend: str,
) -> scipy_optimize.OptimizeResult:
    if backend not in ["scipy", "grurobi"]:
        raise ValueError(f"Unknown backend: {backend}")

    N = costs.shape[0]
    coefficients = costs.reshape(-1)
    A_upperbound = weights.reshape(1, -1)
    A_equality = np.zeros_like(weights, shape=(N,) + weights.shape)
    A_equality[np.arange(N), np.arange(N), :] = 1.0
    A_equality = A_equality.reshape(N, -1)

    if backend == "grurobi":
        grurobi_model = gp.Model()
        grurobi_model.setParam(paramname="Timelimit", newval=60)  # type: ignore
        x = grurobi_model.addMVar(shape=coefficients.shape, vtype=gp.GRB.BINARY, name="x")
        grurobi_model.setObjective(coefficients @ x, gp.GRB.MINIMIZE)
        grurobi_model.addConstr((A_upperbound @ x) <= budget, name="upperbounds")
        grurobi_model.addConstr((A_equality @ x) == 1.0, name="equalities")
        grurobi_model.optimize()
        return scipy_optimize.OptimizeResult(x=x.X, fun=grurobi_model.ObjVal)

    raise ValueError


def get_ilp_data(args):
    names = []
    params = []
    qconfigs = args.qconfigs.split(",")
    num_ranks = args.ranks

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    for name, submodule in model.named_sublayers():
        if "_proj" in name:
            names.append(name)
            params.append(submodule.weight)

    costs, weights = prepare_data_for_qconfig(names=names, parameters=params, qconfigs=qconfigs, num_ranks=num_ranks)
    ilp_data = {
        "names": names,
        "shapes": [param.shape for param in params],
        "nparams": sum([param.numel() for param in params]),
        "costs": costs,
        "weights": weights,
        "qconfigs": qconfigs,
    }

    return ilp_data


def get_lqlora_quantize_cfg():
    args = parse_arguments()
    GIGABYTES = 1024.0**3

    ilp_data = get_ilp_data(args)
    costs = ilp_data["costs"]
    weights = ilp_data["weights"]
    num_params = ilp_data["nparams"]
    names = ilp_data["names"]
    qconfigs = ilp_data["qconfigs"]

    normalized_costs = costs / paddle.linalg.norm(costs) * 1000.0
    normalized_budget = args.budget / GIGABYTES * num_params
    normalized_weights = weights / GIGABYTES
    assignments_cost, assignments = compute_qconfig_assignments(
        budget=normalized_budget, costs=normalized_costs, weights=normalized_weights, num_chunks=1
    )

    if not all(
        [
            costs.shape == [len(names), len(qconfigs)],
            weights.shape == [len(names), len(qconfigs)],
            assignments.shape == [len(names), len(qconfigs)],
        ]
    ):
        raise ValueError

    qconfig_dict = {}
    for i0, (i1, qconfig_index) in enumerate(assignments.nonzero().tolist()):
        if i0 != i1:
            raise ValueError
        key0 = names[i1]
        qconfig_dict[key0] = qconfigs[qconfig_index]

    paddle.save(qconfig_dict, os.path.join(args.output_path, "lqlora_quantize_cfg"))


if __name__ == "__main__":
    get_lqlora_quantize_cfg()
