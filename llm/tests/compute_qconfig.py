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

from typing import Tuple

import gurobipy as gp
import numpy as np
import paddle
import scipy.optimize._optimize as scipy_optimize


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


GIGABYTES = 1024.0**3
budget = 6.0

ilp_data = paddle.load("../ilp_data/merge/llama2-7b.ilp.ranks-64.pth")
costs = ilp_data["costs"]
weights = ilp_data["weights"]
num_params = ilp_data["nparams"]
names = ilp_data["names"]
qconfigs = ilp_data["qconfigs"]

normalized_costs = costs / paddle.linalg.norm(costs) * 1000.0
normalized_budget = budget / GIGABYTES * num_params
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

assignments_cost = assignments_cost * paddle.linalg.norm(costs) / 1000.0
assignments_cost_ = (assignments * costs).sum()
assignments_weight_ = (assignments * weights).sum() / num_params
qconfig_dict = {}
for i0, (i1, qconfig_index) in enumerate(assignments.nonzero().tolist()):
    if i0 != i1:
        raise ValueError
    key0 = names[i1]
    qconfig_dict[key0] = qconfigs[qconfig_index]

paddle.save(qconfig_dict, "../ilp_data/merge/qconfig_dict")

print(qconfig_dict)
