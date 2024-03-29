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

import os

import paddle
from paddle import _C_ops
from paddle.base.framework import Variable
from paddle.optimizer import AdamW

from paddlenlp.trainer import Trainer, strtobool


def cast(x, dtype):
    if x.dtype == dtype:
        return x
    return x.cast(dtype)


class GaLoreProjector:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type="std"):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type

    def project(self, full_rank_grad, iter=None, need_cast=True):
        full_rank_grad = full_rank_grad.t()

        if self.proj_type == "std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="right", need_cast=need_cast
                    )
                low_rank_grad = paddle.matmul(full_rank_grad, cast(self.ortho_matrix.t(), full_rank_grad.dtype))
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="left", need_cast=need_cast
                    )
                low_rank_grad = paddle.matmul(cast(self.ortho_matrix.t(), full_rank_grad.dtype), full_rank_grad)
        elif self.proj_type == "reverse_std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="left", need_cast=need_cast
                    )
                low_rank_grad = paddle.matmul(cast(self.ortho_matrix.t(), full_rank_grad.dtype), full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="right", need_cast=need_cast
                    )
                low_rank_grad = paddle.matmul(full_rank_grad, cast(self.ortho_matrix.t(), full_rank_grad.dtype))
        elif self.proj_type == "right":
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="right", need_cast=need_cast
                )
            low_rank_grad = paddle.matmul(full_rank_grad, cast(self.ortho_matrix.t(), full_rank_grad.dtype))
        elif self.proj_type == "left":
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="left", need_cast=need_cast
                )
            low_rank_grad = paddle.matmul(cast(self.ortho_matrix.t(), full_rank_grad.dtype), full_rank_grad)
        elif self.proj_type == "full":
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="full", need_cast=need_cast
                )
            low_rank_grad = paddle.matmul(cast(self.ortho_matrix[0].t(), full_rank_grad.dtype), full_rank_grad) @ cast(
                self.ortho_matrix[1].t(), full_rank_grad.dtype
            )

        return low_rank_grad.t()

    def project_back(self, low_rank_grad):
        low_rank_grad = low_rank_grad.t()

        if self.proj_type == "std":
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = paddle.matmul(low_rank_grad, cast(self.ortho_matrix, low_rank_grad.dtype))
            else:
                full_rank_grad = paddle.matmul(cast(self.ortho_matrix, low_rank_grad.dtype), low_rank_grad)
        elif self.proj_type == "reverse_std":
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:  # note this is different from std
                full_rank_grad = paddle.matmul(cast(self.ortho_matrix, low_rank_grad.dtype), low_rank_grad)
            else:
                full_rank_grad = paddle.matmul(low_rank_grad, cast(self.ortho_matrix, low_rank_grad.dtype))
        elif self.proj_type == "right":
            full_rank_grad = paddle.matmul(low_rank_grad, cast(self.ortho_matrix, low_rank_grad.dtype))
        elif self.proj_type == "left":
            full_rank_grad = paddle.matmul(cast(self.ortho_matrix, low_rank_grad.dtype), low_rank_grad)
        elif self.proj_type == "full":
            full_rank_grad = paddle.matmul(cast(self.ortho_matrix[0], low_rank_grad.dtype), low_rank_grad) @ cast(
                self.ortho_matrix[1], low_rank_grad.dtype
            )

        return (full_rank_grad * self.scale).t()

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type, need_cast=True):
        module_params = weights

        if module_params.dtype != paddle.float32:
            need_cast = True and need_cast
            original_type = module_params.dtype
            matrix = module_params.cast("float32")
        else:
            need_cast = False
            matrix = module_params

        # U, s, Vh = svd(matrix, full_matrices=False)
        U, s, Vh = paddle.linalg.svd(matrix, full_matrices=False)

        # make the smaller matrix always to be orthogonal matrix
        if type == "right":
            # A = U[:, :rank] @ paddle.diag(s[:rank])
            B = Vh[:rank, :]

            if need_cast:
                B = B.cast(original_type)
            return B
        elif type == "left":
            A = U[:, :rank]
            # B = paddle.diag(s[:rank]) @ Vh[:rank, :]
            if need_cast:
                A = A.cast(original_type)
            return A
        elif type == "full":
            A = U[:, :rank]
            B = Vh[:rank, :]
            if need_cast:
                A = A.cast(original_type)
                B = B.cast(original_type)
            return [A, B]
        else:
            raise ValueError("type should be left, right or full")


class GaLoreAdamW(AdamW):
    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-6,
        parameters=None,
        weight_decay=0.01,
        lr_ratio=None,
        apply_decay_param_fun=None,
        grad_clip=None,
        lazy_mode=False,
        multi_precision=False,
        name=None,
    ):
        if not isinstance(parameters, list):
            raise ValueError("parameters should be a list.")
        if not isinstance(parameters[0], dict):
            raise ValueError(
                "parameters should be a list of dict, like `parameters=[{'params': [tensor1, tensor2, tensor3]}]` format."
            )
        self._param_name_to_galore_projector_mapping = {}
        self._master_name_to_param_name_mapping = {}
        self._global_steps = 0
        # if isinstance(parameters[0], paddle.Tensor):
        #     parameters = [{"params": parameters, **kwargs}]
        # parameters = [tensor1, tensor2, tensor3] 的时候，强制改成下面的形式。
        # parameters=[{'params': [tensor1, tensor2, tensor3]}]
        super(GaLoreAdamW, self).__init__(
            learning_rate,
            beta1,
            beta2,
            epsilon,
            parameters,
            weight_decay,
            lr_ratio,
            apply_decay_param_fun,
            grad_clip,
            lazy_mode,
            multi_precision,
            name,
        )
        for param_group in self._param_groups:
            for param in param_group["params"]:
                if "rank" in param_group:
                    if param.stop_gradient:
                        continue
                    if param.ndim == 2:
                        if param.name not in self._param_name_to_galore_projector_mapping:
                            self._param_name_to_galore_projector_mapping[param.name] = GaLoreProjector(
                                param_group["rank"],
                                update_proj_gap=param_group["update_proj_gap"],
                                scale=param_group["scale"],
                                proj_type=param_group["proj_type"],
                            )
                    else:
                        raise ValueError("GaLore only supports 2D parameters, but got {}".format(param.ndim))

    def _add_accumulator(
        self,
        name,
        param,
        dtype=None,
        fill_value=0.0,
        shape=None,
        type=None,
        device=None,
    ):
        param_name = (
            self._master_name_to_param_name_mapping[param.name]
            if param.name in self._master_name_to_param_name_mapping
            else param.name
        )
        galore_projector = self._param_name_to_galore_projector_mapping.get(param_name, None)
        if galore_projector is not None and shape is None:
            sp = param.shape
            rank = galore_projector.rank
            assert rank <= min(sp), f"rank should be less than {min(sp)}"
            if sp[0] > sp[1]:
                shape = [sp[0], rank]
            else:
                shape = [rank, sp[1]]

        return super(GaLoreAdamW, self)._add_accumulator(
            name=name,
            param=param,
            dtype=dtype,
            fill_value=fill_value,
            shape=shape,
            type=type,
            device=device,
        )

    def _create_master_weight(self, param):
        var = super(GaLoreAdamW, self)._create_master_weight(
            param=param,
        )
        if param.name not in self._master_name_to_param_name_mapping and var.name != param.name:
            self._master_name_to_param_name_mapping[var.name] = param.name
        return var

    def _append_optimize_op(self, block, param_and_grad):
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        galore_projector = self._param_name_to_galore_projector_mapping.get(param_and_grad[0].name, None)

        if galore_projector is None:
            # 如果是正常tensor，只需要走原来的优化器逻辑
            # print(f"############# >>>>>>>>>>  Warning: GaLoreAdamW is not used ! {param_and_grad[0].name}")
            return super(GaLoreAdamW, self)._append_optimize_op(block, param_and_grad)

        # Whether we should do weight decay for the parameter.
        with_decay = True
        if self._apply_decay_param_fun is not None and not self._apply_decay_param_fun(param_and_grad[0].name):
            with_decay = False

        moment1 = self._get_accumulator_master(self._moment1_acc_str, param_and_grad[0])
        moment2 = self._get_accumulator_master(self._moment2_acc_str, param_and_grad[0])
        beta1_pow_acc = self._get_accumulator_master(self._beta1_pow_acc_str, param_and_grad[0])
        beta2_pow_acc = self._get_accumulator_master(self._beta2_pow_acc_str, param_and_grad[0])

        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(param_and_grad[0].dtype)
        master_weight = self._master_weights[param_and_grad[0].name] if find_master else None
        lr = self._create_param_lr(param_and_grad)

        lr_ratio_ = 1.0 if self._lr_ratio is None else self._lr_ratio(param_and_grad[0])

        _beta1 = self._beta1 if not isinstance(self._beta1, Variable) else self._beta1.item(0)
        _beta2 = self._beta2 if not isinstance(self._beta2, Variable) else self._beta2.item(0)
        param_grad = galore_projector.project(param_and_grad[1], self._global_steps, need_cast=not find_master)

        zero_param = paddle.zeros_like(param_grad)
        zero_param_master_weight = paddle.zeros(param_grad.shape, dtype="float32") if find_master else None

        # c_ops adamw_ in https://github.com/PaddlePaddle/Paddle/blob/18f00786e402132d2199c5c255a1e0c99ac7b8f7/paddle/phi/kernels/gpu/adamw_kernel.cu#L35
        _, _, _, _, _, _ = _C_ops.adamw_(
            zero_param,
            param_grad,
            lr,
            moment1,
            moment2,
            beta1_pow_acc,
            beta2_pow_acc,
            zero_param_master_weight,
            None,
            _beta1,
            _beta2,
            self._epsilon,
            lr_ratio_,
            0.0,
            False,
            self._lazy_mode,
            1000,
            find_master,
            False,
        )
        new_param = param_and_grad[0] + galore_projector.project_back(zero_param)

        if find_master:
            new_param_master_weight = master_weight + galore_projector.project_back(zero_param_master_weight)

        # add weight decay
        if self._weight_decay > 0 and with_decay:
            new_param *= 1 - cast(lr, new_param.dtype) * lr_ratio_ * self._weight_decay
            if find_master:
                new_param_master_weight *= 1 - cast(lr, new_param_master_weight.dtype) * lr_ratio_ * self._weight_decay

        param_and_grad[0].copy_(new_param, False)
        if find_master:
            self._master_weights[param_and_grad[0].name].copy_(new_param_master_weight, False)
        return None

    @paddle.no_grad()
    def step(self):
        out = super(GaLoreAdamW, self).step()
        self._global_steps += 1
        return out


def set_galore_optimizer(trainer, model):
    if strtobool(os.getenv("use_galore", 0)):
        from paddlenlp.trainer.trainer_utils import OptimizerNames

        def get_optimizer_cls_and_kwargs(args):
            # optimizer_kwargs = {"lr": args.learning_rate}
            optimizer_kwargs = {}
            adam_kwargs = {
                "beta1": args.adam_beta1,
                "beta2": args.adam_beta2,
                "epsilon": args.adam_epsilon,
            }
            if args.optim == OptimizerNames.ADAMW:
                optimizer_cls = GaLoreAdamW
                optimizer_kwargs.update(adam_kwargs)
                print("##########>>>>>>>>>>>>>>>>>> Using GaLoreAdamW optimizer!")
            else:
                raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
            return optimizer_cls, optimizer_kwargs

        Trainer.get_optimizer_cls_and_kwargs = staticmethod(get_optimizer_cls_and_kwargs)

        galore_params = []
        galore_params_names = []
        for module_name, module in model.named_sublayers(include_self=True):
            if "Linear" not in module.__class__.__name__:
                print(module.__class__.__name__, "is not supported")
                continue

            for pname, p in module.named_parameters():
                if not p.stop_gradient and p.ndim == 2:
                    galore_params.append(p)
                    galore_params_names.append(module_name + "." + pname)

        if len(galore_params) == 0:
            raise ValueError("None of the target modules were found!")

        non_galore_params = [
            p for n, p in model.named_parameters() if n not in galore_params_names and not p.stop_gradient
        ]

        optim_args = {}
        galore_optim_kwargs = {
            "rank": optim_args.pop("rank", 256),
            "update_proj_gap": optim_args.pop("update_proj_gap", 200),
            "scale": optim_args.pop("scale", 4.0),
            "proj_type": optim_args.pop("proj_type", "std"),
        }
        # The default args are from the official repository: https://github.com/jiaweizzhao/GaLore
        param_groups = [
            {"params": galore_params, **galore_optim_kwargs},
        ]
        if len(non_galore_params) > 0:
            param_groups.append({"params": non_galore_params})
        print("Galore parameters num: ", len(galore_params))
        print("NonGalore parameters num: ", len(non_galore_params))
        trainer.set_optimizer_grouped_parameters(param_groups)


if __name__ == "__main__":
    import paddle
    import paddle.nn as pnn

    # paddle.seed(42)
    num = 11
    bs = 16
    rank = 64
    lr = 1e-1
    in_feature = 768
    out_feature = 128
    bias = True

    model_pd = pnn.Linear(in_feature, out_feature, bias_attr=bias)
    optimizer_pd = GaLoreAdamW(
        learning_rate=lr,
        weight_decay=1e-2,
        epsilon=1e-6,
        parameters=[
            {"params": model_pd.bias},
            {"params": model_pd.weight, "rank": rank, "update_proj_gap": 200, "scale": 0.25, "proj_type": "std"},
        ],
    )
    inputs_pd = []
    for step in range(num):
        x = paddle.randn((bs, in_feature))
        y = paddle.randn((bs, out_feature))
        inputs_pd.append((x, y))

    outputs_pd = []
    weight_numpy_list = []
    bias_numpy_list = []
    for x, y in inputs_pd:
        weight_numpy_list.append(model_pd.weight.numpy())
        bias_numpy_list.append(model_pd.bias.numpy())
        loss = ((model_pd(x) - y) ** 2).mean()
        loss.backward()
        optimizer_pd.step()
        outputs_pd.append(
            (
                loss.detach().cpu().numpy(),
                model_pd.weight.detach().cpu().numpy(),
                model_pd.bias.detach().cpu().numpy(),
                model_pd.weight.grad.detach().cpu().numpy(),
                model_pd.bias.grad.detach().cpu().numpy(),
            )
        )
        optimizer_pd.clear_grad()

    import numpy as np
    import torch
    import torch.nn as tnn
    from galore_torch import GaLoreAdamW as PTGaLoreAdamW

    def to_torch(x, device):
        if isinstance(x, np.ndarray) or isinstance(x, np.float32):
            return torch.tensor(x).to(device)
        return torch.from_numpy(x.detach().cpu().numpy()).to(device)

    device = "cuda"
    model_pt = tnn.Linear(in_feature, out_feature, bias=bias)
    model_pt.bias.data.zero_()
    model_pt.to(device)
    param_groups = [
        {"params": model_pt.bias},
        {"params": model_pt.weight, "rank": rank, "update_proj_gap": 200, "scale": 0.25, "proj_type": "std"},
    ]
    optimizer_pt = PTGaLoreAdamW(param_groups, lr=lr, correct_bias=True, weight_decay=1e-2, eps=1e-6)
    input_pt = []
    for x, y in inputs_pd:
        input_pt.append((to_torch(x, device), to_torch(y, device)))

    outputs_pt = []
    for (x, y), weight_numpy, bias_numpy in zip(input_pt, weight_numpy_list, bias_numpy_list):
        model_pt.weight.data = torch.from_numpy(weight_numpy.T).to(device)
        model_pt.bias.data = torch.from_numpy(bias_numpy).to(device)
        loss = ((model_pt(x) - y) ** 2).mean()
        loss.backward()
        optimizer_pt.step()
        outputs_pt.append(
            (
                loss.detach().cpu().numpy(),
                model_pt.weight.t().detach().cpu().numpy(),
                model_pt.bias.detach().cpu().numpy(),
                model_pt.weight.grad.t().detach().cpu().numpy(),
                model_pt.bias.grad.detach().cpu().numpy(),
            )
        )
        optimizer_pt.zero_grad()

    for a, b in zip(outputs_pt, outputs_pd):
        for aa, bb in zip(a, b):
            loss_dif = aa - bb
            print(
                to_torch(loss_dif, "cpu").mean(),
            )
        print("=" * 50)
