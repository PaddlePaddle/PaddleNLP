# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import warnings
from typing import Union

import paddle
import paddle.nn as nn


class DisLoRALinear(nn.Linear):
    """
    Paddle implementation of Direct Low-Rank Adaptation (DisLoRA) layer.
    DisLoRA decomposes W into backbone (W_prin) and task-specific (W_res) subspaces via SVD,
    further identifying task-specific directions (W_TSD) for fine tuning.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        dislora_alpha: int = 8,
        dislora_dropout: float = 0.0,
        dash_flag: int = 50,
        s_tsd: int = 8,
        prefer_small_sigma: bool = True,
        merge_weights: bool = False,
        init_lora_weights: Union[bool, str] = True,
        **kwargs
    ):

        if r <= 0:
            raise ValueError(f"`r` must be a positive integer, got {r}")
        if s_tsd <= 0:
            raise ValueError(f"`s_tsd` must be a positive integer, got {s_tsd}")

        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        original_weight = self.weight.clone()
        original_bias = self.bias.clone() if self.bias is not None else None

        self.base_dtype = original_weight.dtype

        delattr(self, "weight")
        if hasattr(self, "bias") and self.bias is not None:
            delattr(self, "bias")

        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            default_initializer=nn.initializer.Assign(original_weight),
            dtype=self.base_dtype,
            attr=paddle.ParamAttr(trainable=False),
        )

        if original_bias is not None:
            self.bias = self.create_parameter(
                shape=[out_features],
                default_initializer=nn.initializer.Assign(original_bias),
                dtype=self.base_dtype,
                attr=paddle.ParamAttr(trainable=True),
            )
        else:
            self.bias = None

        self.r = r
        self.dislora_alpha = dislora_alpha
        self.scaling = dislora_alpha / r
        self.dislora_dropout = nn.Dropout(p=dislora_dropout) if dislora_dropout > 0.0 else nn.Identity()
        self.dash_flag = dash_flag
        self.s_tsd = s_tsd
        self.prefer_small_sigma = prefer_small_sigma
        self.merge_weights = merge_weights
        self.init_lora_weights = init_lora_weights

        self._disable_adapters = False
        self.merged = False

        self.register_buffer("step", paddle.to_tensor(0, dtype="int64"))

        self.U = None
        self.S = None
        self.Vh = None

        self.Direc_Ur = nn.Linear(r, out_features, bias_attr=False)
        self.Direc_Sr = self.create_parameter(
            shape=[r], default_initializer=nn.initializer.Constant(0.0), dtype=self.base_dtype
        )
        self.Direc_Vhr = nn.Linear(in_features, r, bias_attr=False)
        self.Direc_Ur.weight.stop_gradient = False
        self.Direc_Sr.stop_gradient = False
        self.Direc_Vhr.weight.stop_gradient = False

        self.Direc_Utsd = nn.Linear(s_tsd, out_features, bias_attr=False)
        self.Direc_Stsd = self.create_parameter(
            shape=[s_tsd], default_initializer=nn.initializer.Constant(0.0), dtype=self.base_dtype
        )
        self.Direc_Vhtsd = nn.Linear(in_features, s_tsd, bias_attr=False)

        self.Direc_Utsd.weight.stop_gradient = True
        self.Direc_Vhtsd.weight.stop_gradient = True

        self._align_dtypes()

        if init_lora_weights:
            self._init_lora_weights()

    def _align_dtypes(self):
        """Ensure that the data types of all parameters are consistent with those of the base layer."""
        target_dtype = self.base_dtype

        if self.Direc_Ur.weight.dtype != target_dtype:
            self.Direc_Ur.weight.set_value(self.Direc_Ur.weight.astype(target_dtype))
        if self.Direc_Vhr.weight.dtype != target_dtype:
            self.Direc_Vhr.weight.set_value(self.Direc_Vhr.weight.astype(target_dtype))
        if self.Direc_Utsd.weight.dtype != target_dtype:
            self.Direc_Utsd.weight.set_value(self.Direc_Utsd.weight.astype(target_dtype))
        if self.Direc_Vhtsd.weight.dtype != target_dtype:
            self.Direc_Vhtsd.weight.set_value(self.Direc_Vhtsd.weight.astype(target_dtype))
        if self.Direc_Sr.dtype != target_dtype:
            self.Direc_Sr.set_value(self.Direc_Sr.astype(target_dtype))
        if self.Direc_Stsd.dtype != target_dtype:
            self.Direc_Stsd.set_value(self.Direc_Stsd.astype(target_dtype))

    def _init_lora_weights(self):
        """
        Initialize LoRA weights using SVD
        Decompose the original weight W into W_prin (frozen backbone) + W_res (trainable residual)
        Note: The shape of the Linear weight in PaddlePaddle is [in_features, out_features]
        """
        weight_float32 = self.weight.astype("float32")

        weight_transposed = weight_float32.T

        U, S, Vh = paddle.linalg.svd(weight_transposed, full_matrices=False)

        self.U = U.astype(self.base_dtype)
        self.S = S.astype(self.base_dtype)
        self.Vh = Vh.astype(self.base_dtype)

        if self.prefer_small_sigma:
            _, indices = paddle.topk(S, self.r, largest=False)
        else:
            _, indices = paddle.topk(S, self.r, largest=True)

        self.Direc_Ur.weight.set_value(U[:, indices].T.astype(self.base_dtype))
        self.Direc_Sr.set_value(S[indices].astype(self.base_dtype))

        self.Direc_Vhr.weight.set_value(Vh[indices, :].T.astype(self.base_dtype))
        self.Direc_Ur.weight.stop_gradient = False
        self.Direc_Sr.stop_gradient = False
        self.Direc_Vhr.weight.stop_gradient = False
        self.Direc_Stsd.stop_gradient = False

        S_diag = paddle.diag(self.Direc_Sr)  # [r, r]
        W_res_T = self.Direc_Ur.weight.T @ S_diag @ self.Direc_Vhr.weight.T  # [out_features, in_features]
        W_res = W_res_T.T * self.scaling  # [in_features, out_features]

        if W_res.shape != self.weight.shape:
            raise ValueError(f"Expected W_res shape {self.weight.shape}, but got {W_res.shape}.")

        self.weight.set_value(self.weight - W_res.astype(self.base_dtype))
        self.weight.stop_gradient = True

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Forward propagation: W_prin @ x + W_res @ x + W_TSD @ x
        - W_prin is calculated through the base_layer
        - W_res is calculated through the trainable LoRA structure
        - W_TSD is calculated through the frozen dynamic vector (after warmup)
        """
        if self._disable_adapters:
            if self.merged:
                self.unmerge()
            return super().forward(x)

        if self.merged:
            return super().forward(x)

        result = super().forward(x)

        temp = self.dislora_dropout(x)
        temp = self.Direc_Vhr(temp)
        temp = temp * self.Direc_Sr
        temp = self.Direc_Ur(temp)
        result += temp * self.scaling

        if self.step < self.dash_flag:
            pass
        elif self.step == self.dash_flag:
            self._initialize_dynamic_vectors()
        else:
            temp = self.dislora_dropout(x)
            temp = self.Direc_Vhtsd(temp)
            temp = temp * self.Direc_Stsd
            temp = self.Direc_Utsd(temp)
            result += temp * self.scaling

        if self.training:
            with paddle.no_grad():
                self.step += 1

        return result

    def _initialize_dynamic_vectors(self):
        """
        After the warm-up steps, initialize the dynamic singular vector W_TSD.
        Based on the current change of W_res, select the most important s_tsd directions.
        """
        with paddle.no_grad():

            S_diag = paddle.diag(self.Direc_Sr)  # [r, r]
            deltaW_T = self.Direc_Ur.weight.T @ S_diag @ self.Direc_Vhr.weight.T  # [out_features, in_features]

            delta_sigma = paddle.diag(self.U.T @ deltaW_T @ self.Vh.T)

            top_indices = self.calculate_change_rate(
                self.S, delta_sigma, self.s_tsd, largest=not self.prefer_small_sigma
            )

            self.Direc_Utsd.weight.set_value(self.U[:, top_indices].T.astype(self.base_dtype))
            self.Direc_Stsd.set_value(self.S[top_indices].astype(self.base_dtype))
            self.Direc_Vhtsd.weight.set_value(self.Vh[top_indices, :].T.astype(self.base_dtype))

            self.Direc_Utsd.weight.stop_gradient = True
            self.Direc_Vhtsd.weight.stop_gradient = True

    def calculate_change_rate(self, a: paddle.Tensor, b: paddle.Tensor, s: int, largest: bool = True) -> paddle.Tensor:
        """
        Calculate the rate of change of singular values and
        select the top-s index change_rate = |b| / (|a| + eps)
        """
        with paddle.no_grad():

            change_rate = paddle.abs(b) / (paddle.abs(a) + 1e-8)

            _, top_s_indices = paddle.topk(change_rate, s, largest=largest)
        return top_s_indices

    def merge(self):
        """
        Merge the trainable W_res into the base weights.
        After merging: base_layer.weight = W_prin + W_res
        Note: W_TSD remains frozen and does not participate in the merge.
        """
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return

        if self.r > 0:

            delta_weight = self.get_delta_weight()
            orig_weights = self.weight.clone()
            orig_weights += delta_weight
            self.weight.set_value(orig_weights)

        self.merged = True

    def unmerge(self):
        """
        Remove the merging of W_res from the base weights.
        After the merging is removed: base_layer.weight = W_prin
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        if self.r > 0:
            delta_weight = self.get_delta_weight()
            self.weight.set_value(self.weight - delta_weight)

        self.merged = False

    def get_delta_weight(self) -> paddle.Tensor:
        """
        Calculate the trainable LoRA incremental weights
        It consists of two parts:
        1. W_res = Ur @ diag(Sr) @ Vhr * scaling (transposed)
        2. W_tsd = Utsd @ diag(Stsd) @ Vhtsd * scaling (transposed)
        Return the incremental weights with the shape of [in_features, out_features]
        """

        S_diag_r = paddle.diag(self.Direc_Sr)  # [r, r]
        delta_weight_T = self.Direc_Ur.weight.T @ S_diag_r @ self.Direc_Vhr.weight.T  # [out_features, in_features]
        delta_weight = delta_weight_T.T * self.scaling  # [in_features, out_features]

        if not paddle.all(self.Direc_Stsd == 0.0):
            S_diag_tsd = paddle.diag(self.Direc_Stsd)  # [s_tsd, s_tsd]
            delta_weight_tsd_T = (
                self.Direc_Utsd.weight.T @ S_diag_tsd @ self.Direc_Vhtsd.weight.T
            )  # [out_features, in_features]
            delta_weight += delta_weight_tsd_T.T * self.scaling  # [in_features, out_features]

        return delta_weight.astype(self.base_dtype)

    def enable_adapters(self):
        """Enable the adapter"""
        self._disable_adapters = False

    def disable_adapters(self):
        """Disable adapter"""
        self._disable_adapters = True

    def __repr__(self) -> str:
        rep = super().__repr__()
        return rep
