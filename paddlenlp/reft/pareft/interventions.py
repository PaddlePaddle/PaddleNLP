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

import math

import paddle
import paddle.nn as nn
from paddle import ParamAttr

from paddlenlp.reft.pavenv import (
    DistributedRepresentationIntervention,
    SourcelessIntervention,
    TrainableIntervention,
)

from .layers import LowRankRotateLayer


def linear_act(x):
    return x


ACT2FN = {
    "linear": linear_act,
    "relu": nn.ReLU(),
}


class LoreftIntervention(SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        print('kwargs["embed_dim"]', kwargs["embed_dim"])

        print(type(self.embed_dim))
        rotate_layer = LowRankRotateLayer(kwargs["embed_dim"], kwargs["low_rank_dimension"])
        self.rotate_layer = rotate_layer  # Paddle doesn't have a direct orthogonal parametrization utility
        self.learned_source = nn.Linear(
            kwargs["embed_dim"],
            kwargs["low_rank_dimension"],
            weight_attr=ParamAttr(initializer=nn.initializer.Orthogonal()),
        )
        if "dtype" in kwargs:
            # print("kwargs['dtype']", kwargs["dtype"])
            self.learned_source = self.learned_source.astype(kwargs["dtype"])
        else:
            self.learned_source = self.learned_source.astype(paddle.bfloat16)
        self.dropout = nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = (
            ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        )

    def forward(self, base, source=None, subspaces=None):
        # print("loreft invention forward")
        # print("weight:", self.rotate_layer.weight[0][0:8])
        rotated_base = self.rotate_layer(base)
        # print("self.act_fn", self.act_fn)
        output = base + paddle.matmul(
            (
                self.act_fn(
                    self.learned_source(
                        base,
                    )
                )
                - rotated_base
            ),
            self.rotate_layer.weight.T,
        )
        return self.dropout(output.astype(base.dtype))

    # def state_dict(self, *args, **kwargs):
    #     """
    #     Overwrite for data-efficiency.
    #     """
    #     state_dict = OrderedDict()
    #     for k, v in self.learned_source.state_dict().items():
    #         state_dict[k] = v
    #     state_dict["rotate_layer"] = self.rotate_layer.weight.numpy()
    #     return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        # self.learned_source.set_state_dict(state_dict)
        self.learned_source.weight.data = state_dict["learned_source.weight"]
        self.learned_source.bias.data = state_dict["learned_source.bias"]

        overload_w = state_dict["rotate_layer.weight"]
        overload_w_width = overload_w.shape[-1]
        with paddle.no_grad():
            self.rotate_layer.weight[:, :overload_w_width] = paddle.to_tensor(overload_w)
        print("self.rotate_layer.weight", self.rotate_layer.weight)
        return


class TinyIntervention(SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        print('kwargs["embed_dim"]', kwargs["embed_dim"])

        print(type(self.embed_dim))

        self.rank = kwargs["low_rank_dimension"]
        self.hidden_size = kwargs["embed_dim"]

        dropout = 0.0
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

        self.scaling = 1

        # Actual trainable parameters
        self.param_A = self.create_parameter(
            shape=[self.hidden_size, self.rank],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu"),
        )
        self.param_B = self.create_parameter(
            shape=[self.rank, self.hidden_size],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.Constant(value=0.0),
        )

        self.param_a = self.create_parameter(
            shape=[self.rank],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.Constant(value=1),
        )

        self.param_b = self.create_parameter(
            shape=[self.hidden_size],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.Constant(value=1),
        )

        # Freezing the A B  weight matrix and bias vector
        self.param_A.stop_gradient = False
        self.param_B.stop_gradient = False

    def forward(self, base, source=None, subspaces=None):
        diag_b = paddle.diag(self.param_b)
        diag_a = paddle.diag(self.param_a)
        result = (self.dropout(base) @ self.param_A @ diag_a @ self.param_B @ diag_b) * self.scaling
        return self.dropout(base + result.astype(base.dtype))
