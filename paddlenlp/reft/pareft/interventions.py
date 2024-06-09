import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from collections import OrderedDict
from .layers import LowRankRotateLayer
from paddlenlp.reft.pavenv import (
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)


def linear_act(x):
    return x


ACT2FN = {
    "linear": linear_act,
}


class LoreftIntervention(
    SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention
):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        print('kwargs["embed_dim"]', kwargs["embed_dim"])

        print(type(self.embed_dim))
        rotate_layer = LowRankRotateLayer(
            kwargs["embed_dim"], kwargs["low_rank_dimension"]
        )
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
            ACT2FN["linear"]
            if "act_fn" not in kwargs or kwargs["act_fn"] is None
            else ACT2FN[kwargs["act_fn"]]
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
            self.rotate_layer.weight[:, :overload_w_width] = paddle.to_tensor(
                overload_w
            )
        print("self.rotate_layer.weight", self.rotate_layer.weight)
        return
