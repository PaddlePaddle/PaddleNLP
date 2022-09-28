# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class GraphSageConv(nn.Layer):
    """ GraphSAGE is a general inductive framework that leverages node feature
    information (e.g., text attributes) to efficiently generate node embeddings
    for previously unseen data.

    Paper reference:
    Hamilton, Will, Zhitao Ying, and Jure Leskovec.
    "Inductive representation learning on large graphs."
    Advances in neural information processing systems. 2017.
    """

    def __init__(self, input_size, hidden_size, learning_rate, aggr_func="sum"):
        super(GraphSageConv, self).__init__()
        assert aggr_func in ["sum", "mean", "max", "min"], \
            "Only support 'sum', 'mean', 'max', 'min' built-in receive function."
        self.aggr_func = "reduce_%s" % aggr_func

        self.self_linear = nn.Linear(
            input_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(learning_rate=learning_rate))
        self.neigh_linear = nn.Linear(
            input_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(learning_rate=learning_rate))

    def forward(self, graph, feature, act=None):

        def _send_func(src_feat, dst_feat, edge_feat):
            return {"msg": src_feat["h"]}

        def _recv_func(message):
            return getattr(message, self.aggr_func)(message["msg"])

        msg = graph.send(_send_func, src_feat={"h": feature})
        neigh_feature = graph.recv(reduce_func=_recv_func, msg=msg)

        self_feature = self.self_linear(feature)
        neigh_feature = self.neigh_linear(neigh_feature)
        output = self_feature + neigh_feature
        if act is not None:
            output = getattr(F, act)(output)

        output = F.normalize(output, axis=1)
        return output


class ErnieSageV2Conv(nn.Layer):
    """ ErnieSage (abbreviation of ERNIE SAmple aggreGatE), a model proposed by the PGL team.
    ErnieSageV2: Ernie is applied to the EDGE of the text graph.
    """

    def __init__(self,
                 ernie,
                 input_size,
                 hidden_size,
                 learning_rate,
                 cls_token_id=1,
                 aggr_func='sum'):
        """ErnieSageV2: Ernie is applied to the EDGE of the text graph.

        Args:
            ernie (nn.Layer): the ernie model.
            input_size (int): input size of feature tensor.
            hidden_size (int): hidden size of the Conv layers.
            learning_rate (float): learning rate.
            aggr_func (str): aggregate function. 'sum', 'mean', 'max' avaliable.
        """
        super(ErnieSageV2Conv, self).__init__()
        assert aggr_func in ["sum", "mean", "max", "min"], \
            "Only support 'sum', 'mean', 'max', 'min' built-in receive function."
        self.aggr_func = "reduce_%s" % aggr_func
        self.cls_token_id = cls_token_id
        self.self_linear = nn.Linear(
            input_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(learning_rate=learning_rate))
        self.neigh_linear = nn.Linear(
            input_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(learning_rate=learning_rate))

        self.ernie = ernie

    def ernie_send(self, src_feat, dst_feat, edge_feat):
        """ Apply ernie model on the edge.

        Args:
            src_feat (Tensor Dict): src feature tensor dict.
            dst_feat (Tensor Dict): dst feature tensor dict.
            edge_feat (Tensor Dict): edge feature tensor dict.

        Returns:
            Tensor Dict: tensor dict which use 'msg' as the key.
        """
        # input_ids
        cls = paddle.full(shape=[src_feat["term_ids"].shape[0], 1],
                          dtype="int64",
                          fill_value=self.cls_token_id)
        src_ids = paddle.concat([cls, src_feat["term_ids"]], 1)

        dst_ids = dst_feat["term_ids"]

        # sent_ids
        sent_ids = paddle.concat(
            [paddle.zeros_like(src_ids),
             paddle.ones_like(dst_ids)], 1)
        term_ids = paddle.concat([src_ids, dst_ids], 1)

        # build position_ids
        input_mask = paddle.cast(term_ids > 0, "int64")
        position_ids = paddle.cumsum(input_mask, axis=1) - 1

        outputs = self.ernie(term_ids, sent_ids, position_ids)
        feature = outputs[1]
        return {"msg": feature}

    def send_recv(self, graph, term_ids):
        """Message Passing of erniesage v2.

        Args:
            graph (Graph): the Graph object.
            feature (Tensor): the node feature tensor.

        Returns:
            Tensor: the self and neighbor feature tensors.
        """

        def _recv_func(message):
            return getattr(message, self.aggr_func)(message["msg"])

        msg = graph.send(self.ernie_send, node_feat={"term_ids": term_ids})
        neigh_feature = graph.recv(reduce_func=_recv_func, msg=msg)

        cls = paddle.full(shape=[term_ids.shape[0], 1],
                          dtype="int64",
                          fill_value=self.cls_token_id)
        term_ids = paddle.concat([cls, term_ids], 1)
        term_ids.stop_gradient = True
        outputs = self.ernie(term_ids, paddle.zeros_like(term_ids))
        self_feature = outputs[1]

        return self_feature, neigh_feature

    def forward(self, graph, term_ids, act='relu'):
        """Forward funciton of Conv layer.

        Args:
            graph (Graph): Graph object.
            feature (Tensor): node feture.
            act (str, optional): activation function. Defaults to 'relu'.

        Returns:
            Tensor: feature after conv.
        """

        self_feature, neigh_feature = self.send_recv(graph, term_ids)
        self_feature = self.self_linear(self_feature)
        neigh_feature = self.neigh_linear(neigh_feature)
        output = self_feature + neigh_feature
        if act is not None:
            output = getattr(F, act)(output)
        output = F.normalize(output, axis=1)
        return output
