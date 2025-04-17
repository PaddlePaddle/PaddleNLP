# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddleslim


def get_pruned_params(model):
    params = []
    for sublayer in model.sublayers():
        for param in sublayer.parameters(include_sublayers=False):
            if (
                isinstance(sublayer, paddle.nn.layer.common.Linear)
                or isinstance(sublayer, paddle.distributed.fleet.layers.mpu.mp_layers.ColumnParallelLinear)
                or isinstance(sublayer, paddle.distributed.fleet.layers.mpu.mp_layers.RowParallelLinear)
            ):
                if len(param.shape) != 2:
                    continue

                # NOTE(minghaoBD):
                # 1. param.shape[1] == 3 * param.shape[0]： prune fused-qkv's weight and its next weight: out-linear's weight
                # 2. param.shape[1] == 4 * param.shape[0]： prune ffn1's weight and its next weight: ffn2's weight
                # If your model has a different architecture, like your qkv's weights are not fused or ffn1_weight.shape[1] != 4*ffn1_weight.shape[0], you may need to customize this function to suit your model.
                if param.shape[1] == 3 * param.shape[0] or param.shape[1] == 4 * param.shape[0]:
                    params.append(param.name)

    return params


def prune_model(model, configs, inputs_desc=[]):
    prune_criterion = configs.criterion
    ratio = configs.ratio
    shapes, dtypes = [], []
    for input_desc in inputs_desc:
        dtypes.append(input_desc.dtype)
        new_shape = [10 if item == -1 else item for item in input_desc.shape]
        shapes.append(new_shape)
    # TODO(minghaoBD): support ViT and other model architectures in the future
    num_attention_heads = model.gpt.decoder.layers[0].self_attn.num_heads

    if prune_criterion == "l1_norm":
        pruner = paddleslim.L1NormFilterPruner(
            model, shapes, skip_leaves=False, prune_type="fc", input_dtype=dtypes[0], num_head=num_attention_heads
        )
    elif prune_criterion == "l2_norm":
        pruner = paddleslim.L2NormFilterPruner(
            model, shapes, skip_leaves=False, prune_type="fc", input_dtype=dtypes[0], num_head=num_attention_heads
        )
    params = get_pruned_params(model)
    ratios = {}
    for param in params:
        ratios[param] = ratio
    # NOTE(minghaoBD): hidden size in Layernorm must be 768/1024/2048/4096 for best inference performace, and when axis=0, the hidden size in layernorm will be changed accordingly. So axis=1 is required.
    pruner.prune_vars(ratios, [1])


def quant_model(model, configs):
    quanter = paddleslim.dygraph.quant.QAT(configs)
    return quanter.quantize(model), quanter
