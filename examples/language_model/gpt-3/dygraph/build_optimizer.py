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

import inspect

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer import (
    DygraphShardingOptimizer,
)


def is_new_version_sharding_stage1_optimizer():
    signature_keys = set(inspect.signature(DygraphShardingOptimizer).parameters.keys())
    return "inner_optimizer_class" not in signature_keys


def apply(model, args, lr_scheduler, clip, decay_params, strategy):
    if args.sharding_stage == 1 and args.sharding_degree > 1 and not is_new_version_sharding_stage1_optimizer():
        # for backward compatibility.
        # this call will raise, if sharding stage1 is handled in HybridParallelOptimizer,
        # in which case, the logic follows will handle it
        optimizer = DygraphShardingOptimizer(
            hcg=fleet.get_hybrid_communicate_group(),
            user_defined_strategy=strategy,
            params=model.parameters(),
            inner_optimizer_class=paddle.optimizer.AdamW,
            learning_rate=lr_scheduler if lr_scheduler is not None else args.max_lr,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            epsilon=args.adam_epsilon,
            weight_decay=args.weight_decay,
            grad_clip=clip,
            apply_decay_param_fun=lambda x: x in decay_params,
            multi_precision=args.use_pure_fp16,
        )
    else:
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler if lr_scheduler is not None else args.max_lr,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            epsilon=args.adam_epsilon,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            grad_clip=clip,
            apply_decay_param_fun=lambda x: x in decay_params,
            # TODO: remove 'multi_precision' in definition of optimizer
            # and add it to 'paddle.amp.decorate'
            multi_precision=args.use_pure_fp16,
        )
    return optimizer
