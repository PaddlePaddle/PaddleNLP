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
import json
from dataclasses import dataclass, field

from .trainer_utils import split_parallel_config
from .training_args import TrainingArguments
from .utils import add_start_docstrings


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class AutoTrainingArguments(TrainingArguments):
    """
    Training Arguments for auto_parallel.
    """

    fused_linear: bool = field(
        default=False,
        metadata={"help": "Enable fused linear op, which will fuse matmul and bias add together."},
    )

    fused_linear_param_grad_add: bool = field(
        default=False,
        metadata={
            "help": "Enable fused_linear_param_grad pass, which should replace add_n_op with add_op for gradients accumulation."
        },
    )
    fuse_allreduce_split_to_reducescatter: bool = field(
        default=False,
        metadata={"help": "Enable fuse_allreduce_split_to_reducescatter pass."},
    )
    eliminate_transpose: bool = field(
        default=False,
        metadata={
            "help": "Enable eliminate_transpose pass, which should replace transpose with reshape when sequence parallel is enabled."
        },
    )
    use_intermediate_api: bool = field(
        default=False,
        metadata={"help": "Weather to use auto_parallel intermediate api"},
    )
    refined_ops_patterns: str = field(default=None, metadata={"help": "The pattern of refined recompute."})

    def __post_init__(self):
        super().__post_init__()
        assert self.enable_auto_parallel

        fused_passes = self.strategy.fused_passes

        if self.fused_linear_param_grad_add:
            fused_passes.enable = True
            fused_passes.fused_passes_list.append("fused_linear_param_grad_add_pass")

        if self.fuse_allreduce_split_to_reducescatter:
            fused_passes.enable = True
            fused_passes.fused_passes_list.append("fuse_allreduce_split_to_reducescatter_pass")

        if self.eliminate_transpose:
            fused_passes.enable = True
            fused_passes.fused_passes_list.append("eliminate_transpose")

        if self.fused_linear:
            fused_passes.enable = True
            fused_passes.fused_passes_list.append("fused_gemm_epilogue_pass")

        mp_configs = split_parallel_config(self.tensor_parallel_config)
        if "replace_with_parallel_cross_entropy" in mp_configs:
            self.strategy.mp_optimization.replace_with_parallel_cross_entropy = True

        if self.recompute:
            recompute = self.strategy.recompute
            recompute.enable = True
            recompute.refined_ops_patterns = []
            if type(self.refined_ops_patterns) == str:
                recompute.refined_ops_patterns = json.loads(self.refined_ops_patterns)
            else:
                recompute.refined_ops_patterns = (
                    self.refined_ops_patterns if self.refined_ops_patterns is not None else []
                )
