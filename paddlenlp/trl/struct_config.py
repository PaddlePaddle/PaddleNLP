# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from dataclasses import dataclass, field

__all__ = ["StructConfig"]


@dataclass
class StructConfig:
    """
    StructConfig is used to re-config the model structure after loading pretrained config
    if restruct_model is True, we will re-config the model structure according to the following parameters which are not None.
    """

    restruct_model: bool = field(
        default=False,
        metadata={"help": "Whether to reorganize the model structure after loading pretrained config"},
    )

    # model related parameters, which can used to re-config the model
    num_hidden_layers: int = field(
        default=None, metadata={"help": "num_hidden_layers, when it is None, we will use the model config value."}
    )
    num_nextn_predict_layers: int = field(
        default=None,
        metadata={"help": "num_nextn_predict_layers, when it is None, we will use the model config value."},
    )
    num_nextn_predict_lambda: int = field(
        default=None,
        metadata={"help": "num_nextn_predict_lambda, when it is None, we will use the model config value."},
    )
    hidden_size: int = field(
        default=None, metadata={"help": "hidden_size, when it is None, we will use the model config value."}
    )
    moe_intermediate_size: int = field(
        default=None, metadata={"help": "moe_intermediate_size, when it is None, we will use the model config value."}
    )
    intermediate_size: int = field(
        default=None, metadata={"help": "intermediate_size, when it is None, we will use the model config value."}
    )
    num_attention_heads: int = field(
        default=None, metadata={"help": "num_attention_heads, when it is None, we will use the model config value."}
    )
    num_key_value_heads: int = field(
        default=None, metadata={"help": "num_key_value_heads, when it is None, we will use the model config value."}
    )
    n_shared_experts: int = field(
        default=None, metadata={"help": "n_shared_experts, when it is None, we will use the model config value."}
    )
    n_routed_experts: int = field(
        default=None, metadata={"help": "n_routed_experts, when it is None, we will use the model config value."}
    )
    hidden_act: str = field(
        default=None, metadata={"help": "hidden_act, when it is None, we will use the model config value."}
    )
    tie_word_embeddings: bool = field(
        default=None, metadata={"help": "tie_word_embeddings, when it is None, we will use the model config value."}
    )

    # moe related parameters
    capacity_factor: float = field(
        default=1.0,
        metadata={
            "help": "Capacity factor for MoE (Mixture of Experts), used to adjust the capacity of expert networks."
        },
    )
    eval_capacity_factor: float = field(
        default=1.0,
        metadata={
            "help": "Capacity factor for MoE during evaluation, used to adjust the capacity of expert networks in evaluation phase."
        },
    )
    min_capacity: int = field(
        default=1, metadata={"help": "Minimum capacity for MoE, setting the minimum capacity of expert networks."}
    )
    max_capacity: int = field(
        default=pow(2, 32),
        metadata={"help": "Maximum capacity for MoE, setting the maximum capacity of expert networks."},
    )
    global_aux_loss: bool = field(
        default=False, metadata={"help": "Whether to use a global auxiliary loss function to train the MoE model."}
    )
    expert_drop: bool = field(
        default=False, metadata={"help": "Whether to randomly drop experts within the expert network during training."}
    )
    noisy_gate_policy: str = field(
        default=None,
        metadata={"help": "Set the noisy gate policy to control the gating mechanism of the expert network."},
    )
    drop_tokens: bool = field(
        default=False, metadata={"help": "Whether to randomly drop tokens from the input sequence during training."}
    )
    use_rts: bool = field(default=True, metadata={"help": "Whether to use RTS (Random Token Selection) mechanism."})
    top2_2nd_expert_sampling: bool = field(
        default=True, metadata={"help": "Whether to use Top-2 sampling strategy for the second expert."}
    )
    drop_policy: str = field(
        default="probs",
        metadata={"help": "Set the drop policy for expert networks, e.g., dropping based on probabilities."},
    )
    topk_method: str = field(
        default="greedy", metadata={"help": "Set the method for selecting Top-K experts, e.g., greedy algorithm."}
    )
    top_k: int = field(default=1, metadata={"help": "Set the value of K for selecting Top-K experts."})
    n_group: int = field(default=1, metadata={"help": "Set the number of groups for expert networks."})
    topk_group: int = field(
        default=1, metadata={"help": "Set the value of K for selecting Top-K experts within each group."}
    )
    norm_topk_prob: bool = field(
        default=False, metadata={"help": "Whether to normalize the selection probabilities of Top-K experts."}
    )
    routed_scaling_factor: float = field(
        default=1.0, metadata={"help": "Whether to normalize the selection probabilities of Top-K experts."}
    )
