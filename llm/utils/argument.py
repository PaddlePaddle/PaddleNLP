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
from dataclasses import dataclass, field


@dataclass
class ReftArgument:
    layers: str = field(default="all", metadata={"help": "Layer configuration for the model."})
    position: str = field(default="f7+l7", metadata={"help": "Position parameter for model."})
    intervention_type: str = field(default="LoreftIntervention", metadata={"help": "Type of intervention."})
    rank: int = field(default=8, metadata={"help": "Rank parameter for model."})
    act_fn: str = field(default="linear", metadata={"help": "Activation function."})
    add_bias: bool = field(default=False, metadata={"help": "Flag indicating whether to add bias."})
    dropout: float = field(default=0.0, metadata={"help": "Dropout rate."})


@dataclass
class GenerateArgument:
    top_k: int = field(
        default=1,
        metadata={
            "help": "The number of highest probability tokens to keep for top-k-filtering in the sampling strategy"
        },
    )
    top_p: float = field(
        default=1.0, metadata={"help": "The cumulative probability for top-p-filtering in the sampling strategy."}
    )
