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
from typing import List, Optional


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


@dataclass
class EmbeddingArgument:
    max_query_len: int = field(
        default=1,
        metadata={
            "help": "The number of highest probability tokens to keep for top-k-filtering in the sampling strategy"
        },
    )
    max_passage_len: int = field(
        default=1.0, metadata={"help": "The cumulative probability for top-p-filtering in the sampling strategy."}
    )
    group_size: int = field(
        default=8,
        metadata={
            "help": (
                "Number of total positive and negative samples associated with " "each query for embedding training."
            )
        },
    )
    query_template: str = field(
        default="Query: {text}\nUse one word to summarize the query's relevant information. The word is: \"",
        metadata={
            "help": (
                "Query template. Ensure the template includes the placeholder "
                "'{text}' to insert the actual query text."
            )
        },
    )
    passage_template: str = field(
        default="Text: {text}\nUse one word to summarize the text's content. The word is: \"",
        metadata={
            "help": (
                "Passage template. Ensure the template includes the placeholder "
                "'{text}' to insert the actual passage text."
            )
        },
    )
    embedding_temperature: float = field(
        default=0.02,
        metadata={"help": "The temperature used in embedding learning."},
    )
    embedding_negatives_cross_device: bool = field(
        default=True,
        metadata={"help": "Whether to share the negatives across all GPUs."},
    )
    embedding_matryoshka_dims: Optional[List[int]] = field(
        default=None,
        metadata={"help": "The dims for matryoshka training."},
    )
