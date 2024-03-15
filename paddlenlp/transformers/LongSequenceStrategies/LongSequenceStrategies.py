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

import importlib


class LongSequenceStrategies:
    @classmethod
    def build_long_sequence_strategy(cls, strategy_type=None, stratety_name=None, **init_args):
        """

        **init_args:   head_dim,
                       max_position_embeddings,
                       rope_scaling_type,
                       rope_scaling_factor,
                       ...

        strategy_type: "None" ---------------走原始的build-in模块
                       "EmbeddingStrategies"
                       "AttentionStrategies"
                       ...

        stratety_name: "RotaryEmbedding"、
                       "LinearScalingRotaryEmbedding"、
                       "NTKScalingRotaryEmbedding"、
                       "DynamicNTKScalingRotaryEmbedding"
                       "AttentionWithLinearBias"
                       ...

        """

        """
        paddlenlp.transformers.LongSequenceStrategies.{strategy_type<->import_class)}.{stratety_name<->strategy_class)}
        paddlenlp.transformers.LongSequenceStrategies.{EmbeddingStrategies}.{RoPE,...}
        paddlenlp.transformers.LongSequenceStrategies.{AttentionStrategies}.{ALiBi,...}
        """
        try:
            import_class = importlib.import_module(f"paddlenlp.transformers.LongSequenceStrategies.{strategy_type}")
        except ValueError:
            raise ValueError(f"Wrong strategy type {strategy_type}.")
        try:
            strategy_class = getattr(import_class, stratety_name)
            strategy_instance = strategy_class(**init_args)
            return strategy_instance
        except AttributeError:
            all_strategy_classes = import_class.__all__
            raise AttributeError(
                f"module '{import_class.__name__}' only supports the following classes: "
                + ", ".join(m for m in all_strategy_classes)
            )
