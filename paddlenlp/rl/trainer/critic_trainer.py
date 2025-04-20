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


from __future__ import annotations

from typing import Any, Dict

import paddle

from ...transformers import PretrainedTokenizer
from ..models.ppo_model_utils import RLHFValueLoss, create_startend_row_indices
from ..utils.comm_utils import CriticStages
from ..utils.offload_utils import reload_and_offload_scope
from ..utils.timer_utils import TimerScope
from .rl_trainer import RLTrainer


class CriticTrainer(RLTrainer):
    loss_cls = RLHFValueLoss
    trainer_type = "value"
    # define loss name for logging
    loss_identifier = lambda self, inputs: "reward_critic_loss"

    def compute_value(
        self,
        input_ids: paddle.Tensor,
        position_ids: paddle.Tensor = None,
        input_ids_tokenizer: PretrainedTokenizer = None,
        **kwargs,
    ) -> Dict[str, paddle.Tensor]:
        # TODO: confirm actor_tokenizer or reward_tokenizer or critic_tokenizer
        # need retokenize?
        attn_mask_startend_row_indices = create_startend_row_indices(input_ids, self.tokenizer.pad_token_id)
        reward_value = self.model(
            input_ids,
            attention_mask=None,
            position_ids=position_ids,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )[0]
        reward_value = reward_value.squeeze(axis=-1)
        reward_value = reward_value[:, :-1]

        return reward_value

    def update_critic(self, rl_batch: Dict[str, paddle.Tensor]) -> Dict[str, Any]:
        """
        Update the parameters of the critic (reward function).

        This function takes a dictionary as input, containing the following key-value pairs:
            - input_ids (paddle.Tensor): IDs of the input sequences, shape (src+tgt, batch).
            - attention_mask (paddle.Tensor): Attention mask for the input sequences, shape (src+tgt, batch).
            - position_ids (paddle.Tensor): Position IDs of the input sequences, shape (src+tgt, batch).
            - old_reward_values (paddle.Tensor): Reward values from the previous time step, shape (src+tgt-1, batch).
            - reward_returns (paddle.Tensor): Reward returns, shape (src+tgt-1, batch).
            - sequence_mask (paddle.Tensor): Sequence mask, shape (src+tgt-1, batch).

        Returns (Dict[str, Any]):
            - train_value_loss (float): Training loss of the critic (reward function).
        """
        # Inputs shared by policy and value trainer
        input_ids = rl_batch["input_ids"].contiguous()  # length: src+tgt
        attention_mask = rl_batch["attention_mask"]  # length: src+tgt
        position_ids = rl_batch["position_ids"]  # length: src+tgt
        sequence_mask = rl_batch["sequence_mask"]  # length: src+tgt(-1)

        # Inputs used by value trainer
        old_reward_values = rl_batch["reward_values"]  # length: src+tgt(-1)
        reward_returns = rl_batch["reward_returns"]  # length: src+tgt(-1)

        value_trainer_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "old_reward_values": old_reward_values,
            "reward_returns": reward_returns,
            "sequence_mask": sequence_mask,
        }

        with TimerScope(
            self.timers, CriticStages.MODEL_ENABLE_DISABLE, minus_names=[CriticStages.CRITIC_TRAINING_STEP]
        ):
            with reload_and_offload_scope(self, self.model, self.optimizer):
                with TimerScope(self.timers, CriticStages.CRITIC_TRAINING_STEP):
                    reward_critic_loss = self.full_training_step(**value_trainer_inputs)

        return reward_critic_loss
