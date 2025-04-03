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

from typing import Dict

import paddle
from models.ppo_model_utils import RLHFValueLoss, create_startend_row_indices

from paddlenlp.transformers import PretrainedTokenizer
from trainer.rl_trainer import RLTrainer


class CriticTrainer(RLTrainer):
    loss_cls = RLHFValueLoss
    trainer_type = "value"
    # define loss name for logging
    loss_identifier = lambda self, inputs: "reward_critic_loss"

    def compute_reward(
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
