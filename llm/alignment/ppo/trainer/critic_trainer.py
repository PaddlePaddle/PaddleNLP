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

from models.ppo_model_utils import RLHFValueLoss
from trainer.rl_trainer import RLTrainer


class CriticTrainer(RLTrainer):
    loss_cls = RLHFValueLoss
    trainer_type = "value"
    # define loss name for logging
    loss_identifier = lambda self, inputs: "reward_critic_loss"
