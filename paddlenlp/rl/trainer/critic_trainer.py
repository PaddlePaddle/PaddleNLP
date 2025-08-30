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

import paddle

from ...datasets.rlhf_datasets.protocol import DataProto
from ...transformers import PretrainedTokenizer
from ..models.ppo_model_utils import RLHFValueLoss, create_startend_row_indices

# from ..utils.comm_utils import CriticStages
# from ..utils.offload_utils import reload_and_offload_scope
# from ..utils.timer_utils import TimerScope
from .rl_trainer import RLTrainer


class CriticTrainer(RLTrainer):
    loss_cls = RLHFValueLoss
    trainer_type = "value"
    # define loss name for logging
    loss_identifier = lambda self, inputs: "reward_critic_loss"

    @paddle.no_grad()
    def compute_value(
        self,
        batch: DataProto,
        input_ids_tokenizer: PretrainedTokenizer = None,
    ) -> DataProto:
        self.model.eval()

        input_ids = batch.batch["input_ids"]
        position_ids = batch.batch["position_ids"]

        values_list = []
        batch_size, sequence_length = input_ids.shape
        per_device_value_batch_size = self.args.per_device_value_batch_size
        num_batches = (batch_size + per_device_value_batch_size - 1) // per_device_value_batch_size
        startend_row_indices = create_startend_row_indices(input_ids, self.tokenizer.pad_token_id)
        response_start = batch.batch["prompt"].shape[-1] - 1 if "prompt" in batch.batch else 0
        for i in range(num_batches):
            start_index = i * per_device_value_batch_size
            end_index = min(start_index + per_device_value_batch_size, batch_size)

            # Extract the current batch
            current_input_ids = input_ids[start_index:end_index]
            current_startend_row_indices = (
                startend_row_indices[start_index:end_index] if startend_row_indices is not None else None
            )
            current_position_ids = position_ids[start_index:end_index] if position_ids is not None else None
            if self.args.use_remove_padding:
                from ..utils.bert_padding import prepare_flashmask_inputs

                update_inputs = prepare_flashmask_inputs(
                    current_input_ids,
                    current_position_ids,
                    self.tokenizer.pad_token_id,
                    self.model.config.sequence_parallel,
                    self.model.config.tensor_parallel_degree,
                )
                current_input_ids = update_inputs["input_ids"]
                current_position_ids = update_inputs["position_ids"]
                current_startend_row_indices = update_inputs["attn_mask_startend_row_indices"]
                indices = update_inputs["indices"]
                raw_input_shape = update_inputs["raw_input_shape"]
                pad_size = update_inputs["pad_size"]
            reward_value = self.model(
                current_input_ids,
                position_ids=current_position_ids,
                attn_mask_startend_row_indices=current_startend_row_indices,
                use_cache=False,
            )[0]
            reward_value = reward_value.squeeze(0)
            if self.model.config.sequence_parallel:
                from paddle.distributed.fleet.utils.sequence_parallel_utils import (
                    GatherOp,
                )

                reward_value = GatherOp.apply(reward_value)

            if self.args.use_remove_padding:
                from ..utils.bert_padding import pad_input

                if pad_size > 0:
                    reward_value = reward_value[:-pad_size, :]
                reward_value = pad_input(
                    reward_value.squeeze(0).unsqueeze(-1), indices, batch=raw_input_shape[0], seqlen=raw_input_shape[1]
                ).squeeze(-1)
                reward_value = reward_value[:, response_start:-1].contiguous()
            values_list.append(reward_value.squeeze(-1))
            reward_value = None
            paddle.device.cuda.empty_cache()

        return DataProto.from_single_dict({"reward_values": paddle.concat(values_list, axis=0)})

    def update_critic(self, rl_batch: DataProto) -> DataProto:
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
        self.model.train()
        # Inputs shared by policy and value trainer
        input_ids = rl_batch.batch["input_ids"].contiguous()  # length: src+tgt
        position_ids = rl_batch.batch["position_ids"]  # length: src+tgt
        sequence_mask = rl_batch.batch["eos_mask"]  # length: src+tgt(-1)
        if self.args.use_fp32_compute and sequence_mask.dtype != paddle.float32:
            sequence_mask = sequence_mask.cast(paddle.float32)

        # Inputs used by value trainer
        old_reward_values = rl_batch.batch["reward_values"]  # length: src+tgt(-1)
        reward_returns = rl_batch.batch["reward_returns"]  # length: src+tgt(-1)

        attn_mask_startend_row_indices = create_startend_row_indices(input_ids, self.tokenizer.pad_token_id)
        value_trainer_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "old_reward_values": old_reward_values,
            "reward_returns": reward_returns,
            "sequence_mask": sequence_mask,
            "response_start": rl_batch.batch["prompt"].shape[-1] - 1,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
        }

        reward_critic_loss = self.full_training_step(**value_trainer_inputs)

        # return DataProto(meta_info={"metrics": {"train_value_loss": reward_critic_loss}})
        return {"train_value_loss": reward_critic_loss}
