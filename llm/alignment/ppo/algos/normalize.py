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

from paddle.distributed import fleet
from utils.comm_utils import gather_and_pad

from paddlenlp.trainer import Trainer


def normalize_batch_data_ppo(trainer: Trainer, micro_batches):
    micro_batches = trainer.normalize_batch_data(
        micro_batches,
        use_tgt_len_value=trainer.args.use_tgt_len_value,
    )

    return micro_batches


def normalize_batch_data_reinforce_plus_plus(trainer: Trainer, micro_batches):
    old_log_probs = [micro_batch["log_probs"] for micro_batch in micro_batches]
    ref_log_probs = [micro_batch["ref_log_probs"] for micro_batch in micro_batches]
    rewards = [micro_batch["rewards"] for micro_batch in micro_batches]
    eos_mask = [
        (micro_batch["input_ids"] != trainer.tokenizer.pad_token_id)[:, micro_batch["prompt"].shape[-1] :].to(
            old_log_probs[0].dtype
        )
        for micro_batch in micro_batches
    ]
    shapes = [micro_batch["log_probs"].shape for micro_batch in micro_batches]
    try:
        hcg = fleet.get_hybrid_communicate_group()
        sd_group = hcg.get_sharding_parallel_group()
        dp_group = hcg.get_data_parallel_group()
    except AttributeError:
        pass
    new_batch = {
        "rewards": gather_and_pad(rewards, dp_group, sd_group, pad=False),
        "log_probs": gather_and_pad(old_log_probs, dp_group, sd_group),
        "ref_log_probs": gather_and_pad(ref_log_probs, dp_group, sd_group),
        "eos_mask": gather_and_pad(eos_mask, dp_group, sd_group),
    }
    new_batches = trainer.normalize_batch_data([new_batch], use_tgt_len_value=trainer.args.use_tgt_len_value)
    local_data = {
        "reward_advantages": trainer.get_rank_data(new_batches[0]["reward_advantages"]),
        "rewards": trainer.get_rank_data(new_batches[0]["rewards"]),
        "ori_rewards": trainer.get_rank_data(new_batches[0]["ori_rewards"]),
        "reward_returns": trainer.get_rank_data(new_batches[0]["reward_returns"]),
        "kl_rewards": trainer.get_rank_data(new_batches[0]["kl_rewards"]),
        "rewards_with_kl": trainer.get_rank_data(new_batches[0]["rewards_with_kl"]),
        "eos_mask": trainer.get_rank_data(new_batches[0]["eos_mask"]),
    }
    offset = 0
    for idx, batch in enumerate(micro_batches):
        for k, v in local_data.items():
            if local_data[k][offset].ndim < 1:
                micro_batches[idx].update(
                    {k: local_data[k][offset : offset + len(batch["log_probs"])][: shapes[idx][-1]]}
                )
            else:
                micro_batches[idx].update(
                    {k: local_data[k][offset : offset + len(batch["log_probs"])][:, : shapes[idx][-1]]}
                )
        offset += len(batch["log_probs"])

    return micro_batches
