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
import uuid
from typing import Any, Dict, List

import numpy as np
import paddle

from ..models.ppo_model_utils import RLHFPPOMixedLoss
from .rl_trainer import RLTrainer
from .trainer_utils import guard_set_args


class ActorReferenceTrainer(RLTrainer):
    loss_cls = RLHFPPOMixedLoss
    trainer_type = "policy"

    def loss_identifier(self, inputs: Dict) -> str:
        """
        根据输入的字典，判断是否使用ptx损失函数和演员损失函数。如果有标签（labels），则返回"ptx_loss"；否则返回"actor_loss"。
        参数：
            inputs (Dict): 包含两个键值对，分别为"inputs"和"labels"，其中"inputs"是模型的输入，"labels"是可选的，表示是否使用ptx损失函数。默认值为None。
            返回值 (str): 返回一个字符串，分别为"ptx_loss"或"actor_loss"，表示是否使用ptx损失函数和演员损失函数。
        """
        return "actor_loss"

    @paddle.no_grad()
    def generate_sequences(self, prompt_only_batch: Dict, do_eval=False) -> List[Dict[str, Any]]:
        """Rollout a batch of experiences."""
        input_ids = prompt_only_batch["input_ids"]
        # attention_mask = prompt_only_batch["attention_mask"]
        if do_eval:
            train_num_return_sequences = self.args.num_return_sequences
            self.args.num_return_sequences = 1

        # position_ids = (
        #     prompt_only_batch["position_ids"]
        #     if "position_ids" in prompt_only_batch
        #     else make_position_ids(attention_mask)
        # )

        if self.args.num_return_sequences > 1:
            input_ids = input_ids.repeat_interleave(self.args.num_return_sequences, axis=0)
            # raw_dtype = attention_mask.dtype
            # attention_mask = (
            #     attention_mask.cast("int32").repeat_interleave(self.args.num_return_sequences, axis=0).cast(raw_dtype)
            # )
            # position_ids = position_ids.repeat_interleave(self.args.num_return_sequences, axis=0)

        with guard_set_args(self._model_config, {"use_fused_head_and_loss_fn": False}):
            sequences = self.model.generate(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                generation_config=self.generation_config,
                synced_gpus=ShardingOption.FULL_SHARD in self.actor_trainer.args.sharding,
                do_eval=do_eval,
            )[0]

        if self.args.use_rm_server:
            label_ids = prompt_only_batch["label_ids"]
            if self.args.num_return_sequences > 1:
                label_ids = label_ids.repeat_interleave(self.args.num_return_sequences, axis=0)

        sequences = sequences.reshape(
            [input_ids.shape[0] // self.args.num_return_sequences, self.args.num_return_sequences, -1]
        )
        if do_eval:
            self.args.num_return_sequences = train_num_return_sequences
            sequences = sequences.transpose([1, 0, 2])
        # prompt, sequence, attention_mask
        return [
            {
                "prompt": input_ids,
                "input_ids": seq,
                **({"label_ids": label_ids[idx * len(seq) : (idx + 1) * len(seq)]} if self.args.use_rm_server else {}),
                "index": np.array([str(uuid.uuid4())] * len(seq), dtype=object),
                # "attention_mask": make_attention_mask(
                #     seq,
                #     pad_id=self.tokenizer.pad_token_id,
                #     eos_id=None,
                #     unk_id=self.tokenizer.unk_token_id,
                #     causal_mask=True,
                # ).cast(self._model_config.dtype),
                # "sequence_mask": make_attention_mask(
                #     seq,
                #     pad_id=self.tokenizer.pad_token_id,
                #     eos_id=None,
                #     unk_id=self.tokenizer.unk_token_id,
                #     causal_mask=False,
                # ).cast(self._model_config.dtype),
            }
            for idx, seq in enumerate(sequences)
        ]
