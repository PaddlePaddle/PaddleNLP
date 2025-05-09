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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
from paddle import nn
from paddle.distributed.fleet.meta_parallel import ParallelCrossEntropy
from paddle.io import Dataset

from ...data import DataCollator
from ...generation import GenerationConfig
from ...trainer.trainer import (
    EvalPrediction,
    ShardingOption,
    TrainerCallback,
    TrainingArguments,
)
from ...transformers import PretrainedModel, PretrainedTokenizer
from ..models.ppo_model_utils import (
    RLHFPPOMixedLoss,
    create_startend_row_indices,
    gather_log_probabilities,
)
from .rl_trainer import RLTrainer
from .trainer_utils import guard_set_args


class ActorReferenceTrainer(RLTrainer):
    loss_cls = RLHFPPOMixedLoss
    trainer_type = "policy"

    def __init__(
        self,
        model: Union[PretrainedModel, nn.Layer] = None,
        criterion: nn.Layer = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
        tokenizer: Optional[PretrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]] = None,
    ):
        super().__init__(
            model,
            criterion,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.max_dec_len,
            num_return_sequences=self.args.num_return_sequences,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=0,  # to disable top_k sampling, default is 50
            repetition_penalty=self.args.repetition_penalty,
            min_length=self.args.min_dec_len,
            do_sample=True,
            # allow generation output to contain input
            trunc_input=False,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.cls_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def loss_identifier(self, inputs: Dict) -> str:
        """
        Identify whether to use the ptx loss function or the actor loss function based on the input dictionary.
        If labels are present, return "ptx_loss"; otherwise, return "actor_loss".

        Args:
            inputs (Dict): A dictionary containing two key-value pairs, "inputs" and "labels".
                           "inputs" represents the model's input, while "labels" is optional and indicates whether to use the ptx loss function.
                           The default value for "labels" is None.

        Returns:
            str: A string indicating whether to use the ptx loss function or the actor loss function, either "ptx_loss" or "actor_loss".
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

        with guard_set_args(self.model.config, {"use_fused_head_and_loss_fn": False}):
            sequences = self.get_model(False).generate(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                generation_config=self.generation_config,
                synced_gpus=ShardingOption.FULL_SHARD in self.args.sharding,
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

    @paddle.no_grad()
    def compute_logprob(self, input_ids: paddle.Tensor, position_ids: paddle.Tensor = None, **kwargs) -> paddle.Tensor:
        """
        Computes the log probability of each token during the rollout process.

        Args:
            input_ids (paddle.Tensor, shape [batch_size, sequence_length]):
                Input sequences where each element is an int representing the ID of the respective token.
            attention_mask (paddle.Tensor, shape [batch_size, sequence_length]):
                Attention mask for the input sequences where each element is 0 or 1, indicating which tokens should be considered by the model.
            position_ids (paddle.Tensor, optional, shape [batch_size, sequence_length], defaults to None):
                Position IDs for each token in the input sequences, defaults to None.
            kwargs (Dict[str, Any], optional, defaults to {}):
                Optional arguments, currently not used.

        Returns:
            Dict[str, paddle.Tensor]:
                - log_probs (paddle.Tensor, shape [batch_size, sequence_length - 1]):
                    Log probability of each token during the rollout process.
                - ref_log_probs (paddle.Tensor, shape [batch_size, sequence_length - 1]):
                    Reference log probability of each token during the rollout process.

        Raises:
            None.
        """
        log_probs_list = []
        batch_size, sequence_length = input_ids.shape
        if self.args.rollout_logprob_batch_size is None:
            rollout_logprob_batch_size = batch_size
        else:
            if str(self.args.rollout_logprob_batch_size).lower() == "auto":
                # Auto compute
                if sequence_length > 4096 - 128:
                    rollout_logprob_batch_size = 2
                elif sequence_length > 2048 - 128:
                    rollout_logprob_batch_size = 4
                else:
                    rollout_logprob_batch_size = batch_size
            else:
                rollout_logprob_batch_size = int(self.args.rollout_logprob_batch_size)

        num_batches = (batch_size + rollout_logprob_batch_size - 1) // rollout_logprob_batch_size

        # Pipe model outputs a logits tensor with LMHead, while non-pipe model
        # outputs a tuple with logits tensor as the only one element.
        startend_row_indices = create_startend_row_indices(input_ids, self.tokenizer.pad_token_id)
        response_start = kwargs["prompt"].shape[-1] - 1 if "prompt" in kwargs else 0

        for i in range(num_batches):
            # Calculate the start and end indices for the current batch
            start_index = i * rollout_logprob_batch_size
            end_index = min(start_index + rollout_logprob_batch_size, batch_size)

            # Extract the current batch
            current_input_ids = input_ids[start_index:end_index]
            current_startend_row_indices = (
                startend_row_indices[start_index:end_index] if startend_row_indices is not None else None
            )
            current_position_ids = position_ids[start_index:end_index] if position_ids is not None else None

            logits = self.model(
                current_input_ids,
                position_ids=current_position_ids,
                attn_mask_startend_row_indices=current_startend_row_indices,
            )
            if not isinstance(logits, paddle.Tensor):
                logits = logits[0]  # [2, 355, 12544]

            if self.args.use_fp32_compute and logits.dtype != paddle.float32:
                logits = logits.cast(paddle.float32)
            logits = logits / self.args.temperature if self.args.temperature > 0.0 else logits

            if self.model.config.tensor_parallel_degree > 1 and self.model.config.tensor_parallel_output:
                log_probs = (
                    -ParallelCrossEntropy()(
                        logits[:, response_start:-1].astype("float32"), current_input_ids[:, response_start + 1 :]
                    )
                    .squeeze(axis=-1)
                    .astype(logits.dtype)
                )
            else:
                log_probs = gather_log_probabilities(
                    logits[:, response_start:-1], current_input_ids[:, response_start + 1 :]
                )

            log_probs_list.append(log_probs)
            # Set logits to None to save memory
            logits = None
            paddle.device.cuda.empty_cache()

        return paddle.concat(log_probs_list, axis=0)

    def update_actor(self, rl_batch: Dict[str, paddle.Tensor]) -> Dict[str, Any]:
        # inputs shared by policy and value trainer
        input_ids = rl_batch["input_ids"].contiguous()  # length: src+tgt
        position_ids = rl_batch["position_ids"]  # length: src+tgt
        sequence_mask = rl_batch["eos_mask"]  # length: tgt(-1)
        if self.args.use_fp32_compute and sequence_mask.dtype != paddle.float32:
            sequence_mask = sequence_mask.cast(paddle.float32)
        # inputs used by policy trainer
        old_log_probs = rl_batch["log_probs"]  # length: tgt(-1)
        reward_advantages = rl_batch["reward_advantages"]  # length: tgt(-1)

        response_start = rl_batch["prompt"].shape[-1] - 1

        attn_mask_startend_row_indices = create_startend_row_indices(input_ids, self.tokenizer.pad_token_id)
        policy_trainer_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "old_log_probs": old_log_probs,
            "reward_advantages": reward_advantages,
            "sequence_mask": sequence_mask,
            "response_start": response_start,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
        }

        if self.args.rl_algorithm == "grpo":
            policy_trainer_inputs.update({"ref_log_probs": rl_batch["ref_log_probs"]})
        else:
            policy_trainer_inputs.update({"ref_log_probs": None})

        actor_loss = self.full_training_step(**policy_trainer_inputs)

        # metric
        with paddle.no_grad():
            rewards = rl_batch["rewards"].mean()
            ori_rewards = rl_batch["ori_rewards"].mean()
            mask_cast = sequence_mask.cast(paddle.float32)
            if self.args.rl_algorithm in ["ppo", "reinforce_plus_plus"]:
                kl_rewards = (rl_batch["kl_rewards"] * mask_cast).sum() / mask_cast.sum()
                rewards_with_kl = (rl_batch["rewards_with_kl"] * mask_cast).sum() / mask_cast.sum()
                if self.args.rl_algorithm == "ppo":
                    values = (rl_batch["reward_values"] * mask_cast).sum() / mask_cast.sum()
                returns = (rl_batch["reward_returns"] * mask_cast).sum() / mask_cast.sum()
            ref_log_probs = rl_batch["ref_log_probs"]
            kl_divergence = ((old_log_probs - ref_log_probs) * mask_cast).sum() / mask_cast.sum()
            mean_generated_length = mask_cast.sum(axis=-1).mean()
            max_generated_length = mask_cast.sum(axis=-1).max()
            min_generated_length = mask_cast.sum(axis=-1).min()

        return {
            # when using PipelienParallel, the loss returned is 0 when not reach
            # accumulated step and the loss returned at accumulated step is a
            # mixed loss.
            "train_policy_loss": actor_loss,
            **(
                {
                    "train_pure_policy_loss": self.info_buffer.get("pure_policy_loss"),
                    "train_kl_loss": self.info_buffer.get("kl_loss"),
                    "train_entropy_loss": self.info_buffer.get("entropy_loss"),
                }
                if self.args.rl_algorithm == "grpo"
                else {}
            ),
            "train_reward": ori_rewards,  # use original reward to log
            **(
                {
                    "train_norm_reward": rewards,
                    "train_kl_reward": kl_rewards,
                    "train_norm_reward_with_kl": rewards_with_kl,
                    "train_pure_policy_loss": self.info_buffer.get("pure_policy_loss"),
                    "train_entropy_loss": self.info_buffer.get("entropy_loss"),
                    **({"train_values": values} if self.args.rl_algorithm == "ppo" else {}),
                    "train_returns": returns,
                }
                if self.args.rl_algorithm in ["ppo", "reinforce_plus_plus"]
                else {}
            ),
            "train_kl_divergence": kl_divergence,
            "train_mean_generated_length": mean_generated_length,
            "train_max_generated_length": max_generated_length,
            "train_min_generated_length": min_generated_length,
        }
