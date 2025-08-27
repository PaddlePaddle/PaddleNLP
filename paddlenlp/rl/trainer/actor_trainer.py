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
from typing import List

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.meta_parallel import ParallelCrossEntropy

from ...datasets.rlhf_datasets.protocol import DataProto
from ..models.ppo_model_utils import (
    RLHFPPOMixedLoss,
    create_startend_row_indices,
    gather_log_probabilities,
)
from .rl_trainer import RLTrainer
from .trainer_utils import guard_set_args


class ActorReferenceTrainerBase(RLTrainer):
    loss_cls = RLHFPPOMixedLoss
    trainer_type = "policy"
    loss_identifier = lambda self, inputs: "actor_loss"

    @paddle.no_grad()
    def compute_logprob(self, batch: DataProto, key) -> DataProto:
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
        input_ids = batch.batch["input_ids"]
        position_ids = batch.batch["position_ids"]
        prompt = batch.batch.get("prompt", None)
        if self.args.use_fused_head_and_loss_fn:
            return self.compute_fused_logprob(
                input_ids=input_ids,
                key=key,
                position_ids=position_ids,
                prompt=prompt,
            )

        log_probs_list = []
        batch_size, sequence_length = input_ids.shape
        per_device_logprob_batch_size = self.args.per_device_logprob_batch_size
        num_batches = (batch_size + per_device_logprob_batch_size - 1) // per_device_logprob_batch_size

        # Pipe model outputs a logits tensor with LMHead, while non-pipe model
        # outputs a tuple with logits tensor as the only one element.
        startend_row_indices = create_startend_row_indices(input_ids, self.tokenizer.pad_token_id)
        response_start = prompt.shape[-1] - 1 if prompt is not None else 0

        for i in range(num_batches):
            # Calculate the start and end indices for the current batch
            start_index = i * per_device_logprob_batch_size
            end_index = min(start_index + per_device_logprob_batch_size, batch_size)

            # Extract the current batch
            current_input_ids = input_ids[start_index:end_index]
            current_startend_row_indices = (
                startend_row_indices[start_index:end_index] if startend_row_indices is not None else None
            )
            current_position_ids = position_ids[start_index:end_index] if position_ids is not None else None
            current_labels = current_input_ids[:, response_start + 1 :]
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
                current_input_ids_rmpad_rolled = update_inputs["input_ids_rmpad_rolled"]
                indices = update_inputs["indices"]
                raw_input_shape = update_inputs["raw_input_shape"]
                pad_size = update_inputs["pad_size"]

            logits = self.model(
                current_input_ids,
                position_ids=current_position_ids,
                attn_mask_startend_row_indices=current_startend_row_indices,
            )
            if not isinstance(logits, paddle.Tensor):
                logits = logits[0]

            if self.args.use_fp32_compute and logits.dtype != paddle.float32:
                logits = logits.cast(paddle.float32)

            if self.args.temperature > 0.0:
                # use inplace method to save gpu memory
                logits.scale_(1 / self.args.temperature)

            if self.args.use_remove_padding:
                from ..utils.bert_padding import pad_input

                if self.model.config.tensor_parallel_degree > 1 and self.model.config.tensor_parallel_output:
                    log_probs = (
                        -ParallelCrossEntropy()(logits.astype("float32"), current_input_ids_rmpad_rolled)
                        .squeeze(axis=-1)
                        .astype(logits.dtype)
                    )
                else:
                    log_probs = gather_log_probabilities(logits, current_input_ids_rmpad_rolled)

                if pad_size > 0:
                    log_probs = log_probs[:, :-pad_size]
                log_probs = pad_input(
                    log_probs.squeeze(0).unsqueeze(-1), indices, batch=raw_input_shape[0], seqlen=raw_input_shape[1]
                ).squeeze(-1)
                log_probs = log_probs[:, response_start:-1].contiguous()
            else:
                if self.model.config.tensor_parallel_degree > 1 and self.model.config.tensor_parallel_output:
                    log_probs = (
                        -ParallelCrossEntropy()(logits[:, response_start:-1].astype("float32"), current_labels)
                        .squeeze(axis=-1)
                        .astype(logits.dtype)
                    )
                else:
                    log_probs = gather_log_probabilities(logits[:, response_start:-1], current_labels)

            log_probs_list.append(log_probs)
            # Set logits to None to save memory
            logits = None
            paddle.device.cuda.empty_cache()

        return DataProto.from_single_dict(
            {key: paddle.concat(log_probs_list, axis=0)}, meta_info={"temperature": self.args.temperature}
        )

    @paddle.no_grad()
    def compute_fused_logprob(
        self, input_ids: paddle.Tensor, key, position_ids: paddle.Tensor = None, prompt=None, loop_chunk_size=1024
    ) -> DataProto:
        log_probs_list = []
        batch_size, sequence_length = input_ids.shape
        per_device_logprob_batch_size = self.args.per_device_logprob_batch_size
        num_batches = (batch_size + per_device_logprob_batch_size - 1) // per_device_logprob_batch_size

        # Pipe model outputs a logits tensor with LMHead, while non-pipe model
        # outputs a tuple with logits tensor as the only one element.
        startend_row_indices = create_startend_row_indices(input_ids, self.tokenizer.pad_token_id)
        response_start = (prompt.shape[-1] - 1) if prompt is not None else 0

        num_embeddings = self.model.config.vocab_size
        tensor_parallel_degree = self.model.config.tensor_parallel_degree
        tensor_parallel_output = self.model.config.tensor_parallel_output

        for i in range(num_batches):
            # Calculate the start and end indices for the current batch
            start_index = i * per_device_logprob_batch_size
            end_index = min(start_index + per_device_logprob_batch_size, batch_size)

            # Extract the current batch
            current_input_ids = input_ids[start_index:end_index]
            current_startend_row_indices = (
                startend_row_indices[start_index:end_index] if startend_row_indices is not None else None
            )
            current_position_ids = position_ids[start_index:end_index] if position_ids is not None else None
            current_labels = current_input_ids[:, response_start + 1 :]

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

            # NOTE: for use_fused_head_and_loss_fn
            self.model.training = True
            hidden_states, lm_head_weight, lm_head_bias, transpose_y = self.model(
                current_input_ids,
                position_ids=current_position_ids,
                attn_mask_startend_row_indices=current_startend_row_indices,
            )

            self.model.training = False

            if self.args.use_remove_padding:
                if pad_size > 0:
                    hidden_states = hidden_states[:, :-pad_size]

                from ..utils.bert_padding import pad_input

                hidden_states = pad_input(
                    hidden_states.squeeze(0), indices, batch=raw_input_shape[0], seqlen=raw_input_shape[1]
                ).contiguous()

            if self.args.use_fp32_compute and hidden_states.dtype != paddle.float32:
                hidden_states = hidden_states.cast(paddle.float32)
                lm_head_weight = lm_head_weight.cast(paddle.float32)
                if lm_head_bias is not None:
                    lm_head_bias = lm_head_bias.cast(paddle.float32)

            # Recover
            hidden_states = hidden_states[:, response_start:-1, :]
            dtype = hidden_states.dtype
            original_shape = hidden_states.shape
            if tensor_parallel_degree > 1:
                assert tensor_parallel_output, (
                    "When tensor_parallel_degree > 1 and use_fused_head_and_loss_fn, "
                    "tensor_parallel_output needs to be set to True."
                )
            # Parallel Configuration
            if tensor_parallel_degree > 1 and tensor_parallel_output:
                hcg = fleet.get_hybrid_communicate_group()
                model_parallel_group = hcg.get_model_parallel_group()
                tensor_parallel_degree = hcg.get_model_parallel_world_size()

            # reshape
            hidden_states = hidden_states.reshape([-1, original_shape[-1]])
            labels = current_labels.reshape([-1])

            n_tokens = hidden_states.shape[0]
            n_classes = lm_head_weight.shape[0] if transpose_y else lm_head_weight.shape[1]

            # convert dtype of weights and biases of lm_head
            lm_head_weight_cast = lm_head_weight.astype(dtype)
            if lm_head_bias is not None:
                lm_head_bias_cast = lm_head_bias.astype(dtype)

            # use indices to distinguish the devices.
            if tensor_parallel_degree > 1 and tensor_parallel_output:
                rank = hcg.get_model_parallel_rank()
                per_part_size = num_embeddings // tensor_parallel_degree
                indices = paddle.arange(
                    rank * per_part_size,
                    rank * per_part_size + n_classes,
                    dtype=labels.dtype,
                ).unsqueeze(0)
            else:
                indices = paddle.arange(num_embeddings, dtype=labels.dtype).unsqueeze(0)

            log_prob_chunks = []
            for ci in range(0, n_tokens, loop_chunk_size):
                token_start_idx = ci
                token_end_idx = min(ci + loop_chunk_size, n_tokens)
                hidden_states_chunk = hidden_states[token_start_idx:token_end_idx]
                labels_chunk = labels[token_start_idx:token_end_idx]

                # Calculate the current logits_chunk,  not fused linear
                logits_chunk_cast = paddle.matmul(hidden_states_chunk, lm_head_weight_cast, transpose_y=transpose_y)
                if lm_head_bias is not None:
                    logits_chunk_cast += lm_head_bias_cast

                logits_chunk = logits_chunk_cast.astype("float32")
                logits_chunk = logits_chunk / self.args.temperature

                # rewritten as cross entropy
                if tensor_parallel_degree > 1 and tensor_parallel_output:
                    token_loss_chunk = mp_ops._c_softmax_with_cross_entropy(
                        logits_chunk,
                        labels_chunk,
                        group=model_parallel_group,
                        return_softmax=False,
                    )
                else:
                    token_loss_chunk = F.cross_entropy(logits_chunk, labels_chunk, reduction="none")
                log_prob_chunk = -token_loss_chunk.squeeze(axis=-1)
                log_prob_chunks.append(log_prob_chunk)

            log_probs = paddle.concat(log_prob_chunks, axis=-1).reshape(original_shape[:-1])
            log_probs_list.append(log_probs)

            log_prob_chunks = None
            paddle.device.cuda.empty_cache()

        return DataProto.from_single_dict(
            {key: paddle.concat(log_probs_list, axis=0)}, meta_info={"temperature": self.args.temperature}
        )

    def update_actor(self, rl_batch: DataProto) -> DataProto:
        # inputs shared by policy and value trainer
        input_ids = rl_batch.batch["input_ids"].contiguous()  # length: src+tgt
        position_ids = rl_batch.batch["position_ids"]  # length: src+tgt
        sequence_mask = rl_batch.batch["eos_mask"]  # length: tgt(-1)
        if self.args.use_fp32_compute and sequence_mask.dtype != paddle.float32:
            sequence_mask = sequence_mask.cast(paddle.float32)
        # inputs used by policy trainer
        old_log_probs = rl_batch.batch["log_probs"]  # length: tgt(-1)
        reward_advantages = rl_batch.batch["reward_advantages"]  # length: tgt(-1)

        response_start = rl_batch.batch["prompt"].shape[-1] - 1

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
            policy_trainer_inputs.update({"ref_log_probs": rl_batch.batch["ref_log_probs"]})
        else:
            policy_trainer_inputs.update({"ref_log_probs": None})

        actor_loss = self.full_training_step(**policy_trainer_inputs)

        # metric
        with paddle.no_grad():
            rewards = rl_batch.batch["rewards"].mean()
            ori_rewards = rl_batch.batch["ori_rewards"].mean()
            mask_cast = sequence_mask.cast(paddle.float32)
            if self.args.rl_algorithm in ["ppo", "reinforce_plus_plus"]:
                kl_rewards = (rl_batch.batch["kl_rewards"] * mask_cast).sum() / mask_cast.sum()
                rewards_with_kl = (rl_batch.batch["rewards_with_kl"] * mask_cast).sum() / mask_cast.sum()
                if self.args.rl_algorithm == "ppo":
                    values = (rl_batch.batch["reward_values"] * mask_cast).sum() / mask_cast.sum()
                returns = (rl_batch.batch["reward_returns"] * mask_cast).sum() / mask_cast.sum()
            ref_log_probs = rl_batch.batch["ref_log_probs"]
            kl_divergence = ((old_log_probs - ref_log_probs) * mask_cast).sum() / mask_cast.sum()
            mean_generated_length = mask_cast.sum(axis=-1).mean()
            max_generated_length = mask_cast.sum(axis=-1).max()
            min_generated_length = mask_cast.sum(axis=-1).min()

        return DataProto(
            meta_info={
                "metrics": {
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
            }
        )


class ActorReferenceTrainer(ActorReferenceTrainerBase):
    @paddle.no_grad()
    def generate_sequences(self, prompt_only_batch: DataProto, do_eval=False) -> List[DataProto]:
        """Rollout a batch of experiences."""
        input_ids = prompt_only_batch.batch["input_ids"]

        repeat_num = 1 if do_eval else self.args.rollout_n

        with guard_set_args(self.model.config, {"use_fused_head_and_loss_fn": False}):
            sequences = self.get_model(False).generate(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                do_eval=do_eval,
                repeat_num=repeat_num,
            )[0]

        if repeat_num > 1:
            input_ids = input_ids.repeat_interleave(repeat_num, axis=0)

        if self.args.use_rm_server or self.args.use_rule_reward:
            label_ids = prompt_only_batch.batch["label_ids"]
            if repeat_num > 1:
                label_ids = label_ids.repeat_interleave(repeat_num, axis=0)

        sequences = sequences.reshape([input_ids.shape[0] // repeat_num, repeat_num, -1])
        if do_eval:
            sequences = sequences.transpose([1, 0, 2])
        # prompt, sequence, attention_mask
        return [
            DataProto.from_single_dict(
                {
                    "prompt": input_ids[idx * len(seq) : (idx + 1) * len(seq)],  # src prompt
                    "input_ids": seq,
                    **(
                        {"label_ids": label_ids[idx * len(seq) : (idx + 1) * len(seq)]}
                        if self.args.use_rm_server or self.args.use_rule_reward
                        else {}
                    ),  # tgt response
                    "index": np.array([str(uuid.uuid4())] * len(seq), dtype=object),
                }
            )
            for idx, seq in enumerate(sequences)
        ]
