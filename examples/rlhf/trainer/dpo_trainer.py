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

import warnings
from collections import defaultdict

import paddle
import paddle.nn.functional as F
from data.collator import DPODataCollatorWithPadding
from utils.utils import pad_to_length

from paddlenlp.trainer import Trainer


def disable_dropout_in_model(model: paddle.nn.Layer) -> None:
    for module in model.children():
        if isinstance(module, paddle.nn.Dropout):
            module.p = 0


class DPOTrainer(Trainer):
    def __init__(
        self,
        model,
        data_collator,
        ref_model=None,
        beta: float = 0.1,
        disable_dropout: bool = True,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        max_length=None,
        **kwargs
    ):
        super().__init__(model, data_collator=data_collator, **kwargs)

        self.beta = beta
        if ref_model:
            self.ref_model = ref_model
        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)
        self.ref_model.eval()

        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value
        self.beta = beta
        self.max_length = max_length
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        if isinstance(data_collator.__class__, DPODataCollatorWithPadding):
            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

    def concatenated_inputs(self, batch):
        concatenated_batch = {}
        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], paddle.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)

        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], paddle.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = paddle.concat(
                    [concatenated_batch[concatenated_key], pad_to_length(batch[k], max_length, pad_value=pad_value)],
                    axis=0,
                )

        return concatenated_batch

    def dpo_loss(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        reference_free: bool = False,
    ):
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios
        losses = -F.log_sigmoid(self.beta * logits)

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards

    def _get_batch_logps(self, logits, labels, average_log_prob=False):
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id
        labels[labels == self.label_pad_token_id] = 0
        per_token_logps = paddle.take_along_axis(
            F.log_softmax(logits, axis=-1), axis=2, indices=labels.unsqueeze(2)
        ).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(self, model, batch):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(batch)
        len_chosen = batch["chosen_labels"].shape[0]

        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        )[0].astype(paddle.float32)

        all_logps = self._get_batch_logps(
            all_logits, concatenated_batch["concatenated_labels"], average_log_prob=False
        )
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_metrics(self, model, batch, train_eval="train"):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        with paddle.no_grad():
            if self.ref_model is not None:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    reference_chosen_logits,
                    reference_rejected_logits,
                ) = self.concatenated_forward(self.ref_model, batch)
            else:
                raise ValueError("No reference model provided.")

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        reward_accuracies = (chosen_rewards > rejected_rewards).astype(paddle.float32)

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().numpy().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().numpy().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().numpy().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().numpy().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().numpy().mean()
        metrics_list = [
            chosen_rewards,
            rejected_rewards,
            reward_accuracies,
            (chosen_rewards - rejected_rewards),
            policy_rejected_logps.detach(),
            policy_chosen_logps.detach(),
        ]

        return losses.mean(), metrics, metrics_list

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the DPO loss for the given batch of inputs."""
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        loss, metrics, _ = self.get_batch_metrics(model, inputs, train_eval="train")

        if return_outputs:
            return (loss, metrics)

        return loss

    def get_batch_samples(self, model, batch):

        policy_output = model.generate(
            batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        reference_output = self.ref_model.generate(
            batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with paddle.no_grad():
            loss, metrics, metric_list = self.get_batch_metrics(model, inputs, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }

        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = paddle.to_tensor(logits)
        labels = metric_list
        return (loss.detach(), logits, labels)
