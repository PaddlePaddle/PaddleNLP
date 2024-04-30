""" DPO Trainer """
from collections import defaultdict

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddlenlp.trainer import Trainer
from paddlenlp.transformers.model_utils import unwrap_model

def disable_dropout_in_model(model: paddle.nn.Layer) -> None:
    """ "disable dropout"""
    for module in model.children():
        if isinstance(module, paddle.nn.Dropout):
            module.p = 0

class DPOTrainer(Trainer):
    """
    Initialize DPOTrainer.
    """

    def __init__(
        self, model, data_collator, ref_model=None, disable_dropout: bool = True, padding_value: int = 0, **kwargs
    ):
        super().__init__(model, data_collator=data_collator, **kwargs)

        if ref_model:
            self.ref_model = ref_model
        else:
            raise ValueError("No reference model provided.")
        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)
        self.ref_model.eval()

        self.padding_value = padding_value
        self.logprobs = nn.CrossEntropyLoss(reduction="none")
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.effi_token_cnt = 0
        self.all_token_cnt = 0

    def concatenated_inputs(self, batch):
        """Concatenate the chosen and rejected inputs into a single tensor."""
        concatenated_batch = {}
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], paddle.Tensor):
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = batch[k]

        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], paddle.Tensor):
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = paddle.concat(
                    [concatenated_batch[concatenated_key], batch[k]], axis=0
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
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios
        loss = (-F.log_sigmoid(self.args.dpo_beta * logits)).mean()
        return loss

    def _get_batch_logps(self, logits, labels):
        """
        Compute the log probabilities of the given labels under the given logits.
        """

        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")
        logits = logits.astype(paddle.float32)

        # labels: 0 represent the padding token id
        loss_mask = labels != 0
        logps = F.log_softmax(logits, axis=-1)
        per_token_logps = paddle.take_along_axis(logps, axis=2, indices=labels.unsqueeze(2), broadcast=False).squeeze(
            2
        )
        return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(self, model, batch):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(batch)

        len_chosen = batch["chosen_input_ids"].shape[0]
        all_logits = model(concatenated_batch["concatenated_input_ids"])
        if isinstance(all_logits, tuple):
            all_logits = all_logits[0]
        else:
            all_logits = all_logits

        all_logps = self._get_batch_logps(all_logits, concatenated_batch["concatenated_labels"])
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        return (chosen_logps, rejected_logps)

    def cal_chosen_rejected_logps(self, batch, logits):
        """DPO logprobs"""
        labels = batch["chosen_labels"] + batch["rejected_labels"]
        logits = logits.astype("float32")
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        per_token_logps = -self.logprobs(logits, labels.unsqueeze(2)).squeeze(2)
        chosen_logps = paddle.stack(
            [
                (per_token_logps[response_index[0]][response_index[1] : response_index[2]]).sum()
                if response_index[3] != 0
                else paddle.zeros([])
                for response_index in batch["response_indexs"]
            ],
            axis=0,
        )
        rejected_logps = paddle.stack(
            [
                (per_token_logps[response_index[0]][response_index[2] + 1: response_index[3]]).sum()
                if response_index[3] != 0
                else paddle.zeros([])
                for response_index in batch["response_indexs"]
            ],
            axis=0,
        )
        return chosen_logps, rejected_logps

    def get_batch_metrics(self, model, batch, train_eval="train"):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        with paddle.no_grad():
            ref_logits = self.ref_model(
                batch["input_ids"],
                position_ids=batch["position_ids"],
                attention_mask=batch["attention_mask"],
            )[0]
        policy_logits = model(
            batch["input_ids"],
            position_ids=batch["position_ids"],
            attention_mask=batch["attention_mask"],
        )[0]
        policy_chosen_logps, policy_rejected_logps = self.cal_chosen_rejected_logps(batch, policy_logits)
        reference_chosen_logps, reference_rejected_logps = self.cal_chosen_rejected_logps(batch, ref_logits)

        loss = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        policy_chosen_logps, policy_rejected_logps = policy_chosen_logps.detach(), policy_rejected_logps.detach()

        chosen_rewards = self.args.dpo_beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.args.dpo_beta * (policy_rejected_logps - reference_rejected_logps)
        reward_accuracies = (chosen_rewards > rejected_rewards).astype(paddle.float32)

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean()

        for key in metrics:
            metrics[key] = self._nested_gather(paddle.tile(metrics[key], repeat_times=[1, 1])).mean().cpu()
        
        if train_eval == "train":
            self.effi_token_cnt += batch["effi_token_cnt"]
            self.all_token_cnt += batch["all_token_cnt"]

        return loss, metrics

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the DPO loss for the given batch of inputs."""
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")
        if self.args.should_save:
            self.store_metrics(metrics, train_eval="train")
        if return_outputs:
            return (loss, metrics)

        return loss

    def _wrap_ref_model(self, model):
        """Wrap reference model."""
        if unwrap_model(model) is not model:
            return model
        model = fleet.distributed_model(model)

        return model

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):

        """prediction_step"""
        if self.args.pipeline_parallel_degree > 1:
            self.ref_model = self._wrap_ref_model(self.ref_model)
            model = self._wrap_ref_model(model)

            # hack for pipeline mode
            inputs = self._prepare_inputs(inputs)
            return self.prediction_pipeline_step(model, inputs)
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with paddle.no_grad():
            loss, metrics = self.get_batch_metrics(model, inputs, train_eval="eval")

        if self.args.should_save:
            self.store_metrics(metrics, train_eval="eval")
        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }

        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = paddle.to_tensor(logits)
        labels = paddle.zeros(logits.shape[0])
        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics, train_eval="train"):
        """store_metrics"""
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs, **kwargs):
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = paddle.to_tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        if self.state.epoch is not None and train_eval == 'train':
            self.state.epoch *= self.args.num_train_epochs
        return super().log(logs, **kwargs)
