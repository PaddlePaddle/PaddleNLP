# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

""" DPO Trainer """
import types
from collections import OrderedDict, defaultdict

import paddle
import paddle.nn.functional as F
from paddle import framework
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import ParallelCrossEntropy

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

    def __init__(self, model, data_collator, ref_model=None, disable_dropout: bool = True, **kwargs):
        super().__init__(model, data_collator=data_collator, **kwargs)

        self.reference_free = kwargs.pop("reference_free", False)
        if ref_model:
            self.ref_model = ref_model
            self.ref_model = self._wrap_ref_model(self.ref_model)
            self.ref_model.eval()
        elif not self.reference_free:
            raise ValueError("Please provide a reference model.")
        if self.reference_free and self.args.dpo_loss_type not in ["sigmoid", "hinge", "ipo"]:
            raise ValueError(f"{self.args.dpo_loss_type} is not a valid loss type for DPO reference_free.")
        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)
        if self.model.config.tensor_parallel_output and self.model.config.tensor_parallel_degree > 1:
            self.logprobs = ParallelCrossEntropy()
        else:
            self.logprobs = paddle.nn.CrossEntropyLoss(reduction="none")
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

    def dpo_loss(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps=None,
        reference_rejected_logps=None,
    ):
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.reference_free:
            ref_logratios = 0
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        if self.args.dpo_loss_type == "sigmoid":
            loss = (
                -F.log_sigmoid(self.args.dpo_beta * logits) * (1 - self.args.dpo_label_smoothing)
                - F.log_sigmoid(-self.args.dpo_beta * logits) * self.args.dpo_label_smoothing
            )
        elif self.args.dpo_loss_type == "hinge":
            loss = F.relu(1 - self.args.dpo_beta * logits)
        elif self.args.dpo_loss_type == "ipo":
            # parameter for the IPO loss, denoted by tau in the paper.
            loss = (logits - 1 / (2 * self.args.dpo_beta)) ** 2
        elif self.args.dpo_loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clip(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clip(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for
            # chosen (rejected) is estimated using the rejected (chosen) half.
            loss = paddle.concat(
                (
                    1 - F.sigmoid(self.args.dpo_beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.args.dpo_beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        elif self.args.dpo_loss_type == "sppo_hard":
            # In the paper (https://arxiv.org/pdf/2405.00675), SPPO employs a soft probability
            # approach, estimated using the PairRM score. The probability calculation is
            # conducted outside of the trainer class. The version described here is the hard
            # probability version, where P in Equation (4.7) of Algorithm 1 is set to 1 for
            # the winner and 0 for the loser.
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            loss = (a - 0.5 / self.args.dpo_beta) ** 2 + (b + 0.5 / self.args.dpo_beta) ** 2
        elif self.args.dpo_loss_type == "nca_pair":
            chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * self.args.dpo_beta
            rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * self.args.dpo_beta
            loss = (
                -F.log_sigmoid(chosen_rewards)
                - 0.5 * F.log_sigmoid(-chosen_rewards)
                - 0.5 * F.log_sigmoid(-rejected_rewards)
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.args.dpo_loss_type}. "
                "Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair', 'sppo_hard', 'nca_pair']"
            )
        return loss.mean()

    def get_batch_logps(
        self,
        batch,
        logits,
        average_log_prob=False,
    ):
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
                (per_token_logps[response_index[0]][response_index[2] + 1 : response_index[3]]).sum()
                if response_index[3] != 0
                else paddle.zeros([])
                for response_index in batch["response_indexs"]
            ],
            axis=0,
        )
        if average_log_prob:
            chosen_response_length = batch["response_indexs"][:, 2] - batch["response_indexs"][:, 1]
            rejected_response_length = batch["response_indexs"][:, 3] - batch["response_indexs"][:, 2]
            chosen_logps /= chosen_response_length
            rejected_logps /= rejected_response_length
        return chosen_logps, rejected_logps

    def get_batch_metrics(self, model, batch, train_eval="train"):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        if hasattr(self.model.config, "dpo") and self.model.config.dpo:
            dpo_inputs = {
                "input_ids": batch["input_ids"],
                "position_ids": batch["position_ids"],
                "chosen_labels": batch["chosen_labels"],
                "rejected_labels": batch["rejected_labels"],
                "response_indexs": batch["response_indexs"],
            }
            if "attention_mask" in batch:
                dpo_inputs["attention_mask"] = batch["attention_mask"]
            if "attn_mask_startend_row_indices" in batch:
                dpo_inputs["attn_mask_startend_row_indices"] = batch["attn_mask_startend_row_indices"]
            if self.reference_free:
                reference_chosen_logps, reference_rejected_logps = None, None
            else:
                with paddle.no_grad():
                    reference_chosen_logps, reference_rejected_logps = self.ref_model(**dpo_inputs)
                dpo_inputs["reference_chosen_logps"] = reference_chosen_logps
                dpo_inputs["reference_rejected_logps"] = reference_rejected_logps
            loss, policy_chosen_logps, policy_rejected_logps = model(**dpo_inputs)
        else:
            dpo_inputs = {
                "input_ids": batch["input_ids"],
                "position_ids": batch["position_ids"],
            }
            if "attention_mask" in batch:
                dpo_inputs["attention_mask"] = batch["attention_mask"]
            if "attn_mask_startend_row_indices" in batch:
                dpo_inputs["attn_mask_startend_row_indices"] = batch["attn_mask_startend_row_indices"]
            if self.reference_free:
                reference_chosen_logps, reference_rejected_logps = None, None
            else:
                with paddle.no_grad():
                    ref_logits = self.ref_model(**dpo_inputs)[0]
                    reference_chosen_logps, reference_rejected_logps = self.get_batch_logps(
                        batch,
                        ref_logits,
                        average_log_prob=self.args.dpo_loss_type == "ipo",
                    )
            policy_logits = model(**dpo_inputs)[0]
            policy_chosen_logps, policy_rejected_logps = self.get_batch_logps(
                batch,
                policy_logits,
                average_log_prob=self.args.dpo_loss_type == "ipo",
            )

            loss = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )

        policy_chosen_logps, policy_rejected_logps = policy_chosen_logps.detach(), policy_rejected_logps.detach()
        if self.reference_free:
            chosen_rewards = self.args.dpo_beta * (policy_chosen_logps)
            rejected_rewards = self.args.dpo_beta * (policy_rejected_logps)
            reward_accuracies = (chosen_rewards > rejected_rewards).astype(paddle.float32)
        else:
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
        return loss, metrics

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the DPO loss for the given batch of inputs."""
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")
        if self.args.should_save:
            self.store_metrics(metrics, train_eval="train")
        if return_outputs:
            return (loss, metrics)

        return loss

    def _wrap_model(self, model, training=True):
        """Wrap model."""
        model = super()._wrap_model(model, training)
        if self.args.pipeline_parallel_degree > 1:
            model._prepare_pipeline_inputs_func = prepare_pipeline_dpo_inputs_func
            model.eval_dpo_batch = types.MethodType(eval_dpo_batch, model)
            model._forward_step = types.MethodType(_forward_step, model)
            model.broadcast_pp_final_output = types.MethodType(broadcast_pp_final_output, model)
        return model

    def _wrap_ref_model(self, model):
        """Wrap reference model."""
        if unwrap_model(model) is not model:
            return model
        self.amp_dtype = "float16" if self.args.fp16 else "bfloat16"
        model = paddle.amp.decorate(
            models=model,
            level=self.args.fp16_opt_level,
            dtype=self.amp_dtype,
        )
        model = fleet.distributed_model(model)
        if self.args.pipeline_parallel_degree > 1:
            model._prepare_pipeline_inputs_func = prepare_pipeline_dpo_inputs_func
            model.eval_dpo_batch = types.MethodType(eval_dpo_batch, model)
            model._forward_step = types.MethodType(_forward_step, model)
            model.broadcast_pp_final_output = types.MethodType(broadcast_pp_final_output, model)

        return model

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """evaluate"""
        self.model_wrapped = self._wrap_ref_model(self.model_wrapped)
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):

        """prediction_step"""
        if self.args.pipeline_parallel_degree > 1:
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
        if self.state.epoch is not None and train_eval == "train":
            self.state.epoch *= self.args.num_train_epochs
        return super().log(logs, **kwargs)

    def split_response_indexs_for_pipeline(self, batch):
        """
        split response indexs for pipeline parallel mode.
        """
        batch_response_indexs = []
        response_indexs = None
        response_num = [0] * batch["input_ids"].shape[0]
        last_batch = -1
        if batch["response_indexs"][0][1] == 0:
            use_sparse_head_and_loss_fn = True
        else:
            use_sparse_head_and_loss_fn = False
        last_batch_response_length = 0

        for response_index in batch["response_indexs"]:
            if response_index[0] == last_batch:
                response_index -= last_batch_response_length
                response_index[0] = 0
                response_indexs.append(response_index)
            else:
                last_batch += 1
                if use_sparse_head_and_loss_fn:
                    last_batch_response_length = response_index[1]
                if response_indexs is not None:
                    batch_response_indexs.append(response_indexs)
                response_index -= last_batch_response_length
                response_index[0] = 0
                response_indexs = [response_index]
            response_num[last_batch] += 1
        batch_response_indexs.append(response_indexs)
        max_response_num = max(response_num)
        for i in range(len(response_num)):
            for _ in range(max_response_num - response_num[i]):
                batch_response_indexs[i].append(paddle.to_tensor([0, 0, -1, 0], dtype="int64"))

        return paddle.to_tensor(batch_response_indexs)

    def prediction_pipeline_step(
        self,
        model,
        batch,
    ):
        """
        prediction_step function for pipeline parallel mode.
        """
        config_backup = model.micro_batch_size, model.accumulate_steps
        model.accumulate_steps = batch["input_ids"].shape[0]
        model.micro_batch_size = 1
        if not self.reference_free:
            self.ref_model.accumulate_steps = model.accumulate_steps
            self.ref_model.micro_batch_size = model.micro_batch_size
        # [1, total_response_indexs] -> [bs, response_indexs]
        batch["response_indexs"] = self.split_response_indexs_for_pipeline(batch)
        batch["reference_chosen_logps"] = None
        batch["reference_rejected_logps"] = None
        total_response_num = batch["response_indexs"].shape[0] * batch["response_indexs"].shape[1]

        inputs, labels = model._prepare_pipeline_inputs_func(batch)
        with paddle.no_grad():
            with self.autocast_smart_context_manager():
                policy_chosen_logps, policy_rejected_logps = model.eval_dpo_batch(
                    data=[inputs, labels], total_response_num=total_response_num
                )
                policy_chosen_logps = paddle.masked_select(policy_chosen_logps, policy_chosen_logps != 0)
                policy_rejected_logps = paddle.masked_select(policy_rejected_logps, policy_rejected_logps != 0)
                if not self.reference_free:
                    reference_chosen_logps, reference_rejected_logps = self.ref_model.eval_dpo_batch(
                        [inputs, labels], total_response_num=total_response_num
                    )
                    reference_chosen_logps = paddle.masked_select(reference_chosen_logps, reference_chosen_logps != 0)
                    reference_rejected_logps = paddle.masked_select(
                        reference_rejected_logps, reference_rejected_logps != 0
                    )
                else:
                    reference_chosen_logps, reference_rejected_logps = None, None

        loss = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        policy_chosen_logps, policy_rejected_logps = policy_chosen_logps.detach(), policy_rejected_logps.detach()
        if not self.reference_free:
            chosen_rewards = self.args.dpo_beta * (policy_chosen_logps - reference_chosen_logps)
            rejected_rewards = self.args.dpo_beta * (policy_rejected_logps - reference_rejected_logps)
        else:
            chosen_rewards = self.args.dpo_beta * (policy_chosen_logps)
            rejected_rewards = self.args.dpo_beta * (policy_rejected_logps)

        reward_accuracies = (chosen_rewards > rejected_rewards).astype(paddle.float32)
        metrics = {}
        metrics["eval_rewards/chosen"] = chosen_rewards.mean()
        metrics["eval_rewards/rejected"] = rejected_rewards.mean()
        metrics["eval_rewards/accuracies"] = reward_accuracies.mean()
        metrics["eval_rewards/margins"] = (chosen_rewards - rejected_rewards).mean()
        metrics["eval_logps/rejected"] = policy_rejected_logps.mean()
        metrics["eval_logps/chosen"] = policy_chosen_logps.mean()
        for key in metrics:
            metrics[key] = self._nested_gather(paddle.tile(metrics[key], repeat_times=[1, 1])).mean().cpu()
        if self.args.should_save:
            self.store_metrics(metrics, train_eval="eval")
        model.micro_batch_size, model.accumulate_steps = config_backup
        if not self.reference_free:
            self.ref_model.micro_batch_size, self.ref_model.accumulate_steps = config_backup
        return (loss, None, None)

    def training_pipeline_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.
        """
        # accumulation data
        if not hasattr(self, "_pp_data_buffer"):
            self._pp_data_buffer = []
        self._pp_data_buffer.append(inputs)
        if len(self._pp_data_buffer) != self.args.gradient_accumulation_steps:
            return paddle.zeros([])
        response_num = [
            len(self._pp_data_buffer[i]["response_indexs"]) for i in range(self.args.gradient_accumulation_steps)
        ]
        max_response_num = max(response_num)
        for i in range(self.args.gradient_accumulation_steps):
            self._pp_data_buffer[i]["response_indexs"] = paddle.concat(
                [
                    self._pp_data_buffer[i]["response_indexs"],
                    paddle.to_tensor((max_response_num - response_num[i]) * [[0, 0, -1, 0]], dtype="int64"),
                ],
                axis=0,
            )
        total_response_num = self.args.gradient_accumulation_steps * max_response_num
        concatenated_inputs = {}
        for key in self._pp_data_buffer[i].keys():
            concatenated_inputs[key] = [
                self._pp_data_buffer[i][key] for i in range(self.args.gradient_accumulation_steps)
            ]
        concatenated_inputs["reference_chosen_logps"] = None
        concatenated_inputs["reference_rejected_logps"] = None

        self._pp_data_buffer = []
        inputs, labels = model._prepare_pipeline_inputs_func(concatenated_inputs)
        model_config_backup = model.micro_batch_size, model.accumulate_steps
        model.micro_batch_size = self.args.per_device_train_batch_size
        model.accumulate_steps = self.args.gradient_accumulation_steps
        if not self.reference_free:
            ref_model_config_backup = self.ref_model.micro_batch_size, self.ref_model.accumulate_steps
            self.ref_model.accumulate_steps = model.accumulate_steps
            self.ref_model.micro_batch_size = model.micro_batch_size
            with paddle.no_grad():
                with self.autocast_smart_context_manager():
                    reference_chosen_logps, reference_rejected_logps = self.ref_model.eval_dpo_batch(
                        data=[inputs, labels], total_response_num=total_response_num
                    )
            labels = (
                labels[0],
                labels[1],
                labels[2],
                reference_chosen_logps.split(num_or_sections=model.accumulate_steps, axis=0),
                reference_rejected_logps.split(num_or_sections=model.accumulate_steps, axis=0),
            )
        train_inputs = [inputs, labels]
        train_inputs = model._prepare_training(train_inputs, self.optimizer, self.lr_scheduler)
        model.optimizer = None  # we do not use `PipelineParallel` to handler optimizer step
        model.lr_scheduler = None
        with self.autocast_smart_context_manager():
            loss = model.forward_backward_pipeline(train_inputs, self.scaler if self.do_grad_scaling else None)
        model.micro_batch_size, model.accumulate_steps = model_config_backup
        if not self.reference_free:
            self.ref_model.micro_batch_size, self.ref_model.accumulate_steps = ref_model_config_backup
        return loss.detach()


def prepare_pipeline_dpo_inputs_func(inputs):
    """Prepare pipeline inputs"""
    if "attention_mask" in inputs:
        first_stage_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
        ]
    else:
        first_stage_keys = [
            "input_ids",
            "attn_mask_startend_row_indices",
            "position_ids",
        ]

    last_stage_keys = [
        "chosen_labels",
        "rejected_labels",
        "response_indexs",
        "reference_chosen_logps",
        "reference_rejected_logps",
    ]

    def get_expected_keys(inputs, keys):
        ret = tuple([inputs.pop(k) for k in keys if k in inputs])
        if len(ret) == 1:
            ret = ret[0]
        return ret

    if type(inputs) is dict or type(inputs) is OrderedDict:
        return [
            get_expected_keys(inputs, first_stage_keys),
            get_expected_keys(inputs, last_stage_keys),
        ]

    keys = list(inputs[0].keys())
    inputs_batch = {key: [data.pop(key) for data in inputs] for key in keys}
    return [
        get_expected_keys(inputs_batch, first_stage_keys),
        get_expected_keys(inputs_batch, last_stage_keys),
    ]


def eval_dpo_batch(self, data, total_response_num):
    """eval_dpo_batch"""
    # reset the virtual pp rank for each run
    self.set_virtual_pipeline_rank(0)

    self._layers.eval()

    # store data id for micro_batch
    self.micro_batch_id = 0

    # store total loss of entire batch
    self.total_loss = None

    startup_steps = self.num_stages - self.stage_id - 1
    startup_steps = min(startup_steps, self.accumulate_steps)
    steady_steps = self.accumulate_steps - startup_steps

    input_buffers = []
    output_buffers = []

    # convert to micro dataset
    micro_dataset = self._wrap_data(data)

    for step_id in range(startup_steps):
        input_tensor = self._p2p_helper.recv_forward(self.is_pipeline_first_stage())

        output_tensor = self._forward_step(input_tensor, micro_dataset)
        self._p2p_helper.send_forward(output_tensor, self.is_pipeline_last_stage(), skip_check_meta=True)

        input_buffers.append(input_tensor)
        output_buffers.append(output_tensor)

    if steady_steps > 0:
        input_tensor = self._p2p_helper.recv_forward(self.is_pipeline_first_stage())

    for i in range(steady_steps):
        last_iter = i == (steady_steps - 1)

        output_tensor = self._forward_step(input_tensor, micro_dataset)
        self._p2p_helper.send_forward(output_tensor, self.is_pipeline_last_stage(), skip_check_meta=True)

        input_buffers.append(input_tensor)
        output_buffers.append(output_tensor)

        if not last_iter:
            input_tensor = self._p2p_helper.recv_forward(self.is_pipeline_first_stage())
    return self.broadcast_pp_final_output(output_buffers, total_response_num)


def _forward_step(self, input_tensor, micro_dataset, chunk_id=None):
    if self._enable_timer:
        self.timers("forward_step").start()
    if self.is_pipeline_first_stage():
        input_tensor = next(micro_dataset)[0]
        self._check_micro_batch_data_valid(input_tensor)

    assert chunk_id is None or isinstance(chunk_id, int)

    output_tensor = self._layers.forward(input_tensor, chunk_id=chunk_id)

    if self.is_pipeline_last_stage():
        assert self._layers._loss_fn is not None, "loss function should exist to compute loss"
        labels = next(micro_dataset)[1]
        self._check_micro_batch_data_valid(labels)
        for idx, loss_fn in enumerate(self._layers._loss_fn):
            output_tensor = loss_fn(output_tensor, labels[0], labels[1], labels[2], labels[3], labels[4])
            if labels[3] is not None and labels[4] is not None:
                assert isinstance(
                    output_tensor, (paddle.Tensor, framework.core.eager.Tensor)
                ), "Currently, loss_fn should obtain Paddle.Tensor dtype"

                with paddle.amp.auto_cast(enable=False):
                    if self.accumulate_steps > 1 and not self._delay_scale_loss:
                        output_tensor = output_tensor / self.accumulate_steps

                    if self.total_loss is None:
                        self.total_loss = []
                    if len(self.total_loss) <= idx:
                        self.total_loss.append(paddle.zeros_like(output_tensor))
                    self.total_loss[idx] += output_tensor.detach()
                if idx == self.loss_fn_idx:
                    loss_tensor = output_tensor

    if self.is_pipeline_first_stage() or self.is_pipeline_last_stage():
        # Only increase micro batch id at virtual first/last pp stage.
        # The micro batch id is used to load data, therefore, only increase it when load data.
        self.micro_batch_id += 1
    if self._enable_timer:
        self.timers("forward_step").stop()
    if self.is_pipeline_last_stage() and labels[3] is not None and labels[4] is not None:
        return loss_tensor
    else:
        return output_tensor


def broadcast_pp_final_output(self, output_buffers, total_response_num):
    """broadcast_pp_final_output"""
    # Since the last backward run in interleave will set the virtual rank to 0,
    # here we need to check last stage ignoring virtual stage.
    if self.is_pipeline_last_stage(ignore_virtual=True):
        chosen_logps = paddle.concat([buffer[0] for buffer in output_buffers], axis=0)
        rejected_logps = paddle.concat([buffer[1] for buffer in output_buffers], axis=0)
        paddle.distributed.broadcast(chosen_logps, src=self.global_rank, sync_op=True, group=self.pp_group)
        paddle.distributed.broadcast(rejected_logps, src=self.global_rank, sync_op=True, group=self.pp_group)
    else:
        chosen_logps = paddle.zeros(shape=[total_response_num], dtype="float32")
        rejected_logps = paddle.zeros(shape=[total_response_num], dtype="float32")
        paddle.distributed.broadcast(
            chosen_logps,
            src=self._hcg.get_rank_from_stage(self.num_stages - 1),
            sync_op=True,
            group=self.pp_group,
        )
        paddle.distributed.broadcast(
            rejected_logps,
            src=self._hcg.get_rank_from_stage(self.num_stages - 1),
            sync_op=True,
            group=self.pp_group,
        )
    return chosen_logps, rejected_logps
