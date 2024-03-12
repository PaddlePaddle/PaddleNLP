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

import copy
import itertools
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
from data import DummyDataset, PromptOnlyBatch
from infer_utils import InferEvalModel, infer_guard
from models.ppo_model_utils import (
    RLHFPPOMixedLoss,
    RLHFValueLoss,
    create_loss,
    gather_log_probabilities,
    make_position_ids,
)
from paddle.distributed import fleet
from paddle.io import DataLoader, Dataset, DistributedBatchSampler
from paddle.utils import map_structure
from rich.console import Console
from rich.table import Table
from trainer_utils import (
    MuteDefaultFlowCallback,
    PipeEvalModel,
    batch_retokenize,
    guard_set_args,
    is_same_tokenizer,
)

from paddlenlp.data import DataCollator
from paddlenlp.generation import GenerationConfig
from paddlenlp.trainer.trainer import (
    EvalLoopOutput,
    EvalPrediction,
    ShardingOption,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    logger,
    speed_metrics,
)
from paddlenlp.transformers import PretrainedModel, PretrainedTokenizer


class StepTrainer(Trainer):
    """
    Trainer enhanced with step-level training combining with patches of Trianer.
    We can use this to do training whose step is composed of multi models (by
    multiple instances of StepTrainer, such as PPO. Additionally, using a mixed
    loss and get the separated loss metrics is supported.
    """

    # used to create criterion for trainer
    loss_cls: type
    # Moreover, a model/StepTrainer instance may use a mixed loss which uses a
    # different loss for different step and inputs, while we often want to get
    # the separated loss metric. We use a callable discriminator using inputs
    # (dict) as arguments and returning corresponding loss name to identify
    # current loss. NOTE: please make the loss name ends with "_loss". `tr_loss`
    # is the default loss name used in trainer.train.
    loss_identifier: callable
    # refer to mark_step_loss. NOTE: This is transparent to users
    loss_step_indice: Dict
    # When using multiple instances of StepTrainer collaborate to do one training
    # step, each should use its own vars such as loss/model/step_control which are
    # local vars in Trainer.train, we define these vars by `train_step_vars`. They
    # are vars needed by full_training_step for training control, as following:
    # tr_loss, model, epoch, step, step_control. NOTE: This is transparent to users.
    # some vars such as `epoch` are meaningless, they are needed just because
    # full_training_step copies code from Trainer.train which is designed for
    # complete training process.
    # TODO(guosheng): use namedtuple or dataclass to make it more readable.
    train_step_vars: Dict

    def __init__(
        self,
        model: Union[PretrainedModel, nn.Layer] = None,
        criterion: nn.Layer = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
        tokenizer: Optional[PretrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
        preprocess_logits_for_metrics: Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor] = None,
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
        # criterion is only used for non-PipelineParallel models. criterion is
        # included in model for PipelineParallel.
        if getattr(self, "loss_cls", None) and self.criterion is None:
            self.criterion = self.create_criterion()

    def create_criterion(self):
        """loss creator for trainer."""
        criterion = create_loss(self.loss_cls, self.model.config, self.args, merge_labels=True)
        return criterion

    def loss_identifier(self, inputs: Dict) -> str:
        """
        Moreover, a model/StepTrainer instance may use a mixed loss which uses a
        different loss for different step and inputs, while we often want to get
        the separated loss metric. We use a callable discriminator using inputs
        (dict) as arguments and returning corresponding loss name to identify
        current loss. NOTE: please make the loss name ends with "_loss". `tr_loss`
        is the default loss name used in trainer.train.
        """
        return "tr_loss"

    def get_model(self, train=False):
        """
        model visitor wrapps PipelineParalle and Inference model to do evaulation
        and generation.
        """
        if train:
            return self.model_wrapped
        model = getattr(self, "_eval_model", None)
        if model is not None:
            return model
        if self.args.pipeline_parallel_degree > 1:
            # Only accept wrapped model for pipeline_parallel mode
            model = PipeEvalModel(self)
            self._eval_model = model
        else:
            model = InferEvalModel(self)
            self._eval_model = model
        return model

    def get_train_step_vars(self, vars: Dict = None) -> Dict:
        """
        return `train_step_vars`. If not exists, create it first. If `vars` is
        not None, update `train_step_vars` with it.
        """
        if not hasattr(self, "train_step_vars"):
            # should be called after model is wrapped since the model field should
            # use model_wrapped.

            assert self.model is not self.model_wrapped
            self.train_step_vars = {
                # meaningless vars can pass from outter, dummy value is enough
                "epoch": 0,  # meaningless for step training
                "step": 0,  # meaningless for step training
                "steps_in_epoch": 100000,  # meaningless for step training
                "step_control": 0,  # to control training process
                "model": self.model_wrapped,
                # "tr_loss": paddle.to_tensor(0.0),  # lazy create
            }
        if vars:
            self.train_step_vars.update(vars)
        return self.train_step_vars

    def full_training_step(self, **inputs) -> paddle.Tensor:
        """
        Accept any valid key word arguments of model and loss as inputs, they
        would be sent to model and then loss. Mostly it is similar to output from
        data collator.
        Return loss var. However when using PipelienParallel, the loss returned
        is 0 when not reach accumulated step and the loss returned at accumulated
        step is a mixed loss. We can use `get_step_loss` to get the actual loss.
        """
        # if model has multi losses which are combined into one mixed criterion,
        # loss statistic var may change for different training steps according
        # to inputs.
        train_step_vars = self.get_train_step_vars()
        loss_name = self.loss_identifier(inputs)
        loss_var = train_step_vars.get(loss_name, None)
        if loss_var is None:
            loss_var = paddle.to_tensor(0.0)
            train_step_vars[loss_name] = loss_var
        # trainer.train use `tr_loss` as loss var
        train_step_vars["tr_loss"] = loss_var

        new_train_step_vars = super().full_training_step(inputs, **train_step_vars)

        # minimally update
        train_step_vars = self.get_train_step_vars(
            {"step_control": new_train_step_vars["step_control"], loss_name: new_train_step_vars["tr_loss"]}
        )
        if loss_name != "tr_loss":
            train_step_vars.pop("tr_loss")

        self.mark_step_loss(loss_name)

        return train_step_vars[loss_name]

    def _prepare_inputs(self, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> Dict[str, Union[paddle.Tensor, Any]]:
        """
        trainer.criterion only support criterion(prediction, labels), so we need
        to reorganize the inputs to extract label data into one argument. This is
        only used in non-PipelineParallel model training since loss is included
        in PipelineLayer.
        """
        inputs = super()._prepare_input(inputs)
        if self.criterion is None or getattr(self.criterion, "label_names", None) is None:
            return inputs
        # criterion created by create_loss has `label_names` and `label_default_values`
        label_names = self.criterion.__class__.label_names
        # some data fields are used both in model and loss
        shared_fields = set(["input_ids", "attention_mask"])
        labels = []
        for name in label_names:
            if name not in inputs:
                label = self.criterion.__class__.label_default_values.get(name, None)
            elif name in shared_fields:
                label = inputs[name]
            else:
                label = inputs.pop(name)
            labels.append(label)
        # "labels" is the pre-defined label name in Trainer
        inputs["labels"] = labels
        # NOTE: TensorParallel model requires non-Tensor inputs to be lists and
        # broadcast them, thus do not or optionally use these inputs. labels use
        # in criterion not send to model can workaround this.
        return inputs

    def mark_step_loss(self, loss_name):
        """
        When using a mixed loss we often want to get the separated loss metrics,
        thus we mark loss type of each training step to separate them. This is
        not necessary since the loss would be returnd after each training step.
        However when using PipelienParallel, the loss returned is 0 when not reach
        accumulated step and the loss returned at accumulated step is a mixed loss.
        To separate loss metrics in PipelienParallel:
        1. We hack PipelineParallel._forward_step to record actual loss for each
           step in a list.
        2. We mark the loss type only once for each step using `loss_step_indice`
           (dict), then wen can check out the corresponding loss metrics from the
           loss list.
        We assume a static order of multi-losses and mark the loss indice only once.
        """
        self.loss_step_indice = getattr(self, "loss_step_indice", {})
        if loss_name not in self.loss_step_indice:
            self.loss_step_indice[loss_name] = len(self.loss_step_indice)

    def get_step_loss(self, loss_prefix: str = "") -> Dict:
        """
        Return a dict mapping loss name to value of current training step. This
        is mainly to get loss for metric logging, and it would not affect the
        training. Overwrite it when we want to change the logging value.
        """
        model = self.get_model(train=True)
        if not hasattr(self, "loss_dict"):
            self.loss_dict = {}
            for var_name, value in self.get_train_step_vars().items():
                if var_name.endswith("_loss"):
                    self.loss_dict[var_name] = value
        loss_dict = {}  # return a new dict because of new metric names
        if isinstance(model, fleet.model.PipelineParallel) and len(self.loss_dict) > 1:
            # NOTE: PipelineParallel only returns a accumulated loss after
            # accumulated steps, which is a mixed loss of ppo-loss and
            # ptx-loss. We hack PipelineParallel._forward_step to record
            # loss metrics and postprocess the recorded losses here.
            # Maybe better to make the last_stage worker log to reduce
            # comm and for simplicity.
            with paddle.no_grad():
                if model.is_pipeline_last_stage():
                    # loss is 0D tensor, use stack rather than concat
                    mix_loss = paddle.stack(model._step_losses)
                    model._step_losses = None
                else:
                    # The tessor shape is not policy_model.accumulate_steps
                    # (args.accu_steps) but policy_trainer.args.accu_steps,
                    # since policy_model is created with global pp_config
                    # using global args.accu_steps which is only half of
                    # policy_trainer.args.accu_steps, and indeed trainer hack
                    # model.accumulate_steps in training_pipeline_step to use
                    # trainer.args.accu_steps. The dtype is fp32(to be check),
                    # thus no need to broadcast.
                    mix_loss = paddle.empty(shape=[self.args.gradient_accumulation_steps], dtype=paddle.float32)
                paddle.distributed.broadcast(mix_loss, src=model.pp_group.ranks[-1], group=model.pp_group)
                for loss_name in self.loss_dict:
                    # We assume a static order of multi-losses and mark the loss
                    # indice only once.
                    value = mix_loss[self.loss_step_indice[loss_name] :: len(self.loss_dict)].mean()
                    loss_name = loss_prefix + loss_name if loss_prefix else loss_name
                    loss_dict[loss_name] = value
            return loss_dict

        for loss_name in self.loss_dict:
            value = self.get_train_step_vars()[loss_name]
            loss_name = loss_prefix + loss_name if loss_prefix else loss_name
            loss_dict[loss_name] = value
        return loss_dict


class PolicyTrainer(StepTrainer):
    loss_cls = RLHFPPOMixedLoss

    def loss_identifier(self, inputs: Dict) -> str:
        labels = inputs.get("labels", None)
        if labels is not None:  # use ptx
            loss_name = "ptx_loss"
        else:
            loss_name = "actor_loss"
        return loss_name

    def get_step_loss(self, loss_prefix: str = "") -> Dict:
        loss_dict = super().get_step_loss(loss_prefix=loss_prefix)
        # use_ptx would double the gradient_accumulation_steps which causes
        # actor_loss and ptx_loss reduced by half. Moreover, ptx_loss should
        # be divided by ptx_coeff for logging.
        # TODO(guosheng): maybe should consider self._enable_delay_scale_loss()
        # if "ptx_loss" in loss_dict:
        #     loss_dict[loss_prefix + "ptx_loss"] = loss_dict[
        #         "ptx_loss"] * 2 / self.criterion.ptx_coeff
        #     loss_dict[loss_prefix + "actor_loss"] = loss_dict["actor_loss"] * 2
        return loss_dict


class ValueTrainer(StepTrainer):
    loss_cls = RLHFValueLoss
    # define loss name
    loss_identifier = lambda self, inputs: "reward_critic_loss"


class PPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[PretrainedModel, nn.Layer] = None,
        criterion: nn.Layer = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        ptx_dataset: Optional[Dataset] = None,
        eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
        tokenizer: Optional[PretrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
        preprocess_logits_for_metrics: Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor] = None,
    ):
        with guard_set_args(
            args,
            {
                "recompute": False,
                "fp16_opt_level": "O1",
                "pipeline_parallel_degree": 1,  # workaround for pipeline parallel model check
            },
        ):
            # just used to create trival attrs might be used in the training
            # process of trainer, while changing some args to avoid model usage
            # in __init__ such as recompute and AMP-O2
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

        self.train_dataset = train_dataset
        self.ptx_dataset = ptx_dataset
        self.eval_dataset = eval_dataset

        (policy_model, reference_model, reward_model, value_model) = model
        # policy_tokenizer and value_tokenizer should be same
        (policy_tokenizer, reference_tokenizer, reward_tokenizer, value_tokenizer) = tokenizer

        policy_training_args = copy.deepcopy(args)
        self.use_ptx = self.ptx_dataset is not None
        if self.use_ptx:
            policy_training_args.gradient_accumulation_steps *= 2
        self.policy_trainer = PolicyTrainer(
            policy_model,
            criterion,
            policy_training_args,
            data_collator,
            train_dataset,
            eval_dataset,
            policy_tokenizer,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        value_training_args = copy.deepcopy(args)
        for attr_name in [
            "critic_learning_rate",
            "critic_weight_decay",
            "critic_lr_scheduler_type",
            "critic_warmup_ratio",
            "critic_recompute",
        ]:
            if getattr(value_training_args, attr_name, None) is not None:
                setattr(value_training_args, attr_name[len("critic_") :], getattr(value_training_args, attr_name))
        self.value_trainer = ValueTrainer(
            value_model,
            criterion,
            value_training_args,
            data_collator,
            train_dataset,
            eval_dataset,
            value_tokenizer,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        # disable inner trainers' callback/state/control
        self.policy_trainer.add_callback(MuteDefaultFlowCallback)
        self.value_trainer.add_callback(MuteDefaultFlowCallback)

        # use trainer for reference_model/reward_model to enable sharding stage-3
        # and PipelineParallel. maybe we should allow models to use different dist
        # strategies later
        self.reference_trainer = StepTrainer(
            reference_model,
            criterion,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            reference_tokenizer,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.reward_trainer = StepTrainer(
            reward_model,
            criterion,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            reward_tokenizer,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        # TODO(guosheng): sharding stage3 should create master weight optionally
        # instead of creation and clear.
        self.reference_trainer.init_train_model_opt(100, None, clear_master_weight=True)  # dummy max_steps
        self.reward_trainer.init_train_model_opt(100, None, clear_master_weight=True)  # dummy max_steps
        self.reference_model.eval()
        self.reward_model.eval()

        self.reward_tokenizer = reward_tokenizer
        self.tokenizer = policy_tokenizer
        if is_same_tokenizer(self.tokenizer, self.reward_tokenizer):
            self.reward_tokenizer = self.tokenizer

        self.generation_config = GenerationConfig(
            max_length=self.args.max_length,
            num_return_sequences=self.args.num_return_sequences,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            # top_k=self.args.top_k,
            repetition_penalty=self.args.repetition_penalty,
            do_sample=True,
            # allow generation output to contain input
            trunc_input=False,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Those value can be changed
        self.kl_coeff = self.args.kl_coeff
        self.clip_range_score = self.args.clip_range_score
        self.policy_trainer.ptx_coeff = self.ptx_coeff = self.args.ptx_coeff
        self.gamma = 1.0
        self.gae_lambda = 0.95

        # dummy class and object for model to be compaible with methods of
        # Trainer, such as evaluation_loop
        self.DummyPPOModel = type(
            "DummyPPOModel", (object,), {"eval": lambda _: self.set_eval(), "train": lambda _: self.set_train()}
        )
        self.model = self.model_wrapped = self.DummyPPOModel()

    @property
    def reference_model(self):
        return self.reference_trainer.get_model(train=False)

    @property
    def reward_model(self):
        return self.reward_trainer.get_model(train=False)

    @property
    def actor_model(self):
        return self.policy_trainer.get_model(train=self.training)

    @property
    def reward_critic_model(self):
        return self.value_trainer.get_model(train=self.training)

    def set_train(self, mode: bool = True) -> None:
        """Set training mode for all models."""
        if mode:
            self.training = True
            self.actor_model.train()
            self.reward_critic_model.train()
        else:
            self.training = False
            self.actor_model.eval()
            self.reward_critic_model.eval()

    def set_eval(self) -> None:
        """Set model to evaluation mode."""
        self.set_train(mode=False)

    def prediction_step(
        self,
        model: nn.Layer,
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:
        inputs = self._prepare_inputs(inputs)

        with paddle.no_grad():
            with self.autocast_smart_context_manager():
                seq = self.actor_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    position_ids=inputs["position_ids"]
                    if "position_ids" in inputs
                    else make_position_ids(inputs["attention_mask"]),
                    generation_config=self.generation_config,
                    synced_gpus=ShardingOption.FULL_SHARD in self.policy_trainer.args.sharding,
                )[0]
                attention_mask = paddle.logical_and(
                    seq != self.tokenizer.pad_token_id,
                    seq != self.tokenizer.unk_token_id,
                )
                if self.reward_tokenizer is not self.tokenizer:
                    reward_tokenize_output = batch_retokenize(
                        input_ids=seq,
                        src_tokenizer=self.tokenizer,
                        dest_tokenizer=self.reward_tokenizer,
                        skip_special_tokens=True,
                        device=self.args.device,
                    )
                    reward_input_ids = reward_tokenize_output["input_ids"]
                    reward_attention_mask = reward_tokenize_output["attention_mask"]
                else:
                    reward_input_ids = seq
                    reward_attention_mask = attention_mask

                # unify PP with others since PP always return tuple
                reward_score = self.reward_model(
                    reward_input_ids,
                    attention_mask=reward_attention_mask,
                    # return_dict=True,
                )[
                    1
                ]  # .end_scores
                reward_score = reward_score.squeeze(axis=-1).cast(paddle.float32)

        # keep the first batch of eval output sequence to print and check
        prompt = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        generated = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
        for i, text in enumerate(generated):
            self._eval_out_file.write(text + "\n")
        if getattr(self, "_eval_seq", None) is None:
            generated = [text[len(prompt[i]) :] for i, text in enumerate(generated)]
            # prompts.extend(prompt)
            # generateds.extend(generated)
            self._eval_seq = (prompt, generated, reward_score.tolist())

        return reward_score.mean(), None, None

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_eval_iters: Optional[int] = -1,
    ) -> EvalLoopOutput:
        # to save eval generated sequence
        eval_out_file = os.path.join(
            self.args.output_dir, f"eval_out-step{self.state.global_step}-rank{self.args.local_rank}.txt"
        )
        self._eval_out_file = open(eval_out_file, "w")

        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix, max_eval_iters
        )
        output.metrics[f"{metric_key_prefix}/reward"] = output.metrics.pop(f"{metric_key_prefix}_loss")

        columns = ["Prompt", "Generated", "Reward"]
        rows = list(zip(*self._eval_seq))
        rows = [[str(item) for item in row] for row in rows]
        max_num_rows = 5
        table = Table(title="Evaluating...", show_lines=True, title_justify="left")
        for column in columns:
            table.add_column(column)
        for row in rows[:max_num_rows]:
            table.add_row(*row)
        Console(soft_wrap=True, markup=False, emoji=False).print(table)
        self._eval_seq = None

        self._eval_out_file.close()

        return output

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        with guard_set_args(self, {"data_collator": self.eval_dataset.get_collator()}):
            return super().get_eval_dataloader(eval_dataset)

    def _save_checkpoint(self, model, metrics=None):
        # maybe change args.output_dir of policy_trainer/value_trainer directly
        with guard_set_args(self.policy_trainer.args, {"output_dir": os.path.join(self.args.output_dir, "policy")}):
            self.policy_trainer._save_checkpoint(model, metrics)
        with guard_set_args(self.value_trainer.args, {"output_dir": os.path.join(self.args.output_dir, "value")}):
            self.value_trainer._save_checkpoint(model, metrics)

    def init_train_model_opt(
        self: Trainer, max_steps: int, resume_from_checkpoint: bool = False, clear_master_weight: bool = False
    ) -> PretrainedModel:
        # resume should be triggered here
        # maybe change args.output_dir of policy_trainer/value_trainer directly
        with guard_set_args(self.policy_trainer.args, {"output_dir": os.path.join(self.args.output_dir, "policy")}):
            policy_model = self.policy_trainer.init_train_model_opt(
                max_steps,
                os.path.join(resume_from_checkpoint, "policy")
                if isinstance(resume_from_checkpoint, str)
                else resume_from_checkpoint,
            )
        with guard_set_args(self.value_trainer.args, {"output_dir": os.path.join(self.args.output_dir, "value")}):
            value_model = self.value_trainer.init_train_model_opt(
                max_steps,
                os.path.join(resume_from_checkpoint, "value")
                if isinstance(resume_from_checkpoint, str)
                else resume_from_checkpoint,
            )
        return policy_model, value_model

    def get_epoch_iterator(self):
        # TODO(guosheng): support iter dataset
        num_prompt_only_batches = len(self.prompt_only_dataloader)
        num_ptx_batches = len(self.ptx_dataloader)
        num_ptx_replicas = (num_prompt_only_batches + num_ptx_batches - 1) // num_ptx_batches

        def gen_epoch_data():
            for prompt_only_batch, ptx_batch in zip(
                self.prompt_only_dataloader,
                itertools.chain.from_iterable([self.ptx_dataloader] * num_ptx_replicas),
            ):
                # generate batches
                self.set_eval()
                rl_batches = self.split_rl_micro_batches(prompt_only_batch)
                if self.use_ptx:
                    ptx_batches = self.split_ptx_micro_batches(ptx_batch)
                else:
                    ptx_batches = [None for _ in range(len(rl_batches))]
                paddle.device.cuda.empty_cache()

                self.set_train()
                for _ in range(self.args.update_iters):
                    for rl_batch, ptx_batch in zip(rl_batches, ptx_batches):
                        yield rl_batch, ptx_batch

        class EpochIterator:
            def __iter__(self):
                return gen_epoch_data()

        return EpochIterator()

    def init_train_num(self: Trainer, train_dataloader: DataLoader):
        args = self.args

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.dataset_world_size

        len_dataloader = len(train_dataloader)
        num_train_sub_steps = (
            len_dataloader
            * self.args.update_iters
            * self.args.per_device_prompt_batch_size
            * self.args.num_return_sequences
            // self.args.per_device_train_batch_size
        )
        num_update_steps_per_epoch = num_train_sub_steps // args.gradient_accumulation_steps
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            max_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
            num_train_epochs = math.ceil(args.num_train_epochs)
        num_examples = num_train_samples = total_train_batch_size * max_steps

        return (
            total_train_batch_size,
            len_dataloader,
            max_steps,
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
        )

    def is_step_end(self):
        # reach accumulation_steps, value trainer has the same step_control and
        # gradient_accumulation_steps as PPO trainer.
        # if (step_control + 1) % args.gradient_accumulation_steps == 0
        return self.value_trainer.get_train_step_vars()["step_control"] == 0

    def get_step_loss(self, loss_prefix: str = "") -> Dict:
        rl_loss = self.policy_trainer.get_step_loss(loss_prefix)
        value_loss = self.value_trainer.get_step_loss(loss_prefix)
        rl_loss.update(value_loss)
        return rl_loss

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
    ) -> None:
        # ##### The following code try to keep same as the Trainer.train #####
        args = self.args
        self.is_in_train = True

        # ##### trainging data and related num setting #####
        # TODO(guosheng): remove the binding method get_collator of dataset
        with guard_set_args(
            args, {"per_device_train_batch_size": self.args.per_device_prompt_batch_size}
        ), guard_set_args(
            self, {"train_dataset": self.train_dataset, "data_collator": self.train_dataset.get_collator()}
        ):
            train_dataloader = self.prompt_only_dataloader = self.get_train_dataloader()

        if self.use_ptx:
            with guard_set_args(
                args,
                {
                    "per_device_train_batch_size": self.args.per_device_prompt_batch_size
                    * self.args.num_return_sequences
                },
            ), guard_set_args(
                self, {"train_dataset": self.ptx_dataset, "data_collator": self.ptx_dataset.get_collator(shift=True)}
            ):
                self.ptx_dataloader = self.get_train_dataloader()
        else:
            self.ptx_dataloader = DataLoader(DummyDataset(len(self.prompt_only_dataloader)))
        (
            total_train_batch_size,
            len_dataloader,
            max_steps,
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
        ) = self.init_train_num(train_dataloader)

        # ##### model and optimizer related setting #####
        policy_model, value_model = self.init_train_model_opt(max_steps, resume_from_checkpoint)
        paddle.device.cuda.empty_cache()

        # ##### traing statistic logging #####
        # Number of trainable parameters only account for policy_model
        self.init_train_log(
            num_examples, num_train_epochs, total_train_batch_size, max_steps, num_train_samples, policy_model
        )

        # ##### set training state and resume #####
        # consumed_samples used to set train_dataloader.batch_sampler may not be
        # correct. Thus, data cannot be resumed perfectly when not breaking at epoch end.
        epochs_trained, steps_trained_in_current_epoch, steps_trained_progress_bar = self.init_train_state(
            resume_from_checkpoint, train_dataloader, max_steps, num_train_epochs, num_update_steps_per_epoch
        )

        epoch_iterator = self.get_epoch_iterator()
        steps_in_epoch = num_update_steps_per_epoch * args.gradient_accumulation_steps

        # self.callback_handler.model = self.model
        # self.callback_handler.optimizer = self.optimizer
        # self.callback_handler.lr_scheduler = self.lr_scheduler
        # self.callback_handler.train_dataloader = train_dataloader
        self.state.max_steps = int(max_steps)
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        self._globalstep_last_logged = self.state.global_step

        start_time = time.time()
        self._globalstep_last_start_time = start_time
        # self.timers and self.timers("read-data").start()

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                train_dataloader.batch_sampler, DistributedBatchSampler
            ):
                train_dataloader.batch_sampler.set_epoch(epoch)

            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):
                # self.timers and self.timers("read-data").stop()
                # os.environ["TRAINER_GLOBAL_STEP"] = str(self.state.global_step)
                # self.callback_handler.on_load_data_end(args, self.state, self.control, inputs=inputs)
                rl_batch, ptx_batch = inputs
                # TODO(guosheng): make rl_step/ptx_step run with autocast_smart_context_manager
                rl_info = self.rl_step(rl_batch)
                paddle.device.cuda.empty_cache()
                if self.use_ptx:
                    ptx_info = self.ptx_step(ptx_batch)
                    rl_info.update(ptx_info)
                    paddle.device.cuda.empty_cache()

                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch
                if self.is_step_end():
                    rl_info.update(self.get_step_loss(loss_prefix="train/"))
                    # on_step_end
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                else:
                    # on_sub_step_end
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(rl_info, None, epoch, ignore_keys_for_eval, inputs=inputs)

            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({self.state.max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            # argument model is not used in _maybe_log_save_evaluate, thus use None
            self._maybe_log_save_evaluate(rl_info, None, epoch, ignore_keys_for_eval, inputs=inputs)

            if self.control.should_training_stop:
                break
        # TODO(guosheng): add epilogue of training

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs):
        if self.control.should_log:

            logs: Dict[str, float] = {}

            for k, v in tr_loss.items():
                if isinstance(v, paddle.Tensor) and "lr" not in k and "max_generated_length" not in k:
                    v_scalar = self._nested_gather(v).mean().item()
                    # TODO(guosheng): maybe should consider self._enable_delay_scale_loss()
                    # and maybe should merge with loss postprocess in PP
                    if "train/actor_loss" == k and "train/ptx_loss" in tr_loss:
                        # use_ptx would double the gradient_accumulation_steps
                        # which causes actor_loss and ptx_loss reduced by half
                        v_scalar = v_scalar * 2
                    elif "train/ptx_loss" == k:
                        # similar to actor_loss and should double, additionally
                        # it should be divided by ptx_coeff for logging
                        v_scalar = v_scalar * 2 / self.ptx_coeff
                    logs[k] = round(v_scalar / (self.state.global_step - self._globalstep_last_logged), 8)
                    v.subtract_(v)
                    attr_name = "_total_" + k.split("/")[-1] + "_scalar"
                    attr_value = getattr(self, attr_name, 0)
                    setattr(self, attr_name, attr_value + v_scalar)
                elif "max_generated_length" in k:
                    v_scalar = self._nested_gather(v).max().item()
                    logs[k] = v_scalar
                else:
                    logs[k] = float("{0:.3e}".format(v))
            logs["global_step"] = int(self.state.global_step)

            total_train_batch_size = (
                self.args.train_batch_size * self.args.gradient_accumulation_steps * self.args.dataset_world_size
            )
            num_steps = self.state.global_step - self._globalstep_last_logged
            logs.update(
                speed_metrics(
                    "interval",
                    self._globalstep_last_start_time,
                    num_samples=total_train_batch_size * num_steps,
                    num_steps=num_steps,
                )
            )

            self._globalstep_last_logged = self.state.global_step
            self._globalstep_last_start_time = time.time()

            self.log(logs, **kwargs)

        # To trigger evaluation and save but avoid log again
        with guard_set_args(self.control, {"should_log": False}):
            super()._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval)

    def add_kl_divergence_regularization(
        self,
        prompt: paddle.Tensor,  # size = (B, S) # pylint: disable=unused-argument
        log_probs: paddle.Tensor,  # size = (B, L)
        ref_log_probs: paddle.Tensor,  # size = (B, L)
        reward_score: paddle.Tensor,  # size = (B,)
        sequence_mask: paddle.Tensor,  # size = (B, L)
    ) -> paddle.Tensor:
        kl_divergence_estimate = -self.kl_coeff * (log_probs - ref_log_probs)  # size = (B, L)
        rewards = kl_divergence_estimate  # size = (B, L)
        reward_clip = paddle.clip(  # size = (B,)
            reward_score,
            min=-self.clip_range_score,
            max=self.clip_range_score,
        )
        batch_size = log_probs.shape[0]
        for i in range(batch_size):
            end_index = sequence_mask[i].nonzero()[-1]
            # rewards[i, end_index] += reward_clip[i]
            rewards[i, end_index] = rewards[i, end_index] + reward_clip[i]

        return rewards

    def get_advantages_and_returns(
        self,
        values: paddle.Tensor,
        rewards: paddle.Tensor,
        sequence_mask: paddle.Tensor,
        start: int,
        use_tgt_len_return: bool = True,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Compute advantages and returns using Generalized Advantage Estimation (GAE)."""
        # Modified from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
        last_gae_lambda = 0.0
        advantages_reversed = []
        values = values * sequence_mask
        rewards = rewards * sequence_mask
        length = rewards.shape[-1]
        if use_tgt_len_return and start > 0:
            # consistent with Beaver
            # values length is src+tgt-1, start is src-1, return length is tgt
            pass
        elif use_tgt_len_return:
            # values length is tgt, start is 0, return length is tgt
            assert start == 0
        else:
            # values length is src+tgt-1, start is src-1, return length is src+tgt-1
            pass
        for t in reversed(range(start, length)):  # pylint: disable=invalid-name
            next_values = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * next_values - values[:, t]
            last_gae_lambda = delta + self.gamma * self.gae_lambda * last_gae_lambda
            advantages_reversed.append(last_gae_lambda)
        advantages = paddle.stack(advantages_reversed[::-1], axis=1)
        returns = advantages + values[:, start:]
        if not use_tgt_len_return:
            advantages = paddle.concat(
                [paddle.zeros([advantages.shape[0], start], dtype=advantages.dtype), advantages], -1
            )
            returns = paddle.concat([paddle.zeros([returns.shape[0], start], dtype=returns.dtype), returns], -1)
        return advantages.detach(), returns

    def rl_step(self, rl_batch: Dict[str, paddle.Tensor]) -> Dict[str, Any]:
        # inputs shared by policy and value trainer
        input_ids = rl_batch["input_ids"]  # length: src+tgt
        attention_mask = rl_batch["attention_mask"]  # length: src+tgt
        position_ids = rl_batch["position_ids"]  # length: src+tgt
        sequence_mask = rl_batch["sequence_mask"]  # length: src+tgt(-1)
        # inputs used by policy trainer
        old_log_probs = rl_batch["log_probs"]  # length: src+tgt(-1)
        reward_advantages = rl_batch["reward_advantages"]  # length: src+tgt(-1)
        # inputs used by value trainer
        old_reward_values = rl_batch["reward_values"]  # length: src+tgt(-1)
        reward_returns = rl_batch["reward_returns"]  # length: src+tgt(-1)

        policy_trainer_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "old_log_probs": old_log_probs,
            "reward_advantages": reward_advantages,
            "sequence_mask": sequence_mask,
        }
        actor_loss = self.policy_trainer.full_training_step(**policy_trainer_inputs)

        value_trainer_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "old_reward_values": old_reward_values,
            "reward_returns": reward_returns,
            "sequence_mask": sequence_mask,
        }
        reward_critic_loss = self.value_trainer.full_training_step(**value_trainer_inputs)

        # metric
        rewards = rl_batch["rewards"]
        rewards = rewards.mean()
        ref_log_probs = rl_batch["ref_log_probs"]
        kl_divergence = ((old_log_probs - ref_log_probs) * sequence_mask).sum(axis=-1).mean()
        mean_generated_length = sequence_mask.cast(paddle.float32).sum(axis=-1).mean()
        max_generated_length = sequence_mask.cast(paddle.float32).sum(axis=-1).max()

        return {
            # when using PipelienParallel, the loss returned is 0 when not reach
            # accumulated step and the loss returned at accumulated step is a
            # mixed loss.
            "train/actor_loss": actor_loss,
            "train/reward_critic_loss": reward_critic_loss,
            "train/reward": rewards,
            "train/kl_divergence": kl_divergence,
            "train/mean_generated_length": mean_generated_length,
            "train/max_generated_length": max_generated_length,
            "train/actor_lr": self.policy_trainer._get_learning_rate(),
            "train/reward_critic_lr": self.value_trainer._get_learning_rate(),
        }

    def ptx_step(self, ptx_batch: Dict[str, paddle.Tensor]) -> Dict[str, Any]:
        """Perform a single update step with PTX loss."""
        ptx_loss = self.policy_trainer.full_training_step(**ptx_batch)
        return {
            "train/ptx_loss": ptx_loss,
        }

    def split_ptx_micro_batches(
        self,
        ptx_batch: Dict[str, paddle.Tensor],
    ) -> List[Dict[str, paddle.Tensor]]:
        """Split a batch of PTX samples into micro-batches."""
        micro_batches = []
        total_batch_size = ptx_batch["input_ids"].shape[0]
        micro_batch_size = self.args.per_device_train_batch_size
        for i in range(0, total_batch_size, micro_batch_size):
            micro_batch = map_structure(
                # pylint: disable-next=cell-var-from-loop
                lambda tensor: tensor[i : i + micro_batch_size],  # noqa: B023
                ptx_batch,
            )
            micro_batches.append(micro_batch)
        return micro_batches

    def split_rl_micro_batches(
        self,
        prompt_only_batch: PromptOnlyBatch,
    ) -> List[PromptOnlyBatch]:
        """Split a batch of RL samples into micro-batches."""
        total_batch_size = prompt_only_batch["input_ids"].shape[0]
        micro_batch_size = self.args.per_device_train_batch_size
        micro_batches = []
        for i in range(0, total_batch_size, micro_batch_size):
            micro_batch = {}
            micro_batch = map_structure(
                lambda tensor: tensor[i : i + micro_batch_size],
                prompt_only_batch,
            )
            micro_batches.extend(self.rollout(micro_batch))
        return micro_batches

    @paddle.no_grad()
    def rollout(self, prompt_only_batch: PromptOnlyBatch) -> List[Dict[str, Any]]:
        """Rollout a batch of experiences."""
        input_ids = prompt_only_batch["input_ids"]
        attention_mask = prompt_only_batch["attention_mask"]
        position_ids = (
            prompt_only_batch["position_ids"]
            if "position_ids" in prompt_only_batch
            else make_position_ids(attention_mask)
        )
        with infer_guard(self.policy_trainer):
            sequences = self.actor_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                generation_config=self.generation_config,
                synced_gpus=ShardingOption.FULL_SHARD in self.policy_trainer.args.sharding,
            )[0]
        sequences = sequences.reshape([input_ids.shape[0], self.args.num_return_sequences, -1]).transpose([1, 0, 2])

        return [
            # TODO(guosheng): move post_rollout out to split_rl_micro_batches
            # to allow infer model generate multi times consecutively and then
            # convert weights, otherwise we have to convert weights multi times
            # when need multi batch rollout data.
            self.post_rollout(
                input_ids,
                seq,
                attention_mask=paddle.logical_and(
                    seq != self.tokenizer.pad_token_id,
                    seq != self.tokenizer.unk_token_id,
                ),
            )
            for seq in sequences
        ]

    @paddle.no_grad()
    def post_rollout(
        self,
        prompt: paddle.Tensor,
        sequence: paddle.Tensor,
        attention_mask: paddle.Tensor,
    ) -> Dict[str, Any]:
        if self.reward_tokenizer is not self.tokenizer:
            # right padding
            reward_tokenize_output = batch_retokenize(
                sequence,
                src_tokenizer=self.tokenizer,
                dest_tokenizer=self.reward_tokenizer,
                skip_special_tokens=True,
            )
            reward_seq = reward_tokenize_output["input_ids"]
            reward_attention_mask = reward_tokenize_output["attention_mask"]
        else:
            # for text in self.tokenizer.batch_decode(sequence, skip_special_tokens=True):
            #     print(text)
            reward_seq = sequence
            reward_attention_mask = attention_mask
        # position_ids is necessary for non-right padding
        # If using right padding source + left padding target, make padding positions
        # in source be 0, since reward model use position_ids plus with padding size
        # (number of 0s) in source to calculate end offsets.
        position_ids = make_position_ids(attention_mask)

        # pipe model outputs a logits tensor with LMHead, while non-pipe model
        # outputs a tuple with logits tensor as the only one element.
        logits = self.actor_model(
            sequence,
            attention_mask=attention_mask,
            position_ids=position_ids,
            # return_dict=True,
        )  # .logits
        if not isinstance(logits, paddle.Tensor):
            logits = logits[0]
        ref_logits = self.reference_model(
            sequence,
            attention_mask=attention_mask,
            position_ids=position_ids,
            # return_dict=True,
        )  # .logits
        if not isinstance(ref_logits, paddle.Tensor):
            ref_logits = ref_logits[0]

        reward_score = self.reward_model(
            reward_seq,
            attention_mask=reward_attention_mask,
            position_ids=position_ids,
            # return_dict=True,
        )[
            1
        ]  # .end_scores
        reward_value = self.reward_critic_model(
            sequence,
            attention_mask=attention_mask,
            position_ids=position_ids,
            # return_dict=True,
        )[
            0
        ]  # .scores
        reward_score = reward_score.squeeze(axis=-1)
        reward_value = reward_value.squeeze(axis=-1)
        reward_value = reward_value[:, :-1]
        log_probs = gather_log_probabilities(logits[:, :-1], sequence[:, 1:])
        ref_log_probs = gather_log_probabilities(ref_logits[:, :-1], sequence[:, 1:])
        rollout_data = {
            "prompt": prompt,
            "input_ids": sequence,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "rewards": reward_score,
            "reward_values": reward_value,
            "log_probs": log_probs,
            "ref_log_probs": ref_log_probs,
        }
        rollout_data = self.normalize_data(rollout_data, use_tgt_len_value=False)
        return rollout_data

    @paddle.no_grad()
    def normalize_data(
        self,
        rl_batch: Dict[str, paddle.Tensor],
        use_tgt_len_value: bool = False,
    ) -> Dict[str, Any]:
        """
        data dispatch comm among devices needs padding, while the lengths of
        all data fields are different and related, and it's hard to pad.
        """
        prompt = rl_batch["prompt"]  # length: src
        attention_mask = rl_batch["attention_mask"]  # length: src + tgt
        old_log_probs = rl_batch["log_probs"]  # length: src + tgt -1
        ref_log_probs = rl_batch["ref_log_probs"]  # length: src + tgt -1
        rewards = rl_batch["rewards"]  # length: 1
        old_reward_values = rl_batch["reward_values"]  # length: src + tgt -1

        # Beaver uses label data with target length, while we do not slice from
        # inputs and use label data with target length:
        # 1. Sometimes we cannot use label data with target length, mostly because
        # it is hard to pad acorss batches. Think in some cases one batch might
        # have the longest prompt+target length but the shortest target lengh, which
        # might cause mismatch between inputs with prompt+target length and labels
        # with target length. Padding acorss batches is needed in PP and data comm.
        # 2. Additionally, when using flash_attn with casual mask and right padding
        # we cannot use label data with target length.
        start = prompt.shape[-1] - 1
        # sequence_mask is for label masking, make source be masked out
        # clone to avoid to change attention_mask
        sequence_mask = attention_mask[:, 1:].clone()  # length: src + tgt -1
        sequence_mask[:, :start] = False
        if use_tgt_len_value:
            ref_log_probs = ref_log_probs[:, start:]
            old_log_probs = old_log_probs[:, start:]
            old_reward_values = old_reward_values[:, start:]
            sequence_mask = sequence_mask[:, start:]
        old_rewards = self.add_kl_divergence_regularization(
            None,  # prompt,
            old_log_probs,
            ref_log_probs,
            rewards,
            sequence_mask,
        )  # length: tgt if use_tgt_len_value src + tgt -1
        reward_advantages, reward_returns = self.get_advantages_and_returns(
            old_reward_values,
            old_rewards,
            sequence_mask,
            start=0 if use_tgt_len_value else start,
            use_tgt_len_return=use_tgt_len_value,
        )  # length: tgt if use_tgt_len_value src + tgt -1

        rl_batch.update(
            {
                "log_probs": old_log_probs,
                "reward_values": old_reward_values,
                "reward_advantages": reward_advantages,
                "reward_returns": reward_returns,
                "sequence_mask": sequence_mask,
                "ref_log_probs": ref_log_probs,
                "rewards": rewards,
            }
        )
        # pop out to reduce data dispatch comm overhead
        rl_batch.pop("prompt")
        return rl_batch
