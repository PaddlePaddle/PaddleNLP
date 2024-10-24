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

from __future__ import annotations

import inspect
import os
import time
from contextlib import contextmanager
from typing import Dict

import numpy as np
import paddle
import tqdm
from paddle.distributed import fleet
from paddle.io import DataLoader

from paddlenlp.generation.utils import GenerationMixin
from paddlenlp.trainer.trainer import (
    TRAINER_STATE_NAME,
    HybridParallelOptimizer,
    NlpDistributedBatchSampler,
    ShardingOption,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    _obtain_optimizer_parameters_list,
    distributed_file,
    distributed_isfile,
    fused_allreduce_gradients,
    logger,
    reshard_util,
    split_inputs_sequence_dim,
)
from paddlenlp.transformers import BatchEncoding, PretrainedModel, PretrainedTokenizer
from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.transformers.model_outputs import ModelOutput
from paddlenlp.transformers.tokenizer_utils_base import (
    PaddingStrategy,
    TruncationStrategy,
)


# ########## patches for Trianer ##########
def init_train_model_opt(
    self: Trainer, max_steps: int, resume_from_checkpoint: bool = False, clear_master_weight: bool = False
) -> PretrainedModel:
    # Copy of model/optimizer init and resuming related code in `Trainer.train`.
    # NOTE: this `_load_from_checkpoint` is indeed to load model states in the
    # following elif-else branches, though they are apart away in `Trainer.train`.
    if not self.args.should_load_sharding_stage1_model:
        self._load_from_checkpoint(resume_from_checkpoint)

    # delay_optimizer_creation = (
    #     self.sharding is not None
    #     and ShardingOption.SHARD_OP in self.args.sharding
    # )
    delay_optimizer_creation = False

    if not delay_optimizer_creation:
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    if self.args.should_load_sharding_stage1_model:
        model = self._wrap_model_and_load_sharded_checkpoint(resume_from_checkpoint)
    elif self.args.should_save_sharding_stage1_model:
        # In the non-sharded mode, should invoke _load_from_checkpoint before _wrap_model.
        # In this mode, the rank0 load all params and the _wrap_model implicitly broadcast params from rank0 to the other ranks.
        model = self._wrap_model(self.model_wrapped)
        if self.sharding_io is not None:
            assert delay_optimizer_creation is False, "delay_optimizer_creation should be False"
            # the self.optimizer should be wrapped and it is done in _wrap_model
            self.sharding_io.set_optimizer(self.optimizer)
        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model
        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
    else:
        model = self._wrap_model(self.model_wrapped)
        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model
        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

    if ShardingOption.FULL_SHARD in self.args.sharding and clear_master_weight:
        # for inference model to use Trainer sharding stage3, clear master_weight
        # which is created in GroupShardedStage3.__init__
        self.optimizer._master_weights = None

    if self.args.device == "npu" and self.args.flatten_param_grads:
        from .plugins.npu_plugin import npu_accelerate_plugin

        npu_accelerate_plugin(self.optimizer)

    return model


def init_train_state(
    self: Trainer,
    resume_from_checkpoint: bool,
    train_dataloader: DataLoader,
    max_steps: int,
    num_train_epochs: int,
    num_update_steps_per_epoch: int,
):
    args = self.args

    self.state = TrainerState()
    self.state.epoch = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    steps_trained_progress_bar = None

    # Check if continuing training from a checkpoint
    if resume_from_checkpoint is not None and distributed_isfile(
        os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    ):
        self.state = TrainerState.load_from_json(
            distributed_file(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
        )
        epochs_trained = self.state.global_step // num_update_steps_per_epoch
        if not args.ignore_data_skip:
            steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
            steps_trained_in_current_epoch *= args.gradient_accumulation_steps
        else:
            steps_trained_in_current_epoch = 0

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {self.state.global_step}")
        if not args.ignore_data_skip:
            logger.info(
                f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                "flag to your launch command, but you will resume the training on data already seen by your model."
            )
            if self.is_local_process_zero() and not args.disable_tqdm:
                steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                steps_trained_progress_bar.set_description("Skipping the first batches")
        if not args.ignore_data_skip:
            if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                train_dataloader.batch_sampler, NlpDistributedBatchSampler
            ):
                consumed_samples = (
                    self.state.global_step
                    * args.train_batch_size
                    * args.gradient_accumulation_steps
                    * args.dataset_world_size
                )
                train_dataloader.batch_sampler.set_epoch(consumed_samples=consumed_samples)
                logger.info(f"Set DistributedBatchSampler consumed_samples to {consumed_samples}")

    self.state.max_steps = int(max_steps)
    self.state.num_train_epochs = num_train_epochs
    self.state.is_local_process_zero = self.is_local_process_zero()
    self.state.is_world_process_zero = self.is_world_process_zero()

    return epochs_trained, steps_trained_in_current_epoch, steps_trained_progress_bar


def init_train_log(
    self: Trainer,
    num_examples: int,
    num_train_epochs: int,
    total_train_batch_size: int,
    max_steps: int,
    num_train_samples: int,
    model: PretrainedModel,
):
    args = self.args

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples:,}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_steps:,}")
    logger.info(f"  Total num train samples = {num_train_samples:,}")
    # per_device_trainable_numel = sum(p.numel().item() for p in model.parameters() if not p.stop_gradient)
    # TODO: Temporary fix since Tensor.numel() not supported in distributed mode
    per_device_trainable_numel = sum(np.prod(p.shape) for p in model.parameters() if not p.stop_gradient)
    logger.info(f"  Number of trainable parameters = {per_device_trainable_numel:,} (per device)")
    if self.args.use_hybrid_parallel:
        # todo fix for pipeline_parallel_degree
        parts_num = max(self.args.tensor_parallel_degree, 1) * max(self.args.pipeline_parallel_degree, 1)
        if parts_num > 1:
            all_reduce_dtype = "int64"
            if paddle.get_device().split(":")[0] in ["npu", "xpu"]:
                # TODO(duanyanhui): fix when NPU all_reduce supports int64
                all_reduce_dtype = "float32"
            trainable_numel_tensor = paddle.to_tensor(per_device_trainable_numel, dtype=all_reduce_dtype)
            paddle.distributed.all_reduce(trainable_numel_tensor)
            trainable_numel = int(trainable_numel_tensor.item()) // self.args.dataset_world_size
            # the numel is roughly, because the tensor parallel still hold own bias or layer_norm weight without splited
            # so, the trainable numel is a little bigger than real.
            logger.info(f"  Number of trainable parameters = {trainable_numel:,} (all devices, roughly)")


def full_training_step(self: Trainer, inputs: Dict[str, paddle.Tensor], **kwargs):
    """
    Just a copy of single training step complete code in Trainer.train while loop
    which including forward+backward+step, while wraps the inputs and outputs to
    make the complicated copied code no need to change. Maybe a better way is to
    add fine-grained methods including these steps to Trainer which is similar to
    DeepSpeed engine.
    """

    # TODO(guosheng): step, steps_trained_in_current_epoch and steps_trained_progress_bar
    # should use reference since they would be overwrite.
    # for state update
    epoch = kwargs.get("epoch", 0)
    step = kwargs.get("step", 0)
    steps_in_epoch = kwargs.get("steps_in_epoch", 0)
    step_control = kwargs.get("step_control", 0)
    # for step and progress update when resuming data
    train_dataloader = kwargs.get("train_dataloader", None)
    resume_from_checkpoint = kwargs.get("resume_from_checkpoint", None)
    steps_trained_in_current_epoch = kwargs.get("steps_trained_in_current_epoch", 0)
    steps_trained_progress_bar = kwargs.get("steps_trained_progress_bar", None)
    # for eval output ignore to gather
    ignore_keys_for_eval = kwargs.get("ignore_keys_for_eval", None)
    tr_loss = kwargs.get("tr_loss", 0.0)
    model = kwargs.get("model", self.model_wrapped)
    # needed in _maybe_log_save_evaluate
    self._globalstep_last_logged = getattr(self, "_globalstep_last_logged", 0)
    self._globalstep_last_start_time = getattr(self, "_globalstep_last_start_time", time.time())

    args = self.args

    if self.args.use_hybrid_parallel and self.args.sep_parallel_degree > 1:
        inputs = split_inputs_sequence_dim(inputs)
    # self.timers and self.timers("read-data").stop()
    os.environ["TRAINER_GLOBAL_STEP"] = str(self.state.global_step)
    self.callback_handler.on_load_data_end(args, self.state, self.control, inputs=inputs)

    # Skip past any already trained steps if resuming training
    # for paddlenlp.utils.batch_sampler.DistributedBatchSampler
    # We use consumed_samples to reset the status
    if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
        train_dataloader.batch_sampler, NlpDistributedBatchSampler
    ):
        if step == 0:
            if steps_trained_progress_bar is not None:
                steps_trained_progress_bar.update(steps_trained_in_current_epoch)
                steps_trained_progress_bar.close()
                steps_trained_progress_bar = None
            self._load_rng_state(resume_from_checkpoint)
        step += steps_trained_in_current_epoch
    elif steps_trained_in_current_epoch > 0:
        steps_trained_in_current_epoch -= 1
        if steps_trained_progress_bar is not None:
            steps_trained_progress_bar.update(1)
        if steps_trained_in_current_epoch == 0:
            self._load_rng_state(resume_from_checkpoint)
        # continue
        final_local_vars = locals()
        for k in kwargs.keys():
            if k in final_local_vars:
                kwargs[k] = final_local_vars[k]
        return kwargs
    elif steps_trained_progress_bar is not None:
        steps_trained_progress_bar.close()
        steps_trained_progress_bar = None

    if step_control % args.gradient_accumulation_steps == 0:
        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
        self.timers and self.timers("forward-backward").start()

    dp_enabled = self.args.data_parallel_degree > 1 if self.args.use_hybrid_parallel else args.local_rank != -1
    forbidden_no_sync = False
    # stage2 and stage3 should not no_sync, because the is no DDP wrapper and no_sync API
    # hybrid_parallel (tp or pp or sharding stage 1) should not no_sync
    if self.args.use_hybrid_parallel:
        forbidden_no_sync = True

    availiable_no_sync = dp_enabled and not forbidden_no_sync

    is_no_sync = (
        ((step_control + 1) % args.gradient_accumulation_steps != 0)
        and availiable_no_sync
        and args._no_sync_in_gradient_accumulation
    ) or (args.recompute and availiable_no_sync)
    # sharding
    # stage1. the same as ddp
    # stage2. manualy collect gradient on dp group

    dp_master_grad = self.args.world_size > 1 and self.args.amp_master_grad and not self.args.use_hybrid_parallel
    if dp_master_grad:
        is_no_sync = True

    if is_no_sync:
        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
        with model.no_sync():
            tr_loss_step = self.training_step(model, inputs)
    else:
        tr_loss_step = self.training_step(model, inputs)

    tr_loss += tr_loss_step

    if (step_control + 1) % args.gradient_accumulation_steps == 0 or (
        # last step in epoch but step is always smaller than gradient_accumulation_steps
        steps_in_epoch <= args.gradient_accumulation_steps
        and (step + 1) == steps_in_epoch
    ):
        if self.args.pipeline_parallel_degree <= 1 and self._enable_delay_scale_loss():
            tr_loss /= self.args.gradient_accumulation_steps

        self.timers and self.timers("forward-backward").stop()
        # Maunally collect gradients
        # Case 1: Use recompute and dp
        # Case 2: Hack dp with master_grad
        # Case 3: Pipeline or sharding overlap
        # local_rank != -1 don't means dp in networks.
        self.timers and self.timers("all-reduce").start()

        # Case 1: Use recompute and dp / sharding stage1,
        # manualy collect gradient for dp.
        if args.recompute and availiable_no_sync:
            fused_allreduce_gradients(list(model.parameters()), None)

        # Case 2: hack dp with master_grad
        if dp_master_grad and not (args.recompute and availiable_no_sync):
            fused_allreduce_gradients(list(model.parameters()), None)

        # Pipeline parallel mode,  handle gradient reduce here to overlap
        enable_dp_comm_overlap = False
        enable_release_grads = False
        if args.pipeline_parallel_degree > 1:
            enable_dp_comm_overlap = "enable_dp_comm_overlap" in args.pipeline_parallel_config
            enable_release_grads = "enable_release_grads" in args.pipeline_parallel_config

        # Case 3: Pipeline parallel mode, overlap with dp
        if isinstance(self.optimizer, HybridParallelOptimizer) and not self.do_grad_scaling:
            parameters_list = _obtain_optimizer_parameters_list(self.optimizer._inner_opt)

            if not enable_dp_comm_overlap:
                if self.optimizer._sharding_enable:
                    assert reshard_util.is_sharding_opt(self.optimizer)
                    self.optimizer._inner_opt.reduce_gradients(list(parameters_list), self.optimizer._hcg)

                if self.optimizer._dp_enable or getattr(self.optimizer, "_sep_enable", False):
                    fused_allreduce_gradients(list(parameters_list), self.optimizer._hcg)

        self.timers and self.timers("all-reduce").stop()
        self.timers and self.timers("optimizer-step").start()

        if self.args.gradient_accumulation_steps > 1 and self._enable_delay_scale_loss():
            for p in model._layers.parameters():
                with paddle.no_grad():
                    if hasattr(p, "main_grad") and p.main_grad is not None:
                        assert p.grad is None
                        p.main_grad.scale_(1.0 / self.args.gradient_accumulation_steps)
                    elif p.grad is not None:
                        p.grad.scale_(1.0 / self.args.gradient_accumulation_steps)

        # Optimizer step
        self.callback_handler.on_optimizer_begin(
            args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
        )
        optimizer_was_run = True
        if self.do_grad_scaling:
            scale_before = paddle.assign(self.scaler._scale)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler._scale
            # Compatible with paddlepaddle 2.6.0 using typo word.
            if hasattr(self.scaler, "_cache_founf_inf"):
                optimizer_was_run = not self.scaler._cache_founf_inf
            else:
                optimizer_was_run = not self.scaler._cache_found_inf
            if not optimizer_was_run:
                scale_before_value = scale_before.cpu().numpy()
                scale_after_value = scale_after.cpu().numpy()
                logger.warning(
                    f"optimizer not run, scale_before: {scale_before_value[0]}, scale_after: {scale_after_value[0]}"
                )
        elif isinstance(self.optimizer, HybridParallelOptimizer):
            self.optimizer._step(parameters_list)
        else:
            self.optimizer.step()

        self.timers and self.timers("optimizer-step").stop()

        if optimizer_was_run:
            self.lr_scheduler.step()

        if enable_release_grads and args.pipeline_parallel_degree > 1:
            self.optimizer.clear_grad(set_to_zero=False)
            for _, buffers in model._chunk_2_comm_buffers.items():
                for buffer in buffers:
                    buffer._clear_grad_storage()
        else:
            self.optimizer.clear_grad(set_to_zero=False)

        self.callback_handler.on_optimizer_end(
            args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
        )

        self.state.global_step += 1
        self.state.epoch = epoch + (step + 1) / steps_in_epoch
        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
        self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)
        self._print_timer()
        step_control = 0
    else:
        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
        step_control += 1

    if self.control.should_epoch_stop or self.control.should_training_stop:
        # break
        final_local_vars = locals()
        for k in kwargs.keys():
            if k in final_local_vars:
                kwargs[k] = final_local_vars[k]
        return kwargs
    # self.timers and self.timers("read-data").start()

    final_local_vars = locals()
    for k in kwargs.keys():
        if k in final_local_vars:
            kwargs[k] = final_local_vars[k]
    return kwargs


Trainer.init_train_model_opt = init_train_model_opt
Trainer.init_train_log = init_train_log
Trainer.init_train_state = init_train_state
Trainer.full_training_step = full_training_step
# ########## patches for Trianer ##########


class MuteDefaultFlowCallback(TrainerCallback):
    """
    Add this callback can cencel logging/evaluation/saving by DefaultFlowCallback.
    Use this when having multi trainer.
    """

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_save = False
        control.should_evaluate = False
        control.should_log = False
        return control


@contextmanager
def guard_set_args(args, arg_name_values):
    for k, v in arg_name_values.items():
        old_value = getattr(args, k, None)
        setattr(args, k, v)
        arg_name_values[k] = old_value
    yield
    for k, v in arg_name_values.items():
        old_value = getattr(args, k)
        setattr(args, k, v)
        arg_name_values[k] = old_value


class PipeEvalModel(GenerationMixin):
    """
    Wrapper for PipelineParallel to do evaluate and generate. Currently only
    support .
    """

    def __init__(self, trainer: Trainer):
        eval_model = getattr(trainer, "_inner_eval_model", None)
        self.model: fleet.model.PipelineParallel = trainer.model_wrapped if eval_model is None else eval_model
        self.config: PretrainedConfig = trainer.model.config
        self._is_gen = False
        self.update_model_kwargs_for_generation = (
            self.model._layers._non_pipe_model_class.update_model_kwargs_for_generation
        )

    @property
    def pp_group(self):
        return self.model.pp_group

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def _broadcast_outputs(self, outputs):
        # outputs is PipelineParallel.eval_batch which is a list of batches.
        out = []
        outputs = (outputs,) if isinstance(outputs, paddle.Tensor) else outputs
        for tensors in outputs:
            if not self.model.is_pipeline_last_stage():
                tensor = tensors if isinstance(tensors, paddle.Tensor) else tensors[0]
                head_out_meta = (
                    (self.model._layers.head_out_meta,)
                    if isinstance(self.model._layers.head_out_meta, paddle.static.InputSpec)
                    else self.model._layers.head_out_meta
                )
                tensors = tuple(
                    paddle.empty(
                        shape=[
                            tensor.shape[i] if (meta.shape[i] is None or meta.shape[i] < 0) else meta.shape[i]
                            for i in range(len(meta.shape))
                        ],
                        dtype=tensor.dtype if meta.dtype is None else meta.dtype,
                    )
                    for meta in head_out_meta
                )
            else:
                # Currently use tuple instead of ModelOutput and require the
                # caller use the return result as tuple.
                tensors = (
                    (tensors,)
                    if isinstance(tensors, paddle.Tensor)
                    else tensors.to_tuple()
                    if isinstance(tensors, ModelOutput)
                    else tensors
                )

            # use map_structure seems hung
            for tensor in tensors:
                paddle.distributed.broadcast(tensor, src=self.model.pp_group.ranks[-1], group=self.model.pp_group)
            out.append(tensors[0] if len(tensors) == 1 else tensors)
        return out[0] if len(out) == 1 else out

    def __call__(self, *args, **kwargs):
        model = self.model
        assert self.model.training is False
        if self._is_gen:
            # inputs by `prepare_inputs_for_generation` is a dict with following keys:
            # "input_ids", "position_ids", "past_key_values", "use_cache", "attention_mask"
            # NOTE: 1. cache/past_key_values should be passed across decoding steps
            # by using as model attr rather than input args to reduce comm overhead.
            # Also, pipe model defined for training not support this cache input.
            # 2. ignore use_cache since _check_data_vaild requires tensor if not None.
            # 3. attention_mask can reuse _prepare_decoder_attention_mask in LlamaEmbeddingPipe.
            # 4. position_ids pass through _prepare_pipeline_inputs_func and PipeLayer.
            inputs, labels = model._prepare_pipeline_inputs_func(*args, **kwargs)
            # currently, set accumulate_steps to 1 to avoid multi-batch eval/gen
            with guard_set_args(model, {"_compute_loss": False, "accumulate_steps": 1}):
                outputs = model.eval_batch([inputs, labels], compute_loss=False)
            # TODO(guosheng): Broadcasted logits are used to get next_scores, remove
            # it to reduce comm overhead. Also note that we still need broadcast
            # next_tokens though logits are broadcasted since pp ranks' seeds differs.
            # Currently, just slice the last token to reduce comm overhead.
            outputs = [
                micro_batch_output[:, -1, :].unsqueeze(1)
                if isinstance(micro_batch_output, paddle.Tensor)
                else micro_batch_output[0][:, -1, :].unsqueeze(1)
                for micro_batch_output in outputs
            ]
            outputs = self._broadcast_outputs(outputs)
        else:
            # use _prepare_pipeline_inputs_func to convert pipeline inputs
            inputs, labels = model._prepare_pipeline_inputs_func(*args, **kwargs)
            # NOTE(guosheng): bug seems exist. pp.eval_batch(compute_loss=False)
            # will set pp._compute_loss to False and would not set it back. Thus
            # hack here to set it back.
            with guard_set_args(model, {"_compute_loss": False, "accumulate_steps": 1}):
                outputs = model.eval_batch([inputs, labels], compute_loss=False)
            outputs = self._broadcast_outputs(outputs)
        return outputs

    def generate(self, *args, **kwargs):
        self._is_gen = True
        # patch DecoderLayerPipe to use cache, DecoderLayerPipe is subclass of
        # DecoderLayer, and would call super().forward
        ori_decoder_layer_forward = self.model._layers._non_pipe_decoder_layer_class.forward

        def decoder_layer_forward(layer_self, *args, **kwargs):
            kwargs.update({"use_cache": True, "past_key_value": getattr(layer_self, "_cache", None)})
            outputs = ori_decoder_layer_forward(layer_self, *args, **kwargs)
            output = outputs[0]
            layer_self._cache = outputs[1]
            self._has_cache = True
            return output

        with guard_set_args(self.model._layers._non_pipe_decoder_layer_class, {"forward": decoder_layer_forward}):
            outputs = super().generate(*args, **kwargs)
        self._is_gen = False
        # clear cache of decoder layers, sublayers is incursive thus suitable
        # to both 1F1B and interleave
        for layer in self.model._layers.sublayers():
            if isinstance(layer, self.model._layers._non_pipe_decoder_layer_class):
                layer._cache = None
        self._has_cache = False
        return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        arg_bind = inspect.signature(self.model._layers._non_pipe_model_class.prepare_inputs_for_generation).bind(
            *((self,) + args), **kwargs
        )
        arg_bind.apply_defaults()
        arg_dict = arg_bind.arguments
        last_arg_name, last_arg_value = arg_dict.popitem()
        if arg_bind.signature.parameters[last_arg_name].kind == inspect.Parameter.VAR_KEYWORD:
            arg_dict.update(last_arg_value)
        else:
            arg_dict[last_arg_name] = last_arg_value
        arg_dict.pop("self")
        past_key_values = arg_dict.get("past_key_values", None)
        # prepare_inputs_for_generation use past_key_values to discrimate prefill
        # or decode and slice inputs accordingly.
        if getattr(self, "_has_cache", False):
            arg_dict.update({"past_key_values": True})
        model_inputs = self.model._layers._non_pipe_model_class.prepare_inputs_for_generation(self, **arg_dict)
        model_inputs.update({"past_key_values": past_key_values})
        return model_inputs


def is_same_tokenizer(
    tokenizer: PretrainedTokenizer,
    other_tokenizer: PretrainedTokenizer,
) -> bool:
    """Check if two tokenizers are the same."""
    return tokenizer is other_tokenizer or (
        tokenizer.__class__ == other_tokenizer.__class__ and tokenizer.get_vocab() == other_tokenizer.get_vocab()
    )


def batch_retokenize(
    input_ids: paddle.Tensor,
    src_tokenizer: PretrainedTokenizer,
    dest_tokenizer: PretrainedTokenizer,
    *,
    padding: bool | str | PaddingStrategy = PaddingStrategy.LONGEST,
    truncation: bool | str | TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    skip_special_tokens: bool = True,
) -> BatchEncoding:
    """Re-tokenize a batch of input ids from one tokenizer to another."""
    output = dest_tokenizer(
        [
            text + dest_tokenizer.eos_token
            for text in src_tokenizer.batch_decode(
                input_ids,
                skip_special_tokens=skip_special_tokens,
            )
        ],
        padding=padding,
        truncation=truncation,
        return_tensors="pd",
    )
    return output
