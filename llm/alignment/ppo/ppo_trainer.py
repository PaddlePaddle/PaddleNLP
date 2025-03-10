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
import contextlib
import copy
import itertools
import json
import math
import os
import sys
import time
import types
import uuid
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.distributed as dist
import requests
from comm_utils import (
    ActorStages,
    CriticStages,
    RolloutStages,
    create_data_trans_group,
    data_group_merge,
    data_group_split,
    get_timer_label,
    new_timer_log,
    offload_tensor_to_cpu,
    reload_tensor_to_gpu,
)
from infer_utils import InferEvalModel, infer_guard
from models.ppo_model_utils import (
    RLHFPPOMixedLoss,
    RLHFValueLoss,
    create_loss,
    gather_log_probabilities,
    make_attention_mask,
    make_position_ids,
)
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import ParallelCrossEntropy
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
    process_row,
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
from paddlenlp.trainer.trainer_utils import TrainOutput
from paddlenlp.trainer.utils import distributed_concat
from paddlenlp.transformers import (
    CosineAnnealingWithWarmupDecay,
    LinearAnnealingWithWarmupDecay,
    PretrainedModel,
    PretrainedTokenizer,
)
from paddlenlp.transformers.model_utils import _add_variant
from paddlenlp.utils.env import PADDLE_WEIGHTS_NAME


class StepTrainer(Trainer):
    """
    Features of StepTrainer:
    1. Trainer enhanced with step-level training combining with patches of
    Trianer. We can use this to do training whose step is composed of multi
    models via multiple instances of StepTrainer, such as PPO.
    2. Additionally, using a mixed loss and get the separated loss metrics is
    supported, which is helpful to PipelienParallel with a mixed loss.
    3. EMA is supported.
    """

    # used to create criterion for trainer, please refer to `create_criterion`
    # for details.
    loss_cls: type

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
        # criterion is only used for non-PipelineParallel models. criterion is
        # included in model for PipelineParallel.
        self.info_buffer = {}
        if getattr(self, "loss_cls", None) and self.criterion is None:
            self.criterion = self.create_criterion()

        self.use_fusemt = getattr(args, "use_fusemt", False)
        # ablout 4s slower than infer generation without ema
        self.use_ema = getattr(args, "use_ema", False)
        self.shard_ema = getattr(args, "shard_ema", False)
        self.offload_ema = getattr(args, "offload_ema", True)
        self.ema_beta = getattr(args, "ema_beta", 0.992)
        # if self.timers:
        #     self.timers.log = types.MethodType(new_timer_log, self.timers)

    def create_criterion(self):
        """
        create loss using `loss_cls` for trainer. It would use a wrapped loss_cls
        whose label arguments are merged into one argument, this is useful to
        PipelineParallel and trainer.criterion which limit loss format.
        """
        criterion = create_loss(self.loss_cls, self.model.config, self.args, self.info_buffer, merge_labels=True)
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

    def set_eval_model(self, model):
        """
        To avoid eval/generation with PipelineParallel when training with PP, we
        allow to use an extra eval model to do eval/generation, which would need
        to reshard parameters and dispatch data according to model's distributed
        topo. Currently, the eval model should cancel PP setting and keep the same
        TP setting with training.
        """
        if model is None:
            logger.warning("use None to set eval model for trainer and it would be ignored")
            return
        else:
            self._inner_eval_model = model
        # bind a new comm group for eval model data dispatch
        # param dispatch is binded in `InferEvalModel.enable`
        hcg = fleet.get_hybrid_communicate_group()
        sd_group = hcg.get_sharding_parallel_group()
        dp_group = hcg.get_data_parallel_group()
        global_rank = dist.get_rank()
        eval_tp_size = max(model.config.tensor_parallel_degree, 1)
        eval_tp_rank = max(model.config.tensor_parallel_rank, 0)
        old_dp_workers = self.args.world_size // (max(sd_group.nranks, 1) * max(dp_group.nranks, 1))
        group_nums = self.args.logical_process_index // old_dp_workers * eval_tp_size + eval_tp_rank
        self._data_trans_group = create_data_trans_group(global_rank, group_nums)
        # just for compatiable with old code
        self._policy_model_eval_group = self._data_trans_group

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
        inner_eval_model = getattr(self, "_inner_eval_model", None)
        if (self.args.pipeline_parallel_degree > 1 and inner_eval_model is None) or isinstance(
            inner_eval_model, fleet.model.PipelineParallel
        ):
            # Only accept wrapped model for pipeline_parallel mode
            model = PipeEvalModel(self)
            self._eval_model = model
        else:
            model = InferEvalModel(self)
            self._eval_model = model
        return model

    def get_train_step_vars(self, vars: Optional[Dict] = None) -> Dict:
        """
        NOTE: This is transparent to users.
        When using multiple instances of StepTrainer collaborate to do one training
        step, each should use its own vars such as loss/model/step_control which are
        local vars in Trainer.train, we define these vars by `train_step_vars`. They
        are vars needed by full_training_step for training control, as following:
        tr_loss, model, epoch, step, step_control.
        some vars such as `epoch` are meaningless, they are needed just because
        full_training_step copies code from Trainer.train which is designed for
        complete training process.

        return `train_step_vars` (dict). If not exists, create it first. If `vars`
        is not None, update `train_step_vars` with it.

        TODO(guosheng): use namedtuple or dataclass to make it more readable.
        """
        if not hasattr(self, "train_step_vars"):
            # should be called after model is wrapped since the model field should
            # use model_wrapped.

            if paddle.distributed.get_world_size() > 1:
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

    @property
    def loss_names(self):
        """
        返回所有损失项的名称列表，只在第一次调用时计算。
        如果没有损失项，则返回空列表。

        Returns:
            List[str]: 损失项的名称列表，每个名称以"_loss"结尾。
        """
        if not hasattr(self, "_loss_names"):
            self._loss_names = [var_name for var_name in self.get_train_step_vars() if var_name.endswith("_loss")]
            assert len(self._loss_names) > 0
        return self._loss_names

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
        # trainer.train use `tr_loss` as loss var to accumulate loss.
        # NOTE: `tr_loss` in trainer.train not only accumulate mean loss for
        # steps in one `gradient_accumulation_steps`, but also accumulate for
        # one logging intervel which may contains more than one accumulated steps.
        # However, in StepTrainer we only want to use `tr_loss` to accumulate
        # mean loss for steps in a `gradient_accumulation_steps` range. As for
        # logging intervel loss accumulation is not take into account here and
        # should be considered in outter.
        if loss_var is None:  # the first step of current loss type
            loss_var = paddle.to_tensor(0.0)
            train_step_vars[loss_name] = loss_var
        elif self.is_accumulation_step:  # begin a new accumulation step intervel
            for name in self.loss_names:
                train_step_vars[name] = paddle.to_tensor(0.0)
            loss_var = train_step_vars[loss_name]

        train_step_vars["tr_loss"] = loss_var
        # train_step_vars["timer_name"] = self.__class__.__name__

        new_train_step_vars = super().full_training_step(inputs, **train_step_vars)

        # minimally update
        train_step_vars = self.get_train_step_vars(
            {
                "step_control": new_train_step_vars["step_control"],
                loss_name: new_train_step_vars["tr_loss"],
            }
        )
        if loss_name != "tr_loss":
            train_step_vars.pop("tr_loss")

        self.mark_step_loss(loss_name)

        if self.use_ema and self.is_accumulation_step:
            # TODO(guosheng): assume rollout next thus make ema weights on gpu,
            # but may not, maybe need a way to specify it.
            self.ema_update(
                beta=self.ema_beta,
                offload_ema=self.offload_ema,
                offload_model=not self.offload_ema,
            )

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
        shared_fields = {"input_ids", "attention_mask"}
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
        NOTE: This is transparent to users.
        When using a mixed loss we often want to get the separated loss metrics,
        thus we mark loss type of each training step to separate them. This is
        not necessary since the loss would be returnd after each training step.
        However when using PipelienParallel, the loss returned is 0 when not reach
        accumulated step and the loss returned at accumulated step is a mixed loss.
        To separate loss metrics in PipelienParallel:
        1. We hack PipelineParallel._forward_step to record actual loss for each
           step in a list (only in training and not in evaluation currently).
        2. We mark the loss type only once for each step using `loss_step_indice`
           (dict), then wen can check out the corresponding loss metrics from the
           loss list.
        We assume a static order of multi-losses and mark the loss indice only once.
        """
        self.loss_step_indice = getattr(self, "loss_step_indice", {})
        if loss_name not in self.loss_step_indice:
            self.loss_step_indice[loss_name] = len(self.loss_step_indice)

    @paddle.no_grad()
    def get_step_loss(self, loss_prefix: str = "", loss_accumulator: Dict = {}) -> Dict[str, paddle.Tensor]:
        """
        Return a dict mapping loss name to value of current training step. This
        is mainly to get loss for metric logging, and it would not affect the
        training. This is mostly helpful to PipelienParallel with a mixed loss
        in which the loss returned is 0 when not reach accumulated step and the
        loss returned at accumulated step is a mixed loss.
        NOTE: 1. Only when reaching accumulated step the losses returned are
        accurate, and each loss is a mean loss of steps among one accumulated
        steps range.
        """
        if not self.is_accumulation_step:
            msg = "The loss returned may not be accurate when not reaching accumulated step."
            logger.error(msg)
        model = self.get_model(train=True)
        loss_dict = loss_accumulator if loss_accumulator else {}
        if isinstance(model, fleet.model.PipelineParallel) and len(self.loss_names) > 1:
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
                    mix_loss = paddle.empty(
                        shape=[self.args.gradient_accumulation_steps],
                        dtype=paddle.float32,
                    )
                paddle.distributed.broadcast(mix_loss, src=model.pp_group.ranks[-1], group=model.pp_group)
                for loss_name in self.loss_names:
                    # We assume a static order of multi-losses and mark the loss
                    # indice only once.
                    value = mix_loss[self.loss_step_indice[loss_name] :: len(self.loss_names)].mean()
                    loss_name = loss_prefix + loss_name if loss_prefix else loss_name
                    loss_dict[loss_name] = loss_dict[loss_name].add_(value) if loss_name in loss_dict else value
            return loss_dict
        elif isinstance(model, fleet.model.PipelineParallel):
            model._step_losses = None

        for loss_name in self.loss_names:
            value = self.get_train_step_vars()[loss_name]
            loss_name = loss_prefix + loss_name if loss_prefix else loss_name
            loss_dict[loss_name] = loss_dict[loss_name].add_(value) if loss_name in loss_dict else value
        return loss_dict

    @property
    def is_accumulation_step(self):
        """Indicate whether accumulation steps' training is done."""
        return self.get_train_step_vars()["step_control"] == 0

    def get_sharding_master_weight_structured_names(self, model, optimizer):
        """
        获取分片主机权重的结构化名称列表。
        参数：
            model (torch.nn.Module) - 模型对象，包含需要进行权重分片的参数。
            optimizer (torch.optim.Optimizer) - 优化器对象，包含需要进行权重分片的参数。
        返回值（list[str]）- 一个包含所有参数的结构化名称列表，这些参数在当前分片主机上被训练。
        """
        rank_param_names = [p.name for p in optimizer._rank2params[optimizer._sharding_rank]]
        structured_names = []
        # for pipeline model, use `model.state_dict()` would auto map param name
        # for name, p in model.named_parameters():
        for name, p in model.state_dict().items():
            if p.name in rank_param_names:
                structured_names.append(name)
        return structured_names

    def get_master_weight_state_dict(self, model, optimizer):
        """
        获取模型的权重状态字典，如果使用了AMP且支持pipeline并且存在master weights，则返回master weights。
        否则返回model.state_dict()。

        Args:
            model (nn.Module): 待获取权重状态字典的模型。
            optimizer (Optimizer): 与模型关联的优化器，可选参数，默认为None。

        Returns:
            Union[Dict[str, Tensor], Dict[str, Any]]: 返回一个包含模型权重状态的字典，字典中的键是参数名称，值是对应的Tensor或Any类型的值。
            如果使用了AMP且支持pipeline并且存在master weights，则返回的字典只包含master weights。
        """
        if self.amp_dtype in ["float16", "bfloat16"] and hasattr(optimizer, "_master_weights"):
            master_weights = dict(optimizer._master_weights)
            result = {}
            # for pipeline model, use `model.state_dict()` would auto map param name
            # for name, p in model.named_parameters():
            for name, p in model.state_dict().items():
                if p.name in master_weights:
                    result[name] = master_weights[p.name]
            return result
        else:
            return model.state_dict()

    def ema_init(self, offload_ema=True, offload_model=False, shard_ema=True):
        """should be called after model and optimizer are created and wrapped"""
        self.ema_state_dict = {}
        self.bak_state_dict = {}
        hcg = fleet.get_hybrid_communicate_group()
        sharding_size = hcg.get_sharding_parallel_world_size()
        # NOTE: use optimizer.master_weight instead of model.state_dict to set
        # ema_state_dict would make ema coupled with master_weight reshard.
        structured_names = (
            self.get_sharding_master_weight_structured_names(self.model, self.optimizer)
            if sharding_size > 1 and shard_ema
            else None
        )
        # for pipeline model, use `model.state_dict()` would auto map param name
        # for name, p in self.model.named_parameters():
        for name, p in self.model.state_dict().items():
            if structured_names is None or name in structured_names:
                ema_p = p.detach().cast(dtype=paddle.float32)
                if offload_ema:
                    ema_p = ema_p.pin_memory()
                self.ema_state_dict[name] = ema_p
            if offload_model:
                cpu_p = p.pin_memory()
                cpu_p._share_buffer_to(p)
            self.bak_state_dict[name] = p
        if getattr(self.model, "tie_word_embeddings", False):
            raise NotImplementedError

    @paddle.no_grad()
    def ema_update(self, beta=0.992, offload_ema=True, offload_model=False):
        """
        This would be called automatically in `full_training_step` if `use_ema`
        is True to update ema state when ending an accumulated step intervel.
        """
        model_keys = list(self.ema_state_dict.keys())
        hcg = fleet.get_hybrid_communicate_group()
        sharding_size = hcg.get_sharding_parallel_world_size()
        trainer_state_dict = (
            self.get_master_weight_state_dict(self.model, self.optimizer)
            if sharding_size > 1 and self.shard_ema
            else self.model.state_dict()
        )
        for key in model_keys:
            if getattr(self.model, "tie_word_embeddings", False) and "lm_head" in key:
                raise NotImplementedError
            trainer_data = trainer_state_dict[key].cuda()
            if trainer_data.dtype != paddle.float32:
                # use model state dict instead of master weights
                trainer_data = trainer_data.cast(dtype=paddle.float32)
            ema_data = self.ema_state_dict[key].cuda()
            # update ema & offload ema
            ema_result = (beta * ema_data) + (1.0 - beta) * trainer_data
            self.ema_state_dict[key] = ema_result.pin_memory() if offload_ema else ema_result
            if offload_model:
                cpu_p = trainer_data.pin_memory()
                cpu_p._share_buffer_to(trainer_data)
        if getattr(self.model, "tie_word_embeddings", False):
            raise NotImplementedError

    def ema_apply(self):
        """
        If use sharding and `shard_ema` is true, `ema_state_dict` only includes
        sharded weights, thus we need the completed ema state to apply it to model
        and ema would be coupled with reshard, then we need to reshard here.
        """
        # TODO(guosheng): `bak_state_dict` is indeed trainer.model, allow to use
        # a new model instead of trainer.model as target model.
        # NOTE: if `shard_ema` is True, `ema_state_dict` is just a subset (sharded
        # part) of model state_dict, and ema would coupled with reshard.
        for k, v in self.bak_state_dict.items():
            # TODO(guosheng): reshard here
            value = self.ema_state_dict[k].cuda().cast(dtype=v.dtype)
            value._share_buffer_to(v)

    def ema_restore(self):
        """
        将EMA的权重值还原到模型中，并且将其移动到GPU上。
        如果在初始化时设置了offload_ema=True，则会将EMA的权重值移动到GPU上。

        Returns:
            None, 无返回值，直接修改模型的权重值。
        """
        for k, v in self.bak_state_dict.items():
            value = v.cuda()
            value._share_buffer_to(v)
            if self.offload_ema:  # ema weights always in pin_memory in fact
                ema_v = self.ema_state_dict[k]
                ema_value = ema_v.pin_memory()
                ema_value._share_buffer_to(ema_v)


class ema(paddle.no_grad.__mro__[1]):
    def __init__(self, trainer: StepTrainer):
        """
        Args:
        trainer (StepTrainer): Trainer object to be used for training.
        """
        self.trainer = trainer

    def __enter__(self):
        """
        在进入上下文管理器时，如果使用了EMA，则初始化它。
        如果模型和优化器已经创建并包装，则调用ema_init。
        如果使用了EMA，则应用它。

        Returns:
            None, 无返回值。
        """
        trainer = self.trainer
        if trainer.use_ema and not hasattr(trainer, "ema_state_dict"):
            # call ema_init here since it should be called after model and
            # optimizer are created and wrapped
            trainer.ema_init(
                offload_ema=trainer.offload_ema,
                offload_model=not trainer.offload_ema,
                shard_ema=trainer.shard_ema,
            )
        if self.trainer.use_ema:
            self.trainer.ema_apply()

    def __exit__(self, *args):
        """
        如果使用了EMA，则恢复EMA状态。
        参数：
            args (tuple) - 可选，不填或为空元组，默认值为None。
        返回值：
            None - 无返回值。
        """
        if self.trainer.use_ema:
            self.trainer.ema_restore()


class Enable(paddle.no_grad.__mro__[1]):
    """offload"""

    def __init__(self, args):
        """
        初始化函数，用于设置类属性objs为传入的参数args。
        Args:
            args (Any): 需要传入的参数，将作为类属性objs。
        """
        self.objs = args

    def __enter__(self):
        """
        在进入上下文管理器时，将所有的对象都启用。
        如果对象没有 enable 方法，则使用 reload_tensor_to_gpu 来重新加载到 GPU。

        Returns:
            None, 无返回值。
        """
        for obj in self.objs:
            if hasattr(obj[0], "enable"):
                obj[0].enable()
            else:
                if obj[1] != "":
                    reload_tensor_to_gpu(obj)
        # offload_tensor_to_cpu/reload_tensor_to_gpu use non-blocking copy
        # maybe overlap with compute later
        if len(self.objs) > 0:
            paddle.device.synchronize()

    def __exit__(self, *args):
        """
        当with语句结束时，调用该方法。
        关闭所有的对象，并将其中的张量转换为CPU内存。

        Args:
            args (tuple, optional): 可选参数，默认为None。

            - 第一个元素是错误类型的对象（如果有）。
            - 第二个元素是错误信息（如果有）。
            - 第三个元素是错误的traceback（如果有）。

            这些参数与Python标准库中的__exit__方法相同。

        Returns:
            None: 无返回值。
        """
        for obj in self.objs:
            if hasattr(obj[0], "disable"):
                obj[0].disable()
            else:
                if obj[1] != "":
                    offload_tensor_to_cpu(obj)
        # offload_tensor_to_cpu/reload_tensor_to_gpu use non-blocking copy
        # maybe overlap with compute later
        if len(self.objs) > 0:
            paddle.device.synchronize()


class PolicyTrainer(StepTrainer):
    loss_cls = RLHFPPOMixedLoss
    trainer_type = "policy"

    def loss_identifier(self, inputs: Dict) -> str:
        """
        根据输入的字典，判断是否使用ptx损失函数和演员损失函数。如果有标签（labels），则返回"ptx_loss"；否则返回"actor_loss"。
        参数：
            inputs (Dict): 包含两个键值对，分别为"inputs"和"labels"，其中"inputs"是模型的输入，"labels"是可选的，表示是否使用ptx损失函数。默认值为None。
            返回值 (str): 返回一个字符串，分别为"ptx_loss"或"actor_loss"，表示是否使用ptx损失函数和演员损失函数。
        """
        labels = inputs.get("labels", None)
        if labels is not None:  # use ptx
            loss_name = "ptx_loss"
        else:
            loss_name = "actor_loss"
        return loss_name


class ValueTrainer(StepTrainer):
    loss_cls = RLHFValueLoss
    trainer_type = "value"
    # define loss name for logging
    loss_identifier = lambda self, inputs: "reward_critic_loss"


class PPOMetric:
    def set_metric_meta(self, use_ptx=True):
        """
        设置指标的元信息，包括指标名称和运算方式。
        如果不使用PTX（即不需要计算策略网络的损失），则会从指标名称中移除对应项。

        Args:
            use_ptx (bool, optional): 是否使用PTX（默认为True）. Defaults to True.

        Returns:
            None: 无返回值，直接修改了类属性。
        """
        self.metric_names = [
            "train_" + name
            for name in (
                [
                    "policy_loss",
                    "ptx_loss",
                    "value_loss",
                    "reward",
                    "norm_reward",
                    "kl_reward",
                    "norm_reward_with_kl",
                    "values",
                    "returns",
                    "kl_divergence",
                    "mean_generated_length",
                    "max_generated_length",
                    "min_generated_length",
                ]
                if self.args.rl_algorithm == "ppo"
                else [
                    "policy_loss",
                    "ptx_loss",
                    "pure_policy_loss",
                    "kl_loss",
                    "reward",
                    "kl_divergence",
                    "mean_generated_length",
                    "max_generated_length",
                    "min_generated_length",
                ]
            )
        ]

        self.metric_ops = (
            ["mean"] * 10 + ["max", "min"] if self.args.rl_algorithm == "ppo" else ["mean"] * 7 + ["max", "min"]
        )
        if not use_ptx:
            self.metric_names.pop(1)
            self.metric_ops.pop(1)

    def __init__(self, freq, args, use_stack=True, use_ptx=True):
        """
        Args:
        freq (int): frequency of metrics collection.
        use_stack (bool, optional): whether to stack the metrics into a single tensor. Defaults to True.
        use_ptx (bool, optional): whether to use ptx or not. Defaults to True.

        Raises:
            ValueError: when freq is less than 1.
        """
        self.args = args
        self.set_metric_meta(use_ptx=use_ptx)
        self.freq = freq
        self.counter = 0
        self.use_stack = use_stack
        if use_stack:
            self.metrics = paddle.zeros([freq, len(self.metric_names)], dtype=paddle.float32)
        else:
            self.metrics = [None] * len(self.metric_names)
            for i in range(len(self.metrics)):
                self.metrics[i] = paddle.zeros([freq], dtype=paddle.float32)

    @paddle.no_grad()
    def update(self, metrics: Dict[str, paddle.Tensor]) -> Union[None, Dict[str, float]]:
        """
        If has updated for`freq` times then return metrics (results reduced from
        all worker) and reset metric states, otherwise return `None`.
        """
        for name in self.metric_names:
            # PipelineParallel broadcast loss with shape [1]
            if len(metrics[name].shape) != 0:
                metrics[name] = metrics[name].squeeze()
            if metrics[name].dtype != paddle.float32:
                metrics[name] = metrics[name].cast(paddle.float32)
        if self.use_stack:
            self.metrics[self.counter] = paddle.stack([metrics[name] for name in self.metric_names])
        else:
            for i, name in enumerate(self.metric_names):
                self.metrics[i][self.counter] = metrics[name]

        self.counter += 1
        if self.counter == self.freq:
            metrics = distributed_concat(self.metrics) if paddle.distributed.get_world_size() > 1 else self.metrics

            out_metrics = {}
            if self.use_stack:
                mean_metric = metrics.mean(0)
                max_metric = metrics.max(0)
                min_metric = metrics.min(0)
            for i, (name, op) in enumerate(zip(self.metric_names, self.metric_ops)):
                if op == "max":
                    out_metrics[name] = max_metric[i].item() if self.use_stack else metrics[i].max().item()
                elif op == "min":
                    out_metrics[name] = min_metric[i].item() if self.use_stack else metrics[i].min().item()
                else:
                    out_metrics[name] = mean_metric[i].item() if self.use_stack else metrics[i].mean().item()

            # reset
            self.counter = 0
            if self.use_stack:
                self.metrics.fill_(0.0)
            else:
                for i, name in enumerate(self.metric_names):
                    self.metrics[i].fill_(0.0)
            return out_metrics


def data_dispatch(fun):
    """
    用于将函数转换为一个可以处理数据的函数，该函数会根据策略训练器中的数据分组参数进行数据切分和合并。
    如果策略训练器没有设置数据分组参数，则不进行任何操作。

    Args:
        fun (Callable[[Any, Any], Any]): 需要被转换的函数，接受两个参数：第一个是当前对象，第二个是需要处理的数据。返回值为任意类型。

    Returns:
        Callable[[Any, Any], Any]: 返回一个新的函数，接受两个参数：第一个是当前对象，第二个是需要处理的数据。返回值为任意类型。
    """

    def _impl(self, data):
        gp = getattr(self.policy_trainer, "_data_trans_group", None)
        data = data_group_split(data, group=gp)
        data = fun(self, data)
        data = data_group_merge(data, group=gp)
        return data

    return _impl


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
        preprocess_logits_for_metrics: Optional[Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]] = None,
    ):
        """
        Args:
        model (Union[PretrainedModel, nn.Layer], optional): The model to be trained. If not provided, it will be
            initialized based on the values of ``args``. Defaults to None.
        criterion (nn.Layer, optional): The loss function used for training. Defaults to None.
        args (TrainingArguments, optional): Training arguments. If not provided, it will be initialized with
            default values. Defaults to None.
        data_collator (Optional[DataCollator], optional): The function to batch data samples together into
            mini-batches. If not provided, a simple batching function that drops remaining samples will be used.
            Defaults to None.
        train_dataset (Optional[Dataset], optional): The dataset to be used for training. Defaults to None.
        ptx_dataset (Optional[Dataset], optional): The dataset to be used for ptx. Defaults to None.
        eval_dataset (Union[Dataset, Dict[str, Dataset]], optional): The dataset to be used for evaluation.
            Defaults to None.
        tokenizer (Optional[PretrainedTokenizer], optional): The tokenizer used for encoding. Defaults to None.
        compute_metrics (Optional[Callable[[EvalPrediction], Dict]], optional): The function to compute metrics
            during evaluation. Defaults to None.
        callbacks (Optional[List[TrainerCallback]], optional): A list of callbacks to customize the training
            process. Defaults to None.
        optimizers (Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler], optional): The tuple of
            optimizer and learning rate scheduler. Defaults to (None, None).
        preprocess_logits_for_metrics (Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor], optional): The
            function to preprocess logits before computing metrics. Defaults to None.
        """
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

        (
            policy_model,
            reference_model,
            reward_model,
            value_model,
            policy_model_eval,
            value_model_eval,
        ) = model
        self._model_config = policy_model.config  # use this to change flash attention dynamicly
        self._policy_model_eval = policy_model_eval
        if args.rl_algorithm == "ppo":
            self._value_model_eval = value_model_eval

        # policy_tokenizer and value_tokenizer should be same
        (
            policy_tokenizer,
            reference_tokenizer,
            reward_tokenizer,
            value_tokenizer,
        ) = tokenizer

        policy_training_args = copy.deepcopy(args)
        self.use_ptx = self.ptx_dataset is not None
        if self.use_ptx:
            policy_training_args.gradient_accumulation_steps *= 2
        lr_scheduler = self.get_scheduler(policy_training_args)
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
            [None, lr_scheduler],
            preprocess_logits_for_metrics,
        )
        if args.rl_algorithm == "ppo":
            value_training_args = copy.deepcopy(args)
            for attr_name in [
                "critic_learning_rate",
                "critic_weight_decay",
                "critic_lr_scheduler_type",
                "critic_warmup_ratio",
                "critic_recompute",
            ]:
                if getattr(value_training_args, attr_name, None) is not None:
                    setattr(
                        value_training_args,
                        attr_name[len("critic_") :],
                        getattr(value_training_args, attr_name),
                    )
            lr_scheduler = self.get_scheduler(value_training_args)
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
                [None, lr_scheduler],
                preprocess_logits_for_metrics,
            )
        self.policy_trainer.set_eval_model(policy_model_eval)
        if args.rl_algorithm == "ppo":
            self.value_trainer.set_eval_model(value_model_eval)
        # disable inner trainers' callback/state/control
        self.policy_trainer.add_callback(MuteDefaultFlowCallback)
        if args.rl_algorithm == "ppo":
            self.value_trainer.add_callback(MuteDefaultFlowCallback)
        if not self.args.disable_tqdm:
            from paddlenlp.trainer import ProgressCallback

            self.policy_trainer.pop_callback(ProgressCallback)
            if args.rl_algorithm == "ppo":
                self.value_trainer.pop_callback(ProgressCallback)

        # use trainer for reference_model/reward_model to enable sharding stage-3
        # and PipelineParallel. maybe we should allow models to use different dist
        # strategies later

        from paddle.distributed.fleet.meta_parallel import PipelineLayer

        # allow reference_model/reward_model to use different dist strategy
        with guard_set_args(
            args,
            {
                "recompute": False,
                # "fp16_opt_level": "O1",
                "pipeline_parallel_degree": (
                    args.pipeline_parallel_degree if isinstance(reference_model, PipelineLayer) else 1
                ),  # workaround for pipeline parallel model check
            },
        ):
            self.reference_trainer = StepTrainer(
                reference_model,
                criterion,
                copy.deepcopy(args),
                data_collator,
                train_dataset,
                eval_dataset,
                reference_tokenizer,
                compute_metrics,
                callbacks,
                optimizers,
                preprocess_logits_for_metrics,
            )
            if isinstance(reward_model, PretrainedModel):
                self.reward_trainer = StepTrainer(
                    reward_model,
                    criterion,
                    copy.deepcopy(args),
                    data_collator,
                    train_dataset,
                    eval_dataset,
                    reward_tokenizer,
                    compute_metrics,
                    callbacks,
                    optimizers,
                    preprocess_logits_for_metrics,
                )
            else:
                self.reward_server = reward_model
            # TODO(guosheng): sharding stage3 should create master weight optionally
            # instead of creation and clear.
            from paddlenlp.trainer.trainer_utils import ShardingOption

            if args.pipeline_parallel_degree > 1 or ShardingOption.FULL_SHARD in args.sharding:
                self.reference_trainer.init_train_model_opt(100, None, clear_master_weight=True)  # dummy max_steps
                if isinstance(reward_model, PretrainedModel):
                    self.reward_trainer.init_train_model_opt(100, None, clear_master_weight=True)  # dummy max_steps

        self.reference_model.eval()
        if isinstance(reward_model, PretrainedModel):
            self.reward_model.eval()

        self.reward_tokenizer = reward_tokenizer
        self.tokenizer = policy_tokenizer
        if is_same_tokenizer(self.tokenizer, self.reward_tokenizer):
            self.reward_tokenizer = self.tokenizer

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
        # Those value can be changed
        self.kl_coeff = self.args.kl_coeff
        self.clip_range_score = self.args.clip_range_score
        self.ptx_coeff = self.args.ptx_coeff
        self.gamma = 1.0
        self.gae_lambda = 0.95

        # for reward norm
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.sample_batch_num = 0

        # dummy class and object for model to be compaible with methods of
        # Trainer, such as evaluation_loop
        self.DummyPPOModel = type(
            "DummyPPOModel",
            (object,),
            {
                "eval": lambda _: self.set_eval(),
                "train": lambda _: self.set_train(),
            },
        )
        self.model = self.model_wrapped = self.DummyPPOModel()
        if self.timers:
            self.timers.log = types.MethodType(new_timer_log, self.timers)

    @property
    def reference_model(self):
        """
        获取参考模型，如果没有则返回None。
        该方法只能在初始化后使用，否则会引发异常。

        Returns:
            torch.nn.Module, optional - 参考模型，如果没有则返回None。

        Raises:
            Exception - 当调用此方法前未初始化reference_trainer时，将引发异常。
        """
        return self.reference_trainer.get_model(train=False)

    @property
    def reward_model(self):
        """
        获取奖励模型，如果没有则创建一个。
        返回值：tf.keras.models.Model，奖励模型。
        """
        if hasattr(self, "reward_trainer"):
            return self.reward_trainer.get_model(train=False)
        else:
            return self.reward_server

    @property
    def actor_model(self):
        """
        获取当前的actor模型，如果在训练中则返回训练后的模型，否则返回eval时使用的模型。

        Returns:
            torch.nn.Module, torch.jit.ScriptModule: Actor模型，可以是torch.nn.Module或者torch.jit.ScriptModule类型。
        """
        return self.policy_trainer.get_model(train=self.training)

    @property
    def reward_critic_model(self):
        """
        获取 critic model，仅在使用 value-based 策略时有效。

        Returns:
            tf.keras.Model, optional: critic model，如果没有设置则返回 None。
        """
        return self.value_trainer.get_model(train=self.training)

    def set_train(self, mode: bool = True) -> None:
        """Set training mode for all models."""
        if mode:
            self.training = True
            self.actor_model.train()
            if self.args.rl_algorithm == "ppo":
                self.reward_critic_model.train()
        else:
            self.training = False
            self.actor_model.eval()
            if self.args.rl_algorithm == "ppo":
                self.reward_critic_model.eval()

    def set_eval(self) -> None:
        """Set model to evaluation mode."""
        self.set_train(mode=False)

    def get_scheduler(self, args):
        """
        获取学习率调度器，如果没有设置最小学习率则返回None。
        支持两种类型的学习率调度器："cosine"和"linear"。

        Args:
            args (argparse.Namespace): 命令行参数，包含了学习率相关的参数。

        Returns:
            torch.optim.lr_scheduler._LRScheduler or None, optional: 学习率调度器或者None，默认为None。
        """
        if args.decay_steps is None:
            args.decay_steps = args.max_steps
        if args.warmup_steps > 0:
            warmup_steps = args.warmup_steps
        else:
            warmup_steps = args.warmup_ratio * args.max_steps
        lr_scheduler = None
        if args.min_learning_rate is not None:
            if args.lr_scheduler_type == "cosine":
                lr_scheduler = CosineAnnealingWithWarmupDecay(
                    max_lr=args.learning_rate,
                    min_lr=args.min_learning_rate,
                    warmup_step=warmup_steps,
                    decay_step=args.decay_steps,
                    last_epoch=0,
                )
            elif args.lr_scheduler_type == "linear":
                lr_scheduler = LinearAnnealingWithWarmupDecay(
                    max_lr=args.learning_rate,
                    min_lr=args.min_learning_rate,
                    warmup_step=warmup_steps,
                    decay_step=args.decay_steps,
                    last_epoch=0,
                )
        return lr_scheduler

    @paddle.no_grad()
    def prediction_step(
        self,
        model: nn.Layer,
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:
        """
        预测步骤，用于生成下一个输入序列。

        Args:
            model (nn.Layer): 模型实例，需要是 `paddle.nn.Layer` 的子类。
            inputs (Dict[str, Union[paddle.Tensor, Any]]): 包含输入数据的字典，其中包含以下键：
                - "input_ids" (paddle.Tensor, optional): 输入序列的编号 ID，默认为None。
                - "attention_mask" (paddle.Tensor, optional): 输入序列的注意力掩码，默认为None。
                - "position_ids" (paddle.Tensor, optional): 输入序列的位置ID，默认为None。
            prediction_loss_only (bool): 仅返回预测损失，不返回其他任何值。
            ignore_keys (Optional[List[str]], optional): 忽略的键列表，默认为None。

        Returns:
            Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:
            三元组，包含以下元素：
                - Optional[paddle.Tensor]: 如果 `prediction_loss_only` 为False，则为预测得分，否则为None。
                - Optional[paddle.Tensor]: 当前未定义，始终为None。
                - Optional[paddle.Tensor]: 当前未定义，始终为None。

        Raises:
            ValueError: 如果 `ignore_keys` 不是可选参数或者不是一个列表。
        """
        inputs = self._prepare_inputs(inputs)
        with self.enable(self.actor_model, self.reference_model, self.policy_trainer):
            with infer_guard(self.policy_trainer):
                position_ids = inputs.get("position_ids", make_position_ids(inputs["attention_mask"]))
                prompt_only_batch = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "position_ids": position_ids,
                    **({"label_ids": inputs["label_ids"]} if self.args.use_rm_server else {}),
                }
                generated_seq = self.generate(prompt_only_batch, do_eval=True)[0]["input_ids"]

            if self._model_config.sequence_parallel:
                # pad to max_sequence_length
                seq = self.tokenizer.pad(
                    {"input_ids": [s for s in generated_seq]},
                    padding="max_length",
                    max_length=self._model_config.max_sequence_length,
                    return_attention_mask=False,
                )["input_ids"]
            else:
                seq = generated_seq

            if not self.args.use_rm_server:
                if self.reward_tokenizer is not self.tokenizer:
                    reward_tokenize_output = batch_retokenize(
                        input_ids=seq,
                        src_tokenizer=self.tokenizer,
                        dest_tokenizer=self.reward_tokenizer,
                    )
                    reward_input_ids = reward_tokenize_output["input_ids"]
                    reward_attention_mask = reward_tokenize_output["attention_mask"]
                    reward_position_ids = reward_tokenize_output["position_ids"]
                else:
                    reward_input_ids = seq
                    reward_attention_mask = make_attention_mask(
                        seq,
                        pad_id=self.reward_tokenizer.pad_token_id,
                        eos_id=self.reward_tokenizer.eos_token_id,
                        unk_id=self.reward_tokenizer.unk_token_id,
                        causal_mask=True,
                    )
                    reward_position_ids = make_position_ids(reward_attention_mask)

                # .end_scores
                reward_score = self.reward_model(
                    reward_input_ids,
                    attention_mask=reward_attention_mask,
                    position_ids=reward_position_ids,
                    # return_dict=True,
                )[1]
            else:
                prompt_len = inputs["input_ids"].shape[-1]
                if "label_ids" not in inputs:
                    raise ValueError("Rule-based reward needs labels.")
                src = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
                tgt = self.tokenizer.batch_decode(inputs["label_ids"], skip_special_tokens=True)
                response = self.tokenizer.batch_decode(generated_seq[:, prompt_len:], skip_special_tokens=True)
                reward_score = self.request_reward_server(src, tgt, response)

            reward_score = reward_score.squeeze(axis=-1).cast(paddle.float32)
        # keep the first batch of eval output sequence to print and check
        prompt = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        generated = self.tokenizer.batch_decode(generated_seq, skip_special_tokens=True)  # no padding
        reward_score_list = reward_score.tolist()
        for i, text in enumerate(generated):
            item = {
                "Prompt": text[: len(prompt[i]) - 1],
                "Generated": text[len(prompt[i]) :],
                "Reward": reward_score_list[i],
            }
            self._eval_out_file.write(json.dumps(item, ensure_ascii=False) + "\n")

        if getattr(self, "_eval_seq", None) is None:
            generated = [text[len(prompt[i]) :] for i, text in enumerate(generated)]
            # prompts.extend(prompt)
            # generateds.extend(generated)
            self._eval_seq = (prompt, generated, reward_score_list)
        return reward_score.mean(), reward_score, reward_score

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_eval_iters: Optional[int] = -1,
    ) -> EvalLoopOutput:
        """
        循环访问数据集，并对模型进行评估。

        Args:
            dataloader (DataLoader, optional): 用于评估的数据加载器。默认为None。
            description (str, optional): 描述评估过程的字符串。默认为''.
            prediction_loss_only (Optional[bool], optional): 是否只计算预测损失。默认为None。
            ignore_keys (Optional[List[str]], optional): 要忽略的键列表。默认为None。
            metric_key_prefix (str, optional): 指标键前缀。默认为'eval'.
            max_eval_iters (Optional[int], optional): 最大评估次数。默认为-1，表示无限制。

        Returns:
            EvalLoopOutput: 包含评估结果和指标的类实例。

        Raises:
            ValueError: 如果`prediction_loss_only`不是布尔值，则引发ValueError异常。
        """
        # to save eval generated sequence
        eval_out_file = os.path.join(
            self.args.output_dir,
            f"eval_out-step{self.state.global_step}-rank{self.args.local_rank}.jsonl",
        )
        self._eval_out_file = open(eval_out_file, "w", encoding="utf-8")

        # TODO(guosheng): use _inner_eval_model (if trainer has one) instead of
        # original trainer model to eval, especially when using sharded EMA
        # NOTE: use here rather than in prediction_step since actor_model would
        # be set to eval out of prediction_step
        # with guard_set_args(
        #     self.policy_trainer,  # disable _inner_eval_model
        #     {
        #         "_eval_model": None,  # otherwise would use cached _eval_model
        #         "_inner_eval_model": None,  # otherwise would use _inner_eval_model to create _eval_model
        #     },
        # ):
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
            max_eval_iters,
        )
        output.metrics[f"{metric_key_prefix}_reward"] = output.metrics.pop(f"{metric_key_prefix}_loss")

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
        """
        获取用于评估模型的数据加载器。如果未提供`eval_dataset`，则使用`self.eval_dataset`。
            该函数会设置一个名为"data_collator"的参数，并将其传递给`super().get_eval_dataloader()`。

            Args:
                eval_dataset (Optional[Dataset], optional): 用于评估的数据集. Defaults to None.

            Returns:
                DataLoader: 包含用于评估的数据的DataLoader实例。
        """
        with guard_set_args(self, {"data_collator": self.eval_dataset.get_collator()}):
            return super().get_eval_dataloader(eval_dataset)

    def _save_checkpoint(self, model, metrics=None):
        """
        保存模型和指标到两个不同的 checkpoint，一个是 policy 模型，另一个是 value 模型。
        这里使用了 `guard_set_args` 来防止在调用 `_save_checkpoint` 时修改了原始参数。

        Args:
            model (nn.Module): 需要保存的模型。
            metrics (Optional[Dict], optional): 可选的指标字典，默认为 None。
                key 是指标名称，value 是对应的指标值。

        Returns:
            None.
        """
        # maybe change args.output_dir of policy_trainer/value_trainer directly
        self.runtime_timer.start("checkpoint saving time")
        with guard_set_args(
            self.policy_trainer.args,
            {"output_dir": os.path.join(self.args.output_dir, "policy")},
        ):
            if self.policy_trainer.args.unified_checkpoint:
                if "train_model" in self.policy_trainer.args.offload_level:
                    reload_tensor_to_gpu((self.policy_trainer.model, "train_model"))
                if (
                    "optimizer" in self.policy_trainer.args.offload_level
                    and not self.policy_trainer.args.ignore_save_lr_and_optim
                ):
                    reload_tensor_to_gpu((self.policy_trainer.optimizer, "optimizer"))
            self.policy_trainer._save_checkpoint(model, metrics)
        if self.args.rl_algorithm == "ppo":
            with guard_set_args(
                self.value_trainer.args,
                {"output_dir": os.path.join(self.args.output_dir, "value")},
            ):
                if self.value_trainer.args.unified_checkpoint:
                    if "train_model" in self.value_trainer.args.offload_level:
                        reload_tensor_to_gpu((self.value_trainer.model, "train_model"))
                    if (
                        "optimizer" in self.value_trainer.args.offload_level
                        and not self.value_trainer.args.ignore_save_lr_and_optim
                    ):
                        reload_tensor_to_gpu((self.value_trainer.optimizer, "optimizer"))
                self.value_trainer._save_checkpoint(model, metrics)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                metrics = {
                    "policy": self.policy_trainer.state.best_model_checkpoint,
                    **(
                        {"value": self.value_trainer.state.best_model_checkpoint}
                        if self.args.rl_algorithm == "ppo"
                        else {}
                    ),
                }
                self.state.best_model_checkpoint = json.dumps(metrics)

    def save_model(
        self,
        output_dir: Optional[str] = None,
        merge_tensor_parallel: Optional[bool] = False,
    ):
        """
            保存模型。

        Args:
            output_dir (Optional[str], optional): 输出目录，默认为None，使用命令行参数--output-dir。 Defaults to None.
            merge_tensor_parallel (Optional[bool], optional): 是否合并tensor parallel，默认为False。 Defaults to False.

        Raises:
            ValueError: 如果output_dir不在当前工作目录下，则会引发ValueError异常。
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if "train_model" in self.args.offload_level:
            reload_tensor_to_gpu((self.policy_trainer.model, "model"))
            if self.args.rl_algorithm == "ppo":
                reload_tensor_to_gpu((self.value_trainer.model, "model"))
        self.policy_trainer.save_model(os.path.join(output_dir, "policy"), merge_tensor_parallel)
        if self.args.rl_algorithm == "ppo":
            self.value_trainer.save_model(os.path.join(output_dir, "value"), merge_tensor_parallel)

    def init_train_model_opt(
        self: Trainer,
        max_steps: int,
        resume_from_checkpoint: bool = False,
        clear_master_weight: bool = False,
    ) -> PretrainedModel:
        """
            初始化训练模型和优化器。
        如果`resume_from_checkpoint`为字符串，则将其作为路径，并在该路径下恢复模型和优化器状态；否则，将其视为布尔值，表示是否从最后一个保存的检查点中恢复。
        如果`clear_master_weight`为True，则清除主要权重。

        Args:
            max_steps (int): 最大训练步数。
            resume_from_checkpoint (bool, optional): 是否从检查点中恢复模型和优化器状态（默认为False）。
                如果为字符串，则将其作为路径，并在该路径下恢复模型和优化器状态。
            clear_master_weight (bool, optional): 是否清除主要权重（默认为False）。

        Returns:
            Tuple[PretrainedModel, PretrainedModel]: 返回两个元组，分别包含策略模型和价值函数模型。
        """
        # resume should be triggered here
        # maybe change args.output_dir of policy_trainer/value_trainer directly
        with guard_set_args(
            self.policy_trainer.args,
            {"output_dir": os.path.join(self.args.output_dir, "policy")},
        ):
            policy_model = self.policy_trainer.init_train_model_opt(
                max_steps,
                (
                    os.path.join(resume_from_checkpoint, "policy")
                    if isinstance(resume_from_checkpoint, str)
                    else resume_from_checkpoint
                ),
            )
        if self.args.rl_algorithm == "ppo":
            with guard_set_args(
                self.value_trainer.args,
                {"output_dir": os.path.join(self.args.output_dir, "value")},
            ):
                value_model = self.value_trainer.init_train_model_opt(
                    max_steps,
                    (
                        os.path.join(resume_from_checkpoint, "value")
                        if isinstance(resume_from_checkpoint, str)
                        else resume_from_checkpoint
                    ),
                )
        else:
            value_model = None
        return policy_model, value_model

    def get_epoch_iterator(self):
        """
            获取一个迭代器，该迭代器将生成一个批次的数据。每个批次包含两部分：一个是提示仅批次（prompt only batch），另一个是PTX批次（PTX batch）。
        如果使用了PTX，则PTX批次会在每个RL批次之后进行轮换。

        Args:
            无参数。

        Returns:
            EpochIterator (class): 返回一个类，该类包含一个__iter__方法和一个__len__方法。__iter__方法可以生成一个批次的数据，__len__方法返回总共有多少个批次。

        Raises:
            无异常抛出。
        """

        def gen_epoch_data():
            for prompt_only_batch, ptx_batch in zip(
                self.prompt_only_dataloader,
                itertools.cycle(self.ptx_dataloader),
            ):
                # generate batches
                self.set_eval()

                with (
                    ema(self.policy_trainer),
                    ema(self.value_trainer) if self.args.rl_algorithm == "ppo" else contextlib.nullcontext(),
                ):
                    with guard_set_args(self._model_config, {"use_fused_head_and_loss_fn": False}):
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

            def __len__(self):
                return len(self.prompt_only_dataloader) * (
                    self.args.update_iters
                    * self.args.per_device_prompt_batch_size
                    * self.args.num_return_sequences
                    // self.args.per_device_train_batch_size
                )

        return EpochIterator()

    def init_train_num(self: Trainer, train_dataloader: DataLoader):
        """
            初始化训练数据的批次大小，以及相关参数。

        Args:
            self (Trainer): Trainer实例。
            train_dataloader (DataLoader): 用于训练的DataLoader对象。

        Returns:
            tuple (int, Optional[int], int, int, int, int, int):
                返回一个元组，包含：
                1. total_train_batch_size (int) - 总训练批次大小。
                2. len_dataloader (Optional[int]) - 如果不是可迭代的数据集，则为DataLoader长度；否则为None。
                3. max_steps (int) - 最大训练步数。
                4. num_train_epochs (int) - 训练的最大轮数。
                5. num_update_steps_per_epoch (int) - 每个epoch中更新模型的次数。
                6. num_examples (int) - 训练数据的样本数量。
                7. num_train_samples (int) - 训练数据的样本总数。
        """
        args = self.args

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.dataset_world_size
        len_dataloader = None
        if not self._is_iterable_dataset(self.train_dataset):
            len_dataloader = len(train_dataloader)
            num_train_sub_steps = (
                len_dataloader
                * self.args.update_iters
                * self.args.per_device_prompt_batch_size
                * self.args.num_return_sequences
                // self.args.per_device_train_batch_size
            )
            num_update_steps_per_epoch = num_train_sub_steps // args.gradient_accumulation_steps
            num_examples = len(self.train_dataset)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
                num_train_epochs = math.ceil(args.num_train_epochs)
            num_train_samples = total_train_batch_size * max_steps
        else:
            assert args.max_steps > 0
            max_steps = args.max_steps
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = args.max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size

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
        """
            判断是否到达了步数结尾，当累加步数等于args.gradient_accumulation_steps时返回True。
        返回值：bool，如果到达了步数结尾则返回True，否则返回False。
        """
        # reach accumulation_steps, value trainer has the same step_control and
        # gradient_accumulation_steps as PPO trainer.
        # if (step_control + 1) % args.gradient_accumulation_steps == 0
        if self.args.rl_algorithm == "ppo":
            return self.value_trainer.is_accumulation_step
        return self.policy_trainer.is_accumulation_step

    def get_step_loss(self, loss_prefix: str = "") -> Dict:
        """
            获取当前步骤的损失，包括策略训练和价值函数训练的损失。
        如果提供了loss_prefix参数，则将损失名称加上该前缀。

        Args:
            loss_prefix (str, optional): 损失名称的前缀字符串，默认为"".

        Returns:
            Dict[str, float]: 返回一个字典，包含两个损失项：rl_loss（策略训练的损失）和value_loss（价值函数训练的损失）。
        """
        rl_loss = self.policy_trainer.get_step_loss(loss_prefix)
        if self.args.rl_algorithm == "ppo":
            value_loss = self.value_trainer.get_step_loss(loss_prefix)
            rl_loss.update(value_loss)
        return rl_loss

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
    ) -> None:
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (Optional[Union[str, bool]], optional):
                Checkpoint path from which training should be resumed. If a
                path is given, training will restart from this checkpoint. If
                set to ``True``, the last checkpoint in ``output_dir`` will be
                loaded. If ``False`` or ``None`` (default), training will
                start from scratch. Defaults to ``None``.

            ignore_keys_for_eval (Optional[List[str]], optional):
                List of keys to ignore when computing the metrics during
                evaluation. Defaults to ``None``.

        Returns:
            None:
            Training process is finished, no return value.
        """
        # ##### The following code try to keep same as the Trainer.train #####
        args = self.args
        self.is_in_train = True

        # ##### trainging data and related num setting #####
        # TODO(guosheng): remove the binding method get_collator of dataset
        with (
            guard_set_args(
                args,
                {"per_device_train_batch_size": self.args.per_device_prompt_batch_size},
            ),
            guard_set_args(
                self,
                {
                    "train_dataset": self.train_dataset,
                    "data_collator": self.train_dataset.get_collator(),
                },
            ),
        ):
            train_dataloader = self.prompt_only_dataloader = self.get_train_dataloader()

        if self.use_ptx:
            with (
                guard_set_args(
                    args,
                    {
                        "per_device_train_batch_size": (
                            1
                            if getattr(self.ptx_dataset, "is_intokens", False)
                            else self.args.per_device_prompt_batch_size * self.args.num_return_sequences
                        )
                    },
                ),
                guard_set_args(
                    self,
                    {
                        "train_dataset": self.ptx_dataset,
                        "data_collator": self.ptx_dataset.get_collator(),
                    },
                ),
            ):
                self.ptx_dataloader = self.get_train_dataloader()
        else:
            self.ptx_dataloader = range(100)
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
            num_examples,
            num_train_epochs,
            total_train_batch_size,
            max_steps,
            num_train_samples,
            policy_model,
        )

        # ##### set training state and resume #####
        # consumed_samples used to set train_dataloader.batch_sampler may not be
        # correct. Thus, data cannot be resumed perfectly when not breaking at epoch end.
        (epochs_trained, steps_trained_in_current_epoch, steps_trained_progress_bar,) = self.init_train_state(
            resume_from_checkpoint,
            train_dataloader,
            max_steps,
            num_train_epochs,
            num_update_steps_per_epoch,
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
        metric = PPOMetric(freq=self.args.logging_steps, args=self.args, use_ptx=self.use_ptx)

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
                # logger.info("Doing rl step...")
                self.timers and self.timers(get_timer_label(ActorStages.MODEL_ENABLE_DISABLE)).start()
                with self.enable(self.actor_model, self.policy_trainer.optimizer):
                    self.timers and self.timers(get_timer_label(ActorStages.RL_STEP)).start()
                    rl_info = self.rl_step(rl_batch)
                    self.timers and self.timers(get_timer_label(ActorStages.RL_STEP)).stop()
                    if self.use_ptx:
                        logger.info("Doing ptx step...")
                        self.timers and self.timers(get_timer_label(ActorStages.PTX_STEP)).start()
                        with guard_set_args(
                            self._model_config,
                            {
                                # "set_attn_func": True,
                                "use_flash_attention": True
                            },
                        ):
                            ptx_info = self.ptx_step(ptx_batch)
                        rl_info.update(ptx_info)
                        self.timers and self.timers(get_timer_label(ActorStages.PTX_STEP)).stop()
                if self.timers:
                    self.timers(get_timer_label(ActorStages.MODEL_ENABLE_DISABLE)).stop()
                    self.timers(get_timer_label(ActorStages.MODEL_ENABLE_DISABLE)).elapsed_ -= self.timers(
                        get_timer_label(ActorStages.RL_STEP)
                    ).elapsed_
                    if self.use_ptx:
                        self.timers(get_timer_label(ActorStages.MODEL_ENABLE_DISABLE)).elapsed_ -= self.timers(
                            get_timer_label(ActorStages.PTX_STEP)
                        ).elapsed_

                paddle.device.cuda.empty_cache()
                if self.args.rl_algorithm == "ppo":
                    rl_critic_info = self.rl_critic_step(rl_batch)
                    rl_info.update(rl_critic_info)
                if self.is_step_end():
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    rl_info.update(self.get_step_loss(loss_prefix="train_"))
                    rl_info = metric.update(rl_info)
                    # on_step_end
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                else:
                    # on_sub_step_end
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                self._print_timer()
                self._maybe_log_save_evaluate(rl_info, None, epoch, ignore_keys_for_eval, inputs=inputs)
                paddle.device.cuda.empty_cache()

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

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
        logger.info("\nTraining completed. \n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            if args.local_rank != -1:
                dist.barrier()

            best_model_checkpoint = json.loads(self.state.best_model_checkpoint)

            logger.info(f"Loading best model from {best_model_checkpoint['value']}(score: {self.state.best_metric}).")
            self.load_best_ckpt(best_model_checkpoint["value"], self.value_trainer)

            logger.info(f"Loading best model from {best_model_checkpoint['policy']}(score: {self.state.best_metric}).")
            self.load_best_ckpt(best_model_checkpoint["policy"], self.policy_trainer)

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
        )

        self.is_in_train = False
        self.log(metrics)
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        tr_loss = 0.0
        for history in self.state.log_history:
            if "train_policy_loss" in history:
                tr_loss += history["train_policy_loss"]
        tr_loss = tr_loss / self.state.global_step
        return TrainOutput(self.state.global_step, tr_loss, metrics)

    def load_best_ckpt(self, model_path, trainer, **kwargs):
        """
        Load the best checkpoint from the given path into the specified trainer.

        Args:
            args (TrainingArguments): The arguments object containing the configuration settings.
            model_path (str): The path to the directory where the best checkpoint is located.
            trainer (Trainer): The trainer instance that will receive the loaded weights.
            kwargs (Any, optional): Additional keyword arguments passed to the `load_unified_checkpoint` function.
        """
        from paddlenlp.trainer.utils.helper import broadcast_dataset_rank0_model

        if trainer.args.unified_checkpoint:
            trainer.unified_checkpoint_handler.load_unified_checkpoint(
                trainer.model,
                model_path,
            )
            if trainer.args.sharding_parallel_degree > 1 or trainer.args.data_parallel_degree > 1:
                broadcast_dataset_rank0_model(trainer.model)
        else:
            weight_name = PADDLE_WEIGHTS_NAME
            best_model_path = os.path.join(
                model_path,
                _add_variant(weight_name, trainer.args.weight_name_suffix),
            )
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = paddle.load(best_model_path, return_numpy=True)
                # If the model is on the GPU, it still works!
                trainer._set_state_dict_in_model(state_dict)
            else:
                logger.warning(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs):
        """
        记录、保存和评估，如果需要。
            如果控制变量指示应该记录，则记录损失，并将模型保存到磁盘上。
            如果控制变量指示应该评估，则评估模型并将结果保存到磁盘上。

            Args:
                tr_loss (Optional[Dict[str, float]]): 字典形式的训练损失，包含键'train_policy_loss'和'train_ptx_loss'。
                    如果为None，则不记录任何内容。默认为None。
                model (Model): 用于评估的模型。
                epoch (int): 当前迭代次数。
                ignore_keys_for_eval (List[str]): 在评估时要忽略的键列表。默认为空列表。
                kwargs (Any, optional): 其他可选参数，将被传递给`log()`和`save()`方法。默认为空字典。

            Returns:
                None.

            Raises:
                None.
        """
        if self.control.should_log and tr_loss is not None:
            logs: Dict[str, float] = {}
            # use_ptx would double the gradient_accumulation_steps which causes
            # policy_loss and ptx_loss reduced by half. Moreover, ptx_loss should
            # be divided by ptx_coeff for logging.
            if "train_ptx_loss" in tr_loss:
                tr_loss["train_policy_loss"] = tr_loss["train_policy_loss"] * 2
                tr_loss["train_ptx_loss"] = tr_loss["train_ptx_loss"] * 2 / self.ptx_coeff
            logs.update(tr_loss)
            logs["global_step"] = int(self.state.global_step)
            logs["train_actor_lr"] = float(f"{self.policy_trainer._get_learning_rate():.3e}")
            if self.args.rl_algorithm == "ppo":
                logs["train_reward_critic_lr"] = float(f"{self.value_trainer._get_learning_rate():.3e}")

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
        """
            计算KL散度迭代增益，并将其添加到回报中。
        参数：
            prompt (paddle.Tensor, shape=(B, S)): 输入序列的prompt，未使用。
            log_probs (paddle.Tensor, shape=(B, L)): 当前预测的log概率分布。
            ref_log_probs (paddle.Tensor, shape=(B, L)): 基线预测的log概率分布。
            reward_score (paddle.Tensor, shape=(B,)): 基于prompt和输出序列的基本奖励得分。
            sequence_mask (paddle.Tensor, shape=(B, L)): 序列的mask，用于确定序列的长度。
        返回值（paddle.Tensor, shape=(B, L)}：
            包含KL散度迭代增益的向量。
        """

        kl_divergence_estimate = -self.kl_coeff * (log_probs - ref_log_probs)  # size = (B, L)
        rewards = kl_divergence_estimate  # size = (B, L)
        reward_clip = paddle.clip(  # size = (B,)
            reward_score,
            min=-self.clip_range_score,
            max=self.clip_range_score,
        )
        # TODO(guosheng): use scatter_add/put_along_axis
        index = paddle.cumsum(sequence_mask.cast(paddle.int64), axis=-1).argmax(-1, keepdim=True)

        rewards = paddle.put_along_axis(
            rewards,
            index,
            reward_clip.unsqueeze(axis=-1),
            axis=-1,
            reduce="add",
        )
        return rewards, kl_divergence_estimate

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
        returns = advantages + values[:, start:].contiguous()

        if not use_tgt_len_return:
            advantages = paddle.concat(
                [
                    paddle.zeros([advantages.shape[0], start], dtype=advantages.dtype),
                    advantages,
                ],
                -1,
            )
            returns = paddle.concat(
                [
                    paddle.zeros([returns.shape[0], start], dtype=returns.dtype),
                    returns,
                ],
                -1,
            )

        return advantages.detach(), returns

    def rl_step(self, rl_batch: Dict[str, paddle.Tensor]) -> Dict[str, Any]:
        # inputs shared by policy and value trainer
        input_ids = rl_batch["input_ids"].contiguous()  # length: src+tgt
        attention_mask = rl_batch["attention_mask"]  # length: src+tgt
        position_ids = rl_batch["position_ids"]  # length: src+tgt
        sequence_mask = rl_batch["sequence_mask"]  # length: src+tgt(-1)
        # inputs used by policy trainer
        old_log_probs = rl_batch["log_probs"]  # length: src+tgt(-1)
        reward_advantages = rl_batch["reward_advantages"]  # length: src+tgt(-1)

        policy_trainer_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "old_log_probs": old_log_probs,
            "reward_advantages": reward_advantages,
            "sequence_mask": sequence_mask,
        }

        if self.args.rl_algorithm == "grpo":
            policy_trainer_inputs.update({"ref_log_probs": rl_batch["ref_log_probs"]})

        actor_loss = self.policy_trainer.full_training_step(**policy_trainer_inputs)

        # metric
        with paddle.no_grad():
            rewards = rl_batch["rewards"].mean()
            ori_rewards = rl_batch["ori_rewards"].mean()
            mask_cast = sequence_mask.cast(paddle.float32)
            if self.args.rl_algorithm == "ppo":
                kl_rewards = (rl_batch["kl_rewards"] * mask_cast).sum() / mask_cast.sum()
                rewards_with_kl = (rl_batch["rewards_with_kl"] * mask_cast).sum() / mask_cast.sum()
                values = (rl_batch["reward_values"] * mask_cast).sum() / mask_cast.sum()
                returns = (rl_batch["reward_returns"] * mask_cast).sum() / mask_cast.sum()
            ref_log_probs = rl_batch["ref_log_probs"]
            # kl_divergence = ((old_log_probs - ref_log_probs) * sequence_mask).sum(axis=-1).mean()
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
                    "train_pure_policy_loss": self.policy_trainer.info_buffer.get("pure_policy_loss"),
                    "train_kl_loss": self.policy_trainer.info_buffer.get("kl_loss"),
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
                    "train_values": values,
                    "train_returns": returns,
                }
                if self.args.rl_algorithm == "ppo"
                else {}
            ),
            "train_kl_divergence": kl_divergence,
            "train_mean_generated_length": mean_generated_length,
            "train_max_generated_length": max_generated_length,
            "train_min_generated_length": min_generated_length,
        }

    def rl_critic_step(self, rl_batch: Dict[str, paddle.Tensor]) -> Dict[str, Any]:
        """
        更新评价函数（奖励函数）的参数。
            该函数需要接收一个字典类型的参数，包括以下键值对：
                - input_ids (paddle.Tensor): 输入序列的ID，形状为（src+tgt, batch）。
                - attention_mask (paddle.Tensor): 输入序列的注意力掩码，形状为（src+tgt, batch）。
                - position_ids (paddle.Tensor): 输入序列的位置ID，形状为（src+tgt, batch）。
                - old_reward_values (paddle.Tensor): 上一时间步的奖励值，形状为（src+tgt-1, batch）。
                - reward_returns (paddle.Tensor): 回报返回值，形状为（src+tgt-1, batch）。
                - sequence_mask (paddle.Tensor): 序列掩码，形状为（src+tgt-1, batch）。
        返回值（Dict[str, Any]）：
            - train_value_loss (float): 评价函数（奖励函数）的训练损失。
        """
        # inputs shared by policy and value trainer
        input_ids = rl_batch["input_ids"].contiguous()  # length: src+tgt
        attention_mask = rl_batch["attention_mask"]  # length: src+tgt
        position_ids = rl_batch["position_ids"]  # length: src+tgt
        sequence_mask = rl_batch["sequence_mask"]  # length: src+tgt(-1)
        # inputs used by value trainer
        old_reward_values = rl_batch["reward_values"]  # length: src+tgt(-1)
        reward_returns = rl_batch["reward_returns"]  # length: src+tgt(-1)

        value_trainer_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "old_reward_values": old_reward_values,
            "reward_returns": reward_returns,
            "sequence_mask": sequence_mask,
        }
        self.timers and self.timers(get_timer_label(CriticStages.MODEL_ENABLE_DISABLE)).start()
        with self.enable(self.reward_critic_model, self.value_trainer.optimizer):
            self.timers and self.timers(get_timer_label(CriticStages.CRITIC_TRAINING_STEP)).start()
            reward_critic_loss = self.value_trainer.full_training_step(**value_trainer_inputs)
            self.timers and self.timers(get_timer_label(CriticStages.CRITIC_TRAINING_STEP)).stop()

        if self.timers:
            self.timers and self.timers(get_timer_label(CriticStages.MODEL_ENABLE_DISABLE)).stop()
            self.timers(get_timer_label(CriticStages.MODEL_ENABLE_DISABLE)).elapsed_ -= self.timers(
                get_timer_label(CriticStages.CRITIC_TRAINING_STEP)
            ).elapsed_

        return {"train_value_loss": reward_critic_loss}

    def ptx_step(self, ptx_batch: Dict[str, paddle.Tensor]) -> Dict[str, Any]:
        """Perform a single update step with PTX loss."""
        # sft inputs use right padding, position_ids is optional
        # ptx_batch["position_ids"] = ptx_batch.get(
        #     "position_ids", make_position_ids(ptx_batch["attention_mask"]))
        ptx_loss = self.policy_trainer.full_training_step(**ptx_batch)
        return {
            "train_ptx_loss": ptx_loss,
        }

    def enable(self, *args):
        """
            启用指定的对象或方法。
        如果指定的对象是模型，则会将其设置为训练状态；如果是优化器，则会将其设置为训练状态。
        如果指定的方法是"train_model"，则会将所有需要训练的模型设置为训练状态。
        如果指定的方法是"freeze_model"，则会将所有不需要训练的模型设置为非训练状态。
        如果指定的方法是"optimizer"，则会将所有需要训练的优化器设置为训练状态。
        如果指定的方法是""，则会返回一个包含所有需要训练的对象和方法的元组列表。

        Args:
            args (Tuple[Any], optional): 可选参数，默认值为空元组，表示需要启用所有需要训练的对象和方法。支持多个参数，每个参数只能是一个模型、优化器或方法。

        Returns:
            Union[Tuple[Tuple[Any, str]], Enable]: 如果传入了参数，则返回一个包含所有需要训练的对象和方法的元组列表；否则返回一个Enable实例，用于启用所有需要训练的对象和方法。
        """
        # note: must keep the same model since actor_model, reward_model etc.
        # are property
        enable_map = {
            # maybe use `model: (pattern, enable_method, disable_method)``
            self.actor_model: "train_model",
            self.reference_model: "freeze_model",
            **({self.reward_model: "freeze_model"} if not self.args.use_rm_server else {}),
            self.policy_trainer.optimizer: "optimizer",
        }
        if self.args.rl_algorithm == "ppo":
            enable_map.update(
                {
                    self.reward_critic_model: "train_model",
                    self.value_trainer.optimizer: "optimizer",
                }
            )
        # if use an extra eval model to do eval/generation, switch on actor_model
        # and reward_critic_model; otherwise no need to switch
        if getattr(self.policy_trainer, "_inner_eval_model", None) is not None:
            enable_map.update({self.policy_trainer._inner_eval_model: "freeze_model"})
        if self.args.rl_algorithm == "ppo" and getattr(self.value_trainer, "_inner_eval_model", None) is not None:
            enable_map.update({self.value_trainer._inner_eval_model: "freeze_model"})
        # NOTE(GONGENLEI)： new offload
        objs = [(arg, enable_map.get(arg, "")) for arg in args if enable_map.get(arg, "") in self.args.offload_level]
        return Enable(objs)

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
                lambda tensor: tensor[i : i + micro_batch_size],
                ptx_batch,
            )
            micro_batches.append(micro_batch)
        return micro_batches

    @paddle.no_grad()
    @data_dispatch  # 3.10 static methods are now callable as regular functions.
    def split_rl_micro_batches(
        self,
        prompt_only_batch: Dict,
    ) -> List[Dict]:
        """Split a batch of RL samples into micro-batches."""
        total_batch_size = prompt_only_batch["input_ids"].shape[0]
        # micro_batch_size = self.args.per_device_train_batch_size
        per_device_rollout_batch_size = self.args.per_device_rollout_batch_size
        per_device_train_batch_size = self.args.per_device_train_batch_size
        micro_batches = []

        # TODO(guosheng): clean get_epoch_iterator:
        # 1. scope guard for offload, we would split post_rollout into multiple
        #    sub-methods to offload in-time
        # 2. decorate split_rl_micro_batches to automatically split/merge data

        self.timers and self.timers(get_timer_label(RolloutStages.ACTOR_MODEL_ENABLE_DISABLE)).start()
        with self.enable(self.actor_model, self.reference_model):
            # generate for multi batches and then disable FuseMT model
            cleanup_batches = []
            indices = []
            if self.args.use_rm_server:
                label_ids_batches = []
            self.timers and self.timers(get_timer_label(RolloutStages.GENERATE)).start()
            with infer_guard(self.policy_trainer):
                for i in range(0, total_batch_size, per_device_rollout_batch_size):
                    micro_batch = {}
                    micro_batch = map_structure(
                        lambda tensor: tensor[i : i + per_device_rollout_batch_size],
                        prompt_only_batch,
                    )
                    generated_batches = self.generate(micro_batch)

                    for batch in generated_batches:
                        cleanup_batches.extend(
                            [
                                process_row(
                                    row,
                                    remove_value=self.tokenizer.pad_token_id,
                                    remove_side="right",
                                )
                                for row in batch["input_ids"]
                            ]
                        )
                        if self.args.use_rm_server:
                            label_ids_batches.extend(
                                [
                                    process_row(
                                        row,
                                        remove_value=self.tokenizer.pad_token_id,
                                        remove_side="right",
                                    )
                                    for row in batch["label_ids"]
                                ]
                            )
                        indices.append(batch["index"])
            indices = np.concatenate(indices)
            self.timers and self.timers(get_timer_label(RolloutStages.GENERATE)).stop()
            # get log_probs for multi batches and then disable actor/refer rmodel
            origin_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "right"
            self.timers and self.timers(get_timer_label(RolloutStages.ROLLOUT_LOGPROB)).start()
            for i in range(0, len(cleanup_batches), per_device_train_batch_size):
                # position_ids is necessary for non-right padding
                # If using right padding source + left padding target, make padding positions
                # in source be 0, since reward model use position_ids plus with padding size
                # (number of 0s) in source to calculate end offsets.

                padding_strategy = "longest"
                padding_max_len = None

                if self._model_config.sequence_parallel:
                    padding_strategy = "max_length"
                    padding_max_len = self._model_config.max_sequence_length

                truncate_max_len = self._model_config.max_position_embeddings

                cur_batch = []
                for batch in cleanup_batches[i : i + per_device_train_batch_size]:
                    if len(batch) > truncate_max_len:
                        cur_batch.append(
                            self.tokenizer.truncate_sequences(
                                batch,
                                num_tokens_to_remove=len(batch) - truncate_max_len,
                                truncation_strategy="longest_first",
                            )[0]
                        )
                    else:
                        cur_batch.append(batch)

                input_ids = self.tokenizer.pad(
                    {"input_ids": cur_batch},
                    padding=padding_strategy,
                    max_length=padding_max_len,
                    return_attention_mask=False,
                )["input_ids"]

                sequence_mask = make_attention_mask(
                    input_ids,
                    pad_id=self.tokenizer.pad_token_id,
                    eos_id=None,
                    unk_id=self.tokenizer.unk_token_id,
                    causal_mask=False,
                ).cast(self._model_config.dtype)
                attention_mask = make_attention_mask(
                    input_ids,
                    pad_id=self.tokenizer.pad_token_id,
                    eos_id=None,
                    unk_id=self.tokenizer.unk_token_id,
                    causal_mask=True,
                ).cast(self._model_config.dtype)
                position_ids = make_position_ids(attention_mask)
                prompt = prompt_only_batch["input_ids"][i : i + per_device_train_batch_size]

                micro_batch = {
                    "prompt": prompt,
                    "input_ids": input_ids,
                    "sequence_mask": sequence_mask,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "index": indices[i : i + per_device_train_batch_size],
                    **(
                        {"label_ids": label_ids_batches[i : i + per_device_train_batch_size]}
                        if self.args.use_rm_server
                        else {}
                    ),
                }
                micro_batch.update(self.rollout_logprob(**micro_batch))
                micro_batches.append(micro_batch)
            self.timers and self.timers(get_timer_label(RolloutStages.ROLLOUT_LOGPROB)).stop()
            self.tokenizer.padding_side = origin_padding_side
        if self.timers:
            self.timers(get_timer_label(RolloutStages.ACTOR_MODEL_ENABLE_DISABLE)).stop()
            self.timers(get_timer_label(RolloutStages.ACTOR_MODEL_ENABLE_DISABLE)).elapsed_ -= self.timers(
                get_timer_label(RolloutStages.GENERATE)
            ).elapsed_
            self.timers(get_timer_label(RolloutStages.ACTOR_MODEL_ENABLE_DISABLE)).elapsed_ -= self.timers(
                get_timer_label(RolloutStages.ROLLOUT_LOGPROB)
            ).elapsed_

        # get reward/value for multi batches and then disable reward/value model
        self.timers and self.timers(get_timer_label(RolloutStages.REWARD_MODEL_ENABLE_DISABLE)).start()
        with self.enable(
            self.reward_critic_model if self.args.rl_algorithm == "ppo" else None,
            self.reward_model if not self.args.use_rm_server else None,
        ):
            self.timers and self.timers(get_timer_label(RolloutStages.ROLLOUT_REWARD_VALUE)).start()
            for micro_batch in micro_batches:
                micro_batch.update(self.rollout_reward_value(**micro_batch))
            self.timers and self.timers(get_timer_label(RolloutStages.ROLLOUT_REWARD_VALUE)).stop()
        if self.timers:
            self.timers and self.timers(get_timer_label(RolloutStages.REWARD_MODEL_ENABLE_DISABLE)).stop()
            self.timers(get_timer_label(RolloutStages.REWARD_MODEL_ENABLE_DISABLE)).elapsed_ -= self.timers(
                get_timer_label(RolloutStages.ROLLOUT_REWARD_VALUE)
            ).elapsed_

        micro_batches = self.normalize_batch_data(micro_batches, use_tgt_len_value=self.args.use_tgt_len_value)

        # size of micro_batches (num of training batch) would be:
        # per_device_prompt_batch_size * num_return_sequences // per_device_train_batch_size
        # micro_batches = [self.post_rollout(**micro_batch) for micro_batch in micro_batches]
        return micro_batches

    @paddle.no_grad()
    def generate(self, prompt_only_batch: Dict, do_eval=False) -> List[Dict[str, Any]]:
        """Rollout a batch of experiences."""
        input_ids = prompt_only_batch["input_ids"]
        attention_mask = prompt_only_batch["attention_mask"]
        if do_eval:
            train_num_return_sequences = self.args.num_return_sequences
            self.args.num_return_sequences = 1

        position_ids = (
            prompt_only_batch["position_ids"]
            if "position_ids" in prompt_only_batch
            else make_position_ids(attention_mask)
        )

        if self.args.num_return_sequences > 1:
            input_ids = input_ids.repeat_interleave(self.args.num_return_sequences, axis=0)
            raw_dtype = attention_mask.dtype
            attention_mask = (
                attention_mask.cast("int32").repeat_interleave(self.args.num_return_sequences, axis=0).cast(raw_dtype)
            )
            position_ids = position_ids.repeat_interleave(self.args.num_return_sequences, axis=0)

        sequences = self.actor_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            generation_config=self.generation_config,
            synced_gpus=ShardingOption.FULL_SHARD in self.policy_trainer.args.sharding,
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
                "attention_mask": make_attention_mask(
                    seq,
                    pad_id=self.tokenizer.pad_token_id,
                    eos_id=None,
                    unk_id=self.tokenizer.unk_token_id,
                    causal_mask=True,
                ).cast(self._model_config.dtype),
                "sequence_mask": make_attention_mask(
                    seq,
                    pad_id=self.tokenizer.pad_token_id,
                    eos_id=None,
                    unk_id=self.tokenizer.unk_token_id,
                    causal_mask=False,
                ).cast(self._model_config.dtype),
            }
            for idx, seq in enumerate(sequences)
        ]

    @paddle.no_grad()
    def rollout_logprob(
        self,
        input_ids: paddle.Tensor,
        attention_mask: paddle.Tensor,
        position_ids: paddle.Tensor = None,
        **kwargs,
    ) -> Dict[str, paddle.Tensor]:
        """
        计算rollout过程中每个token的log probability。

            Args:
                input_ids (paddle.Tensor, shape [batch_size, sequence_length]):
                    输入序列，其中每个元素都是一个int，表示各自token的ID。
                attention_mask (paddle.Tensor, shape [batch_size, sequence_length]):
                    输入序列的attention mask，其中每个元素为0或1，用于指示哪些tokens应该被模型考虑。
                position_ids (paddle.Tensor, optional, shape [batch_size, sequence_length], defaults to None):
                    输入序列中每个token的位置ID，默认为None。
                kwargs (Dict[str, Any], optional, defaults to {}):
                    可选参数，目前未使用。

            Returns:
                Dict[str, paddle.Tensor]:
                    - log_probs (paddle.Tensor, shape [batch_size, sequence_length - 1]):
                        每个token在rollout过程中的log probability。
                    - ref_log_probs (paddle.Tensor, shape [batch_size, sequence_length - 1]):
                        每个token在rollout过程中的reference log probability。

            Raises:
                None.
        """
        # pipe model outputs a logits tensor with LMHead, while non-pipe model
        # outputs a tuple with logits tensor as the only one element.

        logits = self.actor_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            # return_dict=True,
        )  # .logits
        if not isinstance(logits, paddle.Tensor):
            logits = logits[0]  # [2, 355, 12544]
        ref_logits = self.reference_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            # return_dict=True,
        )  # .logits

        if not isinstance(ref_logits, paddle.Tensor):
            ref_logits = ref_logits[0]  # [2, 355, 12544]
        logits = logits / self.args.temperature if self.args.temperature > 0.0 else logits
        ref_logits = ref_logits / self.args.temperature if self.args.temperature > 0.0 else ref_logits
        if self.actor_model.config.tensor_parallel_degree > 1 and self.actor_model.config.tensor_parallel_output:
            log_probs = (
                -ParallelCrossEntropy()(logits[:, :-1].astype("float32"), input_ids[:, 1:])
                .squeeze(axis=-1)
                .astype(logits.dtype)
            )
        else:
            log_probs = gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])

        if (
            self.reference_model.config.tensor_parallel_degree > 1
            and self.reference_model.config.tensor_parallel_output
        ):
            ref_log_probs = (
                -ParallelCrossEntropy()(ref_logits[:, :-1].astype("float32"), input_ids[:, 1:])
                .squeeze(axis=-1)
                .astype(ref_logits.dtype)
            )
        else:
            ref_log_probs = gather_log_probabilities(ref_logits[:, :-1], input_ids[:, 1:])

        return {"log_probs": log_probs, "ref_log_probs": ref_log_probs}

    @paddle.no_grad()
    def rollout_reward_value(
        self,
        input_ids: paddle.Tensor,
        attention_mask: paddle.Tensor,
        position_ids: paddle.Tensor = None,
        **kwargs,
    ) -> Dict[str, paddle.Tensor]:
        """
        根据输入的序列，计算每个时间步骤的奖励值和奖励得分。如果模型使用了不同的tokenizer，则先将输入序列转换为目标tokenizer的格式。

            Args:
                input_ids (paddle.Tensor): shape=[batch_size, seq_len], 输入序列的ID，取值范围是[0, vocabulary_size - 1]。
                attention_mask (paddle.Tensor): shape=[batch_size, seq_len], 输入序列的注意力掩码，取值范围是{0, 1}。
                position_ids (Optional, paddle.Tensor, optional): shape=[batch_size, seq_len], 输入序列的位置ID，默认为None。
                kwargs (Dict, optional): 其他可选参数，包括：
                    reward_tokenizer (Tokenizer, optional): 奖励tokenizer，默认为None，表示使用与模型相同的tokenizer。

            Returns:
                Dict[str, paddle.Tensor]: 返回一个字典，包含两个键值对：
                    rewards (paddle.Tensor): shape=[batch_size, seq_len], 每个时间步骤的奖励得分，取值范围是[-inf, inf]。
                    reward_values (paddle.Tensor): shape=[batch_size, seq_len-1], 每个时间步骤的奖励值，取值范围是[0, inf]。
        """
        if not self.args.use_rm_server:
            if self.reward_tokenizer is not self.tokenizer:
                # right padding
                reward_tokenize_output = batch_retokenize(
                    input_ids,
                    src_tokenizer=self.tokenizer,
                    dest_tokenizer=self.reward_tokenizer,
                )
                reward_input_ids = reward_tokenize_output["input_ids"]
                reward_attention_mask = reward_tokenize_output["attention_mask"]
                reward_position_ids = reward_tokenize_output["position_ids"]
            else:
                reward_input_ids = input_ids
                reward_attention_mask = attention_mask
                reward_position_ids = position_ids

            # .end_scores
            reward_score = self.reward_model(
                reward_input_ids,
                attention_mask=reward_attention_mask,
                position_ids=reward_position_ids,
                # return_dict=True,
            )[1]
        else:
            prompt_len = kwargs["prompt"].shape[-1]
            if "label_ids" not in kwargs:
                raise ValueError("Rule-based reward needs labels.")
            src = self.tokenizer.batch_decode(input_ids[:, :prompt_len], skip_special_tokens=True)
            tgt = self.tokenizer.batch_decode(kwargs["label_ids"], skip_special_tokens=True)
            response = self.tokenizer.batch_decode(input_ids[:, prompt_len:], skip_special_tokens=True)
            reward_score = self.request_reward_server(src, tgt, response)

        reward_score = reward_score.squeeze(axis=-1)

        if self.args.rl_algorithm == "grpo":
            return {"rewards": reward_score}

        # .scores
        reward_value = self.reward_critic_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            # return_dict=True,
        )[0]
        reward_value = reward_value.squeeze(axis=-1)
        reward_value = reward_value[:, :-1]

        return {"rewards": reward_score, "reward_values": reward_value}

    def request_reward_server(self, src, tgt, response):
        data = {"src": src, "tgt": tgt, "response": response}

        def post():
            try:
                res = requests.post(self.reward_server, json=data)
                result = json.loads(res.text)
                reward_score = paddle.to_tensor(result["score"], dtype=self._model_config.dtype)
            except:
                logger.warning("Request reward server failed and rewards_score will be set zero.")
                reward_score = paddle.zeros(len(response), dtype=self._model_config.dtype)
            return reward_score

        try:
            hcg = fleet.get_hybrid_communicate_group()
            tp_group = hcg.get_model_parallel_group()
            nranks = tp_group.nranks
            tp_rank = hcg.get_model_parallel_rank()
        except:
            nranks = 1
            tp_rank = 0

        if nranks == 1:
            reward_score = post()
        else:
            if tp_rank == 0:
                reward_score = post()
            else:
                reward_score = paddle.empty(shape=[len(response)], dtype=self._model_config.dtype)
            paddle.distributed.barrier(tp_group)
            paddle.distributed.broadcast(reward_score, src=tp_group.ranks[0], group=tp_group)

        return reward_score.unsqueeze(-1)

    @paddle.no_grad()
    def normalize_batch_data(
        self,
        rl_batches: List[Dict[str, paddle.Tensor]],
        use_tgt_len_value: bool = False,
    ) -> Dict[str, Any]:
        """
        data dispatch comm among devices needs padding, while the lengths of
        all data fields are different and related, and it's hard to pad.
        """
        for rl_batch in rl_batches:
            rl_batch["ori_rewards"] = rl_batch["rewards"].clone()

        use_reward_normalization = self.args.normalize_reward
        use_advantage_normalization = self.args.normalize_advantage

        if use_reward_normalization:
            batch_rewards_list = [rl_batch["rewards"] for rl_batch in rl_batches]
            batch_rewards = paddle.concat(batch_rewards_list, axis=0)
            batch_rewards = batch_rewards.cast(paddle.float32)

            try:
                hcg = fleet.get_hybrid_communicate_group()
                sd_group = hcg.get_sharding_parallel_group()
                dp_group = hcg.get_data_parallel_group()

                if sd_group.nranks > 1:
                    all_gather_batch_rewards = []
                    dist.all_gather(all_gather_batch_rewards, batch_rewards, group=sd_group)
                    batch_rewards = paddle.flatten(paddle.stack(all_gather_batch_rewards))
                if dp_group.nranks > 1:
                    all_gather_batch_rewards = []
                    dist.all_gather(all_gather_batch_rewards, batch_rewards, group=dp_group)
                    batch_rewards = paddle.flatten(paddle.stack(all_gather_batch_rewards))
            except AttributeError:
                pass

            batch_rewards_mean = batch_rewards.mean()
            # batch_rewards_std = batch_rewards.std()
            batch_rewards_var = batch_rewards.var()

            current_batch_num = batch_rewards.shape[0]
            delta = batch_rewards_mean - self.reward_mean
            total_batch_num = self.sample_batch_num + current_batch_num

            new_mean = self.reward_mean + delta * current_batch_num / total_batch_num
            m_a = self.reward_var * self.sample_batch_num
            m_b = batch_rewards_var * current_batch_num
            m2 = m_a + m_b + paddle.square(delta) * (self.sample_batch_num * current_batch_num / total_batch_num)
            new_var = m2 / total_batch_num

            self.reward_mean = new_mean
            self.reward_var = new_var
            self.sample_batch_num = total_batch_num

            for rl_batch in rl_batches:
                reward_mean = self.reward_mean.cast(paddle.bfloat16)
                reward_std = self.reward_var.sqrt().cast(paddle.bfloat16)
                rl_batch["rewards"] = (rl_batch["rewards"] - reward_mean) / (reward_std + 1e-8)

        for rl_batch in rl_batches:
            prompt = rl_batch["prompt"]  # length: src
            attention_mask = rl_batch["attention_mask"]  # length: src + tgt
            if len(attention_mask.shape) == 4:
                # use padding mask instead of causal mask
                attention_mask = rl_batch["sequence_mask"]  # length: src + tgt
            old_log_probs = rl_batch["log_probs"]  # length: src + tgt -1
            ref_log_probs = rl_batch["ref_log_probs"]  # length: src + tgt -1
            rewards = rl_batch["rewards"]  # length: 1
            if self.args.rl_algorithm == "ppo":
                old_reward_values = rl_batch["reward_values"]  # length: src + tgt -1

            start = prompt.shape[-1] - 1
            # sequence_mask is for label masking, make source be masked out
            # clone to avoid to change attention_mask
            sequence_mask = attention_mask[:, 1:].clone()  # length: src + tgt -1
            sequence_mask[:, :start] = False
            if use_tgt_len_value:
                ref_log_probs = ref_log_probs[:, start:].contiguous()
                old_log_probs = old_log_probs[:, start:].contiguous()
                if self.args.rl_algorithm == "ppo":
                    old_reward_values = old_reward_values[:, start:].contiguous()
                sequence_mask = sequence_mask[:, start:].contiguous()
            if self.args.rl_algorithm == "grpo":
                eos_mask = (rl_batch["input_ids"] != self.tokenizer.pad_token_id)[:, 1:].to(old_log_probs.dtype)
                if use_tgt_len_value:
                    eos_mask = eos_mask[:, start:].contiguous()
                reward_advantages = compute_grpo_advantages(
                    rewards, rl_batch["index"], eos_mask, old_log_probs.shape[-1]
                )
            elif self.args.rl_algorithm == "ppo":
                rewards_with_kl, kl_rewards = self.add_kl_divergence_regularization(
                    None,  # prompt,
                    old_log_probs,
                    ref_log_probs,
                    rewards,
                    sequence_mask,
                )  # length: tgt if use_tgt_len_value src + tgt -1
                reward_advantages, reward_returns = self.get_advantages_and_returns(
                    old_reward_values,
                    rewards_with_kl,
                    sequence_mask,
                    start=0 if use_tgt_len_value else start,
                    use_tgt_len_return=use_tgt_len_value,
                )  # length: tgt if use_tgt_len_value src + tgt -1
            else:
                raise ValueError(f"Unknown rl_algorithm: {self.args.rl_algorithm}")

            rl_batch.update(
                {
                    "log_probs": old_log_probs,
                    "reward_advantages": reward_advantages,
                    "sequence_mask": sequence_mask,
                    "ref_log_probs": ref_log_probs,
                    "rewards": rewards,
                }
            )
            if self.args.rl_algorithm == "ppo":
                rl_batch.update(
                    {
                        "reward_values": old_reward_values,
                        "reward_returns": reward_returns,
                        "kl_rewards": kl_rewards,
                        "rewards_with_kl": rewards_with_kl,
                    }
                )

            # pop out to reduce data dispatch comm overhead
            rl_batch.pop("prompt")

        if use_advantage_normalization:
            all_advantages_list = []
            for rl_batch in rl_batches:
                sequence_mask = rl_batch["sequence_mask"].cast(paddle.int64)  # length: src + tgt
                advantages = rl_batch["reward_advantages"]
                all_advantages_list.append(advantages[sequence_mask != 0])
            all_advantages = paddle.concat(all_advantages_list, axis=0)
            all_advantages = all_advantages.cast(paddle.float32)

            try:
                hcg = fleet.get_hybrid_communicate_group()
                sd_group = hcg.get_sharding_parallel_group()
                dp_group = hcg.get_data_parallel_group()

                if sd_group.nranks > 1:
                    object_list = []
                    dist.all_gather_object(object_list, all_advantages.tolist(), group=sd_group)
                    flattened_data = [item for sublist in object_list for item in sublist]
                    all_advantages = paddle.to_tensor(flattened_data, dtype="float32")
                if dp_group.nranks > 1:
                    object_list = []
                    dist.all_gather_object(object_list, all_advantages.tolist(), group=dp_group)
                    flattened_data = [item for sublist in object_list for item in sublist]
                    all_advantages = paddle.to_tensor(flattened_data, dtype="float32")
            except AttributeError:
                pass
            all_advantages_mean = all_advantages.mean()
            all_advantages_std = all_advantages.std()
            for rl_batch in rl_batches:
                all_advantages_mean = all_advantages_mean.cast(paddle.bfloat16)
                all_advantages_std = all_advantages_std.cast(paddle.bfloat16)
                rl_batch["reward_advantages"] = (rl_batch["reward_advantages"] - all_advantages_mean) / (
                    all_advantages_std + 1e-8
                )
                rl_batch["reward_advantages"] = rl_batch["reward_advantages"] * rl_batch["sequence_mask"]

        return rl_batches


@paddle.no_grad()
def compute_grpo_advantages(
    rewards: paddle.Tensor,
    index: np.ndarray,
    sequence_mask: paddle.Tensor,
    response_length: int,
    epsilon: float = 1e-6,
):
    """
    计算每个prompt的GRPO优势。

    Args:
        rewards (paddle.Tensor, shape=[batch_size]): 回报，单位为float。
        index (np.ndarray, shape=[batch_size]): 每个样本对应的prompt索引，类型为int。
        sequence_mask (paddle.Tensor, shape=[batch_size, response_length]): 序列掩码，用于标记每个时间步是否有效，类型为bool。
        response_length (int): 每个样本的响应长度。
        epsilon (float, optional, default=1e-6): 避免除以0的值，默认为1e-6。

    Returns:
        rewards (paddle.Tensor, shape=[batch_size, response_length]): GRPO优势，单位为float。

    Raises:
        ValueError (ValueError): 如果没有在给定的prompt索引中有分数。
    """
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    batch_size = rewards.shape[0]

    for i in range(batch_size):
        id2score[index[i]].append(rewards[i])
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = paddle.to_tensor(0.0, dtype=rewards.dtype)
            id2std[idx] = paddle.to_tensor(1.0, dtype=rewards.dtype)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = paddle.mean(paddle.stack(id2score[idx]))
            id2std[idx] = paddle.std(paddle.stack(id2score[idx]))
        else:
            raise ValueError(f"No score in prompt index: {idx}")
    for i in range(batch_size):
        rewards[i] = (rewards[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    rewards = rewards.unsqueeze(-1).tile([1, response_length]) * sequence_mask
    return rewards
