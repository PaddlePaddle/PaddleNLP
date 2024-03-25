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
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import paddle
import paddle.distributed as dist
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
from paddlenlp.trainer.plugins.unified_checkpoint import flatten_list
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
from paddlenlp.trainer.utils.helper import nested_broadcast_tensor_with_empty
from paddlenlp.utils.distributed import distributed_gather

global_dev_id = 0 if paddle.get_device() == "cpu" else int(paddle.get_device().split(":")[1])
from paddlenlp.transformers import PretrainedModel, PretrainedTokenizer


def offload_tensor_to_cpu(tensors):
    if isinstance(tensors, dict):
        for _, v in tensors.items():
            offload_tensor_to_cpu(v)
    elif isinstance(tensors, paddle.Tensor):
        if tensors.place.is_gpu_place():
            cpu_tensor = tensors._copy_to(paddle.CUDAPinnedPlace(), False)
            tensors.value().get_tensor()._share_data_with(cpu_tensor.value().get_tensor())
    else:
        logger.warning(f"Can't parse for type {type(tensors)}")
        return tensors


def reload_tensor_to_gpu(tensors):
    if isinstance(tensors, dict):
        for _, v in tensors.items():
            reload_tensor_to_gpu(v)
    elif isinstance(tensors, paddle.Tensor):
        if tensors._is_initialized() and not tensors.place.is_gpu_place():
            gpu_tensor = tensors._copy_to(paddle.CUDAPlace(global_dev_id), False)
            tensors.value().get_tensor()._share_data_with(gpu_tensor.value().get_tensor())
    else:
        logger.warning(f"Can't parse for type {type(tensors)}")
        return tensors


def cleanup_tensor_space(tensors):
    if isinstance(tensors, dict):
        for _, v in tensors.items():
            cleanup_tensor_space(v)
    elif isinstance(tensors, paddle.Tensor):
        tensors._clear_data()
    else:
        logger.warning(f"Can't parse for type {type(tensors)}")
        return tensors


def data_group_split(tensors, group):
    if group is None:
        return tensors
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(data_group_split(t, group) for t in tensors)
    elif isinstance(tensors, dict):
        new_dict = {}
        for k, v in tensors.items():
            new_dict[k] = data_group_split(v, group)
        return new_dict
    elif isinstance(tensors, paddle.Tensor):
        return tensors.split(group.nranks)[group.rank]
    else:
        logger.warning(f"Can't parse for type {type(tensors)}")
        return tensors


def data_group_merge(tensors, group):
    if group is None:
        return tensors

    if isinstance(tensors, (list, tuple)):
        return type(tensors)(data_group_merge(t, group) for t in tensors)
    elif isinstance(tensors, dict):
        new_dict = {}
        for k, v in tensors.items():
            new_dict[k] = data_group_merge(v, group)
        return new_dict
    elif isinstance(tensors, paddle.Tensor):
        tensor_list = []
        all_gather_nd(tensor_list, tensors, group=group, padded=True)
        return paddle.concat(tensor_list)
    else:
        logger.warning(f"Can't parse for type {type(tensors)}")
        return tensors


def group_rank_guard(group, rank=0):
    def decorator(func):
        def wrapper_func(*args, **kwargs):
            if group.rank == rank:
                ret = func(*args, **kwargs)
                dist.barrier()
            else:
                ret = None
                dist.barrier()
            ret = nested_broadcast_tensor_with_empty(ret, group=group)
            return ret

        return wrapper_func

    return decorator


def repad_rl_batches(batches, input_lengths):
    if "position_ids" in batches:
        v = batches["position_ids"]
        for x in range(v.shape[0]):
            v[x, input_lengths[x] :] = 1
        batches["position_ids"] = v
    for key in list(batches.keys()):
        if batches[key].shape[0] != input_lengths.shape[0]:
            batches[key] = batches[key].mean()

    return batches


# https://stackoverflow.com/questions/12594148/skipping-execution-of-with-block
class SkipWithBlock(Exception):
    pass


class SkipContextManager:
    def __init__(self, skip):
        self.skip = skip

    def __enter__(self):
        if self.skip:
            sys.settrace(lambda *args, **keys: None)
            frame = sys._getframe(1)
            frame.f_trace = self.trace

    def trace(self, frame, event, arg):
        raise SkipWithBlock()

    def __exit__(self, type, value, traceback):
        if type is None:
            return  # No exception
        if issubclass(type, SkipWithBlock):
            return True  # Suppress special SkipWithBlock exception


def all_gather_nd(tensor_list, tensor, group=None, padded=False):
    """
    Gathers tensor arrays of different lengths in a list.
    The length dimension is 0. This supports any number of extra dimensions in the tensors.
    All the other dimensions should be equal between the tensors.

    Args:
        tensor (Tensor): Tensor to be broadcast from current process.

    Returns:
        (Tensor): output list of tensors that can be of different sizes
    """
    if len(tensor.shape) == 0:
        tensor = tensor.reshape([1])
        dist.all_gather(tensor_list, tensor, group=group)
        return tensor_list

    world_size = group.nranks
    local_size = paddle.to_tensor(tensor.shape, place=tensor.place)
    all_sizes = [paddle.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size, group=group)

    # max_length = max(size[0] for size in all_sizes)

    # length_diff = max_length.item() - local_size[0].item()
    # if length_diff:
    #     pad_size = (length_diff, *tensor.size()[1:])
    #     padding = paddle.zeros(pad_size, place=tensor.place(), dtype=tensor.dtype)
    #     tensor = padle.concat((tensor, padding))

    max_length = max(size[-1] for size in all_sizes)

    length_diff = max_length.item() - local_size[-1].item()
    if length_diff:
        pad_size = (*tensor.shape[:-1], length_diff)
        padding = paddle.zeros(pad_size, dtype=tensor.dtype)
        tensor = paddle.concat([tensor, padding], axis=-1)

    all_tensors_padded = []
    dist.all_gather(all_tensors_padded, tensor, group=group)
    # all_tensors = []
    if padded:
        tensor_list.extend(all_tensors_padded)
        return all_tensors_padded

    for tensor_, size in zip(all_tensors_padded, all_sizes):
        tensor_list.append(tensor_[..., : size[-1]])
    return tensor_list


def export_evaluate_model(self: Trainer, train_model, eval_model, **kwargs):
    if eval_model is None:
        return None

    with_offload = kwargs.pop("with_offload", False)
    train_tp_size = max(train_model.config.tensor_parallel_degree, 1)
    eval_tp_size = max(eval_model.config.tensor_parallel_degree, 1)
    eval_tp_rank = max(eval_model.config.tensor_parallel_rank, 0)

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    pp_group = hcg.get_pipe_parallel_group()
    sd_group = hcg.get_sharding_parallel_group()
    dp_group = hcg.get_data_parallel_group()

    global_rank = paddle.distributed.get_rank()

    train_state_dict = train_model.state_dict()
    eval_state_dict = eval_model.state_dict()

    if dp_group.rank <= 0 and sd_group.rank <= 0:
        train_pp_size = pp_group.nranks
        if eval_tp_size > 1 and train_tp_size != eval_tp_size:
            raise ValueError("Only support for the same tensor_parallel_degree for train and eval model for now.")

        # 单卡情况
        # tp->single
        # tp+pp -> single
        if eval_tp_size == 1:
            if train_pp_size == 1 and train_tp_size > 1:
                # tp ->single
                logger.error("using tp to single eval model.")
                # state = train_model.merge_tensor_parallel()
                tp_actions = train_model.get_tensor_parallel_convert_actions(
                    train_model.config,
                    loaded_state_dict_keys=eval_state_dict.keys(),
                    is_split=False,
                    ignore_error=False,
                )

                is_dst = global_rank == 0
                for key in eval_state_dict.keys():
                    tensor = train_state_dict[key]
                    if key in tp_actions:
                        ret = distributed_gather(tensor, dst=0, group=tp_group, offload=False)
                        action = tp_actions.pop(key)
                        tensor = action(ret) if is_dst else None
                    else:
                        tensor = tensor._copy_to(paddle.CPUPlace(), False) if is_dst else None

                    if tensor is not None:
                        eval_state_dict[key].set_value(tensor)

                    if not eval_state_dict[key]._is_initialized():
                        v = eval_state_dict[key]
                        t = paddle._C_ops.full_like(v, 0, v.dtype, paddle.CUDAPlace(global_dev_id))
                        v.get_tensor()._share_data_with(t.get_tensor())

                    if with_offload:
                        offload_tensor_to_cpu(train_state_dict[key])
            else:
                # single to single
                # tp+pp -> single
                raise ValueError("Not support yet.")

        def create_send_recv_table(train_keys, eval_keys):
            recv_table = []
            send_table = []
            if pp_group.rank == 0:
                for key in eval_keys:
                    recv_table.append((key, global_rank))

            for key in train_keys:
                send_table.append((key, global_rank))

            all_recv, all_send = [], []
            paddle.distributed.all_gather_object(all_recv, [recv_table], group=pp_group)
            paddle.distributed.all_gather_object(all_send, [send_table], group=pp_group)
            all_recv = flatten_list(all_recv)
            all_send = flatten_list(all_send)

            send_dict = {}
            for k, v in all_send:
                send_dict[k] = v

            table = []
            for k, v in all_recv:
                # key, send, recv
                table.append([k, send_dict.pop(k), v])
            assert len(send_dict) == 0, f"Some key can't be recv {send_dict.keys()}"
            return table

            # pp0tp0 -> pp0tp0
            # pp0tp1 -> pp0tp1
            # pp1tp0 -> pp0tp0
            # pp1tp1 -> pp0tp1

        # tp情况
        # tp+pp->tp
        self.timers and self.timers("export-merge-pp").start()
        if eval_tp_size > 1 and train_pp_size > 1:
            table = create_send_recv_table(train_state_dict.keys(), eval_state_dict.keys())

            for key, src_rank, dst_rank in table:
                # Init tensor for model is cleaned
                if not eval_state_dict[key]._is_initialized():
                    v = eval_state_dict[key]
                    t = paddle._C_ops.full_like(v, 0, v.dtype, paddle.CUDAPlace(global_dev_id))
                    v.get_tensor()._share_data_with(t.get_tensor())

                if src_rank == dst_rank and global_rank == src_rank:
                    eval_state_dict[key].copy_(train_state_dict[key], True)
                else:
                    if global_rank == src_rank:
                        dist.stream.send(train_state_dict[key], dst=dst_rank)

                    if global_rank == dst_rank:
                        dist.stream.recv(eval_state_dict[key], src=src_rank)

                # Offload train model if need
                if global_rank == src_rank and with_offload:
                    offload_tensor_to_cpu(train_state_dict[key])

        self.timers and self.timers("export-merge-pp").stop()
        self.timers and self.timers("export-broadcast-pp").start()
        if pp_group.nranks > 1:
            paddle.distributed.parallel.sync_params_buffers(
                eval_model, comm_group=pp_group, src_rank=pp_group.ranks[0], fuse_params=False
            )
        self.timers and self.timers("export-broadcast-pp").stop()
    else:
        # 其他 DP rank 的state dict, 适配 offload 和初始化
        self.timers and self.timers("export-offload-and-init").start()
        if with_offload:
            for key in list(train_state_dict.keys()):
                offload_tensor_to_cpu(train_state_dict[key])
        for k, v in eval_state_dict.items():
            if not v._is_initialized():
                t = paddle._C_ops.full_like(v, 0, v.dtype, paddle.CUDAPlace(global_dev_id))
                v.get_tensor()._share_data_with(t.get_tensor())
        self.timers and self.timers("export-offload-and-init").stop()

    paddle.distributed.barrier()
    self.timers and self.timers("export-broadcast-sd-dp").start()
    if eval_tp_size == 1:
        for _, tensor in eval_state_dict.items():
            paddle.distributed.broadcast(tensor, src=0, group=None, sync_op=True)
    else:
        if sd_group.nranks > 1:
            if dp_group.rank <= 0:
                paddle.distributed.parallel.sync_params_buffers(
                    eval_model, comm_group=sd_group, src_rank=sd_group.ranks[0], fuse_params=False
                )
        if dp_group.nranks > 1:
            paddle.distributed.parallel.sync_params_buffers(
                eval_model, comm_group=dp_group, src_rank=dp_group.ranks[0], fuse_params=False
            )
    self.timers and self.timers("export-broadcast-sd-dp").stop()
    # paddle.save(eval_state_dict, f"./tmp/eval_{sd_group.rank}_tp_{eval_tp_rank}_pp_{pp_group.rank}.pdparams")
    # paddle.save(train_state_dict, f"./tmp/train_{sd_group.rank}_tp_{tp_group.rank}_pp_{pp_group.rank}.pdparams")
    # paddle.distributed.barrier()
    # exit(-1)

    old_dp_workers = self.args.world_size // (max(sd_group.nranks, 1) * max(dp_group.nranks, 1))
    group_nums = self.args.logical_process_index // old_dp_workers * eval_tp_size + eval_tp_rank

    if not hasattr(self, "_policy_model_eval_group") or self._policy_model_eval_group is None:
        self._policy_model_eval_group = create_data_trans_group(global_rank, group_nums)

    return None


def create_data_trans_group(global_rank, group_nums):
    all_split_table = []
    paddle.distributed.all_gather_object(all_split_table, [(global_rank, group_nums)])
    all_split_table = flatten_list(all_split_table)
    split_dict = {}
    for k, v in all_split_table:
        split_dict[k] = v

    split_ranks = {}
    for k, v in all_split_table:
        if v in split_ranks:
            split_ranks[v].append(k)
        else:
            split_ranks[v] = [k]

    group = None
    for k, ranks in split_ranks.items():
        gp = paddle.distributed.new_group(ranks=ranks)
        if global_rank in ranks:
            group = gp

    return group


Trainer.export_evaluate_model = export_evaluate_model


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

        self.use_fusemt = getattr(args, "use_fusemt", False)
        # ablout 4s slower than infer generation without ema
        self.use_ema = getattr(args, "use_ema", False)
        self.shard_ema = getattr(args, "shard_ema", False)
        self.offload_ema = getattr(args, "offload_ema", True)
        self.ema_beta = getattr(args, "ema_beta", 0.992)

    def create_criterion(self):
        """
        create loss using `loss_cls` for trainer. It would use a wrapped loss_cls
        whose label arguments are merged into one argument, this is useful to
        PipelineParallel and trainer.criterion which limit loss format.
        """
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

    def set_eval_model(self, model):
        if model is None:
            logger.warning("use None to set eval model for trainer and it would be ignored")
        else:
            self._inner_eval_model = model

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

    def get_train_step_vars(self, vars: Dict = None) -> Dict:
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

        new_train_step_vars = super().full_training_step(inputs, **train_step_vars)

        # minimally update
        train_step_vars = self.get_train_step_vars(
            {"step_control": new_train_step_vars["step_control"], loss_name: new_train_step_vars["tr_loss"]}
        )
        if loss_name != "tr_loss":
            train_step_vars.pop("tr_loss")

        self.mark_step_loss(loss_name)

        if self.use_ema and self.is_accumulation_step:
            # TODO(guosheng): assume rollout next thus make ema weights on gpu,
            # but may not, maybe need a way to specify it.
            self.ema_update(beta=self.ema_beta, offload_ema=self.offload_ema, offload_model=not self.offload_ema)

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
                    mix_loss = paddle.empty(shape=[self.args.gradient_accumulation_steps], dtype=paddle.float32)
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
        rank_param_names = [p.name for p in optimizer._rank2params[optimizer._sharding_rank]]
        structured_names = []
        # for pipeline model, use `model.state_dict()` would auto map param name
        # for name, p in model.named_parameters():
        for name, p in model.state_dict().items():
            if p.name in rank_param_names:
                structured_names.append(name)
        return structured_names

    def get_master_weight_state_dict(self, model, optimizer):
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
        for k, v in self.bak_state_dict.items():
            value = v.cuda()
            value._share_buffer_to(v)
            if self.offload_ema:  # ema weights always in pin_memory in fact
                ema_v = self.ema_state_dict[k]
                ema_value = ema_v.pin_memory()
                ema_value._share_buffer_to(ema_v)


class ema(paddle.no_grad.__mro__[1]):
    def __init__(self, trainer: StepTrainer):
        self.trainer = trainer

    def __enter__(self):
        trainer = self.trainer
        if trainer.use_ema and not hasattr(trainer, "ema_state_dict"):
            # call ema_init here since it should be called after model and
            # optimizer are created and wrapped
            trainer.ema_init(
                offload_ema=trainer.offload_ema, offload_model=not trainer.offload_ema, shard_ema=trainer.shard_ema
            )
        if self.trainer.use_ema:
            self.trainer.ema_apply()

    def __exit__(self, *args):
        if self.trainer.use_ema:
            self.trainer.ema_restore()


class PolicyTrainer(StepTrainer):
    loss_cls = RLHFPPOMixedLoss

    def loss_identifier(self, inputs: Dict) -> str:
        labels = inputs.get("labels", None)
        if labels is not None:  # use ptx
            loss_name = "ptx_loss"
        else:
            loss_name = "actor_loss"
        return loss_name


class ValueTrainer(StepTrainer):
    loss_cls = RLHFValueLoss
    # define loss name for logging
    loss_identifier = lambda self, inputs: "reward_critic_loss"


class PPOMetric:
    metric_names = [
        "train/" + name
        for name in [
            "actor_loss",
            "ptx_loss",
            "reward_critic_loss",
            "reward",
            "kl_divergence",
            "mean_generated_length",
            "max_generated_length",
        ]
    ]
    metric_ops = ["mean", "mean", "mean", "mean", "mean", "mean", "max"]

    def __init__(self, freq, use_stack=True):
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
        if self.counter + 1 == self.freq:
            from paddlenlp.trainer.utils import distributed_concat

            metrics = distributed_concat(self.metrics)
            out_metrics = {}
            if self.use_stack:
                mean_metric = metrics.mean(0)
                max_metric = metrics.max(0)
            for i, (name, op) in enumerate(zip(self.metric_names, self.metric_ops)):
                if op == "max":
                    out_metrics[name] = max_metric[i].item() if self.use_stack else metrics[i].max().item()
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

        (policy_model, reference_model, reward_model, value_model, policy_model_eval, value_model_eval) = model
        self._policy_model_eval = policy_model_eval
        self._value_model_eval = value_model_eval

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
        self.policy_trainer.set_eval_model(policy_model_eval)
        self.value_trainer.set_eval_model(value_model_eval)
        # disable inner trainers' callback/state/control
        self.policy_trainer.add_callback(MuteDefaultFlowCallback)
        self.value_trainer.add_callback(MuteDefaultFlowCallback)

        # use trainer for reference_model/reward_model to enable sharding stage-3
        # and PipelineParallel. maybe we should allow models to use different dist
        # strategies later

        from paddle.distributed.fleet.meta_parallel import PipelineLayer

        with guard_set_args(
            args,
            {
                "recompute": False,
                "fp16_opt_level": "O1",
                "pipeline_parallel_degree": args.pipeline_parallel_degree
                if isinstance(reference_model, PipelineLayer)
                else 1,  # workaround for pipeline parallel model check
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
            # TODO(guosheng): sharding stage3 should create master weight optionally
            # instead of creation and clear.
            from paddlenlp.trainer.trainer_utils import ShardingOption

            if args.pipeline_parallel_degree > 1 or ShardingOption.FULL_SHARD in args.sharding:
                self.reference_trainer.init_train_model_opt(100, None, clear_master_weight=True)  # dummy max_steps
                self.reward_trainer.init_train_model_opt(100, None, clear_master_weight=True)  # dummy max_steps

        self.reference_model.eval()
        self.reward_model.eval()

        self.reward_tokenizer = reward_tokenizer
        self.tokenizer = policy_tokenizer
        if is_same_tokenizer(self.tokenizer, self.reward_tokenizer):
            self.reward_tokenizer = self.tokenizer

        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.max_length,
            num_return_sequences=self.args.num_return_sequences,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            # top_k=0,  # to disable top_k sampling, default is 50
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
        self.ptx_coeff = self.args.ptx_coeff
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

        # TODO(guosheng): use _inner_eval_model (if trainer has one) instead of
        # original trainer model to eval, especially when using sharded EMA
        # NOTE: use here rather than in prediction_step since actor_model would
        # be set to eval out of prediction_step
        with guard_set_args(
            self.policy_trainer,  # disable _inner_eval_model
            {
                "_eval_model": None,  # otherwise would use cached _eval_model
                "_inner_eval_model": None,  # otherwise would use _inner_eval_model to create _eval_model
            },
        ):
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

                # self.optimizer.offload()
                # if self.args.eval_mode is not None and "optimizer" in self.args.offload_level:
                #     self.timers and self.timers("offload-optimizer").start()
                #     offload_tensor_to_cpu(self.policy_trainer.optimizer.state_dict())
                #     offload_tensor_to_cpu(self.value_trainer.optimizer.state_dict())
                #     self.timers and self.timers("offload-optimizer").stop()

                with ema(self.policy_trainer), ema(self.value_trainer):
                    if self.args.eval_mode is not None:
                        self.timers and self.timers("export-evaluate-model").start()

                        self.policy_trainer.export_evaluate_model(
                            self.policy_trainer.model,
                            self._policy_model_eval,
                            with_offload="train_model" in self.args.offload_level,
                        )
                        self.value_trainer.export_evaluate_model(
                            self.value_trainer.model,
                            self._value_model_eval,
                            with_offload="train_model" in self.args.offload_level,
                        )
                    gp = (
                        self.policy_trainer._policy_model_eval_group
                        if hasattr(self.policy_trainer, "_policy_model_eval_group")
                        else None
                    )
                    # gp = create_data_trans_group(self.args.logical_process_index, paddle.distributed.get_rank(), self._policy_model_eval.config.tensor_parallel_degree)
                    # # todo: zhui
                    self.timers and self.timers("export-evaluate-model").stop()

                    # self.reference_model.reload()
                    # self.reward_model.reload()
                    if "reward" in self.args.offload_level:
                        self.timers and self.timers("reload-reward").start()
                        reload_tensor_to_gpu(self.reference_model.state_dict())
                        reload_tensor_to_gpu(self.reward_model.state_dict())
                        self.timers and self.timers("reload-reward").stop()

                    # todo, split prompt_only_batch
                    # pp2tp2dp2 -> dp4tp2 prompt_only_batch
                    self.timers and self.timers("resplit-data").start()
                    prompt_only_batch = data_group_split(prompt_only_batch, group=gp)
                    self.timers and self.timers("resplit-data").stop()

                    self.timers and self.timers("split-rl-micro-batches").start()
                    # 生成数据
                    # per_train 4, accu 8
                    # prompt 32

                    # 32? [4,4,4,4,4,4,4]
                    rl_batches = self.split_rl_micro_batches(prompt_only_batch)
                    # rl_batches = self.load_sing_gen_data(as_batches=True,
                    #                                      use_counter=True)
                    self.timers and self.timers("split-rl-micro-batches").stop()

                self.timers and self.timers("ptx-batch").start()
                if self.use_ptx:
                    # ptx_batch = data_group_split(ptx_batch, group=gp)
                    ptx_batches = self.split_ptx_micro_batches(ptx_batch)
                    # ptx_batches = data_group_merge(ptx_batches, group=gp)
                else:
                    ptx_batches = [None for _ in range(len(rl_batches))]

                self.timers and self.timers("ptx-batch").stop()

                self.timers and self.timers("merge-data").start()
                # todo, merge data
                if gp is not None:
                    input_ids_length = rl_batches[0]["input_ids"].shape[-1]
                    rl_batches[0]["input_ids_length"] = paddle.to_tensor(
                        [input_ids_length] * rl_batches[0]["input_ids"].shape[0], dtype="int64"
                    )
                    rl_batches = data_group_merge(rl_batches, group=gp)
                    input_ids_length_batchs = rl_batches[0].pop("input_ids_length")
                    rl_batches[0] = repad_rl_batches(rl_batches[0], input_ids_length_batchs)

                paddle.device.cuda.empty_cache()
                self.timers and self.timers("merge-data").stop()

                # # 数据造好, 开始训练
                # self.reference_model.offload()
                # self.reward_model.offload()
                # policy_model_eval.cleanup()
                # value_model_eval.cleanup()
                if self.args.offload_level is not None:
                    if self.args.eval_mode is not None and "eval" in self.args.offload_level:
                        self.timers and self.timers("offload-eval").start()
                        cleanup_tensor_space(self._policy_model_eval.state_dict())
                        cleanup_tensor_space(self._value_model_eval.state_dict())
                        self.timers and self.timers("offload-eval").stop()
                    if "reward" in self.args.offload_level:
                        self.timers and self.timers("offload-reward").start()
                        offload_tensor_to_cpu(self.reference_model.state_dict())
                        offload_tensor_to_cpu(self.reward_model.state_dict())
                        self.timers and self.timers("offload-reward").stop()

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
        return self.value_trainer.is_accumulation_step

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
        metric = PPOMetric(freq=self.args.logging_steps)

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

                # policy_model.reload()
                # value_model.reload()
                self.timers and self.timers("offload-reload").start()
                reload_tensor_to_gpu(self.actor_model.state_dict())
                reload_tensor_to_gpu(self.reward_critic_model.state_dict())
                self.timers and self.timers("offload-reload").stop()

                logger.info("Doing rl step...")
                self.timers and self.timers("rl_step").start()
                rl_info = self.rl_step(rl_batch)
                paddle.device.cuda.empty_cache()
                self.timers and self.timers("rl_step").stop()
                if self.args.eval_mode is not None and "optimizer" in self.args.offload_level:
                    self.timers and self.timers("offload-value-optimizer").start()
                    offload_tensor_to_cpu(self.value_trainer.optimizer.state_dict())
                    self.timers and self.timers("offload-value-optimizer").stop()

                if self.use_ptx:
                    logger.info("Doing ptx step...")
                    self.timers and self.timers("ptx_step").start()
                    ptx_info = self.ptx_step(ptx_batch)
                    rl_info.update(ptx_info)
                    paddle.device.cuda.empty_cache()
                    self.timers and self.timers("ptx_step").stop()

                    self.timers and self.timers("offload-policy-optimizer").start()
                    offload_tensor_to_cpu(self.policy_trainer.optimizer.state_dict())
                    self.timers and self.timers("offload-policy-optimizer").stop()

                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch
                if self.is_step_end():
                    rl_info.update(self.get_step_loss(loss_prefix="train/"))
                    rl_info = metric.update(rl_info)
                    # on_step_end
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                else:
                    # on_sub_step_end
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(rl_info, None, epoch, ignore_keys_for_eval, inputs=inputs)
                self._print_timer()

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
            self._print_timer()

            if self.control.should_training_stop:
                break
        # TODO(guosheng): add epilogue of training

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs):
        if self.control.should_log:

            logs: Dict[str, float] = {}
            # use_ptx would double the gradient_accumulation_steps which causes
            # actor_loss and ptx_loss reduced by half. Moreover, ptx_loss should
            # be divided by ptx_coeff for logging.
            if "train/ptx_loss" in tr_loss:
                tr_loss["train/actor_loss"] = tr_loss["train/actor_loss"] * 2
                tr_loss["train/ptx_loss"] = tr_loss["train/ptx_loss"] * 2 / self.ptx_coeff
            logs.update(tr_loss)
            logs["global_step"] = int(self.state.global_step)
            logs["train/actor_lr"] = float("{0:.3e}".format(self.policy_trainer._get_learning_rate()))
            logs["train/reward_critic_lr"] = float("{0:.3e}".format(self.value_trainer._get_learning_rate()))

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
        # TODO(guosheng): use scatter_add/put_along_axis
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
        with infer_guard(self.policy_trainer):
            for i in range(0, total_batch_size, micro_batch_size):
                micro_batch = {}
                micro_batch = map_structure(
                    lambda tensor: tensor[i : i + micro_batch_size],
                    prompt_only_batch,
                )
                micro_batches.extend(self.generate(micro_batch))
        # size of micro_batches (num of training batch) would be:
        # per_device_prompt_batch_size * num_return_sequences // per_device_train_batch_size
        micro_batches = [self.post_rollout(*micro_batch) for micro_batch in micro_batches]
        return micro_batches

    @paddle.no_grad()
    def generate(self, prompt_only_batch: PromptOnlyBatch) -> List[Dict[str, Any]]:
        """Rollout a batch of experiences."""
        input_ids = prompt_only_batch["input_ids"]
        attention_mask = prompt_only_batch["attention_mask"]
        position_ids = (
            prompt_only_batch["position_ids"]
            if "position_ids" in prompt_only_batch
            else make_position_ids(attention_mask)
        )
        # NOTE: generation output of paddlenlp do not contain prompt, we should
        # change sequences here.

        self.timers and self.timers("actor-model-generate").start()
        sequences = self.actor_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            generation_config=self.generation_config,
            synced_gpus=ShardingOption.FULL_SHARD in self.policy_trainer.args.sharding,
        )[0]

        self.timers and self.timers("actor-model-generate").stop()
        sequences = sequences.reshape([input_ids.shape[0], self.args.num_return_sequences, -1]).transpose([1, 0, 2])

        # prompt, sequence, attention_mask
        return [
            (
                input_ids,
                seq,
                paddle.logical_and(
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
            # actor_model_in_use gen
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
        self.timers and self.timers("actor-model-logit").start()
        logits = self.actor_model(
            sequence,
            attention_mask=attention_mask,
            position_ids=position_ids,
            # return_dict=True,
        )  # .logits
        self.timers and self.timers("actor-model-logit").stop()
        if not isinstance(logits, paddle.Tensor):
            logits = logits[0]
        self.timers and self.timers("reference-model-logit").start()
        ref_logits = self.reference_model(
            sequence,
            attention_mask=attention_mask,
            position_ids=position_ids,
            # return_dict=True,
        )  # .logits
        self.timers and self.timers("reference-model-logit").stop()
        if not isinstance(ref_logits, paddle.Tensor):
            ref_logits = ref_logits[0]

        self.timers and self.timers("reward-model-score").start()
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

        self.timers and self.timers("reward-model-score").stop()
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
