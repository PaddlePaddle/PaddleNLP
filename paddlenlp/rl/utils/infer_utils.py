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

import copy
import inspect
from contextlib import contextmanager

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.utils import try_import

from ...trainer.trainer import Trainer, logger
from ...transformers import (
    AutoInferenceModelForCausalLM,
    PretrainedModel,
    PretrainedTokenizer,
)
from ...transformers.model_utils import dtype_guard
from ..trainer.trainer_utils import process_row
from .offload_utils import offload_tensor_to_cpu, reload_tensor_to_gpu

try:
    from llm.predict.predictor import (
        DygraphBlockInferencePredictor,
        ModelArgument,
        PredictorArgument,
    )
except ImportError:

    class DygraphBlockInferencePredictor(object):
        """
        A dummy class for DygraphBlockInferencePredictor, used when the actual class
        cannot be imported from llm.predict.predictor
        """

        pass

    class ModelArgument(object):
        """
        A dummy class for ModelArgument, used when the actual class
        cannot be imported from llm.predict.predictor
        """

        pass

    class PredictorArgument(object):
        """
        A dummy class for ModelArgument, used when the actual class
        cannot be imported from llm.predict.predictor
        """

        pass


class PolicyPredictor(DygraphBlockInferencePredictor):
    def __init__(
        self, config: PredictorArgument, tokenizer: PretrainedTokenizer = None, model: PretrainedModel = None, **kwargs
    ):
        self.args = kwargs.pop("training_args", None)
        self.is_available = kwargs.pop("is_available", False)
        super().__init__(config, tokenizer, model, **kwargs)

    def enable(self, model, offload_model=True):
        if self.is_available:
            return
        self.set_state_dict(model, offload_model)
        self.is_available = True

    def disable(self, model, onload_model=True):
        for _, param in self.model.state_dict().items():
            param._clear_data()
        if onload_model:
            model.to(paddle.device.get_device())
        self.is_available = False

    @paddle.no_grad()
    def predict(self, input_ids: paddle.Tensor = None, repeat_num=1, **kwargs):
        input_ids_list = []
        for row in input_ids:
            row_ids = process_row(row, remove_value=self.tokenizer.pad_token_id, remove_side="left").tolist()
            input_ids_list.append(row_ids)

        if self.config.dynamic_insert:
            outputs = self.predict_dy_insert(
                input_ids=input_ids_list,
                return_tokens=True,
                all_rank_return=True,
                detokenize=False,
                repeat_num=repeat_num,
                **kwargs,
            )[-1]
            return paddle.to_tensor(outputs, dtype=input_ids.dtype)
        else:
            raise NotImplementedError("dynamic_insert is False is not supported.")

    @paddle.no_grad()
    def set_state_dict(self, model, offload_model=True):
        if offload_model:
            offload_place = paddle.CUDAPinnedPlace()
            state_dict = model.state_dict()
            for k, v in state_dict.items():
                cpu_arg = v._copy_to(offload_place, blocking=False)
                cpu_arg._share_buffer_to(v)
        paddle.device.synchronize()
        paddle.device.cuda.empty_cache()
        with paddle.LazyGuard():
            with dtype_guard(self.config.dtype):
                self.model.set_state_dict(model.state_dict())


policy_predictor: PolicyPredictor = None


def create_predictor(trainer: Trainer):
    predictor_args = PredictorArgument(
        model_name_or_path=trainer.args.actor_model_name_or_path,
        src_length=trainer.args.max_src_len,
        min_length=trainer.args.min_dec_len,
        max_length=trainer.args.max_dec_len,
        total_max_length=trainer.args.max_src_len + trainer.args.max_dec_len,
        batch_size=trainer.args.rollout_max_num_seqs,
        top_p=trainer.args.top_p,
        temperature=trainer.args.temperature,
        repetition_penalty=trainer.args.repetition_penalty,
        append_attn=True,  # currently only support append_attn
        inference_model=True,
        dtype=trainer.amp_dtype,
        output_via_mq=False,
        dynamic_insert=True,
        quant_type=trainer.args.rollout_quant_type,
    )
    model_args = ModelArgument()
    config = copy.deepcopy(trainer.model.config)
    config.sequence_parallel = False
    config.use_fused_head_and_loss_fn = False
    config.use_fused_rms_norm = False

    if getattr(trainer, "reshard_controller", None) is not None:
        trainer.reshard_controller.set_rollout_env("[create_predictor]")
    hcg = fleet.get_hybrid_communicate_group()
    tensor_parallel_degree = hcg.get_model_parallel_world_size()
    tensor_parallel_rank = hcg.get_model_parallel_rank()
    with dtype_guard(predictor_args.dtype):
        model = AutoInferenceModelForCausalLM.from_config(
            config=config,
            predictor_args=predictor_args,
            model_args=model_args,
            dtype=predictor_args.dtype,
            tensor_parallel_degree=tensor_parallel_degree,
            tensor_parallel_rank=tensor_parallel_rank,
            low_cpu_mem_usage=True,
        )
        predictor = PolicyPredictor(
            predictor_args,
            tokenizer=trainer.tokenizer,
            model=model,
            model_args=model_args,
            init_cache_kvs=False,
            training_args=trainer.args,
            is_available=False,
        )
    if getattr(trainer, "reshard_controller", None) is not None:
        trainer.reshard_controller.set_train_env("[after create_predictor]")

    return predictor


@contextmanager
def infer_guard(trainer, offload_model=True):
    # trainer might use an extra model instead of trainer.model for eval
    eval_model = getattr(trainer, "_inner_eval_model", None)
    model = trainer.model if eval_model is None else eval_model

    # PipelineParallel does not support inference speedup
    if not getattr(trainer, "use_fusemt", False) or isinstance(
        model, (dist.fleet.meta_parallel.PipelineLayer, dist.fleet.model.PipelineParallel)
    ):
        yield
        return

    try:
        try_import("paddlenlp_ops")
    except ImportError:
        logger.warning("paddlenlp_ops does not exist, please install paddlenlp_ops for generation speedup.")
        yield
        return

    global policy_predictor
    if policy_predictor is None:
        policy_predictor = create_predictor(trainer)
    with dtype_guard(trainer.amp_dtype):
        if not policy_predictor.is_available:
            policy_predictor.enable(model, offload_model=offload_model)

    is_distributed = True
    try:
        hcg = dist.fleet.get_hybrid_communicate_group()
    except Exception:
        is_distributed = False

    if getattr(trainer, "reshard_controller", None) is not None:
        trainer.reshard_controller.set_rollout_env("[infer_guard hack broadcast & all_reduce]")

        ori_all_reduce = dist.all_reduce
        ori_broadcast = dist.broadcast
        hcg = fleet.get_hybrid_communicate_group()
        tp_group = hcg.get_model_parallel_group()
        dist.all_reduce = lambda x, **kwargs: ori_all_reduce(x, group=tp_group)
        dist.broadcast = lambda x, rank, **kwargs: ori_broadcast(x, src=tp_group.ranks[0], group=tp_group)
        yield
        dist.all_reduce = ori_all_reduce
        dist.broadcast = ori_broadcast
    else:
        if is_distributed:
            ori_all_reduce = dist.all_reduce
            ori_broadcast = dist.broadcast
            dist.all_reduce = lambda x: ori_all_reduce(x, group=hcg.get_model_parallel_group())
            dist.broadcast = lambda x, rank: ori_broadcast(
                x, src=hcg.get_model_parallel_group_src_rank(), group=hcg.get_model_parallel_group()
            )
            yield
            dist.all_reduce = ori_all_reduce
            dist.broadcast = ori_broadcast
        else:
            yield
    policy_predictor.disable(model, onload_model=offload_model)


def get_policy_predictor():
    global policy_predictor
    return policy_predictor


class InferEvalModel:
    """For faster generation, not support PipelineParallel yet."""

    def __init__(self, trainer: Trainer):
        # trainer might use an extra model instead of trainer.model for eval
        eval_model = getattr(trainer, "_inner_eval_model", None)
        self.model: PretrainedModel = trainer.model if eval_model is None else eval_model
        self.tokenizer: PretrainedTokenizer = trainer.tokenizer
        self.trainer = trainer

    def enable(self):
        trainer = self.trainer
        if trainer.model is not self.model:
            reload_tensor_to_gpu((trainer.model, "train_model"))
            reload_tensor_to_gpu((self.model, "freeze_model"))
            trainer.export_evaluate_model(
                trainer.model,
                self.model,
                with_offload="train_model" in trainer.args.offload_level,
            )
            # NOTE(gongenlei): Add offload
            offload_tensor_to_cpu((trainer.model, "train_model"))
        else:
            reload_tensor_to_gpu((self.model, "train_model"))

    def disable(self):
        trainer = self.trainer
        if trainer.model is not self.model:
            offload_tensor_to_cpu((trainer.model, "train_model"))
            offload_tensor_to_cpu((self.model, "freeze_model"))
        else:
            offload_tensor_to_cpu((self.model, "train_model"))

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def __call__(self, *args, **kwargs):
        # assert model is on GPU
        assert policy_predictor is None or not policy_predictor.is_available
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        do_eval = kwargs.pop("do_eval", False)
        repeat_num = kwargs.pop("repeat_num", 1)
        if policy_predictor is None or not policy_predictor.is_available:
            return self.model.generate(*args, **kwargs)

        arg_dict = inspect.signature(self.model.generate).bind(*args, **kwargs).arguments
        input_ids = arg_dict["input_ids"]
        kwargs = {}
        if do_eval:
            # for greedy search
            kwargs.update(
                {
                    "top_p": 0.0,
                    "temperature": 1.0,
                }
            )
        outputs = policy_predictor.predict(input_ids=input_ids, repeat_num=repeat_num, **kwargs)
        if repeat_num > 1:
            input_ids = input_ids.repeat_interleave(repeat_num, axis=0)

        outputs = paddle.concat([input_ids, outputs], axis=-1)
        return (outputs,)
