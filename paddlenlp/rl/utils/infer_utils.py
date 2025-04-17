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

import contextlib
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
from ...trl.llm_utils import init_dist_env
from ..trainer.trainer_utils import process_row
from .offload_utils import offload_tensor_to_cpu, reload_tensor_to_gpu
from .reshard_utils import init_rollout_env

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
        super().__init__(config, tokenizer, model, **kwargs)
        self.args = kwargs["training_args"]

    def enable(self, model, offload_model=True):
        if self.is_available:
            return
        with paddle.LazyGuard():
            self.set_state_dict(model, offload_model)
        self.is_available = True

    def disable(self, model, onload_model=True):
        for _, param in self.model.state_dict().items():
            param._clear_data()
        if onload_model:
            model.to(paddle.device.get_device())
        self.is_available = False

    @contextmanager
    def update_predictor_params(self, **kwargs):
        # update predictor config
        if kwargs:
            old_predictor_config = copy.deepcopy(self.config)
            for key, new_value in kwargs.items():
                if hasattr(self.config, key):
                    old_value = getattr(self.config, key)
                    if old_value != new_value:
                        setattr(self.config, key, new_value)
                        if key == "top_p":
                            self.update_model_inputs("top_p", new_value)
                        if key == "temperature":
                            self.update_model_inputs("temperature", new_value)
        yield
        if kwargs:
            if self.config.top_p != old_predictor_config:
                self.update_model_inputs("top_p", old_predictor_config.top_p)
            if self.config.temperature != old_predictor_config:
                self.update_model_inputs("temperature", old_predictor_config.temperature)
            self.config = old_predictor_config

    def update_model_inputs(self, key, value):
        assert key in self.model_inputs, f"{key} is not in model_inputs!"
        old_value = self.model_inputs.pop(key)
        self.model_inputs[key] = paddle.full(shape=old_value.shape, fill_value=value, dtype=old_value.dtype)

    @paddle.no_grad()
    def predict(self, input_ids: paddle.Tensor = None, **kwargs):
        bs = input_ids.shape[0]
        input_ids_list = []
        for row in input_ids:
            row_ids = process_row(row, remove_value=self.tokenizer.pad_token_id, remove_side="left").tolist()
            input_ids_list.append(row_ids)

        with self.update_predictor_params(**kwargs):
            self._preprocess(input_text=None, input_ids=input_ids_list)
            self.init_cache_kvs()
            all_tokens = []
            if (
                self.args.rollout_tensor_parallel_degree != self.args.tensor_parallel_degree
                or self.args.pipeline_parallel_degree > 1
            ):
                ori_all_reduce = dist.all_reduce
                ori_broadcast = dist.broadcast
                with init_rollout_env(self.args.rollout_tensor_parallel_degree):
                    hcg = fleet.get_hybrid_communicate_group()
                    tp_group = hcg.get_model_parallel_group()
                    dist.all_reduce = lambda x: ori_all_reduce(x, group=tp_group)
                    dist.broadcast = lambda x, rank: ori_broadcast(
                        x, src=tp_group.ranks[0], group=hcg.get_model_parallel_group()
                    )
                    while self.model_inputs["not_need_stop"]:
                        next_tokens = self._infer(self.model_inputs)[:bs]
                        all_tokens.append(next_tokens)
                dist.all_reduce = ori_all_reduce
                dist.broadcast = ori_broadcast
            else:
                while self.model_inputs["not_need_stop"]:
                    next_tokens = self._infer(self.model_inputs)[:bs]
                    all_tokens.append(next_tokens)

        # remove cache kvs
        self.cache_kvs = None
        self.model_inputs["cache_kvs"] = None
        paddle.device.cuda.empty_cache()

        outputs = paddle.concat(all_tokens, axis=-1)
        outputs = paddle.where(
            outputs < 0, paddle.to_tensor(self.tokenizer.pad_token_id, dtype=outputs.dtype), outputs
        )
        return outputs

    @paddle.no_grad()
    def set_state_dict(self, model, offload_model=True):
        self.model.set_state_dict(model.state_dict())
        if offload_model:
            offload_place = paddle.CUDAPinnedPlace()
            state_dict = model.state_dict()
            for k, v in state_dict.items():
                cpu_arg = v._copy_to(offload_place, blocking=False)
                cpu_arg._share_buffer_to(v)
        paddle.device.synchronize()


policy_predictor: PolicyPredictor = None


def create_predictor(trainer: Trainer):
    predictor_args = PredictorArgument(
        model_name_or_path=trainer.args.actor_model_name_or_path,
        src_length=trainer.args.max_src_len,
        min_length=trainer.args.min_dec_len,
        max_length=trainer.args.max_dec_len,
        total_max_length=trainer.args.max_src_len + trainer.args.max_dec_len,
        batch_size=trainer.args.per_device_rollout_batch_size * trainer.args.num_return_sequences,
        top_p=trainer.args.top_p,
        temperature=trainer.args.temperature,
        repetition_penalty=trainer.args.repetition_penalty,
        append_attn=True,  # currently only support append_attn
        inference_model=True,
        dtype=trainer.amp_dtype,
        output_via_mq=False,
    )
    model_args = ModelArgument()
    config = copy.deepcopy(trainer.model.config)
    config.sequence_parallel = False
    config.use_fused_head_and_loss_fn = False
    config.use_fused_rms_norm = False
    need_reshard = (
        trainer.args.rollout_tensor_parallel_degree != trainer.args.tensor_parallel_degree
        or trainer.args.pipeline_parallel_degree > 1
    )
    if need_reshard:
        init_context = init_rollout_env(trainer.args.rollout_tensor_parallel_degree)
    else:
        tensor_parallel_rank, tensor_parallel_degree = init_dist_env()
        init_context = contextlib.nullcontext()
    with init_context:
        if need_reshard:
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
            model.save_output = False
            predictor = PolicyPredictor(
                predictor_args,
                tokenizer=trainer.tokenizer,
                model=model,
                model_args=model_args,
                init_cache_kvs=False,
                training_args=trainer.args,
            )
            predictor.is_available = False
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

    need_reshard = (
        trainer.args.rollout_tensor_parallel_degree != trainer.args.tensor_parallel_degree
        or trainer.args.pipeline_parallel_degree > 1
    )
    if not need_reshard:
        is_distributed = True
        try:
            hcg = dist.fleet.get_hybrid_communicate_group()
        except Exception:
            is_distributed = False

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
    else:
        yield
    policy_predictor.disable(model, onload_model=offload_model)


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
        outputs = policy_predictor.predict(input_ids=input_ids, **kwargs)
        outputs = paddle.concat([input_ids, outputs], axis=-1)
        return (outputs,)
