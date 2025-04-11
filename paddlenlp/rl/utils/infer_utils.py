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

import builtins
import copy
import inspect
import os
import time
from contextlib import contextmanager

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.utils import try_import

from ...trainer.trainer import Trainer, logger
from ...transformers import (
    AutoInferenceModelForCausalLM,
    PretrainedModel,
    PretrainedTokenizer,
)
from ...transformers.model_utils import dtype_guard
from ...trl import llm_utils
from ...trl.llm_utils import init_dist_env
from ..trainer.trainer_utils import process_row
from .offload_utils import offload_tensor_to_cpu, reload_tensor_to_gpu

_original_import = builtins.__import__
_imported_modules = {}
_paddlenlp_ops_updated = False


def custom_import(name, *args, **kwargs):
    global _paddlenlp_ops_updated

    if name in _imported_modules:
        return _imported_modules[name]

    module = _original_import(name, *args, **kwargs)

    if not _paddlenlp_ops_updated and os.getenv("USE_PYBIND", "1").lower() in ["1", "true", "t", "yes", "y"]:
        if name == "paddlenlp_ops":
            logger.info("Using Pybind paddlenlp_ops!")
            module.update_inputs_v2 = module.f_update_inputs_v2
            module.save_output = module.f_save_output
            module.set_preids_token_penalty_multi_scores = module.f_set_preids_token_penalty_multi_scores
            module.rebuild_padding_v2 = module.f_rebuild_padding_v2
            module.append_attention = module.f_append_attention
            _paddlenlp_ops_updated = True

    _imported_modules[name] = module
    return module


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

    @paddle.no_grad()
    def predict_dy_insert(self, input_ids: list[int], **kwargs):
        # pybind
        self.input_ids = input_ids
        assert self.proposer is None, "dynamic insert don't support proposer."

        total_request_num = len(self.input_ids)
        max_batch_size = self.config.batch_size
        self.block_size = self.config.block_size

        self.prefill_blocks = []
        block_id = 0
        for inst in self.input_ids:
            length = len(inst)
            num_blocks = (length + self.block_size - 1) // self.block_size
            self.prefill_blocks.append(list(range(block_id, block_id + num_blocks)))
            block_id += num_blocks
        # print("prefill_blocks", self.prefill_blocks)

        self.tail_blocks = []
        for _ in range(len(self.input_ids)):
            self.tail_blocks.append(block_id)
            block_id += 1
        # print("tail_blocks", self.tail_blocks)

        self.decoder_blocks = []
        for _ in range(max_batch_size):
            num_blocks = (self.config.max_length + self.block_size - 1) // self.block_size
            self.decoder_blocks.append(list(range(block_id, block_id + num_blocks)))
            block_id += num_blocks
        # print("self.decoder_blocks: ", self.decoder_blocks)

        self.cache_k_shapes = []
        self.cache_v_shapes = []

        max_num_blocks_per_row_per_decoding = (self.config.max_length + self.block_size - 1) // self.block_size

        # For decoder_blocks
        max_num_blocks = max_batch_size * max_num_blocks_per_row_per_decoding

        # For prefill_blocks
        for prefill_block in self.prefill_blocks:
            max_num_blocks += len(prefill_block)

        # For tail_blocks
        max_num_blocks += max_batch_size

        for i in range(self.model.config.num_hidden_layers):
            cache_kv_shape = [
                max_num_blocks,
                self.model.config.num_key_value_heads // max(self.model.config.tensor_parallel_degree, 1),
                self.model.config.block_size,
                self.model.config.hidden_size // self.model.config.num_attention_heads,
            ]
            self.cache_k_shapes.append(cache_kv_shape)
            self.cache_v_shapes.append(cache_kv_shape)
        self.init_cache_kvs()

        self.model_inputs["input_ids"] = paddle.full(
            shape=[max_batch_size, self.config.total_max_length], fill_value=0, dtype="int64"
        )

        self.model_inputs["block_tables"] = paddle.full(
            shape=[
                max_batch_size,
                (self.config.total_max_length + self.config.block_size - 1) // self.config.block_size + 1,
            ],
            fill_value=-1,
            dtype="int32",
        )

        self.model_inputs["seq_lens_this_time"] = paddle.zeros(shape=[max_batch_size, 1], dtype="int32")
        self.model_inputs["seq_lens_encoder"] = paddle.zeros(shape=[max_batch_size, 1], dtype="int32")
        self.model_inputs["seq_lens_decoder"] = paddle.zeros(shape=[max_batch_size, 1], dtype="int32")

        self.model_inputs["pre_ids"] = paddle.full(
            shape=[max_batch_size, self.config.max_length], fill_value=-1, dtype="int64"
        )

        # Construct loop cvars
        self.model_inputs["step_idx"] = paddle.full(shape=[max_batch_size, 1], fill_value=0, dtype="int64")
        self.model_inputs["not_need_stop"] = paddle.full(shape=[1], fill_value=True, dtype="bool").cpu()  # cpu
        self.model_inputs["stop_flags"] = paddle.ones(shape=[max_batch_size, 1], dtype="bool")
        self.model_inputs["stop_nums"] = paddle.full(shape=[1], fill_value=max_batch_size, dtype="int64")
        self.model_inputs["result_id"] = paddle.full(shape=[max_batch_size, 1], fill_value=-1).astype("int32")
        self.model_inputs["next_tokens"] = paddle.full(shape=[max_batch_size, 1], fill_value=-1, dtype="int64")

        # output buffers for all inputs
        self.model_inputs["all_token_ids"] = paddle.full(
            shape=[total_request_num, self.config.max_length],
            fill_value=llm_utils.get_eos_token_id(self.tokenizer, self.generation_config)[0],
            dtype="int64",
        )

        s_time = time.time()
        with self.update_predictor_params(**kwargs):
            for i, inst in enumerate(self.input_ids):
                length = len(inst)
                self.model_inputs["input_ids"][0, :length] = np.array(inst)
                self.model_inputs["seq_lens_this_time"][0] = length
                self.model_inputs["seq_lens_encoder"][0] = length
                self.model_inputs["stop_flags"][0] = False

                num_prefill_blocks = (length + self.block_size - 1) // self.block_size
                self.model_inputs["block_tables"][0, :num_prefill_blocks] = np.array(self.prefill_blocks[i])
                self.model_inputs["block_tables"][0, num_prefill_blocks] = np.array(self.tail_blocks[i])
                self.model_inputs["result_id"][0][:1] = np.arange(i, i + 1)

                next_tokens = self._infer(self.model_inputs)
                self.model_inputs["all_token_ids"][i, 0] = next_tokens[0, 0]
                self.model_inputs["seq_lens_this_time"][0] = 0
                self.model_inputs["seq_lens_encoder"][0] = 0
                self.model_inputs["seq_lens_decoder"][0] = 0
                self.model_inputs["stop_flags"][0] = True
                self.model_inputs["step_idx"][0, 0] = 0
                self.model_inputs["block_tables"][0] = -1
                self.model_inputs["result_id"][0] = -1

            unfinished_ids = list(range(total_request_num - 1, -1, -1))
            for cur_bs in range(max_batch_size):
                if len(unfinished_ids) == 0:
                    break
                task_id = unfinished_ids.pop()
                self.insert(cur_bs, task_id)

            if kwargs.pop("max_length", self.config.max_length) > 1:
                while self.model_inputs["not_need_stop"] or len(unfinished_ids) > 0:
                    no_stop_num = max_batch_size - paddle.sum(self.model_inputs["stop_flags"]).item()
                    if no_stop_num < max_batch_size:
                        for i in range(max_batch_size):
                            if self.model_inputs["stop_flags"][i] and len(unfinished_ids) > 0:
                                task_id = unfinished_ids.pop()
                                self.insert(i, task_id)
                    next_tokens = self._infer(self.model_inputs)
        logger.info(f"running spend {time.time() - s_time}")

        self.cache_kvs = None
        self.model_inputs["cache_kvs"] = None
        paddle.device.cuda.empty_cache()

        if not self.rollout_use_fake_outputs:
            output_tokens = self.model_inputs["all_token_ids"]
            output_tokens = paddle.where(
                output_tokens < 0,
                paddle.to_tensor(self.tokenizer.pad_token_id, dtype=output_tokens.dtype),
                output_tokens,
            )
        else:
            output_tokens = (paddle.ones([total_request_num, self.config.max_length]) * 1000).cast("int64")
        return output_tokens

    @paddle.no_grad()
    def predict(self, input_ids: paddle.Tensor = None, **kwargs):
        builtins.__import__ = custom_import
        bs = input_ids.shape[0]
        input_ids_list = []
        for row in input_ids:
            row_ids = process_row(row, remove_value=self.tokenizer.pad_token_id, remove_side="left").tolist()
            input_ids_list.append(row_ids)

        if self.config.dynamic_insert:
            return self.predict_dy_insert(input_ids_list, **kwargs)[:bs]
        else:
            with self.update_predictor_params(**kwargs):
                self._preprocess(input_text=None, input_ids=input_ids_list)
                self.init_cache_kvs()
                if not self.rollout_use_fake_outputs:
                    all_tokens = []
                    while self.model_inputs["not_need_stop"]:
                        next_tokens = self._infer(self.model_inputs)[:bs]
                        all_tokens.append(next_tokens)

            # remove cache kvs
            self.cache_kvs = None
            self.model_inputs["cache_kvs"] = None
            paddle.device.cuda.empty_cache()

            if not self.rollout_use_fake_outputs:
                outputs = paddle.concat(all_tokens, axis=-1)
                outputs = paddle.where(
                    outputs < 0, paddle.to_tensor(self.tokenizer.pad_token_id, dtype=outputs.dtype), outputs
                )
            else:
                outputs = (paddle.ones([bs, self.config.max_length]) * 1000).cast("int64")
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
    eval_model = getattr(trainer, "_inner_eval_model", None)
    if eval_model is not None:
        raise NotImplementedError("Currently do not support _inner_eval_model!")

    predictor_args = PredictorArgument(
        model_name_or_path=trainer.args.actor_model_name_or_path,
        src_length=trainer.args.max_src_len,
        min_length=trainer.args.min_dec_len,
        max_length=trainer.args.max_dec_len,
        total_max_length=trainer.args.max_src_len + trainer.args.max_dec_len,
        batch_size=trainer.args.rollout_continue_batching_batch_size,
        top_p=trainer.args.top_p,
        temperature=trainer.args.temperature,
        repetition_penalty=trainer.args.repetition_penalty,
        append_attn=True,  # currently only support append_attn
        inference_model=True,
        dtype=trainer.amp_dtype,
        output_via_mq=False,
        init_cache_kvs=False,
        dynamic_insert=trainer.args.rollout_use_dynamic_insert,
    )
    model_args = ModelArgument()
    config = copy.deepcopy(trainer.model.config)
    config.sequence_parallel = False
    config.use_fused_head_and_loss_fn = False
    config.use_fused_rms_norm = False
    tensor_parallel_rank, tensor_parallel_degree = init_dist_env()
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
        )
        predictor.rollout_use_fake_outputs = trainer.args.rollout_use_fake_outputs
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

    # TODO(guosheng): patch for dist.all_recude to use tp group, fix it later
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
