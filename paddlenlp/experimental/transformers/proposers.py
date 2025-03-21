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
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np
import paddle

if not paddle.is_compiled_with_xpu():
    from paddlenlp_ops import (
        draft_model_postprocess,
        draft_model_preprocess,
        eagle_get_base_model_hidden_states,
        eagle_get_self_hidden_states,
    )

from paddlenlp.transformers import AutoConfig, AutoInferenceModelForCausalLM
from paddlenlp.trl import llm_utils


@dataclass
class SpeculateArgument:
    model_name_or_path: str = field(default=None, metadata={"help": "The directory of model."})
    quant_type: str = field(
        default="",
        metadata={
            "help": "Quantization type. Supported values: a8w8, a8w8c8, a8w8_fp8, a8w8c8_fp8, weight_only_int4, weight_only_int8"
        },
    )
    cachekv_int8_type: str = field(
        default=None,
        metadata={
            "help": "If cachekv_int8_type set as `dynamic`, cache kv would be quantized to int8 dynamically. If cachekv_int8_type set as `static`, cache kv would be quantized to int8 Statically."
        },
    )
    use_fake_parameter: bool = field(default=False, metadata={"help": "use fake parameter, for ptq scales now."})
    dtype: str = field(default=None, metadata={"help": "Model dtype"})
    max_batch_size: int = field(default=1, metadata={"help": "The max batch size of data."})
    total_max_length: int = field(default=8192, metadata={"help": "the max length for encoding and decoding"})
    min_length: int = field(default=1, metadata={"help": "the min length for decoding."})
    max_length: int = field(default=1024, metadata={"help": "the max length for decoding."})
    temperature: float = field(default=1.0, metadata={"help": "top_p parameter for generation"})
    decode_strategy: str = field(
        default="draft_model_sample",
        metadata={"help": "the decoding strategy of generation, it only supports [draft_model_sample] now"},
    )
    mode: str = field(default="dynamic", metadata={"help": "the type of predictor, it only supports [dynamic] now"})
    inference_model: bool = field(default=True, metadata={"help": "whether use InferenceModel to do generation"})
    block_attn: bool = field(default=True, metadata={"help": "whether use block attention"})
    append_attn: bool = field(default=True, metadata={"help": "whether use append attention"})
    block_size: int = field(default=64, metadata={"help": "the block size for cache_kvs."})
    speculate_method: str = field(
        default=None,
        metadata={"help": "speculate method, it should be one of ['eagle', 'mtp']"},
    )
    speculate_max_draft_token_num: int = field(
        default=1,
        metadata={"help": "the max length of draft tokens for speculate method."},
    )
    speculate_max_ngram_size: int = field(default=1, metadata={"help": "the max ngram size of speculate method."})
    speculate_max_candidate_len: int = field(default=5, metadata={"help": "the max length of candidate tokens."})
    speculate_verify_window: int = field(
        default=2, metadata={"help": "the max length of verify window for speculate method."}
    )
    return_full_hidden_states: int = field(default=False, metadata={"help": "whether return full hidden_states"})
    serving_mode: str = field(default=False, metadata={"help": "whether in serving_mode"})

    mla_use_matrix_absorption: bool = field(default=False, metadata={"help": "implement mla with matrix-absorption."})
    weightonly_group_size: int = field(default=-1, metadata={"help": "the max length of candidate tokens."})
    weight_block_size: List[int] = field(
        default_factory=lambda: [128, 128],
        metadata={"help": "Quantitative granularity of weights. Supported values: [128 128]"},
    )
    moe_quant_type: str = field(
        default="",
        metadata={"help": "Quantization type of moe. Supported values: weight_only_int4, weight_only_int8"},
    )

    @classmethod
    def build_from_predictor(cls, predictor_args):
        args = {}
        args["model_name_or_path"] = predictor_args.draft_model_name_or_path
        args["dtype"] = predictor_args.dtype
        args["quant_type"] = predictor_args.draft_model_quant_type
        args["use_fake_parameter"] = predictor_args.use_fake_parameter

        args["max_batch_size"] = predictor_args.batch_size
        args["total_max_length"] = predictor_args.total_max_length
        args["min_length"] = predictor_args.min_length
        args["max_length"] = predictor_args.max_length
        # temperature=1.0 is the best choice in most of cases
        args["temperature"] = 1.0

        args["speculate_method"] = predictor_args.speculate_method
        args["speculate_max_draft_token_num"] = predictor_args.speculate_max_draft_token_num
        args["speculate_max_candidate_len"] = predictor_args.speculate_max_candidate_len

        args["mla_use_matrix_absorption"] = predictor_args.mla_use_matrix_absorption
        args["weightonly_group_size"] = predictor_args.weightonly_group_size
        args["weight_block_size"] = predictor_args.weight_block_size
        args["moe_quant_type"] = predictor_args.moe_quant_type

        assert args["speculate_method"] in [
            "eagle",
            "mtp",
        ], f"Speculate model only support [eagle, mtp]. But get {args['speculate_method']}"

        return cls(**args)

    @classmethod
    def build_from_serving(
        cls,
        **kwargs,
    ):
        kwargs["serving_mode"] = True
        return cls(**kwargs)


class Proposer(ABC):
    """
    Abstract base class for all proposers that can be used in the speculative decoding framework.
    The subclasses of this class must implement the run method to get the draft tokens that are
    generated by the proposer.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def run(self, model_inputs: dict[str, paddle.Tensor], **kwargs):
        """
        Get the draft tokens that are generated by the proposer.
        """
        raise NotImplementedError()

    @abstractmethod
    def insert_query(self, **kwargs):
        """
        Insert new query
        """
        pass

    @abstractmethod
    def postprocess(self, **kwargs):
        """
        Postprocessing finished query
        """
        pass


class InferenceWithReferenceProposer(Proposer):
    """
    InferenceWithReference(https://arxiv.org/pdf/2304.04487) is one of the speculative decoding method.
    It match tokens in the input and output as draft tokens.
    """

    def __init__(self, max_draft_token_num: int, max_ngram_size: int, max_batch_size: int, max_seq_len: int, **kwargs):
        """
        Args:
        max_draft_token_num (int):
            Maximum number of tokens a proposer can generate at one time.
            The hyperparameter of k in the paper.
        max_ngram_size (int):
            The maximum size of the window used to match inputs and outputs.
            The hyperparameter of n in the paper.
        max_batch_size (int):
            The maximum batch size.
        max_seq_len (int):
            The maximum sequence length.
        """
        super().__init__()
        self.max_ngram_size = max_ngram_size
        self.input_ids_len = paddle.zeros(shape=[max_batch_size, 1], dtype="int64").cpu()
        self.input_ids_cpu = paddle.zeros(shape=[max_batch_size, max_seq_len], dtype="int64").cpu()
        self.max_batch_size = max_batch_size
        self.max_draft_token_num = max_draft_token_num

    def run(self, model_inputs: dict[str, paddle.Tensor], **kwargs):
        """
        Use ngram_match to get draft tokens from the input and output.
        """
        draft_tokens = model_inputs["draft_tokens"].cpu()
        seq_lens_this_time = kwargs["seq_lens_this_time"].cpu()
        seq_lens_encoder = model_inputs["seq_lens_encoder"].cpu()
        seq_lens_decoder = model_inputs["seq_lens_decoder"].cpu()

        from paddlenlp_ops import ngram_match

        ngram_match(
            self.input_ids_cpu,
            self.input_ids_len.cpu(),
            model_inputs["pre_ids"].cpu(),
            model_inputs["step_idx"].cpu(),
            model_inputs["actual_draft_token_num"].cpu(),
            draft_tokens,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            model_inputs["max_length"].cpu(),
            kwargs["real_batch_size"],
            self.max_ngram_size,
            self.max_draft_token_num,
        )

        model_inputs["draft_tokens"][:] = draft_tokens.cuda()
        model_inputs["seq_lens_encoder"][:] = seq_lens_encoder.cuda()
        kwargs["seq_lens_this_time"][:] = seq_lens_this_time.cuda()

    def insert_query(self, **kwargs):
        """
        Insert new query
        """
        pass

    def postprocess(self, **kwargs):
        """
        Postprocessing finished query
        """


class ModelProposer(Proposer):
    """
    用于类 Model 的 Proposer 基类
    在输入输出中匹配符合的tokens作为 draft tokens
    """

    def __init__(self, args: SpeculateArgument, **kwargs):
        super().__init__()
        self.args = args
        self.draft_type = self.args.speculate_method
        self.dtype = self.args.dtype
        assert self.draft_type in (
            "draft_model",
            "eagle",
            "mtp",
        ), f"draft_type support [draft_model, eagle], but get {self.draft_type}"

        self.max_draft_tokens = self.args.speculate_max_draft_token_num
        self.actual_draft_token_num = self.max_draft_tokens
        self.max_batch_size = self.args.max_batch_size
        self.total_max_length = self.args.total_max_length
        self.max_length = self.args.max_length
        self.block_size = self.args.block_size
        self.max_query_block_num = (self.total_max_length + self.block_size - 1) // self.block_size
        self.init_predictor()

        if self.args.serving_mode:
            self.base_model_inputs = kwargs["base_model_inputs"]
            self.create_persistent_inputs()
        else:
            self.base_model_inputs = None

    def build_args(self, args):
        from copy import deepcopy

        draft_model_args = deepcopy(args)
        draft_model_args.quant_type = args.draft_model_quant_type
        draft_model_args.model_name_or_path = args.draft_model_name_or_path
        draft_model_args.decode_strategy = "draft_model_sample"
        draft_model_args.mode = "dynamic"
        draft_model_args.return_full_hidden_states = 0
        return draft_model_args

    def init_predictor(self):
        """
        init_predictor
        """

        tensor_parallel_rank, tensor_parallel_degree = llm_utils.init_dist_env()

        self.config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        paddle.set_default_dtype(self.dtype)
        self.model = AutoInferenceModelForCausalLM.from_pretrained(
            self.args.model_name_or_path,
            config=self.config,
            predictor_args=self.args,
            dtype=self.dtype,
            tensor_parallel_degree=tensor_parallel_degree,
            tensor_parallel_rank=tensor_parallel_rank,
            spec_model_type=self.draft_type,
        )

        # prepare model_inputs
        self.model_inputs = {}

        self.cache_k_shapes, self.cache_v_shapes = self.model.get_cache_kvs_shape(
            self.model.config, self.max_batch_size
        )
        cachekv_dtype = self.dtype if self.config.cachekv_int8_type is None else "uint8"
        self.cache_kvs = []
        if self.cache_k_shapes and self.cache_v_shapes:
            assert len(self.cache_k_shapes) == len(self.cache_v_shapes)
            for cache_k_shape, cache_v_shape in zip(self.cache_k_shapes, self.cache_v_shapes):
                self.cache_kvs.append(paddle.zeros(cache_k_shape, dtype=cachekv_dtype))
                self.cache_kvs.append(paddle.zeros(cache_v_shape, dtype=cachekv_dtype))
        else:
            # for mla's absorption
            assert self.cache_v_shapes is None
            self.cache_kvs = [paddle.zeros(shape, dtype=cachekv_dtype) for shape in self.cache_k_shapes]

        self.max_block_nums = self.cache_k_shapes[0][0]
        self.free_list = list(range(self.max_block_nums))
        self.used_list = [[] for _ in range(self.max_batch_size)]
        self.pre_ids = paddle.to_tensor(np.zeros((self.max_batch_size, self.total_max_length)).astype("int64") - 1)
        self.rope_theta = self.config.get("rope_theta", 10000.0)
        self.rope_scaling = self.config.get("rope_scaling", None)

        self.head_dim = self.cache_k_shapes[0][-1]
        if self.draft_type == "mtp":
            self.rope_emb = None
        else:
            self.rope_emb = llm_utils.get_rotary_position_embedding(
                paddle.arange(self.total_max_length).reshape((1, -1)),
                self.head_dim,
                self.rope_theta,
                self.rope_scaling,
            )

    def create_persistent_inputs(self):
        self.model_inputs = {}
        # same shape/dytpe with base model
        self.model_inputs["block_tables"] = paddle.clone(self.base_model_inputs["block_tables"])

        self.model_inputs["input_ids"] = paddle.clone(self.base_model_inputs["input_ids"])
        self.model_inputs["seq_lens_this_time"] = paddle.full(
            shape=[self.max_batch_size, 1], fill_value=-1, dtype="int32"
        )
        self.model_inputs["seq_lens_encoder"] = paddle.clone(self.base_model_inputs["seq_lens_encoder"])
        self.model_inputs["seq_lens_decoder"] = paddle.clone(self.base_model_inputs["seq_lens_decoder"])
        self.model_inputs["step_idx"] = paddle.clone(self.base_model_inputs["step_idx"])
        self.model_inputs["stop_flags"] = paddle.clone(self.base_model_inputs["stop_flags"])
        self.model_inputs["stop_nums"] = paddle.clone(self.base_model_inputs["stop_nums"])
        self.model_inputs["not_need_stop"] = paddle.to_tensor([False], dtype="bool", place="cpu")
        self.model_inputs["pre_ids"] = paddle.clone(self.base_model_inputs["pre_ids"])
        self.model_inputs["rope_emb"] = self.rope_emb
        self.model_inputs["cache_kvs"] = self.cache_kvs

        # reuse base model inputs
        self.model_inputs["top_p"] = self.base_model_inputs["top_p"]
        self.model_inputs["temperature"] = self.base_model_inputs["temperature"]
        self.model_inputs["eos_token_id"] = self.base_model_inputs["eos_token_id"]
        self.model_inputs["penalty_score"] = self.base_model_inputs["penalty_score"]
        self.model_inputs["frequency_score"] = self.base_model_inputs["frequency_score"]
        self.model_inputs["presence_score"] = self.base_model_inputs["presence_score"]
        self.model_inputs["max_length"] = self.base_model_inputs["max_length"]
        self.model_inputs["min_length"] = self.base_model_inputs["min_length"]
        self.model_inputs["bad_tokens"] = self.base_model_inputs["bad_tokens"]
        self.model_inputs["next_tokens"] = paddle.full(shape=[self.max_batch_size, 1], fill_value=-1, dtype="int64")
        self.model_inputs["base_model_draft_tokens"] = self.base_model_inputs["draft_tokens"]
        self.model_inputs["substep"] = 0

        self.model_inputs["draft_tokens"] = paddle.full(shape=[self.max_batch_size, 2], fill_value=-1, dtype="int64")

        self.first_token_record = paddle.full(shape=[self.max_batch_size, 1], fill_value=-1, dtype="int32")

    def run(self, share_inputs, **kwargs):
        self.run_preprocess(share_inputs, **kwargs)
        self.run_infer(share_inputs, **kwargs)
        self.run_postprocess(share_inputs, **kwargs)

    def create_temporary_inputs(self, real_bs, seq_lens):
        # real_bs = kwargs.get("real_bs")
        # seq_lens = kwargs.get("seq_lens")
        # base_model_inputs = kwargs.get("base_model_inputs")
        base_model_inputs = self.base_model_inputs
        self.model_inputs["block_tables"] = paddle.full_like(
            base_model_inputs["block_tables"], fill_value=-1, dtype="int32"
        )
        self.free_list = list(range(self.max_block_nums))  # Refresh on every new insert
        for i in range(real_bs):
            real_len = seq_lens[i] + self.max_length
            if real_len > self.total_max_length:
                raise ValueError(
                    f"input_len({seq_lens[i]}) + \
max_length({self.max_length}) > total_max_length({self.total_max_length})"
                )
            for j in range((real_len + self.args.block_size - 1) // self.args.block_size):
                used_block_id = self.free_list.pop()
                self.model_inputs["block_tables"][i, j] = used_block_id

        self.model_inputs["input_ids"] = paddle.clone(base_model_inputs["input_ids"])
        self.model_inputs["seq_lens_this_time"] = paddle.clone(base_model_inputs["seq_lens_this_time"])
        self.model_inputs["seq_lens_encoder"] = paddle.clone(base_model_inputs["seq_lens_encoder"])
        self.model_inputs["seq_lens_decoder"] = paddle.clone(base_model_inputs["seq_lens_decoder"])
        self.model_inputs["step_idx"] = paddle.clone(base_model_inputs["step_idx"])
        self.model_inputs["stop_flags"] = paddle.clone(base_model_inputs["stop_flags"])
        self.model_inputs["stop_nums"] = paddle.clone(base_model_inputs["stop_nums"])
        self.model_inputs["not_need_stop"] = paddle.to_tensor([False], dtype="bool", place="cpu")
        self.model_inputs["pre_ids"] = self.pre_ids
        self.model_inputs["rope_emb"] = self.rope_emb
        self.model_inputs["cache_kvs"] = self.cache_kvs
        self.model_inputs["top_p"] = base_model_inputs["top_p"]
        self.model_inputs["temperature"] = base_model_inputs["temperature"]
        self.model_inputs["eos_token_id"] = base_model_inputs["eos_token_id"]
        self.model_inputs["penalty_score"] = base_model_inputs["penalty_score"]
        self.model_inputs["frequency_score"] = base_model_inputs["frequency_score"]
        self.model_inputs["presence_score"] = base_model_inputs["presence_score"]
        self.model_inputs["max_length"] = base_model_inputs["max_length"]
        self.model_inputs["min_length"] = base_model_inputs["min_length"]
        self.model_inputs["bad_tokens"] = base_model_inputs["bad_tokens"]
        self.model_inputs["next_tokens"] = paddle.full(shape=[self.max_batch_size, 1], fill_value=-1, dtype="int64")
        self.model_inputs["base_model_draft_tokens"] = base_model_inputs["draft_tokens"]
        self.model_inputs["draft_tokens"] = paddle.full(shape=[self.max_batch_size, 2], fill_value=-1, dtype="int64")

        self.first_token_record = paddle.full(shape=[self.max_batch_size, 1], fill_value=-1, dtype="int32")
        self.model_inputs["substep"] = 0
        for i in range(real_bs):
            self.model_inputs["pre_ids"][i, 0] = self.model_inputs["input_ids"][i, -1]
            self.first_token_record[i : i + 1] = seq_lens[i]

        if self.draft_type in ["ealge", "mtp"]:
            self.model_inputs["input_ids"][:, :-1] = base_model_inputs["input_ids"][:, 1:]

    def dynamic_insert(self, task, idx):
        # input_ids is different in ['mtp/ealge', 'draft_model'].
        length = len(task["input_ids"])
        if self.draft_type in ["eagle", "mtp"]:
            self.model_inputs["input_ids"][idx : idx + 1, : length - 1] = self.base_model_inputs["input_ids"][
                idx : idx + 1, 1:length
            ]
        else:
            self.model_inputs["input_ids"][idx : idx + 1, :length] = self.base_model_inputs["input_ids"][
                idx : idx + 1, :length
            ]

        self.model_inputs["pre_ids"][idx : idx + 1] = -1
        self.model_inputs["seq_lens_this_time"][idx : idx + 1] = 0
        self.model_inputs["seq_lens_encoder"][idx : idx + 1] = 0
        self.model_inputs["seq_lens_decoder"][idx : idx + 1] = 0
        self.model_inputs["step_idx"][idx : idx + 1] = 0
        self.model_inputs["stop_flags"][idx : idx + 1] = True
        self.first_token_record[idx : idx + 1] = length

        real_len = length + self.model_inputs["max_length"][idx].item()
        real_len = min(real_len, self.total_max_length)
        need_block_num = min((real_len + self.block_size - 1) // self.block_size, self.max_query_block_num)
        for i in range(need_block_num):
            used_block_id = self.free_list.pop()
            self.used_list[idx].append(used_block_id)
        self.model_inputs["block_tables"][idx : idx + 1, :need_block_num] = paddle.to_tensor(
            self.used_list[idx], dtype="int32"
        )

    def insert_query(self, **kwargs):
        if self.args.serving_mode:
            task = kwargs["task"]
            idx = kwargs["idx"]
            self.dynamic_insert(task, idx)
        else:
            real_bs = kwargs["real_bs"]
            seq_lens = kwargs["seq_lens"]
            self.base_model_inputs = kwargs["base_model_inputs"]
            self.create_temporary_inputs(
                real_bs=real_bs,
                seq_lens=seq_lens,
            )

    def run_preprocess(self, share_inputs, **kwargs):
        """
        update draft model parameteds
        """
        # if kwargs.get("insert_step", 0):
        #     self.actual_draft_token_num = 1

        draft_model_preprocess(
            self.model_inputs["draft_tokens"],
            self.model_inputs["input_ids"],
            self.model_inputs["stop_flags"],
            self.model_inputs["seq_lens_this_time"],
            self.model_inputs["seq_lens_encoder"],
            self.model_inputs["seq_lens_decoder"],
            self.model_inputs["step_idx"],
            self.first_token_record,
            self.model_inputs["not_need_stop"],
            share_inputs["accept_tokens"],
            share_inputs["accept_num"],
            share_inputs["seq_lens_encoder"],
            share_inputs["seq_lens_decoder"],
            share_inputs["step_idx"],
            share_inputs["stop_flags"],
            share_inputs["draft_tokens"],
            self.actual_draft_token_num,
            self.draft_type in ["eagle", "mtp"],
        )

    def run_infer(self, share_inputs, **kwargs):
        """
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses mut implement this function")

    def run_postprocess(self, share_inputs, **kwargs):
        """
        Update base model draft_tokens
        """
        draft_model_postprocess(
            share_inputs["draft_tokens"],
            share_inputs["seq_lens_this_time"],
            share_inputs["seq_lens_encoder"],
            share_inputs["stop_flags"],
        )
        # if kwargs.get("insert_step", 0):
        #     self.actual_draft_token_num = 1

    def postprocess(self):
        for i in range(self.max_batch_size):
            if self.base_model_inputs["stop_flags"][i] and len(self.used_list[i]) > 0:
                while len(self.used_list[i]) > 0:
                    block_id = self.used_list[i].pop()
                    self.free_list.append(block_id)


class EagleProposer(ModelProposer):
    """
    用于 Eagle/MTP 的 Proposer
    在输入输出中匹配符合的tokens作为 draft tokens
    """

    def __init__(self, args: SpeculateArgument, **kwargs):
        super().__init__(args, **kwargs)
        self.last_seq_lens_this_time = paddle.full(shape=[self.max_batch_size, 1], fill_value=-1, dtype="int32")

    def run_infer(self, share_inputs, **kwargs):
        base_model_full_hidden_states = kwargs["base_model_full_hidden_states"]
        if self.model_inputs["not_need_stop"]:
            base_model_hidden_states = eagle_get_base_model_hidden_states(
                base_model_full_hidden_states,
                self.model_inputs["seq_lens_this_time"],
                self.model_inputs["seq_lens_encoder"],
                self.model_inputs["seq_lens_decoder"],
                self.model_inputs["stop_flags"],
                share_inputs["accept_num"],
                share_inputs["seq_lens_this_time"],
                share_inputs["seq_lens_encoder"],
                self.actual_draft_token_num,
            )
            self.model_inputs["hidden_states"] = base_model_hidden_states

        with paddle.no_grad():
            self.model_inputs["substep"] = 0
            while self.model_inputs["not_need_stop"] and self.model_inputs["substep"] < self.actual_draft_token_num:
                self.last_seq_lens_this_time[:] = self.model_inputs["seq_lens_this_time"][:]
                output_hidden_states = self.model.generate(**self.model_inputs)
                self.model_inputs["substep"] += 1
                if self.model_inputs["not_need_stop"] and self.model_inputs["substep"] < self.actual_draft_token_num:
                    self.model_inputs["hidden_states"] = eagle_get_self_hidden_states(
                        output_hidden_states,
                        self.last_seq_lens_this_time,
                        self.model_inputs["seq_lens_this_time"],
                        self.model_inputs["step_idx"],
                    )
                else:
                    self.model_inputs["hidden_states"] = None
