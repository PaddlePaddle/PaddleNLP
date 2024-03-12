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

import importlib
import inspect
import types

import paddle
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import LayerDesc

from paddlenlp.transformers import LlamaForCausalLM, LlamaForCausalLMPipe
from paddlenlp.transformers.llama.modeling import LlamaDecoderLayer
from paddlenlp.transformers.llama.modeling_pp import (
    LlamaRMSNormPipe,
    parse_args,
    return_args,
)

from .ppo_model_utils import RLHFPPOMixedLoss, RLHFValueLoss, create_loss
from .score_model_utils import ScoreModelMixin


def print_patch(func, output, *args, **kwargs):
    return
    print("=" * 20, func.__name__, output)


def fwd_step_patch(func, output, self, *args, **kwargs):
    # training patch
    if self.training and self.is_pipeline_last_stage():
        if getattr(self, "_step_losses", None):
            self._step_losses.append(output.detach())
        else:
            self._step_losses = [output.detach()]


# def fwd_step_eval_patch(func, output, self, *args, **kwargs):
#     # eval patch for actor/reference model
#     logits = output
#     # sequence = self.
#     log_probs = gather_log_probabilities(logits[:, :-1], sequence[:, 1:])
#     if self.is_pipeline_first_stage():
#         if getattr(self, "_step_losses", None):
#             self._step_losses.append(output.detach())
#         else:
#             self._step_losses = [output.detach()]
#         print("=" * 20, "fwd_step_patch", len(self._step_losses))


def make_wrapper(func, pre_patch=None, post_patch=None):
    def wrapper(*args, **kwargs):
        if pre_patch is not None:
            pre_patch(func, None, *args, **kwargs)
        output = func(*args, **kwargs)
        # print("=" * 20, func.__name__, output)
        if post_patch is not None:
            post_patch(func, output, *args, **kwargs)
        return output

    return wrapper


funcs = [
    paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.P2pHelper.send_forward_recv_backward,
    paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.P2pHelper.send_backward_recv_forward,
    paddle.distributed.fleet.model.PipelineParallel._backward_step,
    paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.P2pHelper.recv_backward,
    paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.P2pHelper.send_backward,
    (paddle.distributed.fleet.model.PipelineParallel._forward_step, fwd_step_patch),
    paddle.distributed.fleet.meta_parallel.pipeline_parallel.FakeMicroDataset._load_micro_batch,
    paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.P2pHelper.recv_forward,
    paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.P2pHelper.send_forward,
    paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.SendRecvMeta.recv_meta,
    paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.SendRecvMeta.send_meta,
]

for func in funcs:
    if isinstance(func, tuple):
        fun, patch = func
    else:
        fun, patch = func, print_patch
    module = importlib.import_module(fun.__module__)
    cls_name = fun.__qualname__[: -len(fun.__name__) - 1]
    wrap_fun = make_wrapper(fun, post_patch=patch)
    cls_obj = getattr(module, cls_name)
    setattr(cls_obj, fun.__name__, wrap_fun)


# _raw_load_micro_batch = paddle.distributed.fleet.meta_parallel.pipeline_parallel.FakeMicroDataset._load_micro_batch
# _raw_forward_step = paddle.distributed.fleet.model.PipelineParallel._forward_step
# _raw_recv_forward = paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.P2pHelper.recv_forward
# _raw_send_forward = paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.P2pHelper.send_forward
# _raw_recv_meta = paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.SendRecvMeta.recv_meta
# _raw_send_meta = paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.SendRecvMeta.send_meta


# def _load_micro_batch(self, micro_step):
#     output = _raw_load_micro_batch(self, micro_step)
#     print("=" * 20, "_load_micro_batch", output)
#     return output

# def _forward_step(self, input_tensor, micro_dataset, chunk_id=None):
#     if True: # self.is_pipeline_first_stage():
#         print("=" * 20, "_forward_step input", input_tensor, self._p2p_helper._use_cache)
#     output = _raw_forward_step(self, input_tensor, micro_dataset, chunk_id)
#     print("=" * 20, "_forward_step output", output, self._p2p_helper._use_cache)
#     return output


# def recv_forward(self, pp_first_stage, sync_recv=True):
#     input_tensor = _raw_recv_forward(self, pp_first_stage, sync_recv)
#     print("=" * 20, "recv_forward", input_tensor)
#     return input_tensor


# def send_forward(self, output_tensor, pp_last_stage):
#     output = _raw_send_forward(self, output_tensor, pp_last_stage)
#     print("=" * 20, "send_forward", output_tensor)
#     return output


# def recv_meta(self, group):
#     output = _raw_recv_meta(self, group)
#     print("=" * 20, "recv_meta", self.recv_shape_message, self.recv_dtype_message)
#     return output


# def send_meta(self, tensor, group):
#     output = _raw_send_meta(self, tensor, group)
#     print("=" * 20, "send_meta", self.send_shape_message, self.send_dtype_message)
#     return output

# paddle.distributed.fleet.model.PipelineParallel._forward_step = _forward_step
# paddle.distributed.fleet.meta_parallel.pipeline_parallel.FakeMicroDataset._load_micro_batch = _load_micro_batch
# paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.P2pHelper.recv_forward = recv_forward
# paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.P2pHelper.send_forward = send_forward
# paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.SendRecvMeta.recv_meta = recv_meta
# paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication.SendRecvMeta.send_meta = send_meta


def loss_fwd_wrapper(loss_maker):
    def _wrapper(*args, **kwargs):
        loss = loss_maker(*args, **kwargs)
        ori_fwd = loss.forward

        def _fwd(self, output, label_info):
            if isinstance(label_info, tuple):
                loss = ori_fwd(self, output, *label_info)
            else:
                loss = ori_fwd(self, output, label_info)
            return loss

        loss.forward = types.MethodType(_fwd, loss)
        return loss

    return _wrapper


@paddle.no_grad()
def make_position_ids(attention_mask, source=None):
    attention_mask_bool = attention_mask
    attention_mask = attention_mask.cast(paddle.int64)
    position_ids = attention_mask.cumsum(-1) - 1
    # Make padding positions in source be 0, since reward model use position_ids
    # plus with padding size (number of 0s) in source to calculate end offsets.
    # It does not matter when source is left padding and target is right padding
    # which is the output of non-FuseMT generation, while when using FuseMT whose
    # output is right padding source and right padding target, we have to set
    # padding positions in source be 0 to make compatible.
    if source is not None:
        src_len = position_ids[:, source.shape[-1] - 1].unsqueeze(-1)
        position_ids = paddle.where(
            paddle.logical_and(paddle.logical_not(attention_mask_bool), position_ids <= src_len),
            attention_mask,
            position_ids,
        )
        return position_ids
    position_ids = paddle.where(position_ids == -1, attention_mask, position_ids)
    return position_ids


@paddle.no_grad()
def pad_batches_inputs(inputs, padding_value=0, max_len=None, pad_len=None):
    """Pad length for tensors shaped [bs, seq_len] to [bs, max(seq_lens)]"""
    if pad_len is not None:
        pad_len = [pad_len] * len(inputs) if isinstance(pad_len, int) else pad_len
    elif max_len is None:
        # max_len = max([x.shape[-1] for x in inputs if x is not None])
        max_len = max([x.shape[-1] if isinstance(x, paddle.Tensor) else 0 for x in inputs])
        pad_len = [max_len - x.shape[-1] if isinstance(x, paddle.Tensor) else 0 for x in inputs]
    for i in range(len(inputs)):
        x = inputs[i]
        # if x is None or x.shape[-1] == max_len:
        if not isinstance(x, paddle.Tensor) or x.shape[-1] == max_len:
            continue
        inputs[i] = paddle.concat([x, paddle.full([x.shape[0], pad_len[i]], padding_value, dtype=x.dtype)], -1)
    return inputs


def get_expected_keys(inputs, keys):
    ret = tuple([inputs.get(k, None) for k in keys if k in inputs])
    if len(ret) == 1:
        ret = ret[0]
    return ret


# patches for base pipe model
# non-pipe model class, can be used to parse and convert forward args
LlamaForCausalLMPipe._non_pipe_model_class = LlamaForCausalLM
LlamaForCausalLMPipe._non_pipe_decoder_layer_class = LlamaDecoderLayer


def fwd_args_to_dict(fun):
    def _impl(self, *args, **kwargs):
        try:
            return fun(self, *args, **kwargs)
        except TypeError:
            # otherwise, inputs is any valid format of non_pipe_model forward args,
            # convert to dict, to support more args format in prediction_pipeline_step

            arg_dict = (
                inspect.signature(self._non_pipe_model_class.forward).bind(*((self,) + args), **kwargs).arguments
            )
            arg_dict.pop("self")
            return fun(self, arg_dict)

    return _impl


class LlamaPolicyPipe(LlamaForCausalLMPipe):
    # TODO(guosheng): maybe make a Mixin is better

    @fwd_args_to_dict
    def _prepare_pipeline_inputs_func(self, inputs):
        # first_stage_keys = ["input_ids", "attention_mask"]
        first_stage_keys = ["input_ids", "attention_mask", "position_ids"]
        # last_stage_keys = [
        #     "labels", "input_ids", "log_probs", "advantages", "sequence_mask"
        # ]
        # TODO(guosheng): make input keys same with model arg names, maybe we
        # can use inspect and set as global var which can then be used here and
        # in PPOTrainer.
        last_stage_keys = ["labels", "input_ids", "old_log_probs", "reward_advantages", "sequence_mask"]

        if type(inputs) is dict:
            # for left padding, position_ids is nececessary
            if "position_ids" not in inputs:
                inputs["position_ids"] = make_position_ids(inputs["attention_mask"])
            # ppo-loss and ptx-loss need different labels, and data iter provides
            # corrensponding data, thus add the not provided fields here.
            # policy trian and infer has different inputs, infer uses position_ids.
            # for key in last_stage_keys:
            for key in first_stage_keys + last_stage_keys:
                if key not in inputs:
                    inputs[key] = None
            return [
                get_expected_keys(inputs, first_stage_keys),
                get_expected_keys(inputs, last_stage_keys),
            ]

        for data in inputs:
            # for key in last_stage_keys:
            for key in first_stage_keys + last_stage_keys:
                if key not in data:
                    if key == "position_ids":
                        data[key] = make_position_ids(data["attention_mask"])
                        continue
                    data[key] = None
        # keys = list(inputs[0].keys())
        inputs_batch = {key: [data.get(key) for data in inputs] for key in first_stage_keys + last_stage_keys}
        # NOTE(guosheng): PipelineParallel requires send/recv tensors among
        # micro-batches/accu-steps have the same shape. Thus pad here, maybe
        # should make data collator do padding and pad optionally here, since
        # padding strategy may not be clear here.
        # 1. For input_ids/attention_mask/labels (prompt+target) padding:
        # Some data fields, such as input_ids/attention_mask/labels, should
        # have same shape after padding, and each of them cannot pad only
        # according to its own max length which might be different since the
        # filed value is None for different batches/tasks.
        src_tgt_keys = ["input_ids", "attention_mask", "labels", "position_ids"]
        max_len = max([x.shape[-1] for x in inputs_batch["input_ids"]])
        pad_len = [max_len - x.shape[-1] for x in inputs_batch["input_ids"]]
        for key in src_tgt_keys:
            # Do not pad position_ids with 0 since 0s in position_ids has special
            # usage in reward model. We use 1 to pad.
            padding_value = self._ignore_index if key == "labels" else 1 if key == "position_ids" else 0
            inputs_batch[key] = pad_batches_inputs(inputs_batch[key], padding_value, pad_len=pad_len)
        # 2. For old_log_probs/reward_advantages/sequence_mask (target) padding:
        # hard to pad acorss batches, think in some cases one batch might have the
        # longest prompt+target length but the shortest target lengh, which might
        # cause mismatch between inputs with prompt+target length and labels with
        # target length. NOTE: however trick can be used here, label fields with
        # target length such as old_log_probs/reward_advantages/sequence_mask do
        # not need to join comm and thus there is no need to keep same shape among
        # batches of accumulation steps, they just need to pad as prompt+target
        # fields such as input_ids.
        tgt_keys = ["old_log_probs", "reward_advantages", "sequence_mask"]
        for key in tgt_keys:
            padding_value = 0
            inputs_batch[key] = pad_batches_inputs(inputs_batch[key], padding_value, pad_len=pad_len)
        # for key, value in inputs_batch.items():
        #     padding_value = self._ignore_index if key == "labels" else 0
        #     max_len = max_len if key in [
        #         "input_ids", "attention_mask", "labels"
        #     ] else None
        #     inputs_batch[key] = pad_batches_inputs(value, padding_value, max_len)
        return [
            get_expected_keys(inputs_batch, first_stage_keys),
            get_expected_keys(inputs_batch, last_stage_keys),
        ]

    def __init__(self, config, **kwargs):
        # NOTE: make _sequential_layers/_single_to_pp_mapping/_pp_to_single_mapping
        # instance attrs instead of class attrs to support more than one pipeline
        # models. Maybe make all sequential_layers add once.
        self._sequential_layers = []
        self._single_to_pp_mapping = None
        self._pp_to_single_mapping = None
        # To be consistent with score model init and allow hyper-param be passed
        # using __init__/from_pretrained
        self._init_kwargs = kwargs
        super().__init__(config)
        self._ignore_index = self._loss_fn.sft_criterion.ignore_index

    def get_loss_fn(self, config):
        return create_loss(RLHFPPOMixedLoss, config, self._init_kwargs)

    @property
    def head_out_meta(self):
        # None means to use actual data info
        return paddle.static.InputSpec(shape=[None, None, self.config.vocab_size], dtype=None)


class _LlamaRMSNormPipe(LlamaRMSNormPipe):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, args):
        hidden_states, attention_mask, position_ids, alibi = parse_args(args)
        return return_args(self.norm(hidden_states), attention_mask, position_ids)


# LayerDesc of PipelineParallel requires head to be a nn.Layer
class ValueHead(nn.Layer, ScoreModelMixin):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.init_score_head(config, hidden_size=config.hidden_size, **kwargs)

    def forward(self, args):
        # attention_mask passed from pre-stage is shaped (bs, 1, seq_len, seq_len)
        hidden_state, attention_mask, position_ids, alibi = parse_args(args)
        outputs = self.get_score(
            hidden_state, attention_mask=attention_mask, position_ids=position_ids, return_dict=True
        )
        return outputs


class LlamaValuePipe(LlamaForCausalLMPipe):
    # TODO(guosheng): maybe make a Mixin is better

    @fwd_args_to_dict
    def _prepare_pipeline_inputs_func(self, inputs):
        # ValueHead/get_score needs original attention_mask or position_ids,
        # while attention_mask passed from pre-stage is not the original, thus
        # hack for position_ids here.
        # Maybe add position_ids into inputs later and use position_ids instead
        # of attention_mask to get score not only for pipeline parallel.
        first_stage_keys = ["input_ids", "attention_mask", "position_ids"]
        # TODO(guosheng): make input keys same with model arg names, maybe we
        # can use inspect and set as global var which can then be used here and
        # in PPOTrainer.
        last_stage_keys = ["old_reward_values", "reward_returns", "sequence_mask"]

        if type(inputs) is dict:
            if "position_ids" not in inputs:
                inputs["position_ids"] = make_position_ids(inputs["attention_mask"])

            return [
                get_expected_keys(inputs, first_stage_keys),
                get_expected_keys(inputs, last_stage_keys),
            ]

        for data in inputs:
            if "position_ids" not in data:
                data["position_ids"] = make_position_ids(data["attention_mask"])
        # keys = list(inputs[0].keys())
        inputs_batch = {key: [data.get(key) for data in inputs] for key in first_stage_keys + last_stage_keys}
        # 1. For input_ids/attention_mask (prompt+target) padding:
        # src_tgt_keys = ["input_ids", "attention_mask"]
        src_tgt_keys = ["input_ids", "attention_mask", "position_ids"]
        max_len = max([x.shape[-1] for x in inputs_batch["input_ids"]])
        pad_len = [max_len - x.shape[-1] for x in inputs_batch["input_ids"]]
        for key in src_tgt_keys:
            # Do not pad position_ids with 0 since 0s in position_ids has special
            # usage in reward model. We use 1 to pad.
            padding_value = self._ignore_index if key == "labels" else 1 if key == "position_ids" else 0
            inputs_batch[key] = pad_batches_inputs(inputs_batch[key], padding_value, pad_len=pad_len)
        # 2. For old_reward_values/reward_returns/sequence_mask (target) padding:
        tgt_keys = ["old_reward_values", "reward_returns", "sequence_mask"]
        for key in tgt_keys:
            padding_value = 0
            inputs_batch[key] = pad_batches_inputs(inputs_batch[key], padding_value, pad_len=pad_len)
        # for key, value in inputs_batch.items():
        #     inputs_batch[key] = pad_batches_inputs(value, padding_value=0)
        # if "position_ids" not in inputs[0]:
        #     inputs_batch["position_ids"] = [
        #         make_position_ids(attention_mask) for attention_mask in inputs_batch["attention_mask"]
        #     ]
        return [
            get_expected_keys(inputs_batch, first_stage_keys),
            get_expected_keys(inputs_batch, last_stage_keys),
        ]

    def __init__(self, config, **kwargs):
        # NOTE: make _sequential_layers/_single_to_pp_mapping/_pp_to_single_mapping
        # instance attrs instead of class attrs to support more than one pipeline
        # models. Maybe make all sequential_layers add once.
        self._sequential_layers = []
        self._single_to_pp_mapping = None
        self._pp_to_single_mapping = None
        # To be consistent with score model init and allow hyper-param be passed
        # using __init__/from_pretrained
        self._init_kwargs = kwargs
        super().__init__(config)

    def add_head(self, config):
        init_kwargs = self._init_kwargs
        # hack to replace original RMSNormPipe to support ValueHead inputs
        norm_prefix = self._sequential_layers.pop(-1)["name_prefix"]
        self.add_sequential_layer(LayerDesc(_LlamaRMSNormPipe, config=config), norm_prefix)
        self.add_sequential_layer(LayerDesc(ValueHead, config, **init_kwargs), "")

    def get_loss_fn(self, config):
        return create_loss(RLHFValueLoss, config, self._init_kwargs)

    @property
    def head_out_meta(self):
        # None means to use actual data info
        return (
            paddle.static.InputSpec(shape=[None, None, 1], dtype=None),
            paddle.static.InputSpec(shape=[None, 1], dtype=None),
        )
