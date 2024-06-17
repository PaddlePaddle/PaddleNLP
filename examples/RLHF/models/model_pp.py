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

from .pp_model_utils import fwd_args_to_dict, get_expected_keys, pad_batches_inputs
from .ppo_model_utils import (
    RLHFPPOMixedLoss,
    RLHFValueLoss,
    create_loss,
    make_position_ids,
)
from .score_model_utils import ScoreModelMixin

# patches for base pipe model
# non-pipe model class, can be used to parse and convert forward args
# mainly used for generation with PipelienParallel model
LlamaForCausalLMPipe._non_pipe_model_class = LlamaForCausalLM
LlamaForCausalLMPipe._non_pipe_decoder_layer_class = LlamaDecoderLayer


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
        """mainly for eval/generation with PipelineParallel"""
        # None means to use actual data info
        return paddle.static.InputSpec(shape=[None, None, self.config.vocab_size], dtype=None)


class _LlamaRMSNormPipe(LlamaRMSNormPipe):
    """
    We need position_ids for reward model, so wrap LlamaRMSNormPipe to pass position_ids
    """

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
