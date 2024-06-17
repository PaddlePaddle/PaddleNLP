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
import json
import os

import paddle
from paddle import nn
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddle.quantization import PTQ, QAT, QuantConfig
from paddleslim.quant.advanced import (
    GPTQ,
    AutoClip,
    AWQSearch,
    EMASampler,
    MultiStepSampler,
    PieceWiseSearch,
    Shift,
    Smooth,
)
from paddleslim.quant.advanced.utils import find_parent_layer_and_sub_name
from paddleslim.quant.layers import (
    QuantizedColumnParallelLinear,
    QuantizedRowParallelLinear,
)
from paddleslim.quant.observers import (
    AbsMaxChannelWiseWeightObserver,
    AVGObserver,
    GroupWiseWeightObserver,
)
from paddleslim.quant.observers.abs_max_weight import (
    AbsMaxChannelWiseWeightObserverLayer,
)
from paddleslim.quant.observers.avg import AVGObserverLayer
from paddleslim.quant.observers.groupwise import GroupWiseWeightObserverLayer

from paddlenlp.peft import PrefixModelForCausalLM
from paddlenlp.peft.lora import (
    ColumnParallelLoRALinear,
    LoRALinear,
    RowParallelLoRALinear,
)
from paddlenlp.peft.lora.lora_quant_layers import (
    ColumnParallelQuantedLoRALinear,
    QuantedLoRALinear,
    RowParallelQuantedLoRALinear,
)
from paddlenlp.utils.log import logger


def create_qat_model(quant_args, model, dtype):
    from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
    from paddleslim.quant.quanters import (
        FakeQuanterChannelWiseAbsMaxObserver,
        PACTQuanter,
    )

    q_config = QuantConfig(activation=None, weight=None)
    q_config.add_qat_layer_mapping(LoRALinear, QuantedLoRALinear)
    q_config.add_qat_layer_mapping(RowParallelLoRALinear, RowParallelQuantedLoRALinear)
    q_config.add_qat_layer_mapping(ColumnParallelLoRALinear, ColumnParallelQuantedLoRALinear)
    if quant_args.quant_type == "a8w8":
        activation = PACTQuanter(quanter=FakeQuanterWithAbsMaxObserver(), init_value=20.0, dtype=dtype)
        weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=8, dtype="float32")
    elif quant_args.quant_type == "weight_only_int4":
        activation = None
        weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=4, dtype="float32")
    elif quant_args.quant_type == "weight_only_int8":
        activation = None
        weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=8, dtype="float32")
    else:
        raise ValueError("quant_type should be one of ['a8w8', 'weight_only_int4', 'weight_only_int8']")

    q_config.add_type_config(RowParallelLoRALinear, weight=weight, activation=activation)
    q_config.add_type_config(ColumnParallelLoRALinear, weight=weight, activation=activation)
    q_config.add_type_config(LoRALinear, weight=weight, activation=activation)
    q_config.add_type_config(nn.Linear, weight=weight, activation=activation)

    qat = QAT(q_config)
    model = qat.quantize(model, inplace=True)
    return model


def apply_shift(quant_args, trainer, ptq_dataloader, ptq_model_config):
    logger.info("***** Running Shift *****")
    shift_sampler = EMASampler() if quant_args.shift_sampler == "ema" else None
    shift = Shift(
        model=trainer.model,
        model_config=ptq_model_config,
        sample_function=shift_sampler,
        shift_all_linears=quant_args.shift_all_linears,
    )
    with paddle.no_grad():
        trainer.ptq_loop(
            ptq_dataloader,
            description="Shift",
            max_eval_iters=quant_args.shift_step,
        )
        shift.update_weight()
    del shift, shift_sampler
    logger.info("***** Shift done *****")


def apply_smooth(quant_args, trainer, ptq_dataloader, ptq_model_config):

    if quant_args.do_awq:
        logger.info("***** Running AWQ *****")
    else:
        logger.info("***** Running Smooth *****")
    smooth_sampler = MultiStepSampler() if quant_args.smooth_sampler == "multi_step" else None
    if quant_args.smooth_piecewise_search:
        search_func = PieceWiseSearch(
            k_piece=quant_args.smooth_k_piece,
            bits_length=8,
            search_piece=quant_args.smooth_search_piece,
            search_alpha_min=0.2,
            search_alpha_max=0.8,
            search_scale_min=1.0,
            search_scale_max=5.0,
            weight_quant_method="abs_max_channel_wise",
            act_quant_method="avg",
        )
    elif quant_args.do_awq:
        search_func = AWQSearch(
            n_grid=20,
            bits_length=4,
            weight_quant_method=quant_args.weight_quant_method,
        )
    else:
        search_func = None
    smooth = Smooth(
        trainer.model,
        ptq_model_config,
        alpha=0.5,
        smooth_all_linears=quant_args.smooth_all_linears,
        sample_function=smooth_sampler,
        search_function=search_func,
        smooth_method="awq" if quant_args.do_awq else "smoothquant",
    )
    with paddle.no_grad():
        trainer.ptq_loop(
            ptq_dataloader,
            description="Smooth",
            max_eval_iters=quant_args.smooth_step,
        )

        smooth.update_weight()
    del smooth, smooth_sampler, search_func
    logger.info("***** Smooth done *****")


def apply_autoclip(quant_args, trainer, ptq_dataloader):
    """
    AutoClip
    """
    print("-------------------Start AutoClip------------------")
    sampler = MultiStepSampler()
    auto_clip = AutoClip(
        trainer.model,
        weight_bits=4,
        weight_quant_method=quant_args.weight_quant_method,
        sample_function=sampler,
        n_grid=20,
        max_shrink=0.5,
    )
    with paddle.no_grad():
        trainer.ptq_loop(
            ptq_dataloader,
            description="AutoClip",
            max_eval_iters=quant_args.autoclip_step,
        )
        auto_clip.auto_clip()
    del sampler, auto_clip
    logger.info("***** AutoClip done *****")


def apply_ptq(quant_args, trainer, ptq_dataloader):
    logger.info("***** Running PTQ *****")
    q_config = QuantConfig(activation=None, weight=None)
    if quant_args.weight_quant_method == "abs_max_channel_wise":
        weight_observer = AbsMaxChannelWiseWeightObserver
    elif quant_args.weight_quant_method == "groupwise":
        weight_observer = GroupWiseWeightObserver
    else:
        raise ValueError("weight_quant_method should be one of ['abs_max_channel_wise', 'groupwise']")

    if quant_args.quant_type == "a8w8(int)":
        activation = AVGObserver(quant_bits=8) #INT8
        weight = weight_observer(quant_bits=8) #INT8
    if quant_args.quant_type == "a8w8(fp)":
        activation = AbsmaxObserver(quant_bits=(4,3)) #FP8
        weight = AbsmaxObserver(quant_bits=(4,3)) #FP8
    elif quant_args.quant_type == "weight_only_int4":
        activation = None
        weight = weight_observer(quant_bits=4) #INT4
    elif quant_args.quant_type == "weight_only_int8":
        activation = None
        weight = weight_observer(quant_bits=8) #INT8
    else:
        raise ValueError("quant_type should be one of ['a8w8(int)', 'a8w8(fp)' 'weight_only_int4', 'weight_only_int8']")

    q_config.add_qat_layer_mapping(ColumnParallelLinear, QuantizedColumnParallelLinear)
    q_config.add_qat_layer_mapping(RowParallelLinear, QuantizedRowParallelLinear)
    q_config.add_type_config(
        [paddle.nn.Linear, ColumnParallelLinear, RowParallelLinear, QuantedLoRALinear],
        activation=activation,
        weight=weight,
    )

    ptq = PTQ(q_config)
    trainer.model = ptq.quantize(trainer.model, inplace=True)
    trainer.ptq_loop(
        ptq_dataloader,
        description="PTQ",
        max_eval_iters=quant_args.ptq_step,
    )
    weight_scales = {}
    act_scales = {}
    for cur_name, cur_layer in trainer.model.named_sublayers():
        if isinstance(cur_layer, AbsMaxChannelWiseWeightObserverLayer):
            if "_observer" not in cur_name:
                weight_scales[cur_name] = cur_layer.scales().numpy().tolist()
        if isinstance(cur_layer, GroupWiseWeightObserverLayer):
            if "_observer" not in cur_name:
                weight_scales[cur_name] = cur_layer.scales().numpy().tolist()
        if isinstance(cur_layer, AVGObserverLayer):
            if "_observer" not in cur_name:
                act_scales[cur_name] = cur_layer.scales().numpy().tolist()
    weight_scales_path = os.path.join(trainer.args.output_dir, "weight_scales.json")
    with open(weight_scales_path, "w") as f:
        json.dump(weight_scales, f)
    logger.info(f"Weight scales saved in {weight_scales_path}.")

    act_scales_path = os.path.join(trainer.args.output_dir, "act_scales.json")
    with open(act_scales_path, "w") as f:
        json.dump(act_scales, f)
    logger.info(f"Activation scales saved in {act_scales_path}.")

    trainer.model = ptq.convert(trainer.model, inplace=True)
    logger.info("***** PTQ done *****")


def apply_gptq(quant_args, trainer, ptq_dataloader):
    logger.info("***** Running GPTQ *****")
    num_layer = 0
    model = trainer.model
    for cur_name, cur_layer in model.named_sublayers():
        if type(cur_layer) in [paddle.nn.Linear, ColumnParallelLinear, RowParallelLinear]:
            num_layer += 1
            logger.info(f"GPTQ layer: {num_layer}, {cur_name}")
            parent_layer, sub_name = find_parent_layer_and_sub_name(model, cur_name)
            cur_quant_layer = GPTQ(cur_layer)
            setattr(parent_layer, sub_name, cur_quant_layer)
            with paddle.no_grad():
                trainer.ptq_loop(
                    ptq_dataloader,
                    description="GPTQ",
                    max_eval_iters=quant_args.gptq_step,
                )
                cur_quant_layer.fasterquant(percdamp=0.1, groupsize=-1, actorder=True)
            del cur_quant_layer
            setattr(parent_layer, sub_name, cur_layer)
    logger.info("***** GPTQ done *****")


def get_ptq_model_config(model):
    if isinstance(model, PrefixModelForCausalLM):
        base_model_prefix = model.model.base_model_prefix
    else:
        base_model_prefix = model.base_model_prefix

    if base_model_prefix in ["chatglm"]:
        raise NotImplementedError(f"{model} does not support Shift or Smooth.")
    elif base_model_prefix == "chatglm_v2":
        model_config = {"fused_qkv": False, "parallel_ffn": False, "skip_norm_list": ["rms_norm_56"]}
    elif base_model_prefix == "bloom":
        model_config = {"fused_qkv": True, "parallel_ffn": False}
    elif base_model_prefix == "llama":
        model_config = {"fused_qkv": False, "parallel_ffn": True}
    else:
        raise ValueError(
            f"Unknown base_model_prefix: {model.base_model_prefix}. Supported base_model_prefix list: chatglm_V2, bloom, llama."
        )
    return model_config
