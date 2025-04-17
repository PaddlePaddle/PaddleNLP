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
import json
import os

import paddle
from paddle import nn
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddle.quantization import PTQ, QAT, QuantConfig
from paddle.quantization.base_observer import BaseObserver
from paddleslim.common.wrapper_function import FuncWrapper
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
from paddleslim.quant.layers.custom_attention import QuantizedCustomAttentionLayer
from paddleslim.quant.observers import (
    AbsMaxChannelWiseWeightObserver,
    GroupWiseWeightObserver,
)
from paddleslim.quant.observers.abs_max import AbsmaxObserver
from paddleslim.quant.observers.abs_max_headwise import AbsMaxHeadwiseObserver
from paddleslim.quant.observers.avg import AVGObserver
from paddleslim.quant.observers.avg_headwise import AvgHeadwiseObserver
from paddleslim.quant.observers.channel_wise import ChannelWiseObserver

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

ACT_OBSERVER = dict(
    abs_max=AbsmaxObserver,
    avg=AVGObserver,
)

WEIGHT_OBSERVER = dict(
    abs_max_channel_wise=AbsMaxChannelWiseWeightObserver,
    groupwise=GroupWiseWeightObserver,
)

CACHEKV_OBSERVER = dict(
    abs_max_headwise=AbsMaxHeadwiseObserver,
    avg_headwise=AvgHeadwiseObserver,
)

FP8_OBSERVER = dict(
    abs_max=AbsmaxObserver,
    avg=AVGObserver,
)


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
            search_alpha_min=quant_args.search_alpha_min,
            search_alpha_max=quant_args.search_alpha_max,
            search_scale_min=quant_args.search_scale_min,
            search_scale_max=quant_args.search_scale_max,
            weight_quant_method=quant_args.weight_quant_method,
            act_quant_method=quant_args.act_quant_method,
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


def prepare_qconfig(args):
    """
    Prepare qconfig
    """
    args.quant_type = args.quant_type.lower()

    if args.quant_type in ["a8w8_fp8"]:
        use_fp8 = "aw"
        args.quant_type = args.quant_type.replace("_fp8", "")
    else:
        use_fp8 = ""

    weight_observer = (
        WEIGHT_OBSERVER.get(args.weight_quant_method, None)
        if "w" not in use_fp8
        else FP8_OBSERVER.get(args.weight_quant_method, None)
    )
    act_observer = (
        ACT_OBSERVER.get(args.act_quant_method, None)
        if "a" not in use_fp8
        else FP8_OBSERVER.get(args.act_quant_method, None)
    )
    cachekv_observer = CACHEKV_OBSERVER.get(args.cachekv_quant_method, None)

    if "c8" in args.quant_type:
        quant_type = args.quant_type.replace("c8", "")
        cachekv_quant = True
        cachekv_quant_bits = "int8"
    else:
        quant_type = args.quant_type.replace("c16", "")
        cachekv_quant = False

    q_config = QuantConfig(activation=None, weight=None)
    if quant_type in ["a8w8", "w8a8"]:

        if "w" in use_fp8:
            w_quant_bit = (4, 3) if args.fp8_type[use_fp8.index("w")] == "e4m3" else (5, 2)
        else:
            w_quant_bit = 8

        if "a" in use_fp8:
            a_quant_bit = (4, 3) if args.fp8_type[use_fp8.index("a")] == "e4m3" else (5, 2)
        else:
            a_quant_bit = 8
        activation = act_observer(quant_bits=a_quant_bit)
        weight = weight_observer(quant_bits=w_quant_bit)

    elif quant_type in ["wint4", "w4a16", "weight_only_int4"]:
        activation = None
        weight = weight_observer(quant_bits=4)

    elif quant_type in ["wint8", "w8a16", "weight_only_int8"]:
        activation = None
        if "w" in use_fp8:
            weight = weight_observer(quant_bits=(4, 3))
        else:
            weight = weight_observer(quant_bits=8)
    else:
        raise ValueError(
            "quant_type should be in ['weight_only_int8/wint8', 'weight_only_int4/wint4', 'a8w8', 'a8w8c8', 'a8w8_fp8']"
        )

    q_config.add_qat_layer_mapping(ColumnParallelLinear, QuantizedColumnParallelLinear)
    q_config.add_qat_layer_mapping(RowParallelLinear, QuantizedRowParallelLinear)

    cachekv = None
    if cachekv_quant:
        if cachekv_quant_bits == "int8":
            cachekv_quant_bit = 8
            if "headwise" in args.cachekv_quant_method:
                cachekv = [
                    cachekv_observer(quant_bits=cachekv_quant_bit, quant_axis=1),
                    cachekv_observer(quant_bits=cachekv_quant_bit, quant_axis=1),
                ]
            else:
                cachekv = [
                    cachekv_observer(quant_bits=cachekv_quant_bit),
                    cachekv_observer(quant_bits=cachekv_quant_bit),
                ]
            q_config.add_qat_layer_mapping(FuncWrapper, QuantizedCustomAttentionLayer)
        else:
            raise ValueError("cachekv_quant_bits should be int8")

    return activation, weight, cachekv, q_config


def load_quant_model(model, quant_args, load_quant_path, dtype="float32"):
    """
    Load quantized model and its scales
    """
    activation, weight, cachekv, q_config = prepare_qconfig(quant_args)

    if cachekv is not None:
        set_wrapper_for_attn(model)

    skip_list_names = [] if quant_args.skip_list_names is None else quant_args.skip_list_names
    for cur_name, cur_layer in model.named_sublayers():
        skip = False
        for k in skip_list_names:
            if k in cur_name:
                logger.info(f"Skip layer {cur_name}")
                skip = True
        if skip:
            continue

        if type(cur_layer) in [paddle.nn.Linear, ColumnParallelLinear, RowParallelLinear]:
            logger.info(f"PTQ layer: {cur_name}")
            q_config.add_name_config([cur_layer.full_name()], activation=activation, weight=weight)

        if type(cur_layer) in [FuncWrapper] and cachekv is not None:
            logger.info(f"PTQ layer: {cur_name}")
            # set both act and weight for attention, actually act-k and act-v are quantized
            q_config.add_name_config([cur_layer.full_name()], weight=cachekv[0], activation=cachekv[1])

    ptq = PTQ(q_config)
    model = ptq.quantize(model, inplace=True)

    logger.info("Load quant model...")
    if activation is not None:
        with open(f"{load_quant_path}/act_scales.json") as outfile:
            act_scales = json.load(outfile)
    else:
        act_scales = {}

    if cachekv is not None:
        with open(f"{load_quant_path}/cachekv_scales.json") as outfile:
            cachekv_scales = json.load(outfile)
    else:
        cachekv_scales = {}

    with open(f"{load_quant_path}/weight_scales.json") as outfile:
        weight_scales = json.load(outfile)

    for cur_name, cur_layer in model.named_sublayers():
        if hasattr(cur_layer, "scales"):

            if isinstance(cur_layer, ChannelWiseObserver) or isinstance(cur_layer, BaseObserver):
                logger.info(f"Load scale for layer {cur_name}")
                if "attn_func" in cur_name:
                    cur_name = cur_name.replace("attn_func.activation_quanter_v", "cachev_matmul.activation_quanter")
                    cur_name = cur_name.replace("attn_func.activation_quanter_k", "cachek_matmul.activation_quanter")
                    if cur_name in cachekv_scales:
                        cur_layer._scale = paddle.to_tensor(cachekv_scales[cur_name], dtype=dtype)
                        if cur_name + ".zero_point" in cachekv_scales:
                            cur_layer._zero_point = paddle.to_tensor(
                                cachekv_scales[cur_name + ".zero_point"], dtype=dtype
                            )
                        else:
                            cur_layer._zero_point = paddle.to_tensor(0.0, dtype=dtype)
                    else:
                        logger.info(f"No scale found for layer {cur_name}, remove it")
                        parent_layer, sub_name = find_parent_layer_and_sub_name(model, cur_name)
                        setattr(parent_layer, sub_name, None)

                elif "activation_quanter" in cur_name:
                    if cur_name in act_scales:
                        cur_layer._scale = paddle.to_tensor(act_scales[cur_name], dtype=dtype)
                        cur_layer._zero_point = paddle.to_tensor(0.0, dtype=dtype)
                    else:
                        logger.info(f"No scale found for layer {cur_name}, remove it")
                        parent_layer, sub_name = find_parent_layer_and_sub_name(model, cur_name)
                        setattr(parent_layer, sub_name, None)
                elif "weight_quanter" in cur_name:
                    if cur_name in weight_scales:
                        cur_layer._scale = paddle.to_tensor(weight_scales[cur_name], dtype=dtype)
                    else:
                        logger.info(f"No scale found for layer {cur_name}, remove it")
                        parent_layer, sub_name = find_parent_layer_and_sub_name(model, cur_name)
                        setattr(parent_layer, sub_name, None)

    model = ptq.convert(model, inplace=True)
    if os.path.exists(os.path.join(load_quant_path, "model_state.pdparams")):
        logger.info(f"Load model checkpoint from {load_quant_path}")
        model_path = os.path.join(load_quant_path, "model_state.pdparams")
        model_dict = paddle.load(model_path, return_numpy=True)
        model.set_dict(model_dict)
    else:
        raise Exception("Only support load model from pdparams now")


def apply_ptq(quant_args, trainer, ptq_dataloader):
    logger.info("***** Running PTQ *****")
    activation, weight, cachekv, q_config = prepare_qconfig(quant_args)

    if cachekv is not None:
        set_wrapper_for_attn(trainer.model)

    skip_list_names = [] if quant_args.skip_list_names is None else quant_args.skip_list_names

    for cur_name, cur_layer in trainer.model.named_sublayers():
        skip = False
        for k in skip_list_names:
            if k in cur_name:
                logger.info(f"Skip layer {cur_name}")
                skip = True
        if skip:
            continue

        if type(cur_layer) in [paddle.nn.Linear, ColumnParallelLinear, RowParallelLinear]:
            logger.info(f"PTQ layer: {cur_name}")
            q_config.add_name_config([cur_layer.full_name()], activation=activation, weight=weight)

        if cachekv is not None and type(cur_layer) in [FuncWrapper]:
            logger.info(f"PTQ layer: {cur_name}")
            # set both act and weight for attention, actually act-k and act-v are quantized
            q_config.add_name_config([cur_layer.full_name()], weight=cachekv[0], activation=cachekv[1])

    ptq = PTQ(q_config)
    trainer.model = ptq.quantize(trainer.model, inplace=True)

    # enable observer
    enable_observer(trainer.model)
    logger.info("***** PTQ loop start *****")
    trainer.ptq_loop(
        ptq_dataloader,
        description="PTQ",
        max_eval_iters=quant_args.ptq_step,
    )
    # disable observer
    disable_observer(trainer.model)

    weight_scales = {}
    act_scales = {}
    cachekv_scales = {}
    for cur_name, cur_layer in trainer.model.named_sublayers():
        if isinstance(cur_layer, ChannelWiseObserver) or isinstance(cur_layer, BaseObserver):
            if "_observer" not in cur_name:
                if "attn_func" in cur_name:
                    cur_name = cur_name.replace("attn_func.activation_quanter_v", "cachev_matmul.activation_quanter")
                    cur_name = cur_name.replace("attn_func.activation_quanter_k", "cachek_matmul.activation_quanter")
                    cachekv_scales[cur_name] = cur_layer.scales().cast("float32").numpy().tolist()
                elif "activation_quanter" in cur_name:
                    act_scales[cur_name] = cur_layer.scales().cast("float32").numpy().tolist()
                elif "weight_quanter" in cur_name:
                    weight_scales[cur_name] = cur_layer.scales().cast("float32").numpy().tolist()

    weight_scales_path = os.path.join(trainer.args.output_dir, "weight_scales.json")
    with open(weight_scales_path, "w") as f:
        json.dump(weight_scales, f)
    logger.info(f"Weight scales saved in {weight_scales_path}.")

    act_scales_path = os.path.join(trainer.args.output_dir, "act_scales.json")
    with open(act_scales_path, "w") as f:
        json.dump(act_scales, f)
    logger.info(f"Activation scales saved in {act_scales_path}.")

    cachekv_scales_path = os.path.join(trainer.args.output_dir, "cachekv_scales.json")
    with open(cachekv_scales_path, "w") as f:
        json.dump(cachekv_scales, f)
    logger.info(f"CacheKV scales saved in {cachekv_scales_path}.")

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


def set_wrapper_for_attn(model: nn.Layer, attn_name="attn_func"):
    for cur_name, cur_layer in model.named_sublayers():
        if hasattr(cur_layer, attn_name):
            logger.info(f"Set wrapper for {attn_name} in {cur_name}")
            cur_layer.attn_func = FuncWrapper(cur_layer.attn_func)


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
    elif base_model_prefix == "qwen2":
        model_config = {"fused_qkv": False, "parallel_ffn": True}
    else:
        raise ValueError(
            f"Unknown base_model_prefix: {model.base_model_prefix}. Supported base_model_prefix list: chatglm_V2, bloom, llama, qwen2."
        )
    return model_config


def enable_observer(model: nn.Layer):
    # TODO maybe not support pp,tp etc.
    for mod in model.sublayers():
        if hasattr(mod, "observer_enabled"):
            mod.observer_enabled = True


def disable_observer(model: nn.Layer):
    # TODO maybe not support pp,tp etc.
    for mod in model.sublayers():
        if hasattr(mod, "observer_enabled"):
            mod.observer_enabled = False


def add_quant_inp_out_hook(model: nn.Layer, tag_func):
    def get_hook():
        inp_ret = []
        out_ret = []

        def hook(layer, inp, out):
            nonlocal inp_ret, out_ret
            inp_ret.append(inp[0].flatten().numpy())
            out_ret.append(out.flatten().numpy())
            return out

        return hook, inp_ret, out_ret

    inp_dict = dict()
    out_dict = dict()

    handlers = []
    for cur_name, cur_layer in model.named_sublayers():
        if tag_func(cur_name):
            hook, inp_ret, out_ret = get_hook()
            handle = cur_layer.register_forward_post_hook(hook)
            inp_dict[cur_name] = inp_ret
            out_dict[cur_name] = out_ret
            handlers.append(handle)

    return inp_dict, out_dict


def save_dict(inp_dict, file_path):
    import pickle

    with open(file_path, "wb") as f:
        pickle.dump(inp_dict, f)
