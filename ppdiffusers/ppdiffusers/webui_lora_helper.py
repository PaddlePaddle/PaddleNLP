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

import math
import os
from types import MethodType

import paddle
import paddle.nn as nn

from .models.modeling_utils import convert_state_dict
from .pipelines import DiffusionPipeline
from .ppnlp_patch_utils import patch_to
from .utils import is_safetensors_available, is_torch_available
from .utils.constants import PPDIFFUSERS_CACHE
from .utils.download_utils import ppdiffusers_url_download
from .utils.initializer_utils import kaiming_uniform_, zeros_
from .utils.load_utils import smart_load

if is_safetensors_available():
    import safetensors


def convert_pt_to_pd(state, dtype):
    if dtype is None:
        dtype = "float32"
    new_state = {}
    for a, b in safetensors_weight_mapping:
        if a in state:
            val = state[a]
            if val.ndim == 2:
                val = val.T
            if val.ndim == 0:
                val = val.reshape((1,))
            new_state[b] = val.cast(dtype)
        else:
            print(f"We find {a} not in state_dict and we will continue!")
    return new_state


def convert_pd_to_pt(state):
    new_state = {}
    for a, b in safetensors_weight_mapping:
        if b in state:
            val = state[b]
            if val.ndim == 2:
                val = val.T
            # if ".alpha" in a:
            #     val = val.squeeze()

            new_state[a] = val
    return new_state


def extract_lora_weights(model):
    sd = {}
    for k, v in model.state_dict().items():
        if "lora" in k or ".alpha" in k:
            sd[k] = v
    return sd


@patch_to([DiffusionPipeline, nn.Layer])
def save_lora(pipe_or_module, save_directory, WEIGHT_NAME=None):
    if WEIGHT_NAME is None:
        WEIGHT_NAME = "text_encoder_unet_lora.safetensors"
    outdict = {}
    if isinstance(pipe_or_module, nn.Layer):
        outdict.update(extract_lora_weights(pipe_or_module))
    else:
        if hasattr(pipe_or_module, "text_encoder"):
            outdict.update(extract_lora_weights(pipe_or_module.text_encoder))
        if hasattr(pipe_or_module, "unet"):
            outdict.update(extract_lora_weights(pipe_or_module.unet))
    os.makedirs(save_directory, exist_ok=True)

    if is_torch_available():
        save_function = safetensors.torch.save_file
        outdict = convert_state_dict(convert_pd_to_pt(outdict), framework="torch")
    else:
        save_function = safetensors.numpy.save_file
        outdict = convert_state_dict(convert_pd_to_pt(outdict), framework="numpy")

    save_function(outdict, os.path.join(save_directory, WEIGHT_NAME))
    del outdict
    print(f"Model weights saved in {os.path.join(save_directory, WEIGHT_NAME)}")


@patch_to([DiffusionPipeline, nn.Layer])
def set_lora_enabled(pipe_or_module, enable=True):
    def set_lora(self):
        if hasattr(self, "enable_lora"):
            self.enable_lora = enable

    if isinstance(pipe_or_module, nn.Layer):
        pipe_or_module.apply(set_lora)
    else:
        if hasattr(pipe_or_module, "text_encoder"):
            pipe_or_module.text_encoder.apply(set_lora)
        if hasattr(pipe_or_module, "unet"):
            pipe_or_module.unet.apply(set_lora)


@patch_to([DiffusionPipeline, nn.Layer])
def apply_lora(
    pipe_or_module,
    lora_weight_or_path=None,
    rank=4,
    alpha=None,
    multiplier=1.0,
    text_encoder_target_replace_modules=["TransformerEncoderLayer"],
    unet_target_replace_modules=["Transformer2DModel", "Attention"],
    enable_lora=True,
    **kwargs,
):
    resume_download = kwargs.pop("resume_download", False)
    force_download = kwargs.pop("force_download", False)
    paddle_dtype = kwargs.pop("paddle_dtype", None)
    cache_dir = kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)

    if paddle_dtype is None:
        if isinstance(pipe_or_module, nn.Layer):
            paddle_dtype = pipe_or_module.dtype
        else:
            if hasattr(pipe_or_module, "text_encoder"):
                paddle_dtype = pipe_or_module.text_encoder.dtype
            if hasattr(pipe_or_module, "unet"):
                paddle_dtype = pipe_or_module.unet.dtype

    if lora_weight_or_path is not None:
        lora_weight_or_path = str(lora_weight_or_path)
        if os.path.isfile(lora_weight_or_path):
            lora_weight_or_path = lora_weight_or_path
        elif lora_weight_or_path.startswith("http://") or lora_weight_or_path.startswith("https://"):
            lora_weight_or_path = ppdiffusers_url_download(
                lora_weight_or_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                force_download=force_download,
            )
        else:
            raise EnvironmentError(f"Please check your {lora_weight_or_path}.")
        lora_weight_or_path = convert_pt_to_pd(smart_load(lora_weight_or_path), paddle_dtype)

        mayberanklist = []
        maybealphalist = []
        for k, v in lora_weight_or_path.items():
            if "lora_down" in k and "alpha" not in k:
                if v.ndim == 2:
                    mayberanklist.append(v.shape[1])
                elif v.ndim == 4:
                    mayberanklist.append(v.shape[0])

            if "lora_up" in k and "alpha" not in k:
                if v.ndim == 2:
                    mayberanklist.append(v.shape[0])
                elif v.ndim == 4:
                    mayberanklist.append(v.shape[1])

            if "alpha" in k:
                # we must cast it to float32
                maybealphalist.append(v.astype("float32").item())
            if len(mayberanklist) > 20:
                break
        if len(set(mayberanklist)) > 1:
            print(f"Can't guess rank! Here are the rank list {mayberanklist}. We will use default rank {rank}.")
        else:
            rank = mayberanklist[0]
        print(f"|---------------Currently, rank is {rank}!")

        if len(set(maybealphalist)) > 1:
            print(f"Can't guess alpha! Here are the rank list {maybealphalist}. We will use default alpha {alpha}")
        else:
            alpha = maybealphalist[0]
        print(f"|---------------Currently, alpha is {alpha}!")

    waitlist = []
    if isinstance(pipe_or_module, nn.Layer):
        waitlist.append((pipe_or_module, text_encoder_target_replace_modules + unet_target_replace_modules))
    else:
        if hasattr(pipe_or_module, "text_encoder"):
            waitlist.append((pipe_or_module.text_encoder, text_encoder_target_replace_modules))
        if hasattr(pipe_or_module, "unet"):
            waitlist.append((pipe_or_module.unet, unet_target_replace_modules))
    lora_modules = {}
    for each_module, target_replace_modules in waitlist:
        for name1, module in each_module.named_sublayers(include_self=True):
            if module.__class__.__name__ in target_replace_modules:
                for name2, child_module in module.named_sublayers(include_self=True):
                    if not getattr(child_module, "is_lora_linear", False) and (
                        child_module.__class__.__name__ == "Linear"
                        or (child_module.__class__.__name__ == "Conv2D" and list(child_module._kernel_size) == [1, 1])
                    ):
                        # if we apply lora multi
                        if hasattr(child_module, "merged") and child_module.merged:
                            with paddle.no_grad():
                                if child_module.is_conv:
                                    new_weight = (
                                        child_module.weight.squeeze([-1, -2])
                                        - child_module.lora_up.weight.squeeze([-1, -2])
                                        @ child_module.lora_down.weight.squeeze([-1, -2])
                                        * child_module.multiplier
                                        * child_module.scale
                                    ).unsqueeze([-1, -2])
                                else:
                                    new_weight = (
                                        child_module.weight
                                        - child_module.lora_down.weight
                                        @ child_module.lora_up.weight
                                        * child_module.multiplier
                                        * child_module.scale
                                    )
                                child_module.weight.set_value(new_weight)

                        in_features, out_features = child_module.weight.shape[0], child_module.weight.shape[1]
                        child_module.is_conv = False
                        child_module.merged = False

                        if child_module.weight.ndim == 4:
                            child_module.is_conv = True
                            in_features, out_features = out_features, in_features

                        if rank > min(in_features, out_features):
                            raise ValueError(
                                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
                            )

                        if child_module.is_conv:
                            child_module.lora_down = nn.Conv2D(in_features, rank, [1, 1], bias_attr=False)
                            child_module.lora_up = nn.Conv2D(rank, out_features, [1, 1], bias_attr=False)
                        else:
                            child_module.lora_down = nn.Linear(in_features, rank, bias_attr=False)
                            child_module.lora_up = nn.Linear(rank, out_features, bias_attr=False)
                        child_module.lora_down.is_lora_linear = True
                        child_module.lora_up.is_lora_linear = True
                        child_module.rank = rank
                        child_module.enable_lora = enable_lora

                        if paddle.is_tensor(alpha):
                            alpha = alpha.detach().cast("float32").numpy()
                        alpha = rank if alpha is None or alpha == 0 else alpha
                        child_module.scale = alpha / child_module.rank
                        child_module.register_buffer("alpha", paddle.to_tensor(alpha, dtype="float32"))

                        # same as microsoft's
                        kaiming_uniform_(child_module.lora_down.weight, a=math.sqrt(5))
                        zeros_(child_module.lora_up.weight)
                        child_module.multiplier = multiplier

                        if getattr(child_module, "raw_forward", None) is None:
                            child_module.raw_forward = child_module.forward

                        def forward_lora(self, x):
                            if self.training:
                                if self.merged:
                                    with paddle.no_grad():
                                        if self.is_conv:
                                            new_weight = (
                                                self.weight.squeeze([-1, -2])
                                                - self.lora_up.weight.squeeze([-1, -2])
                                                @ self.lora_down.weight.squeeze([-1, -2])
                                                * self.multiplier
                                                * self.scale
                                            ).unsqueeze([-1, -2])
                                        else:
                                            new_weight = (
                                                self.weight
                                                - self.lora_down.weight
                                                @ self.lora_up.weight
                                                * self.multiplier
                                                * self.scale
                                            )
                                        self.weight.set_value(new_weight)
                                        self.merged = False
                                if not self.enable_lora:
                                    return self.raw_forward(x)
                                return (
                                    self.raw_forward(x)
                                    + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
                                )
                            else:
                                if self.enable_lora and not self.merged:
                                    with paddle.no_grad():
                                        if self.is_conv:
                                            new_weight = (
                                                self.weight.squeeze([-1, -2])
                                                + self.lora_up.weight.squeeze([-1, -2])
                                                @ self.lora_down.weight.squeeze([-1, -2])
                                                * self.multiplier
                                                * self.scale
                                            ).unsqueeze([-1, -2])
                                        else:
                                            new_weight = (
                                                self.weight
                                                + self.lora_down.weight
                                                @ self.lora_up.weight
                                                * self.multiplier
                                                * self.scale
                                            )
                                        self.weight.set_value(new_weight)
                                        self.merged = True

                                if not self.enable_lora and self.merged:
                                    with paddle.no_grad():
                                        if self.is_conv:
                                            new_weight = (
                                                self.weight.squeeze([-1, -2])
                                                - self.lora_up.weight.squeeze([-1, -2])
                                                @ self.lora_down.weight.squeeze([-1, -2])
                                                * self.multiplier
                                                * self.scale
                                            ).unsqueeze([-1, -2])
                                        else:
                                            new_weight = (
                                                self.weight
                                                - self.lora_down.weight
                                                @ self.lora_up.weight
                                                * self.multiplier
                                                * self.scale
                                            )
                                        self.weight.set_value(new_weight)
                                        self.merged = False
                                return self.raw_forward(x)

                        child_module.forward = MethodType(forward_lora, child_module)
                        child_module.lora_down.training = child_module.training
                        child_module.lora_up.training = child_module.training
                        child_module.to(dtype=paddle_dtype)
                        # we will return lora_modules
                        lora_modules[name1 + "." + name2] = child_module

    if lora_weight_or_path is not None:
        if isinstance(pipe_or_module, nn.Layer):
            pipe_or_module.set_dict(lora_weight_or_path)
        else:
            if hasattr(pipe_or_module, "text_encoder"):
                pipe_or_module.text_encoder.set_dict(lora_weight_or_path)
                pipe_or_module.text_encoder.eval()
            if hasattr(pipe_or_module, "unet"):
                pipe_or_module.unet.set_dict(lora_weight_or_path)
                pipe_or_module.unet.eval()

        del lora_weight_or_path
        print("Loading lora_weights successfully!")
    return lora_modules


safetensors_weight_mapping = [
    [
        "lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight",
        "text_model.transformer.layers.0.self_attn.q_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight",
        "text_model.transformer.layers.0.self_attn.k_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_0_self_attn_v_proj.lora_down.weight",
        "text_model.transformer.layers.0.self_attn.v_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_0_self_attn_out_proj.lora_down.weight",
        "text_model.transformer.layers.0.self_attn.out_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",
        "text_model.transformer.layers.0.linear1.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_0_mlp_fc2.lora_down.weight",
        "text_model.transformer.layers.0.linear2.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight",
        "text_model.transformer.layers.0.self_attn.q_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_up.weight",
        "text_model.transformer.layers.0.self_attn.k_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_0_self_attn_v_proj.lora_up.weight",
        "text_model.transformer.layers.0.self_attn.v_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_0_self_attn_out_proj.lora_up.weight",
        "text_model.transformer.layers.0.self_attn.out_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_up.weight",
        "text_model.transformer.layers.0.linear1.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_0_mlp_fc2.lora_up.weight",
        "text_model.transformer.layers.0.linear2.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_0_self_attn_q_proj.alpha",
        "text_model.transformer.layers.0.self_attn.q_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_0_self_attn_k_proj.alpha",
        "text_model.transformer.layers.0.self_attn.k_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_0_self_attn_v_proj.alpha",
        "text_model.transformer.layers.0.self_attn.v_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_0_self_attn_out_proj.alpha",
        "text_model.transformer.layers.0.self_attn.out_proj.alpha",
    ],
    ["lora_te_text_model_encoder_layers_0_mlp_fc1.alpha", "text_model.transformer.layers.0.linear1.alpha"],
    ["lora_te_text_model_encoder_layers_0_mlp_fc2.alpha", "text_model.transformer.layers.0.linear2.alpha"],
    [
        "lora_te_text_model_encoder_layers_1_self_attn_q_proj.lora_down.weight",
        "text_model.transformer.layers.1.self_attn.q_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_1_self_attn_k_proj.lora_down.weight",
        "text_model.transformer.layers.1.self_attn.k_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_1_self_attn_v_proj.lora_down.weight",
        "text_model.transformer.layers.1.self_attn.v_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_1_self_attn_out_proj.lora_down.weight",
        "text_model.transformer.layers.1.self_attn.out_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_1_mlp_fc1.lora_down.weight",
        "text_model.transformer.layers.1.linear1.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_1_mlp_fc2.lora_down.weight",
        "text_model.transformer.layers.1.linear2.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_1_self_attn_q_proj.lora_up.weight",
        "text_model.transformer.layers.1.self_attn.q_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_1_self_attn_k_proj.lora_up.weight",
        "text_model.transformer.layers.1.self_attn.k_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_1_self_attn_v_proj.lora_up.weight",
        "text_model.transformer.layers.1.self_attn.v_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_1_self_attn_out_proj.lora_up.weight",
        "text_model.transformer.layers.1.self_attn.out_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_1_mlp_fc1.lora_up.weight",
        "text_model.transformer.layers.1.linear1.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_1_mlp_fc2.lora_up.weight",
        "text_model.transformer.layers.1.linear2.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_1_self_attn_q_proj.alpha",
        "text_model.transformer.layers.1.self_attn.q_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_1_self_attn_k_proj.alpha",
        "text_model.transformer.layers.1.self_attn.k_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_1_self_attn_v_proj.alpha",
        "text_model.transformer.layers.1.self_attn.v_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_1_self_attn_out_proj.alpha",
        "text_model.transformer.layers.1.self_attn.out_proj.alpha",
    ],
    ["lora_te_text_model_encoder_layers_1_mlp_fc1.alpha", "text_model.transformer.layers.1.linear1.alpha"],
    ["lora_te_text_model_encoder_layers_1_mlp_fc2.alpha", "text_model.transformer.layers.1.linear2.alpha"],
    [
        "lora_te_text_model_encoder_layers_2_self_attn_q_proj.lora_down.weight",
        "text_model.transformer.layers.2.self_attn.q_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_2_self_attn_k_proj.lora_down.weight",
        "text_model.transformer.layers.2.self_attn.k_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_2_self_attn_v_proj.lora_down.weight",
        "text_model.transformer.layers.2.self_attn.v_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_2_self_attn_out_proj.lora_down.weight",
        "text_model.transformer.layers.2.self_attn.out_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_2_mlp_fc1.lora_down.weight",
        "text_model.transformer.layers.2.linear1.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_2_mlp_fc2.lora_down.weight",
        "text_model.transformer.layers.2.linear2.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_2_self_attn_q_proj.lora_up.weight",
        "text_model.transformer.layers.2.self_attn.q_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_2_self_attn_k_proj.lora_up.weight",
        "text_model.transformer.layers.2.self_attn.k_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_2_self_attn_v_proj.lora_up.weight",
        "text_model.transformer.layers.2.self_attn.v_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_2_self_attn_out_proj.lora_up.weight",
        "text_model.transformer.layers.2.self_attn.out_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_2_mlp_fc1.lora_up.weight",
        "text_model.transformer.layers.2.linear1.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_2_mlp_fc2.lora_up.weight",
        "text_model.transformer.layers.2.linear2.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_2_self_attn_q_proj.alpha",
        "text_model.transformer.layers.2.self_attn.q_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_2_self_attn_k_proj.alpha",
        "text_model.transformer.layers.2.self_attn.k_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_2_self_attn_v_proj.alpha",
        "text_model.transformer.layers.2.self_attn.v_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_2_self_attn_out_proj.alpha",
        "text_model.transformer.layers.2.self_attn.out_proj.alpha",
    ],
    ["lora_te_text_model_encoder_layers_2_mlp_fc1.alpha", "text_model.transformer.layers.2.linear1.alpha"],
    ["lora_te_text_model_encoder_layers_2_mlp_fc2.alpha", "text_model.transformer.layers.2.linear2.alpha"],
    [
        "lora_te_text_model_encoder_layers_3_self_attn_q_proj.lora_down.weight",
        "text_model.transformer.layers.3.self_attn.q_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_3_self_attn_k_proj.lora_down.weight",
        "text_model.transformer.layers.3.self_attn.k_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_3_self_attn_v_proj.lora_down.weight",
        "text_model.transformer.layers.3.self_attn.v_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_3_self_attn_out_proj.lora_down.weight",
        "text_model.transformer.layers.3.self_attn.out_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_3_mlp_fc1.lora_down.weight",
        "text_model.transformer.layers.3.linear1.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_3_mlp_fc2.lora_down.weight",
        "text_model.transformer.layers.3.linear2.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_3_self_attn_q_proj.lora_up.weight",
        "text_model.transformer.layers.3.self_attn.q_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_3_self_attn_k_proj.lora_up.weight",
        "text_model.transformer.layers.3.self_attn.k_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_3_self_attn_v_proj.lora_up.weight",
        "text_model.transformer.layers.3.self_attn.v_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_3_self_attn_out_proj.lora_up.weight",
        "text_model.transformer.layers.3.self_attn.out_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_3_mlp_fc1.lora_up.weight",
        "text_model.transformer.layers.3.linear1.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_3_mlp_fc2.lora_up.weight",
        "text_model.transformer.layers.3.linear2.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_3_self_attn_q_proj.alpha",
        "text_model.transformer.layers.3.self_attn.q_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_3_self_attn_k_proj.alpha",
        "text_model.transformer.layers.3.self_attn.k_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_3_self_attn_v_proj.alpha",
        "text_model.transformer.layers.3.self_attn.v_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_3_self_attn_out_proj.alpha",
        "text_model.transformer.layers.3.self_attn.out_proj.alpha",
    ],
    ["lora_te_text_model_encoder_layers_3_mlp_fc1.alpha", "text_model.transformer.layers.3.linear1.alpha"],
    ["lora_te_text_model_encoder_layers_3_mlp_fc2.alpha", "text_model.transformer.layers.3.linear2.alpha"],
    [
        "lora_te_text_model_encoder_layers_4_self_attn_q_proj.lora_down.weight",
        "text_model.transformer.layers.4.self_attn.q_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_4_self_attn_k_proj.lora_down.weight",
        "text_model.transformer.layers.4.self_attn.k_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_4_self_attn_v_proj.lora_down.weight",
        "text_model.transformer.layers.4.self_attn.v_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_4_self_attn_out_proj.lora_down.weight",
        "text_model.transformer.layers.4.self_attn.out_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_4_mlp_fc1.lora_down.weight",
        "text_model.transformer.layers.4.linear1.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_4_mlp_fc2.lora_down.weight",
        "text_model.transformer.layers.4.linear2.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_4_self_attn_q_proj.lora_up.weight",
        "text_model.transformer.layers.4.self_attn.q_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_4_self_attn_k_proj.lora_up.weight",
        "text_model.transformer.layers.4.self_attn.k_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_4_self_attn_v_proj.lora_up.weight",
        "text_model.transformer.layers.4.self_attn.v_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_4_self_attn_out_proj.lora_up.weight",
        "text_model.transformer.layers.4.self_attn.out_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_4_mlp_fc1.lora_up.weight",
        "text_model.transformer.layers.4.linear1.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_4_mlp_fc2.lora_up.weight",
        "text_model.transformer.layers.4.linear2.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_4_self_attn_q_proj.alpha",
        "text_model.transformer.layers.4.self_attn.q_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_4_self_attn_k_proj.alpha",
        "text_model.transformer.layers.4.self_attn.k_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_4_self_attn_v_proj.alpha",
        "text_model.transformer.layers.4.self_attn.v_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_4_self_attn_out_proj.alpha",
        "text_model.transformer.layers.4.self_attn.out_proj.alpha",
    ],
    ["lora_te_text_model_encoder_layers_4_mlp_fc1.alpha", "text_model.transformer.layers.4.linear1.alpha"],
    ["lora_te_text_model_encoder_layers_4_mlp_fc2.alpha", "text_model.transformer.layers.4.linear2.alpha"],
    [
        "lora_te_text_model_encoder_layers_5_self_attn_q_proj.lora_down.weight",
        "text_model.transformer.layers.5.self_attn.q_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_5_self_attn_k_proj.lora_down.weight",
        "text_model.transformer.layers.5.self_attn.k_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_5_self_attn_v_proj.lora_down.weight",
        "text_model.transformer.layers.5.self_attn.v_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_5_self_attn_out_proj.lora_down.weight",
        "text_model.transformer.layers.5.self_attn.out_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_5_mlp_fc1.lora_down.weight",
        "text_model.transformer.layers.5.linear1.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_5_mlp_fc2.lora_down.weight",
        "text_model.transformer.layers.5.linear2.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_5_self_attn_q_proj.lora_up.weight",
        "text_model.transformer.layers.5.self_attn.q_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_5_self_attn_k_proj.lora_up.weight",
        "text_model.transformer.layers.5.self_attn.k_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_5_self_attn_v_proj.lora_up.weight",
        "text_model.transformer.layers.5.self_attn.v_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_5_self_attn_out_proj.lora_up.weight",
        "text_model.transformer.layers.5.self_attn.out_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_5_mlp_fc1.lora_up.weight",
        "text_model.transformer.layers.5.linear1.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_5_mlp_fc2.lora_up.weight",
        "text_model.transformer.layers.5.linear2.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_5_self_attn_q_proj.alpha",
        "text_model.transformer.layers.5.self_attn.q_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_5_self_attn_k_proj.alpha",
        "text_model.transformer.layers.5.self_attn.k_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_5_self_attn_v_proj.alpha",
        "text_model.transformer.layers.5.self_attn.v_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_5_self_attn_out_proj.alpha",
        "text_model.transformer.layers.5.self_attn.out_proj.alpha",
    ],
    ["lora_te_text_model_encoder_layers_5_mlp_fc1.alpha", "text_model.transformer.layers.5.linear1.alpha"],
    ["lora_te_text_model_encoder_layers_5_mlp_fc2.alpha", "text_model.transformer.layers.5.linear2.alpha"],
    [
        "lora_te_text_model_encoder_layers_6_self_attn_q_proj.lora_down.weight",
        "text_model.transformer.layers.6.self_attn.q_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_6_self_attn_k_proj.lora_down.weight",
        "text_model.transformer.layers.6.self_attn.k_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_6_self_attn_v_proj.lora_down.weight",
        "text_model.transformer.layers.6.self_attn.v_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_6_self_attn_out_proj.lora_down.weight",
        "text_model.transformer.layers.6.self_attn.out_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_6_mlp_fc1.lora_down.weight",
        "text_model.transformer.layers.6.linear1.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_6_mlp_fc2.lora_down.weight",
        "text_model.transformer.layers.6.linear2.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_6_self_attn_q_proj.lora_up.weight",
        "text_model.transformer.layers.6.self_attn.q_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_6_self_attn_k_proj.lora_up.weight",
        "text_model.transformer.layers.6.self_attn.k_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_6_self_attn_v_proj.lora_up.weight",
        "text_model.transformer.layers.6.self_attn.v_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_6_self_attn_out_proj.lora_up.weight",
        "text_model.transformer.layers.6.self_attn.out_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_6_mlp_fc1.lora_up.weight",
        "text_model.transformer.layers.6.linear1.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_6_mlp_fc2.lora_up.weight",
        "text_model.transformer.layers.6.linear2.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_6_self_attn_q_proj.alpha",
        "text_model.transformer.layers.6.self_attn.q_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_6_self_attn_k_proj.alpha",
        "text_model.transformer.layers.6.self_attn.k_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_6_self_attn_v_proj.alpha",
        "text_model.transformer.layers.6.self_attn.v_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_6_self_attn_out_proj.alpha",
        "text_model.transformer.layers.6.self_attn.out_proj.alpha",
    ],
    ["lora_te_text_model_encoder_layers_6_mlp_fc1.alpha", "text_model.transformer.layers.6.linear1.alpha"],
    ["lora_te_text_model_encoder_layers_6_mlp_fc2.alpha", "text_model.transformer.layers.6.linear2.alpha"],
    [
        "lora_te_text_model_encoder_layers_7_self_attn_q_proj.lora_down.weight",
        "text_model.transformer.layers.7.self_attn.q_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_7_self_attn_k_proj.lora_down.weight",
        "text_model.transformer.layers.7.self_attn.k_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_7_self_attn_v_proj.lora_down.weight",
        "text_model.transformer.layers.7.self_attn.v_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_7_self_attn_out_proj.lora_down.weight",
        "text_model.transformer.layers.7.self_attn.out_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_7_mlp_fc1.lora_down.weight",
        "text_model.transformer.layers.7.linear1.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_7_mlp_fc2.lora_down.weight",
        "text_model.transformer.layers.7.linear2.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_7_self_attn_q_proj.lora_up.weight",
        "text_model.transformer.layers.7.self_attn.q_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_7_self_attn_k_proj.lora_up.weight",
        "text_model.transformer.layers.7.self_attn.k_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_7_self_attn_v_proj.lora_up.weight",
        "text_model.transformer.layers.7.self_attn.v_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_7_self_attn_out_proj.lora_up.weight",
        "text_model.transformer.layers.7.self_attn.out_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_7_mlp_fc1.lora_up.weight",
        "text_model.transformer.layers.7.linear1.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_7_mlp_fc2.lora_up.weight",
        "text_model.transformer.layers.7.linear2.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_7_self_attn_q_proj.alpha",
        "text_model.transformer.layers.7.self_attn.q_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_7_self_attn_k_proj.alpha",
        "text_model.transformer.layers.7.self_attn.k_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_7_self_attn_v_proj.alpha",
        "text_model.transformer.layers.7.self_attn.v_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_7_self_attn_out_proj.alpha",
        "text_model.transformer.layers.7.self_attn.out_proj.alpha",
    ],
    ["lora_te_text_model_encoder_layers_7_mlp_fc1.alpha", "text_model.transformer.layers.7.linear1.alpha"],
    ["lora_te_text_model_encoder_layers_7_mlp_fc2.alpha", "text_model.transformer.layers.7.linear2.alpha"],
    [
        "lora_te_text_model_encoder_layers_8_self_attn_q_proj.lora_down.weight",
        "text_model.transformer.layers.8.self_attn.q_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_8_self_attn_k_proj.lora_down.weight",
        "text_model.transformer.layers.8.self_attn.k_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_8_self_attn_v_proj.lora_down.weight",
        "text_model.transformer.layers.8.self_attn.v_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_8_self_attn_out_proj.lora_down.weight",
        "text_model.transformer.layers.8.self_attn.out_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_8_mlp_fc1.lora_down.weight",
        "text_model.transformer.layers.8.linear1.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_8_mlp_fc2.lora_down.weight",
        "text_model.transformer.layers.8.linear2.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_8_self_attn_q_proj.lora_up.weight",
        "text_model.transformer.layers.8.self_attn.q_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_8_self_attn_k_proj.lora_up.weight",
        "text_model.transformer.layers.8.self_attn.k_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_8_self_attn_v_proj.lora_up.weight",
        "text_model.transformer.layers.8.self_attn.v_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_8_self_attn_out_proj.lora_up.weight",
        "text_model.transformer.layers.8.self_attn.out_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_8_mlp_fc1.lora_up.weight",
        "text_model.transformer.layers.8.linear1.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_8_mlp_fc2.lora_up.weight",
        "text_model.transformer.layers.8.linear2.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_8_self_attn_q_proj.alpha",
        "text_model.transformer.layers.8.self_attn.q_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_8_self_attn_k_proj.alpha",
        "text_model.transformer.layers.8.self_attn.k_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_8_self_attn_v_proj.alpha",
        "text_model.transformer.layers.8.self_attn.v_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_8_self_attn_out_proj.alpha",
        "text_model.transformer.layers.8.self_attn.out_proj.alpha",
    ],
    ["lora_te_text_model_encoder_layers_8_mlp_fc1.alpha", "text_model.transformer.layers.8.linear1.alpha"],
    ["lora_te_text_model_encoder_layers_8_mlp_fc2.alpha", "text_model.transformer.layers.8.linear2.alpha"],
    [
        "lora_te_text_model_encoder_layers_9_self_attn_q_proj.lora_down.weight",
        "text_model.transformer.layers.9.self_attn.q_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_9_self_attn_k_proj.lora_down.weight",
        "text_model.transformer.layers.9.self_attn.k_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_9_self_attn_v_proj.lora_down.weight",
        "text_model.transformer.layers.9.self_attn.v_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_9_self_attn_out_proj.lora_down.weight",
        "text_model.transformer.layers.9.self_attn.out_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_9_mlp_fc1.lora_down.weight",
        "text_model.transformer.layers.9.linear1.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_9_mlp_fc2.lora_down.weight",
        "text_model.transformer.layers.9.linear2.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_9_self_attn_q_proj.lora_up.weight",
        "text_model.transformer.layers.9.self_attn.q_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_9_self_attn_k_proj.lora_up.weight",
        "text_model.transformer.layers.9.self_attn.k_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_9_self_attn_v_proj.lora_up.weight",
        "text_model.transformer.layers.9.self_attn.v_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_9_self_attn_out_proj.lora_up.weight",
        "text_model.transformer.layers.9.self_attn.out_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_9_mlp_fc1.lora_up.weight",
        "text_model.transformer.layers.9.linear1.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_9_mlp_fc2.lora_up.weight",
        "text_model.transformer.layers.9.linear2.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_9_self_attn_q_proj.alpha",
        "text_model.transformer.layers.9.self_attn.q_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_9_self_attn_k_proj.alpha",
        "text_model.transformer.layers.9.self_attn.k_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_9_self_attn_v_proj.alpha",
        "text_model.transformer.layers.9.self_attn.v_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_9_self_attn_out_proj.alpha",
        "text_model.transformer.layers.9.self_attn.out_proj.alpha",
    ],
    ["lora_te_text_model_encoder_layers_9_mlp_fc1.alpha", "text_model.transformer.layers.9.linear1.alpha"],
    ["lora_te_text_model_encoder_layers_9_mlp_fc2.alpha", "text_model.transformer.layers.9.linear2.alpha"],
    [
        "lora_te_text_model_encoder_layers_10_self_attn_q_proj.lora_down.weight",
        "text_model.transformer.layers.10.self_attn.q_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_10_self_attn_k_proj.lora_down.weight",
        "text_model.transformer.layers.10.self_attn.k_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_10_self_attn_v_proj.lora_down.weight",
        "text_model.transformer.layers.10.self_attn.v_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_10_self_attn_out_proj.lora_down.weight",
        "text_model.transformer.layers.10.self_attn.out_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_10_mlp_fc1.lora_down.weight",
        "text_model.transformer.layers.10.linear1.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_10_mlp_fc2.lora_down.weight",
        "text_model.transformer.layers.10.linear2.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_10_self_attn_q_proj.lora_up.weight",
        "text_model.transformer.layers.10.self_attn.q_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_10_self_attn_k_proj.lora_up.weight",
        "text_model.transformer.layers.10.self_attn.k_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_10_self_attn_v_proj.lora_up.weight",
        "text_model.transformer.layers.10.self_attn.v_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_10_self_attn_out_proj.lora_up.weight",
        "text_model.transformer.layers.10.self_attn.out_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_10_mlp_fc1.lora_up.weight",
        "text_model.transformer.layers.10.linear1.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_10_mlp_fc2.lora_up.weight",
        "text_model.transformer.layers.10.linear2.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_10_self_attn_q_proj.alpha",
        "text_model.transformer.layers.10.self_attn.q_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_10_self_attn_k_proj.alpha",
        "text_model.transformer.layers.10.self_attn.k_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_10_self_attn_v_proj.alpha",
        "text_model.transformer.layers.10.self_attn.v_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_10_self_attn_out_proj.alpha",
        "text_model.transformer.layers.10.self_attn.out_proj.alpha",
    ],
    ["lora_te_text_model_encoder_layers_10_mlp_fc1.alpha", "text_model.transformer.layers.10.linear1.alpha"],
    ["lora_te_text_model_encoder_layers_10_mlp_fc2.alpha", "text_model.transformer.layers.10.linear2.alpha"],
    [
        "lora_te_text_model_encoder_layers_11_self_attn_q_proj.lora_down.weight",
        "text_model.transformer.layers.11.self_attn.q_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_11_self_attn_k_proj.lora_down.weight",
        "text_model.transformer.layers.11.self_attn.k_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_11_self_attn_v_proj.lora_down.weight",
        "text_model.transformer.layers.11.self_attn.v_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_11_self_attn_out_proj.lora_down.weight",
        "text_model.transformer.layers.11.self_attn.out_proj.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_11_mlp_fc1.lora_down.weight",
        "text_model.transformer.layers.11.linear1.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_11_mlp_fc2.lora_down.weight",
        "text_model.transformer.layers.11.linear2.lora_down.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_11_self_attn_q_proj.lora_up.weight",
        "text_model.transformer.layers.11.self_attn.q_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_11_self_attn_k_proj.lora_up.weight",
        "text_model.transformer.layers.11.self_attn.k_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_11_self_attn_v_proj.lora_up.weight",
        "text_model.transformer.layers.11.self_attn.v_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_11_self_attn_out_proj.lora_up.weight",
        "text_model.transformer.layers.11.self_attn.out_proj.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_11_mlp_fc1.lora_up.weight",
        "text_model.transformer.layers.11.linear1.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_11_mlp_fc2.lora_up.weight",
        "text_model.transformer.layers.11.linear2.lora_up.weight",
    ],
    [
        "lora_te_text_model_encoder_layers_11_self_attn_q_proj.alpha",
        "text_model.transformer.layers.11.self_attn.q_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_11_self_attn_k_proj.alpha",
        "text_model.transformer.layers.11.self_attn.k_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_11_self_attn_v_proj.alpha",
        "text_model.transformer.layers.11.self_attn.v_proj.alpha",
    ],
    [
        "lora_te_text_model_encoder_layers_11_self_attn_out_proj.alpha",
        "text_model.transformer.layers.11.self_attn.out_proj.alpha",
    ],
    ["lora_te_text_model_encoder_layers_11_mlp_fc1.alpha", "text_model.transformer.layers.11.linear1.alpha"],
    ["lora_te_text_model_encoder_layers_11_mlp_fc2.alpha", "text_model.transformer.layers.11.linear2.alpha"],
    [
        "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight",
        "down_blocks.0.attentions.0.proj_in.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight",
        "down_blocks.0.attentions.0.proj_in.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_v.lora_down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_v.lora_up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_2.lora_down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_2.lora_up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_q.lora_down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_q.lora_up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_v.lora_down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_v.lora_up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_proj_out.lora_down.weight",
        "down_blocks.0.attentions.0.proj_out.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_0_proj_out.lora_up.weight",
        "down_blocks.0.attentions.0.proj_out.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_proj_in.lora_down.weight",
        "down_blocks.0.attentions.1.proj_in.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_proj_in.lora_up.weight",
        "down_blocks.0.attentions.1.proj_in.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_q.lora_down.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_q.lora_up.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_k.lora_down.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_k.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_k.lora_up.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_k.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_v.lora_down.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_v.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_v.lora_up.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_v.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_up.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_down.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.proj.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.proj.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_ff_net_2.lora_down.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.ff.net.2.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_ff_net_2.lora_up.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.ff.net.2.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_attn2_to_q.lora_down.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_q.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_attn2_to_q.lora_up.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_q.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_attn2_to_k.lora_down.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_k.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_attn2_to_k.lora_up.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_k.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_attn2_to_v.lora_down.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_v.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_attn2_to_v.lora_up.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_v.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_down.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_up.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_proj_out.lora_down.weight",
        "down_blocks.0.attentions.1.proj_out.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_0_attentions_1_proj_out.lora_up.weight",
        "down_blocks.0.attentions.1.proj_out.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_proj_in.lora_down.weight",
        "down_blocks.1.attentions.0.proj_in.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_proj_in.lora_up.weight",
        "down_blocks.1.attentions.0.proj_in.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_v.lora_down.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_v.lora_up.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_up.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_down.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_2.lora_down.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.2.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_2.lora_up.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.2.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_q.lora_down.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_q.lora_up.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_v.lora_down.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_v.lora_up.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_down.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_up.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_proj_out.lora_down.weight",
        "down_blocks.1.attentions.0.proj_out.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_0_proj_out.lora_up.weight",
        "down_blocks.1.attentions.0.proj_out.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_proj_in.lora_down.weight",
        "down_blocks.1.attentions.1.proj_in.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_proj_in.lora_up.weight",
        "down_blocks.1.attentions.1.proj_in.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_attn1_to_q.lora_down.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_attn1_to_q.lora_up.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_attn1_to_k.lora_down.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_attn1_to_k.lora_up.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_attn1_to_v.lora_down.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_attn1_to_v.lora_up.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_up.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_down.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_ff_net_2.lora_down.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.ff.net.2.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_ff_net_2.lora_up.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.ff.net.2.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_attn2_to_q.lora_down.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_attn2_to_q.lora_up.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_attn2_to_k.lora_down.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_attn2_to_k.lora_up.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_attn2_to_v.lora_down.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_attn2_to_v.lora_up.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_down.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_up.weight",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_proj_out.lora_down.weight",
        "down_blocks.1.attentions.1.proj_out.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_1_attentions_1_proj_out.lora_up.weight",
        "down_blocks.1.attentions.1.proj_out.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_proj_in.lora_down.weight",
        "down_blocks.2.attentions.0.proj_in.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_proj_in.lora_up.weight",
        "down_blocks.2.attentions.0.proj_in.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_v.lora_down.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_v.lora_up.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_up.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_down.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_ff_net_2.lora_down.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.ff.net.2.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_ff_net_2.lora_up.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.ff.net.2.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_q.lora_down.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_q.lora_up.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_v.lora_down.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_v.lora_up.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_down.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_up.weight",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_proj_out.lora_down.weight",
        "down_blocks.2.attentions.0.proj_out.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_0_proj_out.lora_up.weight",
        "down_blocks.2.attentions.0.proj_out.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_proj_in.lora_down.weight",
        "down_blocks.2.attentions.1.proj_in.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_proj_in.lora_up.weight",
        "down_blocks.2.attentions.1.proj_in.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q.lora_down.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q.lora_up.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_k.lora_down.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_k.lora_up.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_v.lora_down.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_v.lora_up.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_up.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_down.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_ff_net_2.lora_down.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.ff.net.2.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_ff_net_2.lora_up.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.ff.net.2.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_q.lora_down.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_q.lora_up.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_k.lora_down.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_k.lora_up.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_v.lora_down.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_v.lora_up.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_down.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_up.weight",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_proj_out.lora_down.weight",
        "down_blocks.2.attentions.1.proj_out.lora_down.weight",
    ],
    [
        "lora_unet_down_blocks_2_attentions_1_proj_out.lora_up.weight",
        "down_blocks.2.attentions.1.proj_out.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_proj_in.lora_down.weight",
        "up_blocks.1.attentions.0.proj_in.lora_down.weight",
    ],
    ["lora_unet_up_blocks_1_attentions_0_proj_in.lora_up.weight", "up_blocks.1.attentions.0.proj_in.lora_up.weight"],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_v.lora_down.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_v.lora_up.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_up.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_down.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_ff_net_2.lora_down.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.ff.net.2.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_ff_net_2.lora_up.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.ff.net.2.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_q.lora_down.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_q.lora_up.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_v.lora_down.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_v.lora_up.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_down.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_up.weight",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_0_proj_out.lora_down.weight",
        "up_blocks.1.attentions.0.proj_out.lora_down.weight",
    ],
    ["lora_unet_up_blocks_1_attentions_0_proj_out.lora_up.weight", "up_blocks.1.attentions.0.proj_out.lora_up.weight"],
    [
        "lora_unet_up_blocks_1_attentions_1_proj_in.lora_down.weight",
        "up_blocks.1.attentions.1.proj_in.lora_down.weight",
    ],
    ["lora_unet_up_blocks_1_attentions_1_proj_in.lora_up.weight", "up_blocks.1.attentions.1.proj_in.lora_up.weight"],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_q.lora_down.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_q.lora_up.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_k.lora_down.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_k.lora_up.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_v.lora_down.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_v.lora_up.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_up.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_down.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_ff_net_2.lora_down.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.ff.net.2.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_ff_net_2.lora_up.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.ff.net.2.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_q.lora_down.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_q.lora_up.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_k.lora_down.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_k.lora_up.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_v.lora_down.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_v.lora_up.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_down.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_up.weight",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_1_proj_out.lora_down.weight",
        "up_blocks.1.attentions.1.proj_out.lora_down.weight",
    ],
    ["lora_unet_up_blocks_1_attentions_1_proj_out.lora_up.weight", "up_blocks.1.attentions.1.proj_out.lora_up.weight"],
    [
        "lora_unet_up_blocks_1_attentions_2_proj_in.lora_down.weight",
        "up_blocks.1.attentions.2.proj_in.lora_down.weight",
    ],
    ["lora_unet_up_blocks_1_attentions_2_proj_in.lora_up.weight", "up_blocks.1.attentions.2.proj_in.lora_up.weight"],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_q.lora_down.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_q.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_q.lora_up.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_q.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_k.lora_down.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_k.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_k.lora_up.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_k.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_v.lora_down.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_v.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_v.lora_up.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_v.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_out_0.lora_up.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_ff_net_0_proj.lora_down.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.ff.net.0.proj.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.ff.net.0.proj.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_ff_net_2.lora_down.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.ff.net.2.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_ff_net_2.lora_up.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.ff.net.2.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_q.lora_down.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_q.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_q.lora_up.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_q.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_k.lora_down.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_k.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_k.lora_up.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_k.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_v.lora_down.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_v.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_v.lora_up.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_v.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_out_0.lora_down.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_out_0.lora_up.weight",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_1_attentions_2_proj_out.lora_down.weight",
        "up_blocks.1.attentions.2.proj_out.lora_down.weight",
    ],
    ["lora_unet_up_blocks_1_attentions_2_proj_out.lora_up.weight", "up_blocks.1.attentions.2.proj_out.lora_up.weight"],
    [
        "lora_unet_up_blocks_2_attentions_0_proj_in.lora_down.weight",
        "up_blocks.2.attentions.0.proj_in.lora_down.weight",
    ],
    ["lora_unet_up_blocks_2_attentions_0_proj_in.lora_up.weight", "up_blocks.2.attentions.0.proj_in.lora_up.weight"],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_attn1_to_v.lora_down.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_attn1_to_v.lora_up.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_up.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_down.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_ff_net_2.lora_down.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.ff.net.2.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_ff_net_2.lora_up.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.ff.net.2.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_attn2_to_q.lora_down.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_attn2_to_q.lora_up.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_attn2_to_v.lora_down.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_attn2_to_v.lora_up.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_down.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_up.weight",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_0_proj_out.lora_down.weight",
        "up_blocks.2.attentions.0.proj_out.lora_down.weight",
    ],
    ["lora_unet_up_blocks_2_attentions_0_proj_out.lora_up.weight", "up_blocks.2.attentions.0.proj_out.lora_up.weight"],
    [
        "lora_unet_up_blocks_2_attentions_1_proj_in.lora_down.weight",
        "up_blocks.2.attentions.1.proj_in.lora_down.weight",
    ],
    ["lora_unet_up_blocks_2_attentions_1_proj_in.lora_up.weight", "up_blocks.2.attentions.1.proj_in.lora_up.weight"],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q.lora_down.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q.lora_up.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_attn1_to_k.lora_down.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_attn1_to_k.lora_up.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_attn1_to_v.lora_down.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_attn1_to_v.lora_up.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_up.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_down.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_ff_net_2.lora_down.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.ff.net.2.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_ff_net_2.lora_up.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.ff.net.2.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_q.lora_down.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_q.lora_up.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_k.lora_down.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_k.lora_up.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_v.lora_down.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_v.lora_up.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_down.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_up.weight",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_1_proj_out.lora_down.weight",
        "up_blocks.2.attentions.1.proj_out.lora_down.weight",
    ],
    ["lora_unet_up_blocks_2_attentions_1_proj_out.lora_up.weight", "up_blocks.2.attentions.1.proj_out.lora_up.weight"],
    [
        "lora_unet_up_blocks_2_attentions_2_proj_in.lora_down.weight",
        "up_blocks.2.attentions.2.proj_in.lora_down.weight",
    ],
    ["lora_unet_up_blocks_2_attentions_2_proj_in.lora_up.weight", "up_blocks.2.attentions.2.proj_in.lora_up.weight"],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_attn1_to_q.lora_down.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_q.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_attn1_to_q.lora_up.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_q.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_attn1_to_k.lora_down.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_k.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_attn1_to_k.lora_up.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_k.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_attn1_to_v.lora_down.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_v.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_attn1_to_v.lora_up.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_v.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_attn1_to_out_0.lora_up.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_ff_net_0_proj.lora_down.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.ff.net.0.proj.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.ff.net.0.proj.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_ff_net_2.lora_down.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.ff.net.2.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_ff_net_2.lora_up.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.ff.net.2.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_attn2_to_q.lora_down.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_q.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_attn2_to_q.lora_up.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_q.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_attn2_to_k.lora_down.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_k.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_attn2_to_k.lora_up.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_k.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_attn2_to_v.lora_down.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_v.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_attn2_to_v.lora_up.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_v.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_attn2_to_out_0.lora_down.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_transformer_blocks_0_attn2_to_out_0.lora_up.weight",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_2_attentions_2_proj_out.lora_down.weight",
        "up_blocks.2.attentions.2.proj_out.lora_down.weight",
    ],
    ["lora_unet_up_blocks_2_attentions_2_proj_out.lora_up.weight", "up_blocks.2.attentions.2.proj_out.lora_up.weight"],
    [
        "lora_unet_up_blocks_3_attentions_0_proj_in.lora_down.weight",
        "up_blocks.3.attentions.0.proj_in.lora_down.weight",
    ],
    ["lora_unet_up_blocks_3_attentions_0_proj_in.lora_up.weight", "up_blocks.3.attentions.0.proj_in.lora_up.weight"],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_q.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_q.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_attn1_to_v.lora_down.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_v.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_attn1_to_v.lora_up.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_v.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_up.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_down.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.ff.net.0.proj.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.ff.net.0.proj.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_ff_net_2.lora_down.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.ff.net.2.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_ff_net_2.lora_up.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.ff.net.2.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_attn2_to_q.lora_down.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_q.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_attn2_to_q.lora_up.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_q.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_k.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_k.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_attn2_to_v.lora_down.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_v.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_attn2_to_v.lora_up.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_v.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_down.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_up.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_0_proj_out.lora_down.weight",
        "up_blocks.3.attentions.0.proj_out.lora_down.weight",
    ],
    ["lora_unet_up_blocks_3_attentions_0_proj_out.lora_up.weight", "up_blocks.3.attentions.0.proj_out.lora_up.weight"],
    [
        "lora_unet_up_blocks_3_attentions_1_proj_in.lora_down.weight",
        "up_blocks.3.attentions.1.proj_in.lora_down.weight",
    ],
    ["lora_unet_up_blocks_3_attentions_1_proj_in.lora_up.weight", "up_blocks.3.attentions.1.proj_in.lora_up.weight"],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_attn1_to_q.lora_down.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_q.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_attn1_to_q.lora_up.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_q.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_attn1_to_k.lora_down.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_k.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_attn1_to_k.lora_up.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_k.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_attn1_to_v.lora_down.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_v.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_attn1_to_v.lora_up.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_v.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_up.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_down.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.ff.net.0.proj.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.ff.net.0.proj.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_ff_net_2.lora_down.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.ff.net.2.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_ff_net_2.lora_up.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.ff.net.2.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_attn2_to_q.lora_down.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_q.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_attn2_to_q.lora_up.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_q.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_attn2_to_k.lora_down.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_k.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_attn2_to_k.lora_up.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_k.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_attn2_to_v.lora_down.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_v.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_attn2_to_v.lora_up.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_v.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_down.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_up.weight",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_1_proj_out.lora_down.weight",
        "up_blocks.3.attentions.1.proj_out.lora_down.weight",
    ],
    ["lora_unet_up_blocks_3_attentions_1_proj_out.lora_up.weight", "up_blocks.3.attentions.1.proj_out.lora_up.weight"],
    [
        "lora_unet_up_blocks_3_attentions_2_proj_in.lora_down.weight",
        "up_blocks.3.attentions.2.proj_in.lora_down.weight",
    ],
    ["lora_unet_up_blocks_3_attentions_2_proj_in.lora_up.weight", "up_blocks.3.attentions.2.proj_in.lora_up.weight"],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_q.lora_down.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_q.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_q.lora_up.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_q.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_k.lora_down.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_k.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_k.lora_up.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_k.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_v.lora_down.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_v.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_v.lora_up.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_v.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_out_0.lora_up.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_ff_net_0_proj.lora_down.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.ff.net.0.proj.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.ff.net.0.proj.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_ff_net_2.lora_down.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.ff.net.2.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_ff_net_2.lora_up.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.ff.net.2.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_q.lora_down.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_q.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_q.lora_up.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_q.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_k.lora_down.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_k.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_k.lora_up.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_k.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_v.lora_down.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_v.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_v.lora_up.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_v.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_out_0.lora_down.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_out_0.lora_up.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_up_blocks_3_attentions_2_proj_out.lora_down.weight",
        "up_blocks.3.attentions.2.proj_out.lora_down.weight",
    ],
    ["lora_unet_up_blocks_3_attentions_2_proj_out.lora_up.weight", "up_blocks.3.attentions.2.proj_out.lora_up.weight"],
    ["lora_unet_mid_block_attentions_0_proj_in.lora_down.weight", "mid_block.attentions.0.proj_in.lora_down.weight"],
    ["lora_unet_mid_block_attentions_0_proj_in.lora_up.weight", "mid_block.attentions.0.proj_in.lora_up.weight"],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_q.lora_down.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_q.lora_up.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_k.lora_down.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_k.lora_up.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_v.lora_down.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_v.lora_down.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_v.lora_up.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_v.lora_up.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_up.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_up.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_down.weight",
        "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj.lora_down.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
        "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj.lora_up.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_ff_net_2.lora_down.weight",
        "mid_block.attentions.0.transformer_blocks.0.ff.net.2.lora_down.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_ff_net_2.lora_up.weight",
        "mid_block.attentions.0.transformer_blocks.0.ff.net.2.lora_up.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn2_to_q.lora_down.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_q.lora_down.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn2_to_q.lora_up.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_q.lora_up.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.lora_down.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.lora_up.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn2_to_v.lora_down.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_v.lora_down.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn2_to_v.lora_up.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_v.lora_up.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_down.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_down.weight",
    ],
    [
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_up.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_up.weight",
    ],
    ["lora_unet_mid_block_attentions_0_proj_out.lora_down.weight", "mid_block.attentions.0.proj_out.lora_down.weight"],
    ["lora_unet_mid_block_attentions_0_proj_out.lora_up.weight", "mid_block.attentions.0.proj_out.lora_up.weight"],
]
