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
#
# Copyright (c) 2023 Daniel Bolya
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/dbolya/tomesd

import math
from typing import Any, Callable, Dict, Tuple, Type, Union

import paddle
import paddle.nn as nn

from ..models.transformer_2d import BasicTransformerBlock
from ..pipelines.pipeline_utils import DiffusionPipeline
from .ppnlp_patch_utils import patch_to

TOME_PREFIX = "ToMe"


def scatter_reduce(
    input: paddle.Tensor,
    dim: int,
    index: paddle.Tensor,
    src: paddle.Tensor,
    reduce: str = "mean",
    include_self: bool = True,
) -> paddle.Tensor:
    # reduce "sum", "prod", "mean",
    # TODO support "amax", "amin" and include_self = False
    if reduce in ["sum", "assign", "add"]:
        if reduce == "sum":
            reduce = "add"
        input.put_along_axis_(indices=index, values=src, axis=dim, reduce=reduce)
    elif reduce == "mean":
        # compute sum first
        input.put_along_axis_(indices=index, values=src, axis=dim, reduce="add")
        # compute div secondly
        input_div = paddle.ones_like(input).put_along_axis(
            indices=index, values=paddle.to_tensor(1.0, dtype=input.dtype), axis=dim, reduce="add"
        )
        input = input / input_div
    elif reduce in ["prod", "mul", "multiply"]:
        input = paddle.put_along_axis(input.cpu(), indices=index.cpu(), values=src.cpu(), axis=dim, reduce="mul")._to(
            device=paddle.get_device()
        )
    else:
        raise NotImplementedError("only support mode in ['add', 'sum', 'prod', 'mul', 'multiply', 'mean', 'assign']!")
    return input


# patch scatter_reduce
paddle.scatter_reduce = scatter_reduce
paddle.Tensor.scatter_reduce = scatter_reduce


def do_nothing(x: paddle.Tensor, mode: str = None):
    return x


def bipartite_soft_matching_random2d(
    metric: paddle.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    no_rand: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    with paddle.no_grad():

        hsy, wsx = h // sy, w // sx

        if no_rand:
            rand_idx = paddle.zeros((hsy, wsx, 1), dtype=paddle.int64)
        else:
            rand_idx = paddle.randint(sy * sx, shape=(hsy, wsx, 1), dtype=paddle.int64)

        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = paddle.zeros([hsy, wsx, sy * sx], dtype=paddle.int64)
        idx_buffer_view.put_along_axis_(
            axis=2, indices=rand_idx, values=-paddle.ones_like(rand_idx, dtype=rand_idx.dtype)
        )
        idx_buffer_view = (
            idx_buffer_view.reshape([hsy, wsx, sy, sx]).transpose([0, 2, 1, 3]).reshape([hsy * sy, wsx * sx])
        )

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = paddle.zeros([h, w], dtype=paddle.int64)
            idx_buffer[: (hsy * sy), : (wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape([1, -1, 1]).argsort(axis=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]  # src
        b_idx = rand_idx[:, :num_dst, :]  # dst

        def split(x):
            C = x.shape[-1]

            src = x.take_along_axis(indices=a_idx.expand([B, N - num_dst, C]), axis=1)
            dst = x.take_along_axis(indices=b_idx.expand([B, num_dst, C]), axis=1)
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(axis=-1, keepdim=True)
        a, b = split(metric)
        scores = paddle.matmul(a, b, transpose_y=True)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # node_max, node_idx = scores.max(axis=-1)
        # top_k vs max argmax
        # Find the most similar greedily
        node_max, node_idx = paddle.topk(scores, k=1, axis=-1)
        # node_max = scores.max(axis=-1)
        # node_idx = scores.argmax(axis=-1)
        edge_idx = node_max.argsort(axis=-2, descending=True)

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens

        dst_idx = node_idx.take_along_axis(indices=src_idx, axis=-2)

    def merge(x: paddle.Tensor, mode="mean") -> paddle.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = src.take_along_axis(indices=unm_idx.expand([n, t1 - r, c]), axis=-2)
        src = src.take_along_axis(indices=src_idx.expand([n, r, c]), axis=-2)

        dst = scatter_reduce(dst, -2, dst_idx.expand([n, r, c]), src, reduce=mode)

        return paddle.concat([unm, dst], axis=1)

    def unmerge(x: paddle.Tensor) -> paddle.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = dst.take_along_axis(indices=dst_idx.expand([B, r, c]), axis=-2)

        # Combine back to the original shape
        out = paddle.zeros([B, N, c], dtype=x.dtype)

        out.put_along_axis_(
            indices=b_idx.expand([B, num_dst, c]),
            values=dst,
            axis=-2,
        )
        out.put_along_axis_(
            indices=a_idx.expand([B, a_idx.shape[1], 1])
            .take_along_axis(indices=unm_idx, axis=1)
            .expand([B, unm_len, c]),
            values=unm,
            axis=-2,
        )
        out.put_along_axis_(
            indices=a_idx.expand([B, a_idx.shape[1], 1]).take_along_axis(indices=src_idx, axis=1).expand([B, r, c]),
            values=src,
            axis=-2,
        )

        return out

    return merge, unmerge


def compute_merge(x: paddle.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]

    if downsample <= args["max_downsample"]:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        r = int(x.shape[1] * args["ratio"])
        # If the batch size is odd, then it's not possible for promted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        m, u = bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], r, not use_rand)
    else:
        m, u = (do_nothing, do_nothing)

    m_a, u_a = (m, u) if args["merge_attn"] else (do_nothing, do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (do_nothing, do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"] else (do_nothing, do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good


def make_tome_block(block_class: Type[nn.Layer]) -> Type[nn.Layer]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBasicTransformerBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self: BasicTransformerBlock,
            hidden_states,
            encoder_hidden_states=None,
            timestep=None,
            attention_mask=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> paddle.Tensor:
            # (1) ToMe
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(hidden_states, self._tome_info)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # (2) ToMe m_a
            norm_hidden_states = m_a(norm_hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # (3) ToMe u_a
            hidden_states = u_a(attn_output) + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # (4) ToMe m_c
                norm_hidden_states = m_c(norm_hidden_states)

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
                # (5) ToMe u_c
                hidden_states = u_c(attn_output) + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            # (6) ToMe m_m
            norm_hidden_states = m_m(norm_hidden_states)

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            # (7) ToMe u_m
            hidden_states = u_m(ff_output) + hidden_states

            return hidden_states

    return ToMeBasicTransformerBlock


def hook_tome_model(model: nn.Layer):
    """Adds a forward pre hook to get the image size. This hook can be removed with remove_patch."""

    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


@patch_to([DiffusionPipeline, nn.Layer])
def remove_tome(model_or_pipe: Union[nn.Layer, DiffusionPipeline], only_return_self: bool = True):
    """Removes a patch from a ToMeXXX module if it was already patched."""
    model_list = []
    if isinstance(model_or_pipe, DiffusionPipeline):
        for _, component in model_or_pipe.components.items():
            if isinstance(component, nn.Layer):
                model_list.append(component)
            elif isinstance(component, (tuple, list)):
                for each_component in component:
                    if isinstance(component, nn.Layer):
                        model_list.append(each_component)
    elif isinstance(model_or_pipe, nn.Layer):
        model_list.append(model_or_pipe)

    for model in model_list:
        for _, module in model.named_sublayers(include_self=True):
            if hasattr(module, "_tome_info"):
                for hook in module._tome_info["hooks"]:
                    hook.remove()
                module._tome_info["hooks"].clear()

            if module.__class__.__name__.startswith(TOME_PREFIX):
                module.__class__ = module._parent

    if only_return_self:
        return model_or_pipe
    return model_or_pipe, model_list


@patch_to([DiffusionPipeline, nn.Layer])
def apply_tome(
    model_or_pipe: Union[nn.Layer, DiffusionPipeline],
    ratio: float = 0.5,
    max_downsample: int = 1,
    sx: int = 2,
    sy: int = 2,
    use_rand: bool = True,
    merge_attn: bool = True,
    merge_crossattn: bool = False,
    merge_mlp: bool = False,
):
    """
    Patches a stable diffusion model_or_pipe with ToMe.
    Apply this to the highest level stable diffusion object (i.e., it should have a .unet).

    Important Args:
     - model_or_pipe: A top level Stable Diffusion module or pipeline to patch in place.
     - ratio: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.
              The maximum value for this is 1-(1/(sx*sy)). By default, the max is 0.75 (I recommend <= 0.5 though).
              Higher values result in more speed-up, but with more visual quality loss.

    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply ToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - sx, sy: The stride for computing dst sets (see paper). A higher stride means you can merge more tokens,
               but the default of (2, 2) works well in most cases. Must divide the image size.
     - use_rand: Whether or not to allow random perturbations when computing dst sets (see paper). Usually
                 you'd want to leave this on, but if you're having weird artifacts try turning this off.
     - merge_attn: Whether or not to merge tokens for attention (recommended).
     - merge_crossattn: Whether or not to merge tokens for cross attention (not recommended).
     - merge_mlp: Whether or not to merge tokens for the mlp layers (very not recommended).

    """
    if ratio >= 1 - (1 / (sx * sy)):
        raise ValueError(f"The tome ratio must be less than {1-(1/(sx*sy))} !")

    # Make sure the model_or_pipe is not currently patched
    model_list = model_or_pipe.remove_tome(only_return_self=False)[1]

    for model in model_list:
        need_patch = False
        model._tome_info = {
            "size": None,
            "hooks": [],
            "args": {
                "ratio": ratio,
                "max_downsample": max_downsample,
                "sx": sx,
                "sy": sy,
                "use_rand": use_rand,
                "merge_attn": merge_attn,
                "merge_crossattn": merge_crossattn,
                "merge_mlp": merge_mlp,
            },
        }
        for _, module in model.named_sublayers(include_self=True):
            # If for some reason this has a different name, create an issue and I'll fix it
            if isinstance(module, BasicTransformerBlock):
                module.__class__ = make_tome_block(module.__class__)
                module._tome_info = model._tome_info
                need_patch = True

        if need_patch:
            hook_tome_model(model)

    return model_or_pipe
