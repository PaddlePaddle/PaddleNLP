# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
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
"""Utilities for inference model."""

import numpy as np
import paddle


def patch_paddlenlp_ops(eos_token_id, pad_token_id):
    import paddlenlp_ops

    paddlenlp_ops.save_with_output = lambda *args, **kwargs: None

    # TODO(guosheng): update the custom op code directly.
    ori_set_ends = paddlenlp_ops.set_stop_value_multi_ends

    def _set_ends(topk_ids, stop_flags, end_ids, mode):
        # infer model uses eos_token_id to pad and discriminate ending,
        # patch to use pad_token_id to pad to unify with non-infer model.
        topk_ids_out, stop_flags_out = ori_set_ends(topk_ids, stop_flags, end_ids, mode)
        if pad_token_id != eos_token_id:
            topk_ids_out = paddle.where(stop_flags, pad_token_id, topk_ids_out)
        return topk_ids_out, stop_flags_out

    paddlenlp_ops.set_stop_value_multi_ends = _set_ends


def patch_infer_generate(eos_token_id, pad_token_id):
    """patches for inference model to make FuseMT adapt"""
    # patch paddlenlp_ops, maybe update the custom op code directly later
    # NOTE: should patch paddlenlp_ops before infer model import
    patch_paddlenlp_ops(eos_token_id, pad_token_id)

    # patch get_weights_mapping for InferenceModel
    patch_infer_model()

    # patch GenerationInferenceModel.sample
    from paddlenlp.experimental.transformers.generation_utils import (
        GenerationInferenceModel,
    )

    ori_update_model_kwargs = GenerationInferenceModel.update_model_kwargs_for_generation

    def _update_model_kwargs(self, *args, **kwargs):
        # update_model_kwargs_for_generation only returns , hack to.
        model_kwargs = ori_update_model_kwargs(self, *args, **kwargs)
        next_tokens = model_kwargs["next_tokens"]
        all_input_ids = paddle.concat([model_kwargs["all_input_ids"], next_tokens], axis=1)
        model_kwargs["next_tokens"] = all_input_ids
        model_kwargs["all_input_ids"] = None
        return model_kwargs

    GenerationInferenceModel.update_model_kwargs_for_generation = _update_model_kwargs


_model_weights_mapping_dict = {}


def register_model(model_cls_name):
    def mark_cls_name(func):
        # Do not register here although we can, otherwise infer model would import
        # before paddlenlp_ops.
        _model_weights_mapping_dict[model_cls_name] = func
        return func

    return mark_cls_name


def patch_infer_model():
    import paddlenlp.experimental.transformers as infer_transformers

    for model_cls_name, get_weights_mapping in _model_weights_mapping_dict.items():
        model_cls = getattr(infer_transformers, model_cls_name)
        model_cls.get_weights_mapping = get_weights_mapping


@register_model("LlamaForCausalLMInferenceModel")
def get_weights_mapping(self):
    """model to infer model"""
    head_size = self.config.hidden_size // self.config.num_attention_heads

    def _concat_qkv(q, k, v):
        if isinstance(q, paddle.Tensor):
            concated_qkv_weight = paddle.concat([q, k, v], axis=-1).T.reshape(
                [
                    3 * (self.config.num_attention_heads // self.config.tensor_parallel_degree) * (head_size),
                    self.config.hidden_size,
                ]
            )
        else:
            concated_qkv_weight = (
                np.concatenate(
                    [q, k, v],
                    axis=-1,
                )
                .transpose(1, 0)
                .reshape(
                    3 * (self.config.num_attention_heads // self.config.tensor_parallel_degree) * (head_size),
                    self.config.hidden_size,
                )
            )

        return concated_qkv_weight

    def _concat_ffn1(w1, w2):
        if isinstance(w1, paddle.Tensor):
            concated_ffn1_weight = paddle.concat([w1, w2], axis=-1)
        else:
            concated_ffn1_weight = np.concatenate([w1, w2], axis=-1)
        return concated_ffn1_weight

    identity = lambda x: x

    weight_mapping = {}
    weight_mapping[self.lm_head.weight] = [
        identity,
        [
            "lm_head.weight",
        ],
    ]
    weight_mapping[self.llama.embed_tokens.weight] = [
        identity,
        [
            "llama.embed_tokens.weight",
        ],
    ]
    weight_mapping[self.llama.norm.weight] = [
        identity,
        [
            "llama.norm.weight",
        ],
    ]
    for idx in range(self.config.num_hidden_layers):
        weight_mapping[self.llama.transformer_block.qkv_weights[idx]] = [
            _concat_qkv,
            [
                f"llama.layers.{idx}.self_attn.q_proj.weight",
                f"llama.layers.{idx}.self_attn.k_proj.weight",
                f"llama.layers.{idx}.self_attn.v_proj.weight",
            ],
        ]
        weight_mapping[self.llama.transformer_block.ffn1_weights[idx]] = [
            _concat_ffn1,
            [
                f"llama.layers.{idx}.mlp.gate_proj.weight",
                f"llama.layers.{idx}.mlp.up_proj.weight",
            ],
        ]
        weight_mapping[self.llama.transformer_block.linear_weights[idx]] = [
            identity,
            [
                f"llama.layers.{idx}.self_attn.o_proj.weight",
            ],
        ]
        weight_mapping[self.llama.transformer_block.ffn2_weights[idx]] = [
            identity,
            [
                f"llama.layers.{idx}.mlp.down_proj.weight",
            ],
        ]
        weight_mapping[self.llama.transformer_block.ln_scales[idx]] = [
            identity,
            [
                f"llama.layers.{idx}.input_layernorm.weight",
            ],
        ]
        weight_mapping[self.llama.transformer_block.ffn_ln_scales[idx]] = [
            identity,
            [
                f"llama.layers.{idx}.post_attention_layernorm.weight",
            ],
        ]
    return weight_mapping
