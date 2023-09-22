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

from typing import List, Union

import paddle
import paddle.nn.functional as F
# from paddlenlp_ops import (
#     get_token_penalty_multi_scores,
#     save_with_output,
#     set_alibi_mask_value,
#     set_mask_value,
#     set_stop_value_multi_ends,
#     set_value_by_flags_and_idx,
# )

from paddlenlp.generation import GenerationMixin, LogitsProcessor, LogitsProcessorList

# try:
#     from paddle import top_p_sampling
# except:
#     from paddlenlp_ops import top_p_sampling


__all__ = ["GenerationInferenceModel"]


class ForcedDecodingEOSTokenLogitsProcessor(LogitsProcessor):
    """
    This `LogitsProcessor` enforces the last generated token to be the selected `forced_eos_token`.

    Args:
        max_length (int): The maximum length of the sequence to be generated.
        forced_eos_token_id (int): The id of the token to be generated as the last token.
    """

    def __init__(self, max_decoding_step: int, forced_eos_token_id: Union[int, List[int]]):
        self.max_decoding_step = max_decoding_step
        self.forced_eos_token_id = forced_eos_token_id

    def __call__(self, input_ids, scores, decoding_step):
        if decoding_step == self.max_decoding_step:
            scores[:] = paddle.finfo(scores.dtype).min
            scores[:, self.forced_eos_token_id] = 0
        return scores


class GenerationInferenceModel(GenerationMixin):
    @classmethod
    def get_cache_kvs_shape(cls, max_batch_size: int = None, max_length: int = None) -> list[list[int]]:
        raise NotImplementedError

    def to_static(self, output_path: str, config: dict):
        dtype = config.get("dtype", paddle.get_default_dtype())

        cache_kvs_shapes = self.get_cache_kvs_shape(self.config, max_length=config.get("max_length", None))
        export_precache = config.get("export_precache", False)
        if export_precache:
            precache_input_spec = [
                paddle.static.InputSpec(shape=[2, None, None, None, None], dtype=dtype, name=f"pre_caches_{i}")
                for i in range(len(cache_kvs_shapes))
            ]
        else:
            precache_input_spec = None

        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),  # input_ids
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype=dtype, name="attention_mask"),  # attention_mask
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="position_ids"),  # position_ids
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="penalty_score"),  # penalty_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="frequency_score"),  # frequency_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="presence_score"),  # presence_score
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="min_length"),  # min_decode_length
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="max_length"),  # max_decode_length
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="temperature"),  # temperature
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="top_p"),  # top_p
            paddle.static.InputSpec(shape=[None], dtype="int64", name="eos_token_id"),  # eos_token_id
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_len_encoder"),  # seq_len_encoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_len_decoder"),  # seq_len_decoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="step_idx"),  # step_idx
            paddle.static.InputSpec(shape=[None, 1], dtype="bool", name="stop_flags"),  # stop_flags
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="tgt_ids"),  # tgt_ids
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="tgt_pos"),  # tgt_pos
            paddle.static.InputSpec(
                shape=[None, 1, 1, None], dtype=dtype, name="tgt_generation_mask"
            ),  # tgt_generation_mask
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="pre_ids"),  # pre_ids
            paddle.static.InputSpec(shape=[1], dtype="int64", name="stop_nums"),  # stop_nums
            [
                paddle.static.InputSpec(
                    shape=shape,
                    dtype=dtype,
                    name="cache_kvs_{}".format(i),
                )
                for i, shape in enumerate(cache_kvs_shapes)
            ],  # cache_kvs
            None,  # inputs_embeds
            config.get("logits_processors", None),
            precache_input_spec,
        ]
        if self.config["model_type"] and "chatglm" in self.config.model_type:
            input_spec[2] = paddle.static.InputSpec(
                shape=[None, None, None], dtype="int64", name="position_ids"
            )  # position_ids
            input_spec[16] = paddle.static.InputSpec(shape=[None, 2, 1], dtype="int64", name="tgt_pos")  # tgt_pos
        elif self.config["model_type"] and "gpt" in self.config.model_type:
            input_spec[2] = paddle.static.InputSpec(shape=[None], dtype="int64", name="position_ids")  # position_ids
        model = paddle.jit.to_static(self.generate, input_spec=input_spec)
        paddle.jit.save(
            model, output_path, skip_prune_program=True
        )  # Note(Zhengzekang): If we prune program it may cause some inference error.

    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        seq_len = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
            seq_len = encoder_output.shape[1]
        return paddle.ones([batch_size, seq_len], dtype="int64") * bos_token_id

    @paddle.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        penalty_score=None,
        frequency_score=None,
        presence_score=None,
        min_length=None,
        max_length=None,
        temperature=None,
        top_p=None,
        eos_token_id=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        step_idx=None,
        stop_flags=None,
        tgt_ids=None,
        tgt_pos=None,
        tgt_generation_mask=None,
        pre_ids=None,
        stop_nums=None,
        cache_kvs=[],
        inputs_embeds=None,
        logits_processors=None,
        pre_caches=None,
        **model_kwargs,
    ):

        model_kwargs["position_ids"] = position_ids
        model_kwargs["attention_mask"] = attention_mask

        model_kwargs["seq_len_encoder"] = seq_len_encoder
        model_kwargs["seq_len_decoder"] = seq_len_decoder
        model_kwargs["tgt_ids"] = tgt_ids
        model_kwargs["tgt_generation_mask"] = tgt_generation_mask
        model_kwargs["tgt_pos"] = tgt_pos
        model_kwargs["step_idx"] = step_idx
        model_kwargs["stop_flags"] = stop_flags
        model_kwargs["pre_ids"] = pre_ids
        model_kwargs["min_dec_len"] = min_length
        model_kwargs["max_dec_len"] = max_length
        model_kwargs["stop_nums"] = stop_nums
        model_kwargs["penalty_score"] = penalty_score
        model_kwargs["frequency_score"] = frequency_score
        model_kwargs["presence_score"] = presence_score
        model_kwargs["logits_processors"] = logits_processors or LogitsProcessorList()
        model_kwargs["pre_caches"] = pre_caches

        ret = self.sample(
            input_ids,
            eos_token_id,
            top_p=top_p,
            cache_kvs=cache_kvs,
            temperature=temperature,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )
        return ret

    def update_model_kwargs_for_generation(self, cache, just_decoder, next_tokens, eos_token_id, model_kwargs):
        if cache is None:
            model_kwargs["step_idx"] = paddle.where(
                model_kwargs["seq_len_encoder"] == 0,
                model_kwargs["step_idx"],
                model_kwargs["step_idx"] + 1,
            )
        else:
            model_kwargs["step_idx"] = paddle.where(
                model_kwargs["stop_flags"],
                model_kwargs["step_idx"],
                model_kwargs["step_idx"] + 1,
            )
        length_cond = paddle.greater_equal(model_kwargs["step_idx"], model_kwargs["max_dec_len"])
        model_kwargs["stop_flags"] = paddle.logical_or(model_kwargs["stop_flags"], length_cond)
        if cache is None:
            next_tokens = paddle.where(just_decoder, paddle.full_like(next_tokens, -1), next_tokens)
        next_tokens, model_kwargs["stop_flags"] = set_stop_value_multi_ends(
            next_tokens, model_kwargs["stop_flags"], eos_token_id, 2
        )  # multi ends

        if cache is None:
            # encoder's generation
            model_kwargs["tgt_ids"] = paddle.where(just_decoder, model_kwargs["tgt_ids"], next_tokens)
            if self.config["position_encoding_2d"] and self.config.position_encoding_2d is True:
                tgt_pos = model_kwargs["tgt_pos"]
                new_position_id = tgt_pos[:, 0, :].clone()
                new_block_id = tgt_pos[:, 1, :].clone()
                new_block_id = new_block_id + 1

                model_kwargs["tgt_pos"] = paddle.concat(
                    [new_position_id.unsqueeze(1), new_block_id.unsqueeze(1)], axis=1
                )
            else:
                model_kwargs["tgt_pos"] = paddle.where(
                    just_decoder, model_kwargs["tgt_pos"], model_kwargs["tgt_pos"] + 1
                )
            if "bloom" in self.config.architectures[0].lower():
                model_kwargs["seq_len_decoder"] = set_alibi_mask_value(
                    model_kwargs["tgt_generation_mask"],
                    model_kwargs["stop_flags"],
                    model_kwargs["seq_len_decoder"],
                    model_kwargs["position_ids"],
                    model_kwargs["tgt_pos"],
                )
            else:
                model_kwargs["seq_len_decoder"] = set_mask_value(
                    model_kwargs["tgt_generation_mask"],
                    model_kwargs["stop_flags"],
                    model_kwargs["seq_len_decoder"],
                )
        else:
            model_kwargs["tgt_ids"] = next_tokens
            if self.config["position_encoding_2d"] and self.config.position_encoding_2d is True:
                tgt_pos = model_kwargs["tgt_pos"]
                new_position_id = tgt_pos[:, 0, :].clone()
                new_block_id = tgt_pos[:, 1, :].clone()
                new_block_id = new_block_id + 1

                model_kwargs["tgt_pos"] = paddle.concat(
                    [new_position_id.unsqueeze(1), new_block_id.unsqueeze(1)], axis=1
                )
            else:
                model_kwargs["tgt_pos"] = paddle.where(
                    model_kwargs["stop_flags"],
                    model_kwargs["tgt_pos"],
                    model_kwargs["tgt_pos"] + 1,
                )

            model_kwargs["seq_len_decoder"] = paddle.where(
                model_kwargs["stop_flags"],
                model_kwargs["seq_len_decoder"],
                model_kwargs["seq_len_decoder"] + 1,
            )
            if "bloom" in self.config.architectures[0].lower():
                model_kwargs["seq_len_decoder"] = set_alibi_mask_value(
                    model_kwargs["tgt_generation_mask"],
                    model_kwargs["stop_flags"],
                    model_kwargs["seq_len_decoder"],
                    model_kwargs["position_ids"],
                    model_kwargs["tgt_pos"],
                )
            else:
                model_kwargs["seq_len_decoder"] = set_mask_value(
                    model_kwargs["tgt_generation_mask"],
                    model_kwargs["stop_flags"],
                    model_kwargs["seq_len_decoder"],
                )

        model_kwargs["next_tokens"] = next_tokens
        return model_kwargs

    def sample(
        self,
        input_ids=None,
        eos_token_id=None,
        cache_kvs=[],
        top_p=None,
        temperature=None,
        inputs_embeds=None,
        **model_kwargs,
    ):
        step_idx_ori = paddle.full(shape=[1], dtype="int64", fill_value=1)
        batch_idx = paddle.full(shape=[1], dtype="int32", fill_value=-1)

        # fake temp next_tokens
        batch = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        next_tokens = paddle.full(shape=[batch, 1], dtype="int32", fill_value=0)

        # let inputs_embeds enter into model_kwargs.
        # because the code below directly use the model_kwargs as a parameter without using inputs_embeds.
        model_kwargs["inputs_embeds"] = inputs_embeds
        model_kwargs["all_input_ids"] = input_ids
        logits_processors = model_kwargs.pop("logits_processors")

        def _forward_(**args):
            # cache_kvs is never empty because it is passed as a parameter in def sample.
            model_inputs = self.prepare_inputs_for_generation(input_ids, cache_kvs, **args)
            return self(**model_inputs)

        def _post_process_(outputs, top_p, temperature, step_idx_ori, model_kwargs):
            cache = model_kwargs.get("cache", None)
            just_decoder = model_kwargs["seq_len_encoder"] == 0
            if cache is None:  # first decoder
                step_idx = paddle.where(
                    just_decoder,
                    paddle.full_like(model_kwargs["step_idx"], -1),
                    model_kwargs["step_idx"],
                )  # not update when continue decode
            else:
                step_idx = model_kwargs["step_idx"]
            model_kwargs["stop_flags"] = set_value_by_flags_and_idx(
                model_kwargs["pre_ids"],
                model_kwargs["tgt_ids"],
                step_idx,
                model_kwargs["stop_flags"],
            )
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            logits = paddle.cast(logits, paddle.float32)
            logits = logits_processors(model_kwargs["all_input_ids"], logits, decoding_step=step_idx_ori)

            logits = get_token_penalty_multi_scores(
                model_kwargs["pre_ids"],
                logits,
                model_kwargs["penalty_score"],
                model_kwargs["frequency_score"],
                model_kwargs["presence_score"],
                step_idx,
                model_kwargs["min_dec_len"],
                eos_token_id,
            )
            # sample
            probs = F.softmax(logits)

            # compute next_tokens, use paddle.top_p_sampling
            logits = logits / temperature

            _, next_tokens = top_p_sampling(probs, top_p, -1)

            if self.config.tensor_parallel_degree > 1:
                paddle.distributed.broadcast(next_tokens, 0)

            model_kwargs = self.update_model_kwargs_for_generation(
                cache, just_decoder, next_tokens, eos_token_id, model_kwargs
            )
            next_tokens = model_kwargs["next_tokens"]

            if model_kwargs["all_input_ids"] is None:
                model_kwargs["all_input_ids"] = next_tokens
            else:
                model_kwargs["all_input_ids"] = paddle.concat([model_kwargs["all_input_ids"], next_tokens], axis=1)

            save_with_output(
                next_tokens,
                batch_idx,
                step_idx_ori,
                "real_time_save.temp_ids",
                self.config.tensor_parallel_rank,
            )

            return next_tokens, model_kwargs

        # encoder
        outputs = _forward_(**model_kwargs)
        # first decoder
        next_tokens, model_kwargs = _post_process_(
            outputs,
            top_p,
            temperature,
            step_idx_ori,
            model_kwargs,
        )
        step_idx_ori += 1

        # gives it a value, means we will entered into decoder phase.
        model_kwargs["cache"] = 0

        # decoder
        while paddle.less_than(
            paddle.sum(paddle.cast(model_kwargs["stop_flags"], "int64")),
            model_kwargs["stop_nums"],
        ):
            next_tokens, model_kwargs = _post_process_(
                _forward_(**model_kwargs),
                top_p,
                temperature,
                step_idx_ori,
                model_kwargs,
            )
            step_idx_ori += 1

        return (
            next_tokens,
            model_kwargs["step_idx"],
            paddle.cast(model_kwargs["stop_flags"], "int32"),
            model_kwargs["seq_len_decoder"],
            model_kwargs["tgt_pos"],
        )
