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
from pyexpat import model
from tkinter.messagebox import NO

from typing import List, Union, Optional

import paddle
import paddle.nn.functional as F
from paddlenlp_ops import (
    get_token_penalty_multi_scores,
    get_token_penalty_multi_scores_v2,
    save_output,
    save_with_output,
    set_stop_value_multi_ends,
    set_stop_value_multi_ends_v2,
    set_value_by_flags_and_idx,
    set_value_by_flags_and_idx_v2,
    update_inputs,
)

from paddlenlp.generation import GenerationMixin, LogitsProcessor, LogitsProcessorList, StoppingCriteriaList

__all__ = ["GenerationInferenceModel", "GenerationBlockInferenceModel"]


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
        # use "==" to distingusih between chatglm and chatglm_v2.
        if self.config["model_type"] and "chatglm" == self.config.model_type.lower():
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
            model_kwargs["seq_len_decoder"] = paddle.where(
                model_kwargs["stop_flags"],
                model_kwargs["seq_len_decoder"] - model_kwargs["seq_len_decoder"],
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

            model_kwargs["seq_len_decoder"] = paddle.where(
                model_kwargs["stop_flags"],
                model_kwargs["seq_len_decoder"] - model_kwargs["seq_len_decoder"],
                model_kwargs["seq_len_decoder"],
            )

        model_kwargs["next_tokens"] = next_tokens
        return model_kwargs



    def assisted_decoding(
        self,
        input_ids,
        target_model_inputs,
        assistant_model_inputs,
        assistant_model: GenerationInferenceModel = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        gamma=5, # the max token_num assistant model generate one stage
        **model_kwargs,
    ):
        print("--------inital input_ids: ", input_ids)
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if eos_token_id is not None and pad_token_id is None:
            raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
        inputs_embeds = target_model_inputs.get("inputs_embeds", None)

        step_idx_ori = paddle.full(shape=[1], dtype="int64", fill_value=1)
        batch = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        target_model_inputs["all_input_ids"] = input_ids
        assistant_model_inputs["all_input_ids"] = input_ids

        eos_token_id = target_model_inputs.pop("eos_token_id", None)
        inputs_embeds = target_model_inputs.pop("inputs_embeds", None)
        # get cache_kvs and attention_mask
        target_model_cache_kvs = target_model_inputs.pop("cache_kvs", None)
        assistant_model_cache_kvs = assistant_model_inputs.pop("cache_kvs", None)
        assistant_model_attention_mask = assistant_model_inputs.get("attention_mask", None)

        def update_draft_model_kwargs_for_generation(cache, just_decoder, next_tokens, eos_token_id, model_kwargs):
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
                model_kwargs["seq_len_decoder"] = paddle.where(
                    model_kwargs["stop_flags"],
                    model_kwargs["seq_len_decoder"] - model_kwargs["seq_len_decoder"],
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
                model_kwargs["seq_len_decoder"] = paddle.where(
                    model_kwargs["stop_flags"],
                    model_kwargs["seq_len_decoder"] - model_kwargs["seq_len_decoder"],
                    model_kwargs["seq_len_decoder"],
                )
            model_kwargs["next_tokens"] = next_tokens
            return model_kwargs


        def update_target_model_kwargs_for_generation(cache, gamma, model_kwargs):
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
                model_kwargs["seq_len_decoder"] += gamma
            length_cond = paddle.greater_equal(model_kwargs["step_idx"], model_kwargs["max_dec_len"])
            model_kwargs["stop_flags"] = paddle.logical_or(model_kwargs["stop_flags"], length_cond)

            return model_kwargs

        def _forward_(model, input_ids, cache_kvs, **args):
            # cache_kvs is never empty because it is passed as a parameter in def sample.
            model_inputs = model.prepare_inputs_for_generation(input_ids, cache_kvs, **args)
            return model(**model_inputs)

        def _post_process_(outputs, step_idx_ori, is_target_model, model_kwargs):
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
            logits = logits_processor(model_kwargs["all_input_ids"], logits, decoding_step=step_idx_ori)

            # if not is_target_model:
            #     logits = get_token_penalty_multi_scores(
            #         model_kwargs["pre_ids"],
            #         logits,
            #         model_kwargs["penalty_score"],
            #         model_kwargs["frequency_score"],
            #         model_kwargs["presence_score"],
            #         step_idx,
            #         model_kwargs["min_dec_len"],
            #         eos_token_id,
            #     )

            if is_target_model:
                model_kwargs = update_target_model_kwargs_for_generation(
                    cache, gamma, model_kwargs
            )
                model_kwargs["cache"] = 0
                return logits
            else:
                logits = paddle.cast(logits, paddle.float32)
                next_tokens = logits.argmax(axis=-1)
                print("-----internal next_tokens: ", next_tokens)
                model_kwargs = update_draft_model_kwargs_for_generation(
                    cache, just_decoder, next_tokens, eos_token_id, model_kwargs
                )
                model_kwargs["cache"] = 0
                next_tokens = model_kwargs["next_tokens"]
                return logits, next_tokens


        eos_token_id_tensor = paddle.to_tensor(eos_token_id) if eos_token_id is not None else None
        max_len = stopping_criteria[0].max_length
        while True:
            # Assistant: main logic start
            cur_len = input_ids.shape[-1]

            #  1. Forecast next N tokens using the assistant model. This `for` block can be replaced with a
            # `.generate()` call if we decide to add `past_key_values` as a possible output of generate, as we
            # need access to the assistant cache to secure strong speedups.
            candidate_input_ids = input_ids
            
            assistant_model_output_logits = None
            if assistant_model_inputs.get("cache", None) is not None:
                assistant_model_inputs["seq_len_decoder"] -= remove_seq_len_decoder_num
            # 👉 小模型推理
            for _ in range(int(gamma)):
                # 1.1. use the assistant model to obtain the next candidate logits
                print("assistant_model's seq_len_encoder: ",assistant_model_inputs["seq_len_encoder"])
                print("assistant_model's seq_len_decoder: ",assistant_model_inputs["seq_len_decoder"])

                draft_model_logits, next_tokens = _post_process_(
                    _forward_(assistant_model, input_ids, assistant_model_cache_kvs, **assistant_model_inputs),
                    step_idx_ori,
                    False, # is_target_model
                    assistant_model_inputs,
                ) # assistant_model_logits: [bsz, vocab_size]
                # gives it a value, means we will entered into decoder phase.
                step_idx_ori += 1
                # 因为投机采样小模型需要输出 gamma 个 token 的 logits，所以需要在 seq_len 维度进行拼接
                if assistant_model_output_logits is None: # assistant_model_output_logits: [bsz, seq_len, vocab_size]
                    assistant_model_output_logits = draft_model_logits
                    # 原来是 [bsz, vocab_size] -> [bsz, 1, vocab_size] (添加一个 seq_len 维度)
                    assistant_model_output_logits = assistant_model_output_logits[:, None, :]
                else:
                    assistant_model_output_logits = paddle.concat((assistant_model_output_logits, draft_model_logits[:,None]), axis=1)
                # 1.2. greedily select the next candidate token
   
                assistant_model_output_logits[:, -1, :] = logits_processor(
                    candidate_input_ids, assistant_model_output_logits[:, -1, :]
                )
                if next_tokens.ndim == 1:
                    next_tokens = next_tokens[None, :]

                candidate_input_ids = paddle.concat((candidate_input_ids, next_tokens), axis=-1)
                candidate_logits = assistant_model_output_logits
                # 1.3. stop assistant generation on EOS
                if eos_token_id_tensor is not None:
                    last_assistant_token_is_eos = next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    last_assistant_token_is_eos = (
                        ~last_assistant_token_is_eos.not_equal(eos_token_id_tensor.unsqueeze(1)).to(paddle.bool)
                    )
                    if last_assistant_token_is_eos:
                        break
                else:
                    last_assistant_token_is_eos = False

            candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
            print("-----本轮 candidate_input_ids: ", candidate_input_ids)

            # 2. Use the original model to obtain the next token logits given the candidate sequence. We obtain
            # `candidate_length + 1` relevant logits from this process: in the event that all candidates are correct,
            # we use this forward pass to also pick the subsequent logits in the original model.

            # 2.1. Run a forward pass on the candidate sequence
            # 👉 大模型推理
            step_idx_ori = paddle.full(shape=[1], dtype="int64", fill_value=1)

            if target_model_inputs.get("cache") is None:
                target_model_input_ids = candidate_input_ids
                target_model_inputs["seq_len_encoder"][0] = target_model_input_ids.shape[1]
                target_model_inputs["seq_len_decoder"][0] = target_model_inputs["seq_len_encoder"][0]
                target_model_inputs["seq_len_decoder"] *= 0
                target_model_inputs["seq_lens_this_time"][0] = target_model_input_ids.shape[1]
            else:
                # target_model_input_ids = candidate_input_ids[:, -gamma:]
                # target_model_inputs["seq_len_encoder"][0] = 0
                # target_model_inputs["seq_len_decoder"][0] = candidate_input_ids.shape[1]
                # target_model_inputs["seq_lens_this_time"][0] = gamma
                pass

            print("target_model's seq_len_encoder: ",target_model_inputs["seq_len_encoder"])
            print("target_model's seq_len_decoder: ",target_model_inputs["seq_len_decoder"])
            print("target_model's seq_lens_this_time: ",target_model_inputs["seq_lens_this_time"])

            target_model_outputs_logits = _post_process_(
                _forward_(self, target_model_input_ids, target_model_cache_kvs, **target_model_inputs),
                step_idx_ori,
                True, # is_target_model
                target_model_inputs,
            )
            # [seq_len, vocab_size] -> [bsz, seq_len, vocab_size]
            target_model_outputs_logits = target_model_outputs_logits[None, :]
            print("!!!!!target model generated logits!!!!!!")
            
            # 2.2. Process the new logits
            new_logits = target_model_outputs_logits[:, -candidate_length-1 :, :]  # excludes the input prompt if present
            print("-------new_logits's shape: ", new_logits.shape)

            for i in range(candidate_length):
                new_logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i], new_logits[:, i])
            # if len(logits_warper) > 0:
            #     for i in range(candidate_length):
            #         new_logits[:, i, :] = logits_warper(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])

            # 3. Obtain the next tokens from the original model logits.
            max_matches = max_len - cur_len - 1
            # valid_tokens = new_token[:,None]
            # n_matches = gamma
            valid_tokens, n_matches = _speculative_sampling(
                candidate_input_ids,
                candidate_logits,
                candidate_length,
                new_logits,
                last_assistant_token_is_eos,
                max_matches,
            )

            # 4. Update variables according to the number of matching assistant tokens. Remember: the token generated
            # by the model after the last candidate match is also valid, as it is generated from a correct sequence.
            # Because of this last token, assisted generation search reduces to a normal greedy search/sample if there
            # is no match.

            # 4.1. Get the valid continuation, after the matching tokens
            input_ids = paddle.concat((input_ids, valid_tokens), axis=-1)
            remove_seq_len_decoder_num = gamma - valid_tokens.shape[-1]
            print("remove_seq_len_decoder_num: ", remove_seq_len_decoder_num)
            cur_len = input_ids.shape[-1]
            print("-----本轮大模型评估过后总的 ids: ", input_ids)

            # 小模型 cache_kvs 擦除
            # assistant_model_cache_kvs: [n_layer, 2, bsz, n_head, max_seq_len, d_head]
            for i in range(len(assistant_model_cache_kvs)):
                assistant_model_cache_kvs[i][:, :, :, cur_len - (gamma -  n_matches):, :] = 0

            # stop if we exceed the maximum length
            if input_ids.shape[-1] >= max_len:
                break

        return input_ids

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

            # compute next_tokens, use paddle.tensor.top_p_sampling
            logits = logits / temperature

            _, next_tokens = paddle.tensor.top_p_sampling(probs, top_p)

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
            # print("-----next_tokens: ", next_tokens)

        return (
            next_tokens,
            model_kwargs["step_idx"],
            paddle.cast(model_kwargs["stop_flags"], "int32"),
            model_kwargs["seq_len_decoder"],
            model_kwargs["tgt_pos"],
        )


class GenerationBlockInferenceModel(GenerationMixin):
    @classmethod
    def get_cache_kvs_shape(cls, max_batch_size: int = None, max_length: int = None) -> list[list[int]]:
        raise NotImplementedError

    def to_static(self, output_path: str, config: dict):
        dtype = config.get("dtype", paddle.get_default_dtype())
        cachekv_dtype = dtype

        cache_kvs_shapes = self.get_cache_kvs_shape(
            self.config, max_batch_size=config.get("max_batch_size", -1), max_length=config.get("max_length", None)
        )
        export_precache = config.get("export_precache", False)
        if export_precache:
            precache_kv_spec = [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype=dtype, name=f"pre_caches_{i}")
                for i in range(len(cache_kvs_shapes))
            ]
        else:
            precache_kv_spec = None
        use_cachekv_int8 = config.get("use_cachekv_int8", "None")

        if use_cachekv_int8 == "static" or use_cachekv_int8 == "dynamic":
            cachekv_dtype = "uint8"

        if use_cachekv_int8 == "dynamic":
            cache_k_quant_scales = [
                paddle.static.InputSpec(
                    shape=[None, self.config.num_attention_heads],
                    dtype="float32",
                    name="k_quant_scales_{}".format(i),
                )
                for i in range(int(len(cache_kvs_shapes) / 2))
            ]

            cache_v_quant_scales = [
                paddle.static.InputSpec(
                    shape=[None, self.config.num_attention_heads],
                    dtype="float32",
                    name="v_quant_scales_{}".format(i),
                )
                for i in range(int(len(cache_kvs_shapes) / 2))
            ]

            cache_k_dequant_scales = [
                paddle.static.InputSpec(
                    shape=[None, self.config.num_attention_heads],
                    dtype="float32",
                    name="k_dequant_scales_{}".format(i),
                )
                for i in range(int(len(cache_kvs_shapes) / 2))
            ]
            cache_v_dequant_scales = [
                paddle.static.InputSpec(
                    shape=[None, self.config.num_attention_heads],
                    dtype="float32",
                    name="v_dequant_scales_{}".format(i),
                )
                for i in range(int(len(cache_kvs_shapes) / 2))
            ]
        else:
            cache_k_quant_scales = None
            cache_v_quant_scales = None
            cache_k_dequant_scales = None
            cache_v_dequant_scales = None

        caches = []
        for i in range(len(cache_kvs_shapes) // 2):
            caches.append(
                paddle.static.InputSpec(
                    shape=cache_kvs_shapes[2 * i], dtype=cachekv_dtype, name="key_caches_{}".format(i)
                )
            )
            caches.append(
                paddle.static.InputSpec(
                    shape=cache_kvs_shapes[2 * i + 1], dtype=cachekv_dtype, name="value_caches_{}".format(i)
                )
            )
        if export_precache:
            src_mask_spec = paddle.static.InputSpec(shape=[None, 1, None, None], dtype=dtype, name="src_mask")
        else:
            src_mask_spec = None

        # bloom model needs src_mask and tgt_mask!
        if "bloom" in self.config.architectures[0].lower():
            src_mask_spec = paddle.static.InputSpec(shape=[None, None, None, None], dtype=dtype, name="src_mask")
            tgt_mask_spec = paddle.static.InputSpec(shape=[None, None, 1, None], dtype=dtype, name="tgt_mask")
        else:
            tgt_mask_spec = None

        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),  # input_ids
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="temperature"),  # temperature
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="top_p"),  # top_p
            paddle.static.InputSpec(shape=[None], dtype="int64", name="eos_token_id"),  # eos_token_id
            src_mask_spec,  # src_mask
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="penalty_score"),  # penalty_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="frequency_score"),  # frequency_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="presence_score"),  # presence_score
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="next_tokens"),  # next_tokens
            paddle.static.InputSpec(shape=[None, 1], dtype="bool", name="is_block_step"),  # is_block_step
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_lens_this_time"),  # seq_lens_this_time
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_lens_encoder"),  # seq_lens_encoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_lens_decoder"),  # seq_lens_decoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="step_idx"),  # step_idx
            paddle.static.InputSpec(shape=[None, 1], dtype="bool", name="stop_flags"),  # stop_flags
            paddle.static.InputSpec(
                shape=[2, None, self.config.max_seq_len, None, None], dtype="float32", name="rope_emb"
            ),  # rope_emb
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="min_length"),  # min_dec_len
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="max_length"),  # max_dec_len
            paddle.static.InputSpec(shape=[1, 1], dtype="int64", name="stop_nums"),  # stop_nums
            paddle.static.InputSpec(shape=[None], dtype="int64", name="bad_tokens"),  # bad_tokens
            paddle.static.InputSpec(shape=[1, 1], dtype="bool", name="not_need_stop"),  # not_need_stop
            paddle.static.InputSpec(shape=[None, None], dtype="int32", name="block_tables"),  # block_tables
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="pre_ids"),  # pre_ids
            precache_kv_spec,
            caches,  # cache_kvs
            cache_k_quant_scales,
            cache_v_quant_scales,
            cache_k_dequant_scales,
            cache_v_dequant_scales,
            tgt_mask_spec,
        ]
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
        temperature=None,
        top_p=None,
        eos_token_id=None,
        src_mask=None,
        penalty_score=None,
        frequency_score=None,
        presence_score=None,
        next_tokens=None,
        is_block_step=None,
        seq_lens_this_time=None,  # update
        seq_lens_encoder=None,  # update
        seq_lens_decoder=None,  # update
        step_idx=None,
        stop_flags=None,
        rope_emb=None,
        min_length=None,
        max_length=None,
        stop_nums=None,
        bad_tokens=None,
        not_need_stop=None,
        block_tables=None,
        pre_ids=None,
        pre_caches=None,
        cache_kvs=[],
        k_quant_scales=None,
        v_quant_scales=None,
        k_dequant_scales=None,
        v_dequant_scales=None,
        tgt_mask=None,
        **model_kwargs,
    ):

        model_kwargs["input_ids"] = input_ids
        model_kwargs["penalty_score"] = penalty_score
        model_kwargs["frequency_score"] = frequency_score
        model_kwargs["presence_score"] = presence_score
        model_kwargs["seq_lens_this_time"] = seq_lens_this_time
        model_kwargs["seq_lens_encoder"] = seq_lens_encoder
        model_kwargs["seq_lens_decoder"] = seq_lens_decoder
        model_kwargs["step_idx"] = step_idx
        model_kwargs["stop_flags"] = stop_flags
        model_kwargs["min_dec_len"] = min_length
        model_kwargs["max_dec_len"] = max_length
        model_kwargs["stop_nums"] = stop_nums
        model_kwargs["rope_emb"] = rope_emb
        model_kwargs["bad_tokens"] = bad_tokens
        model_kwargs["block_tables"] = block_tables
        model_kwargs["pre_ids"] = pre_ids
        model_kwargs["not_need_stop"] = not_need_stop
        model_kwargs["caches"] = cache_kvs
        model_kwargs["k_quant_scales"] = k_quant_scales
        model_kwargs["v_quant_scales"] = v_quant_scales
        model_kwargs["k_dequant_scales"] = k_dequant_scales
        model_kwargs["v_dequant_scales"] = v_dequant_scales
        model_kwargs["pre_caches"] = pre_caches
        model_kwargs["next_tokens"] = next_tokens
        model_kwargs["is_block_step"] = is_block_step
        model_kwargs["src_mask"] = src_mask
        model_kwargs["tgt_mask"] = tgt_mask

        ret = self.sample(
            eos_token_id,
            top_k=0,
            top_p=top_p,
            temperature=temperature,
            **model_kwargs,
        )
        return ret

    def sample(
        self,
        eos_token_id,
        top_k,
        top_p,
        penalty_score,
        frequency_score,
        presence_score,
        temperature=None,
        min_tokens_to_keep=1,
        **model_kwargs
    ):
        def _forward_(**args):
            model_inputs = self.prepare_inputs_for_generation(**args)
            return self(**model_inputs)

        def _post_process_(
            outputs,
            top_k,
            top_p,
            penalty_score,
            frequency_score,
            presence_score,
            temperature,
            model_kwargs,
        ):
            step_idx = model_kwargs["step_idx"]
            set_value_by_flags_and_idx_v2(
                model_kwargs["pre_ids"],
                model_kwargs["input_ids"],
                model_kwargs["seq_lens_this_time"],
                model_kwargs["seq_lens_encoder"],
                model_kwargs["seq_lens_decoder"],
                step_idx,
                model_kwargs["stop_flags"],
            )

            logits = paddle.cast(outputs, paddle.float32)

            # pre-process distribution
            logits = get_token_penalty_multi_scores_v2(
                model_kwargs["pre_ids"],
                logits,
                penalty_score,
                frequency_score,
                presence_score,
                temperature,
                model_kwargs["bad_tokens"],
                step_idx,
                model_kwargs["min_dec_len"],
                eos_token_id,
            )

            # sample
            probs = F.softmax(logits)
            # _, next_tokens = top_p_sampling(probs, top_p, -1)
            _, next_tokens = paddle.topk(probs, 1, -1)
            # print("-------next_tokens: ", next_tokens)
            if self.config.tensor_parallel_degree > 1:
                paddle.distributed.broadcast(next_tokens, 0)

            step_idx = paddle.where(model_kwargs["stop_flags"], model_kwargs["step_idx"], model_kwargs["step_idx"] + 1)
            paddle.assign(step_idx, model_kwargs["step_idx"])
            length_cond = paddle.greater_equal(model_kwargs["step_idx"], model_kwargs["max_dec_len"])
            stop_flags = paddle.logical_or(model_kwargs["stop_flags"], length_cond)
            set_stop_value_multi_ends_v2(
                next_tokens, stop_flags, model_kwargs["seq_lens_this_time"], eos_token_id, model_kwargs["next_tokens"]
            )  # multi ends
            paddle.assign(stop_flags, model_kwargs["stop_flags"])
            # update inputs
            update_inputs(
                model_kwargs["stop_flags"],
                model_kwargs["not_need_stop"],
                model_kwargs["seq_lens_this_time"],
                model_kwargs["seq_lens_encoder"],
                model_kwargs["seq_lens_decoder"],
                model_kwargs["input_ids"],
                model_kwargs["stop_nums"],
                next_tokens,
                model_kwargs["is_block_step"],
            )
            save_output(next_tokens, model_kwargs["not_need_stop"], self.config.tensor_parallel_rank)
            return next_tokens

        # encoder
        outputs = _forward_(**model_kwargs)  # [bs, 1, dim_embed]
        # first decoder
        next_tokens = _post_process_(
            outputs,
            top_k,
            top_p,
            penalty_score,
            frequency_score,
            presence_score,
            temperature,
            model_kwargs,
        )

        return next_tokens


def _speculative_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    last_assistant_token_is_eos,
    max_matches,
):
    """
    Applies sampling as in the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf, algorithm 1). Returns
    the selected tokens, as well as the number of candidate matches.

    NOTE: Unless otherwise stated, the variable names match those in the paper.
    """
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.
    q = F.softmax(candidate_logits, axis=-1)
    # q: [bsz, seq_len, vocab_size]
    q_i = q[:, paddle.arange(candidate_length), new_candidate_input_ids].squeeze()

    p = F.softmax(new_logits, axis=-1)
    # p: [bsz, gamma+1, vocab_size]
    p_i = p[:, paddle.arange(candidate_length), new_candidate_input_ids].squeeze()
    probability_ratio = p_i / q_i

    # When probability_ratio > 1 (i.e. q_i(x) < p_i(x), or "assistant probability of the candidate token is smaller
    # than the model probability for the same token"), keep the token. Otherwise reject with p = 1 - probability_ratio
    # (= keep with p = probability_ratio). Keep all the tokens until the first rejection
    r_i = paddle.rand(probability_ratio.shape)
    is_accepted = r_i <= probability_ratio
    n_matches = ((~is_accepted).astype("int64").cumsum(axis=-1) < 1).sum()  # this is `n` in algorithm 1
    print(f"-----本轮接收 {n_matches} 个token-----")
    # Ensure we don't generate beyond max_len or an EOS token (not in algorithm 1, but needed for correct behavior)
    if last_assistant_token_is_eos and n_matches == candidate_length:
        # Output length is assumed to be `n_matches + 1`. Since we won't generate another token with the target model
        # due to acceptance on EOS we fix `n_matches`
        n_matches -= 1
        valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
    else:
        n_matches = min(n_matches, max_matches)

        # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
        gamma = min(candidate_length, max_matches)
        p_n_plus_1 = p[:, n_matches, :]
        if n_matches < gamma:
            q_n_plus_1 = q[:, n_matches, :]
            p_prime = paddle.clip((p_n_plus_1 - q_n_plus_1), min=0)
            p_prime = paddle.divide(p_prime, p_prime.sum())
        else:
            p_prime = p_n_plus_1
        t = paddle.multinomial(p_prime, num_samples=1)

        # The selected tokens include the matches (if any) plus the next sampled tokens
        if n_matches > 0:
            valid_tokens = paddle.concat((new_candidate_input_ids[:, :n_matches], t), axis=-1)
        else:
            valid_tokens = t

    return valid_tokens, n_matches