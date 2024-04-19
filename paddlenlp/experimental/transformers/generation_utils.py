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
    erase_cache_kv,
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
from paddlenlp.utils.log import logger

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
        do_sample: bool = False,
        logits_processor: Optional[LogitsProcessorList] = None,
        max_generate_length: int = 20,
        eos_token_id: int = None,
        gamma=5,
        r_probability=None,
        **model_kwargs,
    ):
        r"""
        Implementation of speculative decoding. Generates sequences of token ids for target model assisted by a smaller 
        assistant model. Assistant model should use the same vocab as target model.
        NOTE: Only support batch_size=1 for now.

        Parameters:
            input_ids (Tensor of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            target_model_inputs (dict):
                input kwargs for target model.
            assistant_model_inputs (dict):
                input kwargs for assistant model.
            assistant_model: 
                draft model for speculative decoding.
            do_sample (`bool`, *optional*, defaults to `False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            max_generate_length (`int`, *optional*):
                The max generate token number of assisted decoding.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            gamma (`int`, *optional*):
                The max token_num assistant model generate one stage.
            r_probability (`float`, *optional*):
                The parameter r in the paper: https://arxiv.org/pdf/2211.17192.pdf, which will be used to compare p_i(x) / q_i(x)
                with r to determine whether accept the generated token x.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            Generated token ids of assisted decoding.
        """
        if r_probability is not None:
            assert do_sample is True, "do_sample must be true when r_probability is not None"
            assert r_probability >= 0 and r_probability < 1, "r_probability must be in [0, 1)"

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

        target_model_inputs["gamma"] = gamma
        step_idx_ori = paddle.full(shape=[1], dtype="int64", fill_value=1)
        target_model_inputs["all_input_ids"] = input_ids
        assistant_model_inputs["all_input_ids"] = input_ids

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
                model_kwargs["seq_len_encoder"][:] = 0
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
            model_kwargs["seq_lens_this_time"][:] = 1
            return model_kwargs


        def update_target_model_kwargs_for_generation(cache, model_kwargs):
            if cache is None:
                model_kwargs["step_idx"] = paddle.where(
                    model_kwargs["seq_len_encoder"] == 0,
                    model_kwargs["step_idx"],
                    model_kwargs["step_idx"] + 1,
                )
                model_kwargs["seq_len_encoder"][:] = 0
            else:
                model_kwargs["step_idx"] = paddle.where(
                    model_kwargs["stop_flags"],
                    model_kwargs["step_idx"],
                    model_kwargs["step_idx"] + 1,
                )
            length_cond = paddle.greater_equal(model_kwargs["step_idx"], model_kwargs["max_dec_len"])
            model_kwargs["stop_flags"] = paddle.logical_or(model_kwargs["stop_flags"], length_cond)

            return model_kwargs

        def _forward_(model, input_ids, **kwargs):
            model_inputs = model.prepare_inputs_for_generation(input_ids, **kwargs)

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

            if is_target_model:
                model_kwargs = update_target_model_kwargs_for_generation(
                    cache, model_kwargs
            )
                model_kwargs["cache"] = 0
                return logits
            else:
                logits = paddle.cast(logits, paddle.float32)
                next_tokens = logits.argmax(axis=-1)
                model_kwargs = update_draft_model_kwargs_for_generation(
                    cache, just_decoder, next_tokens, eos_token_id, model_kwargs
                )
                model_kwargs["cache"] = 0
                next_tokens = model_kwargs["next_tokens"]
                return logits, next_tokens

        eos_token_id_tensor = paddle.to_tensor(eos_token_id) if eos_token_id is not None else None

        # candidate ids(include promt ids).
        candidate_input_ids = input_ids.clone()

        # total generated tokens after a whole peer(include prompt ids).
        whole_ids_before_assist = input_ids.clone()

        # total generated tokens after a whole peer(exclude prompt ids).
        total_generate_token_num = 0

        while True:
            # Assistant: main logic start

            assistant_model_total_logits_this_peer = None            
            # 👉 run the assistant model
            for i in range(int(gamma)):
                # 1.1. use the assistant model to obtain the next candidate logits
                draft_model_logits, next_tokens = _post_process_(
                    _forward_(assistant_model, input_ids, **assistant_model_inputs),
                    step_idx_ori,
                    False, # is_target_model
                    assistant_model_inputs,
                ) # assistant_model_logits: [bsz, vocab_size]

                step_idx_ori += 1

                if assistant_model_total_logits_this_peer is None:
                    assistant_model_total_logits_this_peer = draft_model_logits
                    # [bsz, vocab_size] -> [bsz, 1, vocab_size], add a seq_len dimension
                    assistant_model_total_logits_this_peer = assistant_model_total_logits_this_peer[:, None, :]
                else:
                    assistant_model_total_logits_this_peer = paddle.concat((assistant_model_total_logits_this_peer, draft_model_logits[:,None]), axis=1)

                assistant_model_total_logits_this_peer[:, -1, :] = logits_processor(
                    candidate_input_ids, assistant_model_total_logits_this_peer[:, -1, :]
                )
                if next_tokens.ndim == 1:
                    next_tokens = next_tokens[None, :]

                candidate_input_ids = paddle.concat((candidate_input_ids, next_tokens), axis=-1)
                candidate_logits = assistant_model_total_logits_this_peer

                # 1.2. stop assistant generation on EOS or acceed max_output_length.
                if eos_token_id_tensor is not None:
                    last_assistant_token_is_eos = paddle.equal_all(eos_token_id_tensor, next_tokens)
                    if last_assistant_token_is_eos:
                        break
                else:
                    last_assistant_token_is_eos = False

                cum_generated_tokens_this_peer = i + 1
                if total_generate_token_num + cum_generated_tokens_this_peer >= max_generate_length:
                    break

            candidate_length = candidate_input_ids.shape[1] - whole_ids_before_assist.shape[1]
            target_model_inputs["candidate_length"] = candidate_length

            # 2. Use the original model to obtain the next token logits given the candidate sequence. 
            # 2.1. Run a forward pass on the candidate sequence
            step_idx_ori = paddle.full(shape=[1], dtype="int64", fill_value=1)

            # Since the input of target model is the candidate sequence, we need to specify the relevant parameter maunually.
            if target_model_inputs.get("cache") is None:
                target_model_inputs["seq_len_encoder"][0] = candidate_input_ids.shape[-1]
                target_model_inputs["seq_len_decoder"] *= 0
                target_model_inputs["seq_lens_this_time"][0] = candidate_input_ids.shape[-1]
            else:
                target_model_inputs["seq_len_decoder"][:] =  candidate_input_ids.shape[-1] - candidate_length - 1
                target_model_inputs["seq_lens_this_time"][0] = candidate_length

            # 👉 run the target model
            target_model_outputs_logits = _post_process_(
                _forward_(
                    self, 
                    candidate_input_ids if target_model_inputs.get("cache") is None else candidate_input_ids[:, -candidate_length-1:-1], 
                    **target_model_inputs),
                step_idx_ori,
                True, # is_target_model
                target_model_inputs,
            )
            target_model_outputs_logits = target_model_outputs_logits[None, :]
            
            # 2.2. Process the new logits
            # in encoder phase, we need excludes the input prompt.
            if target_model_outputs_logits.shape[1] > candidate_length:
                new_logits = target_model_outputs_logits[:, -candidate_length-1:-1, :]
            else:
                new_logits = target_model_outputs_logits

            max_matches = max_generate_length - total_generate_token_num
            # 3. verify logic of speculative decoding.
            valid_tokens, n_matches = _speculative_decoding(
                candidate_input_ids,
                candidate_logits,
                candidate_length,
                new_logits,
                do_sample,
                last_assistant_token_is_eos,
                max_matches,
                r_probability=r_probability,
            )

            total_generate_token_num += int(valid_tokens.shape[-1])

            # 4. Update variables according to the number of matching assistant tokens. Remember: the token generated
            # by the model after the last candidate match is also valid, as it is generated from a correct sequence.
            # Because of this last token, assisted generation search reduces to a normal greedy search/sample if there
            # is no match.

            # 4.1. Get the valid continuation, after the matching tokens
            input_ids = valid_tokens[:, -1]
            if n_matches != candidate_length:
                candidate_input_ids = candidate_input_ids[:, :-candidate_length]
                candidate_input_ids = paddle.concat((candidate_input_ids, valid_tokens), axis=-1)

            # update the parameter
            whole_ids_before_assist = paddle.concat((whole_ids_before_assist, valid_tokens), axis=-1)
            cur_len = whole_ids_before_assist.shape[-1]

            # when we refuse some tokens in the assistant model, so we need to erase the corresponding cache.
            if n_matches != candidate_length:
                if assistant_model_inputs.get("cache", None) is not None:
                    assistant_model_inputs["seq_len_decoder"][0] = cur_len - 1
                    assistant_model_inputs["tgt_pos"][0] = cur_len - 1
                    assistant_model_inputs["tgt_ids"][0] = input_ids

                assistant_model_cache_kvs = assistant_model_inputs.get("cache_kvs", None)
                erase_token_num_in_cache = int(candidate_length - n_matches - 1)

                for i in range(len(assistant_model_cache_kvs)):
                        erase_cache_kv(assistant_model_cache_kvs[i], assistant_model_inputs["seq_len_decoder"][0], erase_token_num_in_cache)

            # stop if we exceed the maximum length or target model has generated an EOS token
            if total_generate_token_num >= max_generate_length or valid_tokens[:, -1].item() == eos_token_id.item():
                break

        return whole_ids_before_assist[:, -total_generate_token_num:]

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


def _speculative_decoding(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    do_sample,
    last_assistant_token_is_eos,
    max_matches,
    r_probability=None,
):
    """
    If do_sample = True, apply sampling method in the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf). 
    Returns the selected tokens, as well as the number of candidate matches.
    If do_sample = False, apply the greedy search.
    """
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    gamma = min(candidate_length, max_matches)

    if do_sample:
        # Gets the probabilities from the logits. q_i and p_i denote the assistant model and target model probabilities of the tokens
        # selected by the assistant model, respectively.
        q = F.softmax(candidate_logits, axis=-1)
        # q: [bsz, gamma, vocab_size]
        q_i = q[:, paddle.arange(candidate_length), new_candidate_input_ids].squeeze()

        p = F.softmax(new_logits, axis=-1)
        # p: [bsz, gamma, vocab_size]
        p_i = p[:, paddle.arange(candidate_length), new_candidate_input_ids].squeeze()
        probability_ratio = p_i / q_i

        # When probability_ratio > 1 (i.e. q_i(x) < p_i(x), or "assistant probability of the candidate token is smaller
        # than the model probability for the same token"), keep the token. Otherwise reject with p = 1 - probability_ratio
        # (= keep with p = probability_ratio). Keep all the tokens until the first rejection
        if probability_ratio is not None:
            r_i = paddle.full_like(probability_ratio, r_probability)
        else:
            r_i = paddle.rand(probability_ratio.shape)
        is_accepted = r_i <= probability_ratio
        n_matches = ((~is_accepted).astype("int64").cumsum(axis=-1) < 1).sum()  # this is `n` in algorithm 1

        if last_assistant_token_is_eos and n_matches == candidate_length:
            valid_tokens = new_candidate_input_ids[:, : n_matches]
            return valid_tokens, n_matches
        else:
            n_matches = min(n_matches, max_matches)

        # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
        if n_matches < gamma:
            p_n_plus_1 = p[:, n_matches, :]
            q_n_plus_1 = q[:, n_matches, :]
            p_prime = paddle.clip((p_n_plus_1 - q_n_plus_1), min=0)
            p_prime = paddle.divide(p_prime, p_prime.sum())
            t = paddle.multinomial(p_prime, num_samples=1)
            valid_tokens = paddle.concat((new_candidate_input_ids[:, :n_matches], t), axis=-1)
        else:
            # NOTE: because we don't have the cache of the last token, so we don't sample an extra token from p_prime when n_matches=gamma
            t = new_candidate_input_ids[:, :n_matches]
            valid_tokens = t
    else:
        target_model_output_token = new_logits.argmax(-1)
        n_matches = ((~(new_candidate_input_ids == target_model_output_token)).astype("int64").cumsum(axis=-1) < 1).sum()
        n_matches = min(n_matches, max_matches)
        if n_matches < candidate_length:
            valid_tokens = target_model_output_token[:, :n_matches+1]
        else:
            valid_tokens = target_model_output_token[:, :n_matches]
    return valid_tokens, n_matches