import paddle


import copy
import random
import paddle.distributed as dist
import paddle.nn.functional as F
import numpy as np
from paddlenlp.transformers import LlamaForCausalLM
from paddlenlp.transformers.model_outputs import ModelOutput
from paddlenlp.transformers.utils import get_scale_by_dtype


from paddlenlp.utils.log import logger

from paddlenlp.generation.configuration_utils import DEFAULT_MAX_NEW_TOKENS, GenerationConfig
from paddlenlp.generation.streamers import BaseStreamer
from paddlenlp.generation.utils import get_unfinished_flag



from paddlenlp.generation.logits_process import (
    LogitsProcessorList,
    TopKProcess,
    TopPProcess,
)
from paddlenlp.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from typing import Optional, Union

threshold = None
small_model = None
uncer_w1 = None
uncer_w2 = None
uncer_w3 = None
randm = None
extra = None

def set_para(thres, small_mod, w1, w2, w3, _randm: bool, ext):
    global threshold
    global small_model
    global uncer_w1
    global uncer_w2
    global uncer_w3
    global randm
    global extra
    threshold = thres
    small_model = small_mod
    uncer_w1 = w1
    uncer_w2 = w2
    uncer_w3 = w3
    randm = _randm
    extra = ext

lex_diver = None
rel_scores = None
def set_para_ano(_lex_diver, _scores):
    global lex_diver
    lex_diver = _lex_diver
    global rel_scores
    rel_scores = _scores



class Drag(LlamaForCausalLM):
    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(input_ids == pad_token_id).item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids == pad_token_id).astype(paddle.get_default_dtype()) * get_scale_by_dtype(
                return_positive=False
            )
        else:
            attention_mask = paddle.zeros_like(input_ids, dtype=paddle.get_default_dtype())
        return attention_mask

    def relative_top_filter(self, scores: paddle.Tensor, relative_top: float = 0.1, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1) -> paddle.Tensor:
        scores_normalized = F.log_softmax(scores, axis=-1)
        sorted_logits = paddle.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1]
        probs_max = paddle.max(scores_normalized, axis=-1)
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = paddle.minimum(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        scores_normalized[scores_normalized < probs_thresh] = filter_value
        return scores_normalized

    def sample_irr(
        self,
        input_ids,
        logits_processors,
        max_length,
        pad_token_id,
        eos_token_id,
        top_k=None,
        top_p=None,
        temperature=None,
        min_tokens_to_keep=1,
        stopping_criteria=None,
        streamer=None,
        fast_ptq_sampling=False,
        trunc_input=True,
        synced_gpus=False,
        input_ids_irr=None,
        attention_mask_irr=None,
        alpha = 10,
        relative_top = 0.1,
        question_token_len = None,
        question_and_doc_len = None,
        **model_kwargs
    ):
        # output_attentions = self.config.output_attentions
        # output_hidden_states = self.config.output_hidden_states
        output_attentions = True
        output_hidden_states = True
        model_kwargs["use_cache"] = model_kwargs.get("use_cache", True)

        model_kwargs_irr = copy.deepcopy(model_kwargs)
        model_kwargs_irr["attention_mask"] = attention_mask_irr
        model_kwargs_irr["use_cache"] = model_kwargs_irr.get("use_cache", True)

        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()

        # max_length will be convert to MaxLengthCriteria
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            # logger.warning(
            #    "`max_length` is deprecated in this function, use"
            #    " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead."
            # )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

        batch_size, cur_len = input_ids.shape
        origin_len = cur_len
        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        generate_end = False

        num_generation = 0
        num_decoding = 0
        decoding_list = []
        risk_scores = []

        def calculate_attention_uncertainty(next_new_token_attn, logits):
            """
            计算基于多头注意力和logits的token不确定性

            :param next_new_token_attn: Tensor of shape (1, 32, n), attention output
            :param logits: Tensor of shape (1, n, vocab_size), logits output for the token prediction
            :return: 不确定性度量的张量 (1, n)
            """
            # 1. 计算多头注意力的权重差异
            # 我们可以计算32个头的标准差或方差
            # 计算标准差, 为每个token在不同头之间的注意力权重计算差异
            # attention_variance = paddle.var(next_new_token_attn, dim=1)  # Variance across the 32 attention heads
            # uncertainty_1 = paddle.sqrt(attention_variance).sum()  # 可以选择使用标准差作为不确定性的度量

            # 2. 计算单个头的注意力权重分布
            # 我们通过计算每个token在每个头的权重分布的熵来衡量分散度
            # 熵较高表示该token的权重分布较为均匀，从而不确定性较高

            if randm:
                return random.random()

            def compute_entropy(attn):
                # 使用softmax来计算概率分布，然后计算熵
                prob_dist = F.softmax(attn, axis=-1)
                entropy = -(prob_dist * prob_dist.log()).sum(axis=-1)
                return entropy
            if uncer_w1 != 0:
                entropy_values = compute_entropy(next_new_token_attn.sum(1).squeeze(0))  # 计算每个token在不同头的熵
                uncertainty_2 = entropy_values  # 熵越大，表示不确定性越高
            else:
                uncertainty_2 = 0

            # 3. 输出token的概率（基于logits）
            # 使用softmax对logits进行归一化，计算每个token的生成概率
            # logits的形状为 (1, n, vocab_size)，我们需要对每个token的logits进行softmax
            if uncer_w2 != 0:
                softmax_probs = F.softmax(logits.squeeze(0), axis=-1)  # (n, vocab_size)

                # # 对每个token，选出最大概率对应的词汇的概率
                max_probs = softmax_probs.max(axis=-1)  # 获取每个token的最大生成概率
                uncertainty_3 = 1 - max_probs  # 概率越大，不确定性越小，取反即为不确定性
            else:
                uncertainty_3 = 0

            if uncer_w3 != 0:
                uncertainty_4 = compute_entropy(logits)
            else:
                uncertainty_4 = 0
            # 综合不确定性：可以选择加权平均或者简单地加总每部分的不确定性
            # total_uncertainty = uncertainty_2#*0.3 + uncertainty_3
            total_uncertainty = uncertainty_2 * uncer_w1 + uncertainty_3 * uncer_w2 + uncertainty_4 * uncer_w3

            return total_uncertainty

        def calc_risk(attn, logits):
            if randm:
                return random.random()
            ret = 0
            attn = attn.sum(1).squeeze(0)
            st = question_token_len
            for j in range(len(question_and_doc_len)):
                en = question_and_doc_len[j]
                assert en <= attn.shape[0], f"{j}---  en:{en} attn_len:{attn.shape[0]}"
                ret += attn[st:en].sum().item() / (1 + rel_scores[j])
                st = en

            softmax_probs = F.softmax(logits.squeeze(0), axis=-1)
            max_probs = softmax_probs.max(axis=-1)
            ret *= (1- max_probs.item())

            if lex_diver > 0:
                ret *= lex_diver
        # else:

            return ret

        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = paddle.to_tensor(0.0 if generate_end else 1.0)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # NOTE: to decrease ref-count and clear outdate cache in-time
            model_kwargs["cache"] = None
            model_kwargs["past_key_values"] = None
            #outputs = self(**model_inputs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            model_kwargs_irr["cache"] = None
            model_kwargs_irr["past_key_values"] = None
            model_inputs_irr = self.prepare_inputs_for_generation(input_ids_irr, **model_kwargs_irr)
            # NOTE: to decrease ref-count and clear outdate cache in-time
            model_kwargs_irr["cache"] = None
            model_kwargs_irr["past_key_values"] = None
            # outputs = self(**model_inputs)

            if synced_gpus and generate_end:
                continue  # don't waste resources running the code we don't need

            if isinstance(outputs, tuple):
                ori_logits = outputs[0]
            elif isinstance(outputs, ModelOutput):
                ori_logits = outputs.logits
            else:
                ori_logits = outputs

            # [batch_size, vocab_size]
            ori_logits = ori_logits[:, -1, :]

            if lex_diver is None:
                risk_score = calculate_attention_uncertainty(outputs['attentions'][-1][:,:,-1,:], logits) # TODO
            else:
                risk_score = calc_risk(outputs['attentions'][-1][:,:,-1,:], ori_logits)

            risk_scores.append(risk_score)
            if risk_score > threshold:
                irr_flag= True
                num_decoding += 1
                decoding_list.append(num_generation)
            else:
                irr_flag = False
            num_generation += 1

            if irr_flag:
                #outputs_irr = self(**model_inputs_irr)
                outputs_irr = self(
                    **model_inputs_irr,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

                if isinstance(outputs_irr, tuple):
                    logits_irr = outputs_irr[0]
                elif isinstance(outputs_irr, ModelOutput):
                    logits_irr = outputs_irr.logits
                else:
                    logits_irr = outputs_irr

                logits_irr = logits_irr[:, -1, :]

                if relative_top > 0.0:
                    ori_logits = self.relative_top_filter(ori_logits, relative_top)
                    logits_irr = F.log_softmax(logits_irr, axis=-1)
                    mask = ori_logits[0] < -1e3
                    logits_irr = paddle.where(mask, paddle.to_tensor(-1e3), logits_irr)
                    # logits_irr[0][mask] = -1e3
                else:
                    ori_logits = F.log_softmax(ori_logits, axis=-1)
                    logits_irr = F.log_softmax(ori_logits, axis=-1)

                logits = ori_logits + alpha * (ori_logits - logits_irr)

            else:
                logits = ori_logits


            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)

            # sample
            origin_probs = F.softmax(logits)
            origin_probs = paddle.log(origin_probs)
            if temperature is not None and temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits)
            if top_k is not None and top_k != 0:
                probs = TopKProcess(probs, top_k, min_tokens_to_keep)
            if top_p is not None and top_p < 1.0:
                probs = TopPProcess(probs, top_p, min_tokens_to_keep)
            if paddle.device.is_compiled_with_custom_device("gcu"):
                probs = paddle.cast(probs, "float32")
            if paddle.device.is_compiled_with_xpu():
                probs = paddle.cast(probs, "float32")

            # multinomial already support fp16 and bf16 currently, fix issue: https://github.com/PaddlePaddle/Paddle/issues/51852
            next_tokens = paddle.multinomial(probs)

            if self.config.tensor_parallel_degree > 1:
                # Maybe no need to broadcast if seed is set correclty.
                from paddle.distributed import fleet

                try:
                    hcg = fleet.get_hybrid_communicate_group()
                    group = hcg.get_model_parallel_group()
                    src = hcg.get_model_parallel_group_src_rank()
                except:
                    group, src = None, 0
                paddle.distributed.broadcast(next_tokens, src=src, group=group)
            # config does not include pipeline_parallel_degree, and pipeline parallel
            # uses trainer.model_wrapped to run in both train and predict mode
            # which has pp_group as a attribute
            # TODO(guosheng): only let the last stage of pipeline to do softmax
            # and sampling, and then broadcast to avoid broadcast logits.
            if getattr(self, "pp_group", None) is not None:
                paddle.distributed.broadcast(
                    next_tokens, src=self.pp_group.ranks[0], group=self.pp_group  # use rank 0 for same seed to check
                )

            next_scores = paddle.index_sample(origin_probs, next_tokens)
            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)

            cur_len += 1
            input_ids = paddle.concat([input_ids, next_tokens], axis=1)
            input_ids_irr = paddle.concat([input_ids_irr, next_tokens], axis=1)

            if streamer is not None:
                if self.config.tensor_parallel_rank == 0:
                    streamer.put(next_tokens.cpu())

            if stopping_criteria(input_ids, scores):
                generate_end = True

            if eos_token_id is not None:
                unfinished_flag = get_unfinished_flag(input_ids, unfinished_flag, eos_token_id)
                if not paddle.any(unfinished_flag):
                    generate_end = True

            # Stop when there is a </s> in all sentences
            if generate_end and not synced_gpus:
                break

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )
            outputs_copy = copy.deepcopy(outputs)
            model_kwargs_irr = self.update_model_kwargs_for_generation(
                outputs_copy, model_kwargs_irr, is_encoder_decoder=self.is_encoder_decoder
            )
            if fast_ptq_sampling:
                break

            if irr_flag:
                del outputs_irr

        if streamer is not None:
            streamer.end()

        return input_ids[:, origin_len:] if trunc_input else input_ids, scores

    @paddle.no_grad()
    def generate(
        self,
        input_ids: paddle.Tensor = None,
        generation_config: GenerationConfig = None,
        stopping_criteria: StoppingCriteria = None,
        streamer: BaseStreamer = None,
        synced_gpus: Optional[bool] = None,
        use_irr: Optional[str] = None,
        inputs_irr: Optional[paddle.Tensor] = None,
        alpha: Optional[float] = 10,
        question_token_len: Optional[int] = None,
        question_and_doc_len: list = None,
        **kwargs,
    ):
        if generation_config is None:
            if self.generation_config is None or self.generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    logger.warning(
                        "model.generation_config is in conflict with model.config, " "model.config is used."
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        # without update model.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)

        assert generation_config.decode_strategy in [
            "greedy_search",
            "sampling",
            "beam_search",
            "sampling_irr",
        ], "`decode_strategy` must be one of 'greedy_search', 'sampling' or 'beam_search' but received {}.".format(
            generation_config.decode_strategy
        )

        if getattr(self, "deprecated_warnings", None) is None:
            self.deprecated_warnings = {}

        use_fast = False
        if "use_faster" in model_kwargs:
            raise ValueError("`use_faster` is deprecated now.")

        if "use_fast" in model_kwargs:
            raise ValueError("`use_fast` is deprecated now.")

        bos_token_id = (
            generation_config.bos_token_id if generation_config.bos_token_id is not None else self.config.bos_token_id
        )
        eos_token_id = (
            generation_config.eos_token_id if generation_config.eos_token_id is not None else self.config.eos_token_id
        )
        pad_token_id = (
            generation_config.pad_token_id if generation_config.pad_token_id is not None else self.config.pad_token_id
        )
        forced_bos_token_id = (
            generation_config.forced_bos_token_id
            if generation_config.forced_bos_token_id is not None
            else self.config.forced_bos_token_id
        )
        forced_eos_token_id = (
            generation_config.forced_eos_token_id
            if generation_config.forced_eos_token_id is not None
            else self.config.forced_eos_token_id
        )
        decoder_start_token_id = (
            generation_config.decoder_start_token_id
            if generation_config.decoder_start_token_id is not None
            else self.config.decoder_start_token_id
        )
        no_repeat_ngram_size = (
            generation_config.no_repeat_ngram_size
            if generation_config.no_repeat_ngram_size is not None
            else self.config.no_repeat_ngram_size
        )

        if getattr(self, "_fast_entry", None) is not False and use_fast:
            fg_args = locals()
            fg_args.pop("self")
            fg_args.pop("__class__", None)
            model_kwargs = fg_args.pop("model_kwargs")
            fg_args.update(model_kwargs)
            try:
                if getattr(self, "_fast_entry", None) is None:
                    self._build_fast(fg_args)
                if self._fast_entry:
                    output = self._fast_entry(**fg_args)
                    if isinstance(output, tuple):
                        output_ids, dummy_srore = output
                    else:
                        output_ids = output
                        # make result and fast result oneconsistent
                        dummy_srore = None
                    if generation_config.decode_strategy == "beam_search":
                        output_ids = output_ids.transpose([1, 2, 0])
                        output_ids = output_ids[:, : generation_config.num_return_sequences, :].reshape(
                            [-1, output_ids.shape[-1]]
                        )
                        if dummy_srore is not None:
                            dummy_srore = dummy_srore[:, : generation_config.num_return_sequences].flatten()
                    else:
                        output_ids = output_ids.transpose([1, 0])
                    return output_ids, dummy_srore

            except Exception as e:
                fg_args["model_kwargs"] = model_kwargs
                # TODO
                # Prevent self._convert_to_fast to throw Exception
                self._convert_to_fast(fg_args)
                logger.warning(e)
                logger.warning("FastGeneration is not available, " "and the original version would be used instead.")

        # input_ids in model_kwargs is supported
        if "input_ids" in model_kwargs:
            _input_ids = model_kwargs.pop("input_ids")
            if input_ids is None:
                input_ids = _input_ids

        # params check
        if input_ids is None and "inputs_embeds" not in model_kwargs:
            # Init `input_ids` with bos_token_id
            input_ids = self.prepare_input_ids_for_generation(bos_token_id)
        elif "inputs_embeds" in model_kwargs:
            # Add input embeds support
            input_ids = self.prepare_input_ids_for_generation(
                bos_token_id, encoder_output=model_kwargs["inputs_embeds"]
            )

        if model_kwargs.get("attention_mask", None) is None:
            # TODO
            # Init `attention_mask` depending on `pad_token_id`
            model_kwargs["attention_mask"] = self.prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )
        self.is_encoder_decoder = self.config.is_encoder_decoder

        if self.is_encoder_decoder:
            model_kwargs = self.prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)
            # set input_ids as decoder_input_ids
            if "decoder_input_ids" in model_kwargs:
                input_ids = model_kwargs.pop("decoder_input_ids")
            else:
                input_ids = self.prepare_decoder_input_ids_for_generation(
                    input_ids, decoder_start_token_id, bos_token_id
                )
        # streamer
        if streamer is not None:
            # streamer couldn't support beam_search strategy
            if generation_config.decode_strategy == "beam_search" or generation_config.num_beams > 1:
                raise ValueError(
                    "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
                )

        pad_token_id = self.set_pad_token_id(pad_token_id, eos_token_id)

        if generation_config.max_length != 0 and generation_config.max_new_tokens == DEFAULT_MAX_NEW_TOKENS:
            logger.warning("`max_length` will be deprecated in future releases, use `max_new_tokens` instead.")
            generation_config.max_new_tokens = generation_config.max_length

        if generation_config.min_length != 0 and generation_config.min_new_tokens == 0:
            logger.warning("`min_length` will be deprecated in future releases, use `min_new_tokens` instead.")
            generation_config.min_new_tokens = generation_config.min_length

        max_length = generation_config.max_new_tokens
        min_length = generation_config.min_new_tokens

        input_len = input_ids.shape[-1]
        min_len = input_len + min_length
        max_len = input_len + max_length

        logits_processors = self.get_logits_processor(
            min_length=min_len if min_length > 0 else None,
            max_length=max_len,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            num_beams=generation_config.num_beams,
            num_beam_groups=generation_config.num_beam_groups,
            diversity_rate=generation_config.diversity_rate,
            repetition_penalty=generation_config.repetition_penalty,
            no_repeat_ngram_size=generation_config.no_repeat_ngram_size,
            logits_processors=model_kwargs["logits_processors"]
            if "logits_processors" in model_kwargs
            and isinstance(model_kwargs["logits_processors"], LogitsProcessorList)
            else None,
        )
        if "logits_processors" in model_kwargs:
            model_kwargs.pop("logits_processors")

        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if use_irr is not None:
            if self.config.is_encoder_decoder:
                logger.warning(
                    "Using irrelevant doc not implemented for encoder-decoder architecture yet."
                )

            input_ids_irr = inputs_irr
            attention_mask_irr = self.prepare_attention_mask_for_generation(
                input_ids_irr, generation_config.pad_token_id, generation_config.eos_token_id
                )


        if generation_config.decode_strategy == "sampling_irr":
            if generation_config.num_return_sequences > 1:
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=generation_config.num_return_sequences, **model_kwargs
                )

            if generation_config.num_return_sequences > 1:
                input_ids_irr, model_kwargs_irr = self.expand_inputs_for_generation(
                    input_ids, expand_size=generation_config.num_return_sequences, **model_kwargs
                )

            return self.sample_irr(
                input_ids,
                logits_processors,
                max_len,
                pad_token_id,
                eos_token_id,
                generation_config.top_k,
                generation_config.top_p,
                generation_config.temperature,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
                fast_ptq_sampling=generation_config.fast_ptq_sampling,
                trunc_input=generation_config.trunc_input,
                synced_gpus=synced_gpus,
                input_ids_irr=input_ids_irr,
                attention_mask_irr=attention_mask_irr,
                alpha=alpha,
                question_token_len=question_token_len,
                question_and_doc_len=question_and_doc_len,
                **model_kwargs,
            )
    def check(self):
        print("Make sure the model is a Drag class")
