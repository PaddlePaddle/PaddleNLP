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

import copy
import logging
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import paddle
import paddle.nn.functional as F

from paddlenlp.generation.configuration_utils import GenerationConfig
from paddlenlp.generation.logits_process import LogitsProcessorList
from paddlenlp.generation.stopping_criteria import StoppingCriteriaList
from paddlenlp.transformers import (
    ChatGLMConfig,
    ChatGLMForConditionalGeneration,
    ChatGLMModel,
    ChatGLMTokenizer,
)
from paddlenlp.transformers.model_outputs import ModelOutput

logger = logging.getLogger(__name__)


def get_unfinished_flag(
    input_ids: paddle.Tensor, unfinished_flag: paddle.Tensor, eos_token_id: Union[int, list[int], list[list[int]]]
) -> paddle.Tensor:
    """get unfinished flag for generation step

    Args:
        input_ids (Tensor): the input_ids
        eos_token_id (Union[int, list[int], list[list[int]]]): the end os sentence flag, which can be:
            * single token id, eg: 10
            * multiple token ids to stop generation, eg: [10, 10]
            * some more tokens to stop generations, eg: [[10], [20, 20], [30, 30, 30]]

    Returns:
        Tensor: the unfinished flag tensor
    """
    if isinstance(eos_token_id, int):
        unfinished_flag = paddle.logical_and(unfinished_flag, input_ids[:, -1:] != eos_token_id)
    elif isinstance(eos_token_id[0], int):
        eos_token_id_tensor = paddle.to_tensor([eos_token_id])
        is_last_tokens_equal = paddle.all(
            paddle.equal(input_ids[:, -len(eos_token_id) :], eos_token_id_tensor), axis=-1
        ).unsqueeze(-1)
        unfinished_flag = paddle.logical_and(unfinished_flag, ~is_last_tokens_equal)
    else:
        batch_unfinish_flag = None
        for batch_eos_token_id in eos_token_id:
            if batch_unfinish_flag is None:
                batch_unfinish_flag = ~get_unfinished_flag(input_ids, unfinished_flag, batch_eos_token_id)
            else:
                batch_unfinish_flag = paddle.logical_or(
                    batch_unfinish_flag, ~get_unfinished_flag(input_ids, unfinished_flag, batch_eos_token_id)
                )

        unfinished_flag = ~batch_unfinish_flag
    return unfinished_flag


class MyChatGLMForConditionalGeneration(ChatGLMForConditionalGeneration):
    def __init__(self, config: ChatGLMConfig):
        super(MyChatGLMForConditionalGeneration, self).__init__(config)

        self.config = config
        self.max_sequence_length = config.max_sequence_length
        self.position_encoding_2d = config.position_encoding_2d
        self.chatglm = ChatGLMModel(config)

        self.lm_head = self.chatglm.get_input_embeddings()
        self.generation_config = GenerationConfig.from_model_config(config)

    @paddle.no_grad()
    def chat(
        self,
        tokenizer,
        query: str,
        history: List[Tuple[str, str]] = None,
        max_length: int = 2048,
        num_beams=1,
        do_sample=True,
        top_p=0.7,
        temperature=0.95,
        logits_processor=None,
        **kwargs
    ):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        # logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "logits_processor": logits_processor,
            **kwargs,
        }
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        inputs = tokenizer([prompt], return_tensors="pd")
        outputs, scores = self.generate(**inputs, **gen_kwargs)
        # breakpoint()
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]) :]
        response = tokenizer.decode(outputs)
        response = self.process_response(response)
        history = history + [(query, response)]
        return response, history

    @paddle.no_grad()
    def stream_chat(
        self,
        tokenizer,
        query: str,
        history: List[Tuple[str, str]] = None,
        max_length: int = 2048,
        do_sample=True,
        top_p=0.7,
        temperature=0.95,
        logits_processor=None,
        **kwargs
    ):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        # logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {
            "max_length": max_length,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "logits_processor": logits_processor,
            **kwargs,
        }
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        inputs = tokenizer([prompt], return_tensors="pd")
        for outputs in self.stream_generate(**inputs, **gen_kwargs):
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]) :]
            response = tokenizer.decode(outputs)
            response = self.process_response(response)
            new_history = history + [(query, response)]
            yield response, new_history

    @paddle.no_grad()
    def stream_generate(
        self,
        input_ids,
        generation_config: Optional[Dict] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, paddle.Tensor], List[int]]] = None,
        **kwargs,
    ):
        batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        eos_token_id = generation_config.eos_token_id

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
            if not has_default_max_length:
                logger.warn(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                    UserWarning,
                )

        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()

        unfinished_flag = paddle.full([input_ids.shape[0], 1], True, dtype="bool")

        batch_size, cur_len = input_ids.shape
        origin_len = cur_len
        pad_token_id = generation_config.pad_token_id
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())
        while True:
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, ModelOutput):
                logits = outputs.logits
            else:
                logits = outputs

            # [batch_size, vocab_size]
            next_token_logits = logits[:, -1, :]

            # pre-process distribution
            next_token_logits = self.adjust_logits_during_generation(next_token_logits)
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # record score before warper
            origin_probs = F.softmax(next_tokens_scores)
            origin_probs = paddle.log(origin_probs)

            next_tokens_scores = logits_warper(input_ids, next_tokens_scores)

            # greedy
            probs = F.softmax(next_tokens_scores)

            # Sampling
            if generation_config.do_sample:
                next_tokens = paddle.multinomial(probs, num_samples=1)
            # Greedy
            else:
                probs = paddle.log(probs)
                next_tokens = paddle.argmax(probs, axis=-1).unsqueeze(-1)
            next_scores = paddle.index_sample(origin_probs.astype("float32"), next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)

            cur_len += 1

            input_ids = paddle.concat([input_ids, next_tokens], axis=1)

            if eos_token_id is not None:
                unfinished_flag = get_unfinished_flag(input_ids, unfinished_flag, eos_token_id)
                unfinished_flag = get_unfinished_flag(input_ids, unfinished_flag, eos_token_id)

            # Stop when there is a </s> in all sentences
            if not paddle.any(unfinished_flag) or stopping_criteria(input_ids, scores):
                break

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            yield input_ids


def run_paddle():
    tokenizer = ChatGLMTokenizer.from_pretrained("THUDM/chatglm-6b")
    model = MyChatGLMForConditionalGeneration.from_pretrained(
        "THUDM/chatglm-6b",  # "/root/paddlejob/workspace/GLM/ChatGLM-6B/",
        load_state_as_np=True,
        low_cpu_mem_usage=True,
        dtype="float16",
    )
    model.eval()
    # normal
    # outputs = model.generate(**inputs, max_length=2048)
    # no streaming
    # outputs = model.chat(tokenizer, query="您好")
    # print("results:", outputs)
    # streaming
    outputs = model.stream_chat(tokenizer, query="你好")
    for texts in outputs:
        print(texts)


if __name__ == "__main__":
    run_paddle()
