"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
"""
import paddle
import paddle.nn as nn
from fastcore.all import patch_to

from paddlenlp.transformers import BlipForConditionalGeneration, BlipProcessor
from paddlenlp.transformers.generation_utils import BeamHypotheses


@patch_to(BeamHypotheses)
def add(self: BeamHypotheses, hyp: paddle.Tensor, sum_logprobs: float, origin_len: int = 0) -> None:
    """
    Add a new hypothesis to the list.
    """
    score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
    if len(self) < self.num_beams or score > self.worst_score:
        self.beams.append((score, hyp))
        if len(self) > self.num_beams:
            sorted_next_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
            del self.beams[sorted_next_scores[0][1]]
            self.worst_score = sorted_next_scores[1][0]
        else:
            self.worst_score = min(score, self.worst_score)


@patch_to(BeamHypotheses)
def is_done(self: BeamHypotheses, best_sum_logprobs: float, cur_len: int, origin_len: int = 0) -> bool:
    """
    If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
    one in the heap, then we are done with this sentence.
    """

    if len(self) < self.num_beams:
        return False
    elif self.early_stopping:
        return True
    else:
        cur_score = best_sum_logprobs / cur_len**self.length_penalty
        ret = self.worst_score >= cur_score
        return ret


class BLIP_Decoder(nn.Layer):
    def __init__(
        self,
        pretrained_model_name_or_path,
        prompt="a picture of ",
    ):
        super().__init__()
        self.text_decoder = BlipForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
        self.text_decoder.eval()
        self.processor = BlipProcessor.from_pretrained(pretrained_model_name_or_path)
        self.processor.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        self.processor.tokenizer.enc_token_id = self.processor.tokenizer.additional_special_tokens_ids[0]
        self.prompt = prompt
        self.prompt_length = len(self.processor.tokenizer(self.prompt).input_ids) - 1

    def generate(
        self,
        image,
        prompt=None,
        sample=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        if prompt is None:
            prompt = self.prompt
            prompt_length = self.prompt_length
        else:
            prompt_length = len(self.processor.tokenizer(prompt).input_ids) - 1
        if not paddle.is_tensor(image):
            model_kwargs = self.processor(images=image, return_tensors="pd")
        else:
            model_kwargs = {"pixel_values": image}
        prompt = [prompt] * model_kwargs["pixel_values"].shape[0]
        input_ids = self.processor.tokenizer(prompt, return_tensors="pd").input_ids

        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length - prompt_length,
                min_length=min_length,
                decode_strategy="sampling",
                top_p=top_p,
                num_return_sequences=1,
                repetition_penalty=repetition_penalty,
                **model_kwargs,
            )[0]
        else:
            if num_beams == 1:
                # greedy search
                outputs = self.text_decoder.generate(
                    input_ids=input_ids,
                    max_length=max_length - prompt_length,
                    min_length=min_length,
                    decode_strategy="greedy_search",
                    **model_kwargs,
                )[0]
            else:
                # beam search
                outputs = self.text_decoder.generate(
                    input_ids=input_ids,
                    max_length=max_length - prompt_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    decode_strategy="beam_search",
                    repetition_penalty=repetition_penalty,
                    length_penalty=1.0,  # note this is not
                    **model_kwargs,
                )[0]

        captions = []
        for output in outputs:
            captions.append(self.processor.decode(output, skip_special_tokens=True))
        return captions
