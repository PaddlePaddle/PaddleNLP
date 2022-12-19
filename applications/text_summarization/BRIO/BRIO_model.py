# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from typing import Callable, Iterable, List, Optional

import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlenlp.transformers.pegasus4BRIO.modeling import PegasusScorer


def RankingLoss(score, summary_score=None, margin=0, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
    ones = paddle.ones_like(score)
    loss_func = paddle.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)
    # candidate loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.reshape([-1])
            neg_score = neg_score.contiguous().reshape([-1])
            ones = paddle.ones_like(pos_score)
            loss_func = paddle.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = paddle.expand_as(summary_score.unsqueeze(-1), score)
    neg_score = score
    pos_score = pos_score.reshape([-1])
    neg_score = neg_score.reshape([-1])
    ones = paddle.ones_like(pos_score)
    loss_func = paddle.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss


class BRIO(nn.Module):
    def __init__(self, mname, pad_token_id):
        super(BRIO, self).__init__()
        self.pegasus = PegasusScorer.from_pretrained(mname)
        self.pad_token_id = pad_token_id

    def forward(
        self, text_id, candidate_id, normalize=True, score_mode="base", length_penalty=1, require_gold=True, adding=0
    ):

        batch_size = text_id.size(0)

        input_mask = text_id != self.pad_token_id
        cand_mask = candidate_id != self.pad_token_id
        cand_mask[:, :, 0] = 1
        output = self.pegasus(
            input_ids=text_id,
            attention_mask=input_mask,
            decoder_input_ids=candidate_id,
            decoder_attention_mask=cand_mask,
            output_hidden_states=True,
        )

        output = output[0]  # [bz x cand_num, seq_len, word_dim]
        output = output.reshape([batch_size, -1, output.size(1), output.size(2)])  # [bz, cand_num, seq_len, word_dim]
        probs = output[:, 0]
        output = output[:, :, :-1]  # truncate last token
        candidate_id = candidate_id[:, :, 1:]  # shift right
        cand_mask = candidate_id != self.pad_token_id
        candidate_id = candidate_id.unsqueeze(-1)
        if normalize:
            if score_mode == "log":
                _output = F.log_softmax(output, axis=3)
            else:
                _output = F.softmax(output, axis=3)
            scores = paddle.gather(_output, candidate_id, 3).squeeze(-1)  # [bz, cand_num, seq_len]
        else:
            scores = paddle.gather(output, candidate_id, 3).squeeze(-1)  # [bz, cand_num, seq_len]
        cand_mask = cand_mask.float()
        scores = paddle.multiply(scores, cand_mask).sum(-1) / (
            (cand_mask.sum(-1) + adding) ** length_penalty
        )  # [bz, cand_num]
        if require_gold:
            output = {"score": scores[:, 1:], "summary_score": scores[:, 0], "probs": probs}
        else:
            output = {"score": scores, "probs": probs}
        return output

    def scoring_mode(self):
        self.pegasus.pegasus.scoring_mode()

    def generation_mode(self):
        self.pegasus.pegasus.generation_mode()

    def generate(
        self,
        input_ids: Optional[paddle.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, paddle.Tensor], List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ):
        return self.pegasus.generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            early_stopping=early_stopping,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            bad_words_ids=bad_words_ids,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            max_time=max_time,
            decoder_start_token_id=decoder_start_token_id,
            use_cache=use_cache,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            remove_invalid_values=remove_invalid_values,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )
