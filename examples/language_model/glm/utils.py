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

from collections import UserDict
from typing import Any, Dict, List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
from paddle import Tensor
from paddle.optimizer.lr import LambdaDecay

from paddlenlp.trainer import Trainer
from paddlenlp.transformers.generation_utils import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
)


class GLMTrainer(Trainer):
    def __init__(self, do_generation: bool, **kwargs):
        super().__init__(**kwargs)
        self.do_generation = do_generation

    def prediction_step(
        self,
        model: nn.Layer,
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:

        if not self.do_generation:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        model.eval()
        with paddle.no_grad():
            tokens = model.generate(
                input_ids=inputs["input_ids"],
                position_ids=inputs["position_ids"],
                attention_mask=inputs["attention_mask"],
                decode_strategy="sampling",
                top_k=1,
                repetition_penalty=2.0,
                bos_token_id=self.tokenizer.sop_token_id,
                eos_token_id=self.tokenizer.eop_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )[0]
            all_preds = []
            for pred_tokens in tokens:
                all_preds.append(pred_tokens[pred_tokens != self.tokenizer.pad_token_id].tolist())
            max_pred_length = max([len(x) for x in all_preds])
            for index, preds in enumerate(all_preds):
                all_preds[index] = preds + [-100] * (max_pred_length - len(preds))

            all_labels = []
            for label, mask in zip(inputs["labels"].numpy(), inputs["loss_mask"].numpy()):
                label = label[mask.astype("bool")]
                label = [x for x in label[label != self.tokenizer.pad_token_id]]
                all_labels.append(label)
            max_label_length = max([len(x) for x in all_labels])
            for index, labels in enumerate(all_labels):
                all_labels[index] = labels + [-100] * (max_label_length - len(labels))

        return (None, paddle.to_tensor(all_preds), paddle.to_tensor(all_labels))

    def create_scheduler(self, num_training_steps: int):
        num_warmup_steps = (
            self.args.warmup_steps if self.args.warmup_steps > 0 else self.args.warmup_ratio * num_training_steps
        )

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                decay_step_ratio = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
                return 1.0 - (1.0 - self.args.lr_decay_ratio) * decay_step_ratio

        if self.lr_scheduler is None:
            self.lr_scheduler = LambdaDecay(self.args.learning_rate, lr_lambda, last_epoch=-1)
        return self.lr_scheduler


@paddle.no_grad()
def generate(
    self,
    input_ids=None,
    position_ids=None,
    attention_mask=None,
    out_seq_length=768,
    tgt_length=256,
    min_tgt_length=5,
    num_beams=1,
    length_penalty=0.0,
    end_token_id=None,
    pad_token_id=None,
    mask_token_id=None,
    no_block_position=True,
    no_repeat_ngram_size=None,
    select_topk=None,
    top_k=0,
    top_p=0.0,
):

    processors = LogitsProcessorList()
    if min_tgt_length > 0:
        processor = MinLengthLogitsProcessor(min_tgt_length, end_token_id)
        processors.append(processor)
    if no_repeat_ngram_size > 0:
        processor = NoRepeatNGramLogitsProcessor(no_repeat_ngram_size)
        processors.append(processor)

    batch_size = input_ids.shape[0]
    beam_scorer = BeamSearchScorer(
        batch_size=batch_size,
        max_length=out_seq_length,
        num_beams=num_beams,
        length_penalty=length_penalty,
        do_early_stopping=False,
    )
    beam_scores = paddle.zeros([batch_size, num_beams], dtype="float32")
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.reshape([batch_size * num_beams])
    # Run the model forward.
    counter = 0
    while counter < tgt_length:
        if counter == 0:
            next_token_logits, mems = self(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                cache=None,
                # use_cache=True,
            )
            seq_length = next_token_logits.shape[1]
            next_token_logits = next_token_logits[:, -1]
            next_token_logits = next_token_logits.unsqueeze(1).tile([1, num_beams, 1])
            next_token_logits = next_token_logits.reshape([batch_size * num_beams, -1])
            mems = [
                mem.unsqueeze(1).tile([1, num_beams, 1, 1]).reshape([batch_size * num_beams, seq_length, -1])
                for mem in mems
            ]
            position_ids = paddle.ones([batch_size, num_beams, 2, 1], dtype=input_ids.dtype)
            for i, text in enumerate(input_ids.tolist()):
                mask_pos = text.index(mask_token_id)
                position_ids[i, :, 0] = mask_pos
            position_ids = position_ids.reshape([batch_size * num_beams, 2, 1])
            input_ids = paddle.zeros([batch_size * num_beams, 0], dtype=input_ids.dtype)
        else:
            if not no_block_position:
                position_ids[:, 1] = counter + 1
            last_token = input_ids[:, -1:]
            cur_attention_mask = paddle.zeros([batch_size * num_beams], dtype=input_ids.dtype)
            next_token_logits, mems = self(
                input_ids=last_token,
                position_ids=position_ids,
                attention_mask=cur_attention_mask,
                cache=mems,
                # use_cache=True,
            )
            next_token_logits = next_token_logits[:, -1]
        next_token_logits = top_k_logits(next_token_logits, top_k=top_k, top_p=top_p)
        next_token_scores = nn.functional.log_softmax(next_token_logits, axis=-1)
        next_token_scores = processors(input_ids, next_token_scores)
        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.reshape([batch_size, num_beams * vocab_size])

        probs = nn.functional.softmax(next_token_scores, axis=-1)
        if select_topk:
            _, next_tokens = paddle.topk(probs, k=2 * num_beams, axis=-1, largest=True)
            # _, next_tokens = paddle.topk(probs, k=num_beams, axis=-1, largest=True)
        else:
            next_tokens = paddle.multinomial(probs, num_samples=2 * num_beams)
        next_token_scores = paddle.take_along_axis(next_token_scores, next_tokens, axis=-1)
        next_token_scores = paddle.sort(next_token_scores, descending=True, axis=1)
        _indices = paddle.argsort(next_token_scores, descending=True, axis=1)
        next_tokens = paddle.take_along_axis(next_tokens, _indices, axis=-1)

        next_indices = next_tokens // vocab_size
        next_tokens = next_tokens % vocab_size
        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            eos_token_id=end_token_id,
            pad_token_id=pad_token_id,
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]
        beam_next_tokens = beam_next_tokens.unsqueeze(-1)
        if input_ids.shape[1] == 0:
            input_ids = beam_next_tokens
        else:
            input_ids = paddle.concat(
                [paddle.stack([input_ids[i, :] for i in beam_idx.tolist()], axis=0), beam_next_tokens], axis=-1
            )
        mems = [mem[beam_idx] for mem in mems] if mems else []
        if beam_scorer.is_done:
            break
        counter += 1
    tokens, _, scores = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        eos_token_id=end_token_id,
        pad_token_id=pad_token_id,
    )
    return tokens, scores


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    # This functiion is from https://github.com/THUDM/GLM/blob/main/generation_utils.py

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < paddle.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        logits = logits.reshape(logits.shape[1:2])
        sorted_logits = paddle.sort(logits, descending=True)
        sorted_indices = paddle.argsort(logits, descending=True)
        cumulative_probs = paddle.cumsum(nn.functional.softmax(sorted_logits, axis=-1), axis=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        # going back to 2D
        logits = logits.reshape([1, -1])

    return logits


class BeamSearchScorer(object):
    r"""
    Implementing standard beam search decoding.
    This class is from https://github.com/THUDM/GLM/blob/main/generation_utils.py.
    Args:
        batch_size (:obj:`int`):
            Batch Size of :obj:`input_ids` for which beam search decoding is run in parallel.
        max_length (:obj:`int`):
            The maximum length of the sequence to be generated.
        num_beams (:obj:`int`):
            Number of beams for beam search.
        length_penalty (:obj:`float`, `optional`, defaults to 1.0):
            Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
            model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
            sequences.
        do_early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
        num_beam_hyps_to_keep (:obj:`int`, `optional`, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            :meth:`~transformer.BeamSearchScorer.finalize`.
    """

    def __init__(
        self,
        batch_size: int,
        max_length: int,
        num_beams: int,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
    ):
        self.max_length = max_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                max_length=self.max_length,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            )
            for _ in range(batch_size)
        ]
        self._done = paddle.to_tensor([False for _ in range(batch_size)], dtype="bool")

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: Tensor,
        next_scores: Tensor,
        next_tokens: Tensor,
        next_indices: Tensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        mems=None,
    ) -> Tuple[Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        assert batch_size == (input_ids.shape[0] // self.num_beams)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        next_beam_scores = paddle.zeros([batch_size, self.num_beams], dtype=next_scores.dtype)
        next_beam_tokens = paddle.zeros([batch_size, self.num_beams], dtype=next_tokens.dtype)
        next_beam_indices = paddle.zeros([batch_size, self.num_beams], dtype=next_indices.dtype)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                assert (
                    len(beam_hyp) >= self.num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(self.num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.num_beams + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() in eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        mems=[mem[[next_index.item()]] for mem in mems] if mems else None,
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.num_beams:
                    break

            if beam_idx < self.num_beams:
                raise ValueError(
                    f"At most {self.num_beams} tokens in {next_tokens[batch_idx]} can be equal to "
                    f"`eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.reshape([-1]),
                "next_beam_tokens": next_beam_tokens.reshape([-1]),
                "next_beam_indices": next_beam_indices.reshape([-1]),
            }
        )

    def finalize(
        self,
        input_ids: Tensor,
        final_beam_scores: Tensor,
        final_beam_tokens: Tensor,
        final_beam_indices: Tensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        mems=None,
    ) -> Tuple[Tensor, List[Tensor]]:
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score, mems=[mem[[batch_beam_idx]] for mem in mems] if mems else None)

        # select the best hypotheses
        sent_lengths = paddle.zeros([batch_size * self.num_beam_hyps_to_keep], dtype=input_ids.dtype)
        best = []

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                score, best_hyp, mems = sorted_hyps.pop()
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)
                best.append((best_hyp, mems, score))

        # prepare for adding eos
        sent_max_len = sent_lengths.max().item()
        decoded: Tensor = paddle.zeros([batch_size * self.num_beam_hyps_to_keep, sent_max_len], dtype=input_ids.dtype)
        scores = paddle.zeros([batch_size * self.num_beam_hyps_to_keep], dtype=final_beam_scores.dtype)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        mems = []
        for i, (hypo, mem, score) in enumerate(best):
            scores[i] = score
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < sent_max_len:
                decoded[i, sent_lengths[i]] = eos_token_id
            mems.append(mem)
        mems = (
            [paddle.concat([mem[i] for mem in mems], axis=0) for i in range(len(mems[0]))]
            if mems and mems[0]
            else None
        )
        return decoded, mems, scores


class BeamHypotheses:
    # This class is from https://github.com/THUDM/GLM/blob/main/generation_utils.py.
    def __init__(self, num_beams: int, max_length: int, length_penalty: float, early_stopping: bool):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: Tensor, sum_logprobs: float, mems=None):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (max(hyp.shape[-1], 1) ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, mems))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
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
