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

from abc import ABC, abstractmethod
from collections import UserDict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
from paddle import Tensor

from paddlenlp.trainer import Trainer


class GLMTrainer(Trainer):
    def __init__(self, do_generation: bool, **kwargs):
        super().__init__(**kwargs)
        self.start_token = self.tokenizer.bos_token_id
        self.end_token = self.tokenizer.eos_token_id
        self.mask_token = self.tokenizer.mask_token_id
        self.pad_token = self.tokenizer.pad_token_id
        self.do_generation = do_generation
        self.processors = LogitsProcessorList()
        if self.args.min_tgt_length > 0:
            processor = MinLengthLogitsProcessor(self.args.min_tgt_length, self.end_token)
            self.processors.append(processor)
        if self.args.no_repeat_ngram_size > 0:
            processor = NoRepeatNGramLogitsProcessor(self.args.no_repeat_ngram_size)
            self.processors.append(processor)

    def compute_loss(
        self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]], return_outputs: bool = False
    ):
        if self.criterion is not None:
            if "labels" in inputs:
                labels = inputs.pop("labels")
            elif self.args.label_names is not None:
                labels = []
                for label in self.label_names:
                    labels.append(inputs.pop(label))
                labels = tuple(labels)
            elif "generator_labels" in inputs:
                labels = inputs["generator_labels"]
        else:
            labels = None
        if "loss_mask" in inputs and inputs["loss_mask"] is not None:
            loss_mask = inputs.pop("loss_mask").reshape([-1])
        else:
            loss_mask = None

        logits, _ = model(**inputs)

        if self.criterion is not None:
            loss = self.criterion(logits, labels)
            if self.args.label_smoothing > 0:
                smooth_loss = (-nn.functional.log_softmax(logits, axis=-1) / logits.shape[2]).sum(axis=-1)
                loss = (1 - self.args.label_smoothing) * loss + self.args.label_smoothing * smooth_loss
            if loss_mask is not None:
                loss = paddle.sum(loss.reshape([-1]) * loss_mask) / paddle.sum(loss_mask)
            outputs = (loss, logits)

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

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
            tokens = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            position_ids = inputs["position_ids"]
            batch_size = tokens.shape[0]
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=self.args.out_seq_length,
                num_beams=self.args.num_beams,
                length_penalty=self.args.length_penalty,
                do_early_stopping=False,
            )
            beam_scores = paddle.zeros([batch_size, self.args.num_beams], dtype="float")
            beam_scores[:, 1:] = -1e9
            beam_scores = beam_scores.reshape(
                [
                    batch_size * self.args.num_beams,
                ]
            )
            # Run the model forward.
            counter = 0
            while counter < self.args.tgt_length:
                if counter == 0:
                    next_token_logits, mems = model(tokens, position_ids, attention_mask, use_cache=True)
                    seq_length = next_token_logits.shape[1]
                    next_token_logits = next_token_logits[:, -1]
                    next_token_logits = next_token_logits.unsqueeze(1).tile([1, self.args.num_beams, 1])
                    next_token_logits = next_token_logits.reshape([batch_size * self.args.num_beams, -1])
                    mems = [
                        mem.unsqueeze(1)
                        .tile([1, self.args.num_beams, 1, 1])
                        .reshape([batch_size * self.args.num_beams, seq_length, -1])
                        for mem in mems
                    ]
                    position_ids = paddle.ones([batch_size, self.args.num_beams, 2, 1], dtype=tokens.dtype)
                    for i, text in enumerate(tokens.tolist()):
                        mask_pos = text.index(self.mask_token)
                        position_ids[i, :, 0] = mask_pos
                    position_ids = position_ids.reshape([batch_size * self.args.num_beams, 2, 1])
                    tokens = paddle.zeros([batch_size * self.args.num_beams, 0], dtype=tokens.dtype)
                else:
                    if not self.args.no_block_position:
                        position_ids[:, 1] = counter + 1
                    last_token = tokens[:, -1:]
                    cur_attention_mask = paddle.zeros([batch_size * self.args.num_beams], dtype=tokens.dtype)
                    next_token_logits, mems = model(last_token, position_ids, cur_attention_mask, mems, use_cache=True)
                    next_token_logits = next_token_logits[:, -1]
                next_token_logits = top_k_logits(next_token_logits, top_k=self.args.top_k, top_p=self.args.top_p)
                next_token_scores = nn.functional.log_softmax(next_token_logits, axis=-1)
                next_token_scores = self.processors(tokens, next_token_scores)
                next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
                vocab_size = next_token_scores.shape[-1]
                next_token_scores = next_token_scores.reshape([batch_size, self.args.num_beams * vocab_size])

                probs = nn.functional.softmax(next_token_scores, axis=-1)
                if self.args.select_topk:
                    _, next_tokens = paddle.topk(probs, k=2 * self.args.num_beams, axis=-1, largest=True)
                else:
                    next_tokens = paddle.multinomial(probs, num_samples=2 * self.args.num_beams)
                next_token_scores = paddle.take_along_axis(next_token_scores, next_tokens, axis=-1)
                next_token_scores = paddle.sort(next_token_scores, descending=True, axis=1)
                _indices = paddle.argsort(next_token_scores, descending=True, axis=1)
                next_tokens = paddle.take_along_axis(next_tokens, _indices, axis=-1)

                next_indices = next_tokens // vocab_size
                next_tokens = next_tokens % vocab_size
                # stateless
                beam_outputs = beam_scorer.process(
                    tokens,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    eos_token_id=self.end_token,
                    pad_token_id=self.pad_token,
                )
                beam_scores = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]
                beam_next_tokens = beam_next_tokens.unsqueeze(-1)
                print(tokens)
                print(beam_idx)
                print(beam_next_tokens)
                if tokens.shape[1] == 0:
                    tokens = beam_next_tokens
                else:
                    tokens = paddle.concat([tokens[beam_idx, :], beam_next_tokens], axis=-1)
                mems = [mem[beam_idx] for mem in mems] if mems else []
                if beam_scorer.is_done:
                    break
                counter += 1
            tokens, _, scores = beam_scorer.finalize(
                tokens,
                beam_scores,
                next_tokens,
                next_indices,
                eos_token_id=self.end_token,
                pad_token_id=self.pad_token,
            )
            all_preds = []
            for i, text in enumerate(tokens.tolist()):
                text = [token for token in text if token not in [self.end_token, self.pad_token]]
                text = self.tokenizer.convert_ids_to_tokens(text)
                all_preds.append(text)

            all_labels = []
            for label, mask in zip(input["labels"].numpy(), attention_mask.numpy()):
                label = self.tokenizer.decode(label[mask.astype("bool")], skip_special_tokens=True).strip()
                label = float(label.replace(" ", ""))
                all_labels.append(label)

        return (None, all_preds, all_labels)


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

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


class LogitsProcessor(ABC):
    """Abstract base class for all logit processors that can be applied during generation."""

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        """Paddle method for processing logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsProcessorList(list):
    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        for processor in self:
            scores = processor(input_ids, scores)
        return scores


class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    Enforcing a min-length by setting EOS probability to 0.
    """

    def __init__(self, min_length: int, eos_token_id: int):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a positive integer, but is {min_length}")

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            scores[:, self.eos_token_id] = -float("inf")
        return scores


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    Enforces no repetition of n-grams. See `Fairseq
    <https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345>`__.
    Args:
        ngram_size (:obj:`int`):
            All ngrams of size :obj:`ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = self._calc_banned_ngram_tokens(input_ids, num_batch_hypotheses, cur_len)

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores

    def _calc_banned_ngram_tokens(self, prev_input_ids: Tensor, num_hypos: int, cur_len: int) -> List[Iterable[int]]:
        """Copied from fairseq for no_repeat_ngram in beam_search"""
        if cur_len + 1 < self.ngram_size:
            # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            return [[] for _ in range(num_hypos)]
        generated_ngrams = [{} for _ in range(num_hypos)]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(*[gen_tokens[i:] for i in range(self.ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

        def _get_generated_ngrams(hypo_idx):
            # Before decoding the next token, prevent decoding of ngrams that have already appeared
            start_idx = cur_len + 1 - self.ngram_size
            ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
            return generated_ngrams[hypo_idx].get(ngram_idx, [])

        banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
        return banned_tokens


class BeamScorer(ABC):
    """
    Abstract base class for all beam scorers that are used for :meth:`~transformers.PretrainedModel.beam_search` and
    :meth:`~transformers.PretrainedModel.beam_sample`.
    """

    @abstractmethod
    def process(
        self, input_ids: Tensor, next_scores: Tensor, next_tokens: Tensor, next_indices: Tensor, **kwargs
    ) -> Tuple[Tensor]:
        raise NotImplementedError("This is an abstract method.")

    @abstractmethod
    def finalize(
        self, input_ids: Tensor, next_scores: Tensor, next_tokens: Tensor, next_indices: Tensor, **kwargs
    ) -> Tensor:
        raise NotImplementedError("This is an abstract method.")


class BeamSearchScorer(BeamScorer):
    r"""
    Implementing standard beam search decoding. Adapted in part from `Facebook's XLM beam search code
    <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.
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
        decoded: Tensor = paddle.zeros([batch_size * self.num_beam_hyps_to_keep], sent_max_len, dtype=input_ids.dtype)
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
