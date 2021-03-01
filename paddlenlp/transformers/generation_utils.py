from typing import List
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from abc import ABC


class BeamHypotheses:
    def __init__(self, num_beams, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
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

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1]**self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being 
        generated can become better than the worst one in the heap, then we 
        are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


class BeamSearchScorer(object):
    """
    implementing standard beam search decoding.
    """

    def __init__(self,
                 batch_size,
                 max_length,
                 num_beams,
                 length_penalty=1.0,
                 do_early_stopping=False,
                 num_beam_hyps_to_keep=1,
                 num_beam_groups=1):
        self.max_length = max_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping)
            for _ in range(batch_size)
        ]
        self._done = paddle.to_tensor(
            [False for _ in range(batch_size)], dtype='bool')

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                "`num_beams` has to be an integer strictly greater than 1, but "
                "received {}. For `num_beams` == 1, one should make use of "
                "`greedy_search` instead.".format(num_beams))

        if not isinstance(num_beam_groups, int) or (
                num_beam_groups > num_beams) or (
                    num_beams % num_beam_groups != 0):
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than "
                "`num_beams` and `num_beams` has to be divisible by "
                "`num_beam_groups`, but received num_beam_groups={}, num_beams="
                "{}.".format(num_beam_groups, num_beams))

    @property
    def is_done(self):
        return paddle.all(self._done)

    def process(self,
                input_ids,
                next_scores,
                next_tokens,
                next_indices,
                pad_token_id=None,
                eos_token_id=None):
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        assert batch_size == (input_ids.shape[0] // self.group_size)

        next_beam_scores = paddle.zeros(
            [batch_size, self.group_size], dtype=next_scores.dtype)
        next_beam_tokens = paddle.zeros(
            [batch_size, self.group_size], dtype=next_tokens.dtype)
        next_beam_indices = paddle.zeros(
            [batch_size, self.group_size], dtype=next_indices.dtype)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                assert (
                    len(beam_hyp) >= self.num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(
                    self.num_beams)
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
            for beam_token_rank, (next_token, next_score,
                                  next_index) in enumerate(
                                      zip(next_tokens[batch_idx], next_scores[
                                          batch_idx], next_indices[batch_idx])):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (
                        next_token.numpy().item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(input_ids[batch_beam_idx].clone(),
                                 next_score.numpy().item())
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    "At most {} tokens in `next_tokens[batch_idx]` can be equal "
                    "to `eos_token_id: {}`. Make sure `next_tokens[batch_idx]` "
                    "are corrected.".format(self.group_size, eos_token_id))

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().numpy().item(), cur_len)

        return {
            "next_beam_scores": next_beam_scores.reshape([-1]),
            "next_beam_tokens": next_beam_tokens.reshape([-1]),
            "next_beam_indices": next_beam_indices.reshape([-1])
        }

    def finalize(self,
                 input_ids,
                 final_beam_scores,
                 final_beam_tokens,
                 final_beam_indices,
                 pad_token_id=None,
                 eos_token_id=None):
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)

        # select the best hypotheses
        sent_lengths = paddle.zeros(
            [batch_size * self.num_beam_hyps_to_keep], dtype=input_ids.dtype)
        best = []

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_score = sorted_hyps.pop()[0]
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)
                best.append([best_hyp, best_score])

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item() + 1, self.max_length)
        decoded = paddle.zeros(
            [batch_size * self.num_beam_hyps_to_keep, sent_max_len],
            dtype=input_ids.dtype)
        # shorter batches are padded if needed
        if sent_lengths.min().numpy().item() != sent_lengths.max().numpy().item(
        ):
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded[:, :] = pad_token_id
        decoded_score = paddle.zeros(
            [batch_size * self.num_beam_hyps_to_keep, 1])

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, (hypo, score) in enumerate(best):
            decoded[i, :sent_lengths[i]] = hypo
            decoded_score[i] = score
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
        return decoded, decoded_score


class GenerationMixin(object):
    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id):
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no "
                             "`input_ids` are provided.")
        return paddle.ones([1, 1]) * bos_token_id

    # TODO
    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id,
                                              eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and (
            pad_token_id in input_ids)
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id))
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            return paddle.cast(input_ids != pad_token_id, dtype='int64')
        return paddle.ones_like(input_ids)

    @staticmethod
    def get_logits_processor(min_length=None, eos_token_id=None):
        processors = LogitsProcessorList()
        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(
                MinLengthLogitsProcessor(min_length, eos_token_id))
        # TODO
        # Add more pre_processing for distribution

        return processors

    @staticmethod
    def expand_inputs_for_generation(input_ids,
                                     expand_size,
                                     attention_mask=None,
                                     **model_kwargs):
        index = paddle.tile(
            paddle.arange(input_ids.shape[0]).unsqueeze(-1),
            [1, expand_size]).reshape([-1])

        input_ids = paddle.index_select(input_ids, index)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = paddle.index_select(attention_mask,
                                                                 index)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.index_select(token_type_ids,
                                                                 index)

        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.index_select(position_ids,
                                                               index)

        return input_ids, model_kwargs

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs):
        # update cache
        if "cache" in outputs:
            model_kwargs["cache"] = outputs["cache"]

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], axis=-1)

        # update position_ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat(
                [position_ids, position_ids[:, -1].unsqueeze(-1) + 1], axis=-1)

        # update attention_mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            batch_size, nhead, seq_len, _ = attention_mask.shape
            new_attention_mask = paddle.zeros(
                [batch_size, nhead, seq_len + 1, seq_len + 1],
                attention_mask.dtype)
            new_attention_mask[:, :, :seq_len, :seq_len] = attention_mask
            new_attention_mask[:, :, -1, :] = new_attention_mask[:, :, -2, :]
            # TODO
            new_attention_mask[:, :, -1, -1] = 0
            model_kwargs["attention_mask"] = new_attention_mask

        return model_kwargs

    @staticmethod
    def reorder_cache(cache, beam_idx):
        """
        This function is used to guarantee the correct cache with the beam_idx 
        at every generation step.
        """
        for i in range(len(cache)):
            cache[i].k = paddle.index_select(cache[i].k, beam_idx)
            cache[i].v = paddle.index_select(cache[i].v, beam_idx)
        return cache

    def adjust_logits_during_generation(self, logits):
        """
        Implement in subclasses for custom behavior to adjust the logits in the
        generate method.
        """
        return logits

    @paddle.no_grad()
    def generate(self,
                 input_ids=None,
                 max_length=20,
                 min_length=0,
                 decode_strategy='greedy_search',
                 temperature=1.0,
                 top_k=0,
                 top_p=1.0,
                 num_beams=1,
                 length_penalty=1.0,
                 early_stopping=False,
                 bos_token_id=None,
                 eos_token_id=None,
                 pad_token_id=None,
                 num_return_sequences=1,
                 use_cache=True,
                 **model_kwargs):
        # params check
        bos_token_id = bos_token_id if bos_token_id is not None else getattr(
            self, 'bos_token_id', None)
        eos_token_id = eos_token_id if eos_token_id is not None else getattr(
            self, 'eos_token_id', None)
        pad_token_id = pad_token_id if pad_token_id is not None else getattr(
            self, 'pad_token_id', None)

        if input_ids is None:
            # Init `input_ids` with bos_token_id
            input_ids = self.prepare_input_ids_for_generation(bos_token_id)

        if model_kwargs.get("attention_mask", None) is None:
            # TODO
            # Init `attention_mask` depending on `pad_token_id`
            model_kwargs[
                "attention_mask"] = self.prepare_attention_mask_for_generation(
                    input_ids, pad_token_id, eos_token_id)

        if pad_token_id is None and eos_token_id is not None:
            print("Setting `pad_token_id` to `eos_token_id`:{} for "
                  "open-end generation.".format(eos_token_id))
            pad_token_id = eos_token_id
        """
        # TODO
        if is_encoder_decoder:
            # Add encoder_outputs to model_kwargs
            # Update input_ids
            raise ValueError(
                "Not support `is_encoder_decoder = True` currently.")
        """

        model_kwargs["use_cache"] = use_cache
        max_length += input_ids.shape[-1]

        logits_processors = self.get_logits_processor(min_length, eos_token_id)

        if decode_strategy == 'greedy_search':
            if num_return_sequences > 1:
                raise ValueError(
                    "`num_return_sequences` has to be 1, but is {} "
                    "when doing greedy search.".format(num_return_sequences))
            return self.greedy_search(input_ids, logits_processors, max_length,
                                      pad_token_id, eos_token_id,
                                      **model_kwargs)
        elif decode_strategy == 'sampling':
            if num_return_sequences > 1:
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_return_sequences, **model_kwargs)
            return self.sample(input_ids, logits_processors, max_length,
                               pad_token_id, eos_token_id, top_k, top_p,
                               temperature, **model_kwargs)
        elif decode_strategy == 'beam_search':
            batch_size = input_ids.shape[0]
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to "
                    "`num_beams`. But received `num_return_sequences` is {}, "
                    "`num_beams` is {}".format(num_return_sequences, num_beams))
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences)
            input_ids, model_kwargs = self.expand_inputs_for_generation(
                input_ids, expand_size=num_beams, **model_kwargs)
            return self.beam_search(input_ids, beam_scorer, logits_processors,
                                    max_length, pad_token_id, eos_token_id,
                                    **model_kwargs)
        else:
            raise ValueError(
                '`decode_strategy` must be one of "greedy_search", "sampling" '
                'and "beam_search".')

    def greedy_search(self, input_ids, logits_processors, max_length,
                      pad_token_id, eos_token_id, **model_kwargs):
        batch_size, cur_len = input_ids.shape
        pred_ids = None
        unfinished_flag = paddle.full([batch_size, 1], 1, dtype='int64')
        scores = paddle.full(
            [batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        while cur_len < max_length:
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                              **model_kwargs)
            outputs = self(**model_inputs)

            # [batch_size, vocab_size]
            logits = outputs['logits'][:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)

            # greedy
            probs = F.log_softmax(logits)
            next_token_ids = paddle.argmax(probs, axis=-1).unsqueeze(-1)
            next_token_scores = paddle.index_sample(probs, next_token_ids)

            next_token_ids = next_token_ids * unfinished_flag + pad_token_id * (
                1 - unfinished_flag)
            next_token_scores = next_token_scores * unfinished_flag + scores * (
                1 - unfinished_flag)
            scores = (scores * cur_len + next_token_scores) / (cur_len + 1)

            unfinished_flag = unfinished_flag * (next_token_ids != eos_token_id)

            cur_len += 1
            input_ids = paddle.concat([input_ids, next_token_ids], axis=1)
            if pred_ids is None:
                pred_ids = next_token_ids
            else:
                pred_ids = paddle.concat([pred_ids, next_token_ids], axis=1)

            # Stop when there is a </s> in all sentences
            if paddle.max(unfinished_flag) == 0:
                break

            model_kwargs = self.update_model_kwargs_for_generation(outputs,
                                                                   model_kwargs)
        return pred_ids, scores

    def sample(self,
               input_ids,
               logits_processors,
               max_length,
               pad_token_id,
               eos_token_id,
               top_k=None,
               top_p=None,
               temperature=None,
               min_tokens_to_keep=1,
               **model_kwargs):
        def TopKProcess(probs, top_k, min_tokens_to_keep):
            top_k = min(max(top_k, min_tokens_to_keep), probs.shape[-1])
            # Remove all tokens with a probability less than the last token of the top-k
            topk_probs, _ = paddle.topk(probs, k=top_k)
            probs = paddle.where(probs >= topk_probs[:, -1:], probs,
                                 paddle.full_like(probs, 0.0))
            return probs

        def TopPProcess(probs, top_p, min_tokens_to_keep):
            sorted_probs = paddle.sort(probs, descending=True)
            sorted_indices = paddle.argsort(probs, descending=True)
            cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)

            # Remove tokens with cumulative probs above the top_p, But keep at 
            # least min_tokens_to_keep tokens
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Set 'min_tokens_to_keep - 1' because the first token is kept
                sorted_indices_to_remove[:, :min_tokens_to_keep - 1] = 0
            # Keep the first token
            sorted_indices_to_remove = paddle.cast(
                sorted_indices_to_remove, dtype='int64')
            sorted_indices_to_remove[:,
                                     1:] = sorted_indices_to_remove[:, :
                                                                    -1].clone()
            sorted_indices_to_remove[:, 0] = 0

            # Scatter sorted tensors to original indexing
            sorted_indices = sorted_indices + paddle.arange(probs.shape[
                0]).unsqueeze(-1) * probs.shape[-1]
            condition = paddle.scatter(sorted_indices_to_remove.flatten(),
                                       sorted_indices.flatten(),
                                       sorted_indices_to_remove.flatten())
            condition = paddle.cast(condition, 'bool').reshape(probs.shape)
            probs = paddle.where(condition, paddle.full_like(probs, 0.0), probs)
            return probs

        batch_size, cur_len = input_ids.shape
        pred_ids = None
        unfinished_flag = paddle.full([batch_size, 1], 1, dtype='int64')
        scores = paddle.full(
            [batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        while cur_len < max_length:
            print('\n\ncur_len: ', cur_len)
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                              **model_kwargs)
            print('input_ids:', model_inputs['input_ids'].shape)
            print(model_inputs['token_type_ids'].shape)
            print(model_inputs['position_ids'].shape)
            print(model_inputs['attention_mask'].shape)
            outputs = self(**model_inputs)

            # [batch_size, vocab_size]
            logits = outputs['logits'][:, -1, :]
            print('logits:', logits)

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)

            # sample
            origin_probs = F.log_softmax(logits)
            if temperature is not None and temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits)
            if top_k is not None and top_k != 0:
                probs = TopKProcess(probs, top_k, min_tokens_to_keep)
            if top_p is not None and top_p < 1.0:
                probs = TopPProcess(probs, top_p, min_tokens_to_keep)
            next_token_ids = paddle.multinomial(probs)
            print('id(next_token_ids):', id(next_token_ids))
            next_token_scores = paddle.index_sample(origin_probs,
                                                    next_token_ids)

            print('next_token_ids: ', next_token_ids)
            # if eos_token_id is not None:
            next_token_ids = next_token_ids * unfinished_flag + pad_token_id * (
                1 - unfinished_flag)
            next_token_scores = next_token_scores * unfinished_flag + scores * (
                1 - unfinished_flag)
            unfinished_flag = unfinished_flag * (next_token_ids != eos_token_id)

            scores = (scores * cur_len + next_token_scores) / (cur_len + 1)

            print('next_token_ids: ', next_token_ids != eos_token_id)
            print('unfinished_flag: ', unfinished_flag)
            cur_len += 1
            input_ids = paddle.concat([input_ids, next_token_ids], axis=1)
            print('before: ', pred_ids)
            if pred_ids is None:
                pred_ids = next_token_ids
            else:
                pred_ids = paddle.concat([pred_ids, next_token_ids], axis=1)
            print('pred_ids: ', pred_ids)

            # Stop when there is a </s> in all sentences
            if paddle.max(unfinished_flag) == 0:
                break

            model_kwargs = self.update_model_kwargs_for_generation(outputs,
                                                                   model_kwargs)
        return pred_ids, scores

    def beam_search(self, input_ids, beam_scorer, logits_processors, max_length,
                    pad_token_id, eos_token_id, **model_kwargs):
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        origin_cur_len = cur_len

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {}, but received {}.".format(
            num_beams * batch_size, batch_beam_size)

        beam_scores = paddle.zeros(
            (batch_size, num_beams), dtype=paddle.get_default_dtype())
        beam_scores[:, 1:] = -1e9
        beam_scores = paddle.reshape(beam_scores, [-1])

        while cur_len < max_length:
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                              **model_kwargs)
            outputs = self(**model_inputs)

            # [batch_size, vocab_size]
            logits = outputs['logits'][:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)

            # beam search
            # [batch_size * num_beams, vocab_size]
            next_token_scores = F.log_softmax(logits)

            next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)
            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.reshape(
                [batch_size, num_beams * vocab_size])

            next_token_scores, next_tokens = paddle.topk(
                next_token_scores, 2 * num_beams, axis=1)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id, )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            cur_len += 1
            input_ids = paddle.concat(
                [
                    paddle.index_select(input_ids, beam_idx),
                    beam_next_tokens.unsqueeze(-1)
                ],
                axis=-1)

            if beam_scorer.is_done:
                break
            model_kwargs = self.update_model_kwargs_for_generation(outputs,
                                                                   model_kwargs)
            if model_kwargs["cache"] is not None:
                model_kwargs["cache"] = self.reorder_cache(
                    model_kwargs["cache"], beam_idx)

        pred_ids, scores = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id)
        return pred_ids[:, origin_cur_len:], scores


class LogitsProcessorList(List):
    def __call__(self, input_ids, logits):
        for processor in self:
            logits = processor(input_ids, logits)
        return logits


class LogitsProcessor(ABC):
    """
    Abstract base class for all logit processors that can be applied during 
    generation.
    """

    def __call__(self, input_ids, logits):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. "
            "Only classes inheriting this class can be called.")


class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    Enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (int): The minimum length of generation sequence.
        eos_token_id (int): The id of the `end-of-sequence` token.
    """

    def __init__(self, min_length, eos_token_id):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(
                "`min_length` should be a positive integer, but get {}".format(
                    min_length))

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(
                "`eos_token_id` should be a positive integer, but get {}".
                format(eos_token_id))

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, logits):
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            logits[:, self.eos_token_id] = -1e9
        return logits
