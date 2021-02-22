from typing import List
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from abc import ABC


class GenerationMixin(object):
    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id):
        if bos_token_id is None:
            raise ValueError("'bos_token_id' should be defined when no "
                             "input_ids' are provided.")
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
            batch_size, seq_len, _ = attention_mask.shape
            new_attention_mask = paddle.zeros(
                [batch_size, seq_len + 1, seq_len + 1], attention_mask.dtype)
            new_attention_mask[:, :seq_len, :seq_len] = attention_mask
            new_attention_mask[:, -1, :] = new_attention_mask[:, -2, :]
            new_attention_mask[:, -1, -1] = 1
            model_kwargs["attention_mask"] = new_attention_mask

        return model_kwargs

    def logits_preprocess(self, logits):
        """
        Implement in subclasses for custom behavior to preprocess logits in the
        generate method.
        """
        return logits

    @paddle.no_grad()
    def generate(self,
                 input_ids=None,
                 max_length=20,
                 min_length=0,
                 decode_strategy='greedy_search',
                 num_beams=1,
                 temperature=1.0,
                 top_k=0,
                 top_p=1.0,
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
                "Not support 'is_encoder_decoder = True' currently.")
        """

        if decode_strategy == 'greedy_search' and num_return_sequences > 1:
            raise ValueError("'num_return_sequences' has to be 1, but is {} "
                             "when doing greedy search.".format(
                                 num_return_sequences))
        if decode_strategy == 'beam_search' and num_return_sequences > num_beams:
            raise ValueError(
                "'num_return_sequences' has to be smaller or "
                "equal to 'num_beams'. But received 'num_return_sequences' is "
                "{}, 'num_beams' is {}".format(num_return_sequences, num_beams))

        # expand input_ids with `num_return_sequences` additional sequences per batch
        if num_return_sequences > 1:
            input_ids, model_kwargs = self.expand_inputs_for_generation(
                input_ids, expand_size=num_return_sequences, **model_kwargs)

        model_kwargs["use_cache"] = use_cache

        logits_processors = self.get_logits_processor(min_length, eos_token_id)

        batch_size = input_ids.shape[0]
        cur_len = 0
        pred_ids = None
        unfinished_flag = paddle.full([batch_size, 1], 1, 'int64')
        scores = paddle.full([batch_size, 1], 0.0, dtype='float32')

        while cur_len < max_length:
            #print('\n\ncur_len =', cur_len, ':')
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                              **model_kwargs)
            """
            for key in model_inputs:
                if key == 'cache':
                    print(key, model_inputs[key] is not None)
                else:
                    print(key, model_inputs[key])
            """
            outputs = self(**model_inputs)
            """
            if cur_len==0:
                print(len(outputs['cache']), outputs['cache'][0].k.shape, outputs['cache'][0].v.shape)
                print(outputs['cache'][0].k, outputs['cache'][0].v)
            """

            # [batch_size, vocab_size]
            logits = outputs['logits'][:, -1, :]
            #print('\nlogits:');print(logits)

            # pre-process distribution
            logits = self.logits_preprocess(logits)
            logits = logits_processors(input_ids, logits)

            if decode_strategy == 'greedy_search':
                next_token_ids, next_token_scores = self.greedy_search(logits)
            elif decode_strategy == 'sampling':
                next_token_ids, next_token_scores = self.sample(
                    logits, top_k, top_p, temperature)
            elif decode_strategy == 'beam_search':
                # TODO
                pass
            else:
                raise ValueError(
                    '"decode_strategy" must be one of '
                    '"greedy_search", "sampling" and "beam_search".')

            #print(next_token_ids);print(next_token_scores)
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

    def greedy_search(self, logits):
        probs = F.log_softmax(logits)
        next_token_ids = paddle.argmax(probs, axis=-1).unsqueeze(-1)
        next_token_scores = paddle.index_sample(probs, next_token_ids)
        return next_token_ids, next_token_scores

    def sample(self,
               logits,
               top_k=None,
               top_p=None,
               temperature=None,
               min_tokens_to_keep=1):
        origin_probs = F.log_softmax(logits)
        if temperature is not None and temperature != 1.0:
            logits = logits / temperature
        probs = F.softmax(logits)
        if top_k is not None and top_k != 0:
            probs = self.TopKProcess(probs, top_k, min_tokens_to_keep)
        if top_p is not None and top_p < 1.0:
            probs = self.TopPProcess(probs, top_p, min_tokens_to_keep)
        next_token_ids = paddle.multinomial(probs)
        next_token_scores = paddle.index_sample(origin_probs, next_token_ids)
        return next_token_ids, next_token_scores

    def TopKProcess(self, probs, top_k, min_tokens_to_keep):
        top_k = min(max(top_k, min_tokens_to_keep), probs.shape[-1])
        # Remove all tokens with a probability less than the last token of the top-k
        topk_probs, _ = paddle.topk(probs, k=top_k)
        probs = paddle.where(probs >= topk_probs[:, -1:], probs,
                             paddle.full_like(probs, 0.0))
        return probs

    def TopPProcess(self, probs, top_p, min_tokens_to_keep):
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
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :
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

    def beam_search(self, num_beams):
        pass


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
