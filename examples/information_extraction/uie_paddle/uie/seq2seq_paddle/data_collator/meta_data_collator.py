#!/usr/bin/env python
# -*- coding:utf-8 -*-
import random
import copy
import math
import numpy as np
from typing import Optional
from collections import OrderedDict
import logging

import paddle
from paddlenlp.data import Pad

from uie.extraction.constants import (
    BaseStructureMarker,
    text_start,
    spot_prompt,
    asoc_prompt, )
from uie.extraction.noiser.spot_asoc_noiser import SpotAsocNoiser
from uie.extraction.record_schema import RecordSchema
from uie.extraction.utils import convert_to_record_function

logger = logging.getLogger("__main__")


class DynamicSSIGenerator():
    """
    根据负例采样动态生成 Prompt
    """

    def __init__(self,
                 tokenizer,
                 schema: RecordSchema,
                 positive_rate=1,
                 negative=-1,
                 ordered_prompt=False) -> None:
        self.spot_dict = self.get_ordered_dict(schema.type_list, tokenizer)
        self.asoc_dict = self.get_ordered_dict(schema.role_list, tokenizer)
        self.spot_list = list(self.spot_dict.keys())
        self.asoc_list = list(self.asoc_dict.keys())
        # Tokenizer of PaddleNLP don't have get_vocab()
        self.spot_prompt = self.get_vocab(tokenizer, spot_prompt)
        self.asoc_prompt = self.get_vocab(tokenizer, asoc_prompt)
        self.text_start = self.get_vocab(tokenizer, text_start)
        # 设置正例比例，为1则不采样。
        # 若采样需要将为采样到的 labels 中对应record删去
        self.positive_rate = positive_rate if positive_rate > 0 and positive_rate < 1 else 1
        self.negative = negative
        self.ordered_prompt = ordered_prompt
        logger.info(f"Meta Sample "
                    f"Negative: {self.negative}, "
                    f"Ordered Prompt: {self.ordered_prompt}")

    @staticmethod
    def get_vocab(tokenizer, token):
        token_encoded = tokenizer.encode(
            token,
            return_token_type_ids=False,
            return_attention_mask=False, )["input_ids"]
        assert len(token_encoded) == 2
        return token_encoded[0]

    @staticmethod
    def get_ordered_dict(schema_name_list, tokenizer):
        schema_ordered_dict = OrderedDict()
        # PaddleNLP has no remove_special_tokens param
        for name in schema_name_list:
            encoded_name = tokenizer.encode(name)
            schema_ordered_dict[name] = encoded_name["input_ids"][:-1:]
        return schema_ordered_dict

    @staticmethod
    def sample_negative(postive, candidates, k=5):
        if k < 0:
            k = len(candidates)
        negative_set = set()
        for index in np.random.permutation(len(candidates))[:k].tolist():
            negative = candidates[index]
            if negative not in postive:
                negative_set.add(negative)

        return list(negative_set)

    def sample_spot(self, positive):
        """ Sample spot

        Args:
            positive (List[str]): Positive Spot List

        Returns:
            List[int]: spot index list
            List[str]: Sampled Positive Spot List
            List[str]: Sampled Negative Spot List
        """
        negative_spot = self.sample_negative(
            postive=positive, candidates=self.spot_list, k=self.negative)
        positive_spot = random.sample(
            positive, math.floor(len(positive) * self.positive_rate))

        # Prefix 中的正负例 Spot
        prefix_spot_candidates = positive_spot + negative_spot
        converted_spot_prefix = self.convert_prefix(
            candidates=prefix_spot_candidates,
            prompt=self.spot_prompt,
            mapper=self.spot_dict,
            ordered_prompt=self.ordered_prompt, )

        return converted_spot_prefix, positive_spot, negative_spot

    def sample_asoc(self, positive):
        """ Sample Asoc

        Args:
            positive (List[str]): Positive Asoc List

        Returns:
            List[int]: asoc index list
            List[str]: Sampled Negative Asoc List
        """
        negative_asoc = self.sample_negative(
            postive=positive, candidates=self.asoc_list, k=self.negative)
        prefix_asoc_candidates = positive + negative_asoc
        converted_asoc_prefix = self.convert_prefix(
            candidates=prefix_asoc_candidates,
            prompt=self.asoc_prompt,
            mapper=self.asoc_dict,
            ordered_prompt=self.ordered_prompt, )
        return converted_asoc_prefix, negative_asoc

    def full_spot(self, shuffle=False):
        # Random Prompt + Shuffle
        if not self.ordered_prompt and shuffle:
            ordered_prompt = False
        else:
            ordered_prompt = True
        return self.convert_prefix(
            candidates=self.spot_list,
            prompt=self.spot_prompt,
            mapper=self.spot_dict,
            ordered_prompt=ordered_prompt, )

    def full_asoc(self, shuffle=False):
        # Random Prompt + Shuffle
        if not self.ordered_prompt and shuffle:
            ordered_prompt = False
        else:
            ordered_prompt = True
        return self.convert_prefix(
            candidates=self.asoc_list,
            prompt=self.asoc_prompt,
            mapper=self.asoc_dict,
            ordered_prompt=ordered_prompt, )

    @staticmethod
    def convert_prefix(candidates, prompt, mapper, ordered_prompt=True):
        prefix = list()

        if ordered_prompt:
            candidate_sorted = sorted(
                [(candidate, index)
                 for index, candidate in enumerate(candidates)])
            index_list = [index for _, index in candidate_sorted]
        else:
            index_list = np.random.permutation(len(candidates)).tolist()

        for index in index_list:
            prefix += [prompt]
            prefix += mapper[candidates[index]]
        return prefix


class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    def __init__(self,
                 tokenizer,
                 negative_sampler: DynamicSSIGenerator,
                 model=None,
                 label_pad_token_id=-100,
                 padding=True,
                 max_source_length: Optional[int]=None,
                 max_target_length: Optional[int]=None,
                 max_prefix_length: Optional[int]=None,
                 pad_to_multiple_of: Optional[int]=None,
                 spot_asoc_nosier: SpotAsocNoiser=None,
                 decoding_format: str='spotasoc',
                 return_tensors=True):

        self.tokenizer = tokenizer
        self.negative_sampler = negative_sampler
        self.model = model
        self.label_pad_token_id = label_pad_token_id
        self.padding = padding
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_prefix_length = max_prefix_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.spot_asoc_nosier = spot_asoc_nosier
        self.decoding_format = decoding_format
        self.return_tensors = return_tensors

    def __call__(self, data, return_tensors=None):

        new_data = []  # To avoid the orgin data being covered

        for ins in data:

            n_ins = {}
            n_ins['spots'] = copy.deepcopy(ins['spots'])
            n_ins['asocs'] = copy.deepcopy(ins['asocs'])
            n_ins['spot_asoc'] = copy.deepcopy(ins['spot_asoc'])

            sample_prompt = ins['sample_prompt']
            if not sample_prompt:
                # 不对 SSI 进行动态采样的情况，Finetune 单个下游任务，设置 sample_prompt 为 False
                # Eval 不 Shuffle
                spot_prefix = self.negative_sampler.full_spot(
                    shuffle=self.model.training)
                asoc_prefix = self.negative_sampler.full_asoc(
                    shuffle=self.model.training)
            else:
                # 对 SSI 进行采样
                spot_prefix, pos_spot, neg_spot = self.negative_sampler.sample_spot(
                    positive=n_ins.get('spots', []))
                asoc_prefix, neg_asoc = self.negative_sampler.sample_asoc(
                    positive=n_ins.get('asocs', []))

                if 'spot_asoc' in ins:
                    # 过滤不在 Pos Spot 中的实例
                    n_ins['spot_asoc'] = list(
                        filter(lambda x: x['label'] in pos_spot, n_ins[
                            'spot_asoc']))

                    # 注入拒绝识别噪声
                    if self.spot_asoc_nosier is not None:
                        n_ins['spot_asoc'] = self.spot_asoc_nosier.add_noise(
                            n_ins['spot_asoc'],
                            spot_label_list=neg_spot,
                            asoc_label_list=neg_asoc, )

                    # 生成新的信息记录
                    target_record = convert_to_record_function[
                        self.decoding_format](
                            n_ins['spot_asoc'],
                            structure_maker=BaseStructureMarker())
                    n_ins["labels"] = self.tokenizer.encode(
                        target_record,
                        return_token_type_ids=False,
                        return_attention_mask=False, )['input_ids']

            n_ins.pop('asocs')
            n_ins.pop('spots')
            n_ins.pop('spot_asoc')

            prefix = spot_prefix + asoc_prefix

            # 超过最大长度之后，对 prefix 进行截断
            if self.max_prefix_length is not None and self.max_prefix_length >= 0:
                prefix = prefix[:self.max_prefix_length]
            # logger.debug(f"SSI: {self.tokenizer.decode(prefix)}")

            n_ins['input_ids'] = prefix \
                + [self.negative_sampler.text_start] \
                + ins['input_ids']

            # 超过最大长度之后，对 input_ids 进行截断
            if self.max_source_length is not None:
                n_ins['input_ids'] = n_ins['input_ids'][:self.max_source_length]

            if self.max_target_length is not None and 'labels' in ins:
                n_ins['labels'] = n_ins['labels'][:self.max_target_length]

            n_ins['attention_mask'] = [1] * len(n_ins['input_ids'])
            n_ins['decoder_attention_mask'] = [1] * len(n_ins['labels'])

            new_data.append(n_ins)

        first = new_data[0]
        assert isinstance(
            first, dict
        ), f'Input pattern not understood. The input of collatot must be a dict with key of input column name and value of data Received input type: {type(first)}'

        labels = [d["labels"]
                  for d in new_data] if "labels" in new_data[0].keys() else None

        batch = {}

        def _pad_function(sequence, pad_value):
            return Pad(axis=0, pad_val=pad_value, dtype='int64')(sequence)

        pad_value_map = {
            'token_type_ids': self.tokenizer.pad_token_type_id,
            'attention_mask': 0,
            'decoder_attention_mask': 0,
            'special_tokens_mask': 1,
            'input_ids': self.tokenizer.pad_token_id,
        }

        for k, v in first.items():
            if k not in ("labels", "label_ids"
                         ) and v is not None and not isinstance(v, str):
                batch[k] = _pad_function(
                    sequence=[d[k] for d in new_data],
                    pad_value=pad_value_map[k], )
            else:
                batch[k] = _pad_function(
                    sequence=[d[k] for d in new_data],
                    pad_value=self.label_pad_token_id, )

        # prepare decoder_input_ids
        if (labels is not None and self.model is not None and
                hasattr(self.model, "prepare_decoder_input_ids_from_labels")):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=batch["labels"])
            if not return_tensors:
                batch["decoder_input_ids"] = decoder_input_ids.numpy()

        if self.return_tensors:
            for k, v in batch.items():
                batch[k] = paddle.to_tensor(v)

        return batch
