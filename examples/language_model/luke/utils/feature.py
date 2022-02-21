# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""feature file"""
import logging
import unicodedata
from argparse import Namespace
from contextlib import closing
from itertools import chain, repeat

import multiprocessing
from multiprocessing.pool import Pool

from tqdm import tqdm

logger = logging.getLogger(__name__)


class InputFeatures:
    """input features"""

    def __init__(
            self,
            unique_id,
            example_index,
            doc_span_index,
            tokens,
            mentions,
            token_to_orig_map,
            token_is_max_context,
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
            start_positions,
            end_positions, ):
        """init fun"""
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.mentions = mentions
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.word_ids = word_ids
        self.word_segment_ids = word_segment_ids
        self.word_attention_mask = word_attention_mask
        self.entity_ids = entity_ids
        self.entity_position_ids = entity_position_ids
        self.entity_segment_ids = entity_segment_ids
        self.entity_attention_mask = entity_attention_mask
        self.start_positions = start_positions
        self.end_positions = end_positions


def convert_examples_to_features(
        examples,
        tokenizer,
        entity_vocab,
        wiki_link_db,
        model_redirect_mappings,
        link_redirect_mappings,
        max_seq_length,
        max_mention_length,
        doc_stride,
        max_query_length,
        min_mention_link_prob,
        segment_b_id,
        add_extra_sep_token,
        is_training,
        pool_size=multiprocessing.cpu_count(),
        chunk_size=30, ):
    """convert examples to features"""
    passage_encoder = PassageEncoder(
        tokenizer,
        entity_vocab,
        wiki_link_db,
        model_redirect_mappings,
        link_redirect_mappings,
        max_mention_length,
        min_mention_link_prob,
        add_extra_sep_token,
        segment_b_id, )

    worker_params = Namespace(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        add_extra_sep_token=add_extra_sep_token,
        passage_encoder=passage_encoder,
        is_training=is_training, )
    features = []
    unique_id = 1000000000
    with closing(
            Pool(
                pool_size,
                initializer=_initialize_worker,
                initargs=(worker_params, ))) as pool:
        with tqdm(total=len(examples)) as pbar:
            for ret in pool.imap(
                    _process_example, enumerate(examples),
                    chunksize=chunk_size):
                for feature in ret:
                    feature.unique_id = unique_id
                    features.append(feature)
                    unique_id += 1
                pbar.update()
    return features


class PassageEncoder:
    """passage encoder"""

    def __init__(
            self,
            tokenizer,
            entity_vocab,
            wiki_link_db,
            model_redirect_mappings,
            link_redirect_mappings,
            max_mention_length,
            min_mention_link_prob,
            add_extra_sep_token,
            segment_b_id, ):
        """passage encoder"""
        self._tokenizer = tokenizer
        self._entity_vocab = entity_vocab
        self._wiki_link_db = wiki_link_db
        self._model_redirect_mappings = model_redirect_mappings
        self._link_redirect_mappings = link_redirect_mappings
        self._max_mention_length = max_mention_length
        self._add_extra_sep_token = add_extra_sep_token
        self._segment_b_id = segment_b_id
        self._min_mention_link_prob = min_mention_link_prob

    def encode(self, title, tokens_a, tokens_b):
        """encode"""
        if self._add_extra_sep_token:
            mid_sep_tokens = [self._tokenizer.sep_token] * 2
        else:
            mid_sep_tokens = [self._tokenizer.sep_token]

        all_tokens = [
            self._tokenizer.cls_token
        ] + tokens_a + mid_sep_tokens + tokens_b + [self._tokenizer.sep_token]

        word_ids = self._tokenizer.convert_tokens_to_ids(all_tokens)
        word_segment_ids = [0] * (len(tokens_a) + len(mid_sep_tokens) + 1
                                  ) + [self._segment_b_id] * (len(tokens_b) + 1)
        word_attention_mask = [1] * len(all_tokens)

        try:
            title = self._link_redirect_mappings.get(title, title)
            mention_candidates = {}
            ambiguous_mentions = set()
            for link in self._wiki_link_db.get(title):
                if link.link_prob < self._min_mention_link_prob:
                    continue

                link_text = self._normalize_mention(link.text)
                if link_text in mention_candidates and mention_candidates[
                        link_text] != link.title:
                    ambiguous_mentions.add(link_text)
                    continue

                mention_candidates[link_text] = link.title

            for link_text in ambiguous_mentions:
                del mention_candidates[link_text]

        except KeyError:
            mention_candidates = {}
            logger.warning("Not found in the Dump DB: %s", title)

        mentions_a = self._detect_mentions(tokens_a, mention_candidates)
        mentions_b = self._detect_mentions(tokens_b, mention_candidates)
        all_mentions = mentions_a + mentions_b
        if all_mentions:
            print(all_mentions)
            exit()

        if not all_mentions:
            entity_ids = [0, 0]
            entity_segment_ids = [0, 0]
            entity_attention_mask = [0, 0]
            entity_position_ids = [[
                -1 for y in range(self._max_mention_length)
            ]] * 2
        else:
            entity_ids = [0] * len(all_mentions)
            entity_segment_ids = [0] * len(mentions_a) + [self._segment_b_id
                                                          ] * len(mentions_b)
            entity_attention_mask = [1] * len(all_mentions)
            entity_position_ids = [
                [-1 for y in range(self._max_mention_length)]
                for x in range(len(all_mentions))
            ]

            offset_a = 1
            offset_b = len(tokens_a) + 2  # 2 for CLS and SEP tokens
            if self._add_extra_sep_token:
                offset_b += 1

            for i, (offset, (entity_id, start, end)) in enumerate(
                    chain(
                        zip(repeat(offset_a), mentions_a),
                        zip(repeat(offset_b), mentions_b))):
                entity_ids[i] = entity_id
                entity_position_ids[i][:end - start] = range(start + offset,
                                                             end + offset)

            if len(all_mentions) == 1:
                entity_ids.append(0)
                entity_segment_ids.append(0)
                entity_attention_mask.append(0)
                entity_position_ids.append(
                    [-1 for y in range(self._max_mention_length)])

        return dict(
            tokens=all_tokens,
            mentions=all_mentions,
            word_ids=word_ids,
            word_segment_ids=word_segment_ids,
            word_attention_mask=word_attention_mask,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
            entity_segment_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask, )

    def _detect_mentions(self, tokens, mention_candidates):
        """detect mentions"""
        mentions = []
        cur = 0
        for start, token in enumerate(tokens):
            if start < cur:
                continue
            if self._is_subword(token):
                continue

            for end in range(
                    min(start + self._max_mention_length, len(tokens)), start,
                    -1):
                if end < len(tokens) and self._is_subword(tokens[end]):
                    continue
                mention_text = self._tokenizer.convert_tokens_to_string(tokens[
                    start:end])
                mention_text = self._normalize_mention(mention_text)
                if mention_text in mention_candidates:
                    cur = end
                    title = mention_candidates[mention_text]
                    title = self._model_redirect_mappings.get(
                        title, title)  # resolve mismatch between two dumps
                    if title in self._entity_vocab:
                        mentions.append((self._entity_vocab[title], start, end))
                    break

        return mentions

    def _is_subword(self, token):
        """is sub sequence word"""
        token = self._tokenizer.convert_tokens_to_string(token)
        return True

    @staticmethod
    def _is_punctuation(char):
        """is punctuation"""
        # obtained from:
        # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
        cp = ord(char)
        if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    @staticmethod
    def _normalize_mention(text):
        """normal mention"""
        return " ".join(text.lower().split(" ")).strip()


params = None


def _initialize_worker(_params):
    """init worker"""
    global params
    params = _params


def _add_process_example(doc_spans, query_tokens, tok_to_orig_index,
                         all_doc_tokens, example, tok_start_positions,
                         tok_end_positions, example_index):
    """add feature"""
    features = []

    for doc_span_index, doc_span in enumerate(doc_spans):
        token_to_orig_map = {}
        token_is_max_context = {}
        answer_tokens = []
        answer_offset = len(query_tokens) + 2
        if params.add_extra_sep_token:
            answer_offset += 1
        for i in range(doc_span["length"]):
            split_token_index = doc_span["start"] + i
            token_to_orig_map[answer_offset + i] = tok_to_orig_index[
                split_token_index]
            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[answer_offset + i] = is_max_context
            answer_tokens.append(all_doc_tokens[split_token_index])
        start_positions = []
        end_positions = []
        if params.is_training:
            if example.is_impossible:
                start_positions = [0]
                end_positions = [0]
            else:
                doc_start = doc_span["start"]
                doc_end = doc_span["start"] + doc_span["length"] - 1
                for tok_start, tok_end in zip(tok_start_positions,
                                              tok_end_positions):
                    if not (tok_start >= doc_start and tok_end <= doc_end):
                        continue
                    doc_offset = len(query_tokens) + 2
                    if params.add_extra_sep_token:
                        doc_offset += 1
                    start_positions.append(tok_start - doc_start + doc_offset)
                    end_positions.append(tok_end - doc_start + doc_offset)
                if not start_positions:
                    start_positions = [0]
                    end_positions = [0]
        features.append(
            InputFeatures(
                unique_id=None,
                example_index=example_index,
                doc_span_index=doc_span_index,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                start_positions=start_positions,
                end_positions=end_positions,
                **params.passage_encoder.encode(example.title, query_tokens,
                                                answer_tokens)))
    return features


def _process_example(args):
    """process example"""
    example_index, example = args
    tokenizer = params.tokenizer
    query_tokens = _tokenize(example.question_text)
    if len(query_tokens) > params.max_query_length:
        query_tokens = query_tokens[0:params.max_query_length]
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for i, token in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = _tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    tok_start_positions = []
    tok_end_positions = []
    if params.is_training and not example.is_impossible:
        for start, end, answer_text in zip(example.start_positions,
                                           example.end_positions,
                                           example.answer_texts):
            tok_start = orig_to_tok_index[start]
            if end < len(example.doc_tokens) - 1:
                tok_end = orig_to_tok_index[end + 1] - 1
            else:
                tok_end = len(all_doc_tokens) - 1
            tok_start, tok_end = _improve_answer_span(
                all_doc_tokens, tok_start, tok_end, tokenizer, answer_text)
            tok_start_positions.append(tok_start)
            tok_end_positions.append(tok_end)
    max_tokens_for_doc = params.max_seq_length - len(query_tokens) - 3
    if params.add_extra_sep_token:
        max_tokens_for_doc -= 1
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(dict(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, params.doc_stride)
    return _add_process_example(doc_spans, query_tokens, tok_to_orig_index,
                                all_doc_tokens, example, tok_start_positions,
                                tok_end_positions, example_index)


def _tokenize(text):
    """token"""
    return params.tokenizer.tokenize(text, add_prefix_space=True)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer.
       Original version was obtained from here:
       https://github.com/huggingface/transformers/blob/23c6998bf46e43092fc59543ea7795074a720f08/src/transformers/data/processors/squad.py#L25
    """
    tok_answer_text = tokenizer.convert_tokens_to_string(
        _tokenize(orig_answer_text)).strip()

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = tokenizer.convert_tokens_to_string(doc_tokens[
                new_start:(new_end + 1)]).strip()
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token.
       Original version was obtained from here:
       https://github.com/huggingface/transformers/blob/23c6998bf46e43092fc59543ea7795074a720f08/src/transformers/data/processors/squad.py#L38
    """
    best_score = None
    best_span_index = None
    for span_index, doc_span in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context,
                    num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
