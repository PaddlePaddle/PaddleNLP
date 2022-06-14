# coding=utf-8
# Copyright 2018 deepset team.
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
"""
Tokenization classes.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Dict, Any, Tuple, Optional, List, Union

import re
import logging
import numpy as np

from paddlenlp.transformers.tokenizer_utils_base import TruncationStrategy

from pipelines.data_handler.samples import SampleBasket

logger = logging.getLogger(__name__)

# Special characters used by the different tokenizers to indicate start of word / whitespace
SPECIAL_TOKENIZER_CHARS = r"^(##|Ġ|▁)"


def tokenize_batch_question_answering(pre_baskets, tokenizer, indices):
    """
    Tokenizes text data for question answering tasks. Tokenization means splitting words into subwords, depending on the
    tokenizer's vocabulary.

    - We first tokenize all documents in batch mode. (When using FastTokenizers Rust multithreading can be enabled by TODO add how to enable rust mt)
    - Then we tokenize each question individually
    - We construct dicts with question and corresponding document text + tokens + offsets + ids

    :param pre_baskets: input dicts with QA info #todo change to input objects
    :param tokenizer: tokenizer to be used
    :param indices: list, indices used during multiprocessing so that IDs assigned to our baskets are unique
    :return: baskets, list containing question and corresponding document information
    """
    assert len(indices) == len(pre_baskets)

    baskets = []
    # Tokenize texts in batch mode
    # tokenized_docs_batch.keys(): dict_keys(['input_ids', 'attention_mask', 'special_tokens_mask', 'offset_mapping'])
    texts = [d["context"] for d in pre_baskets]

    tokenized_docs_batch = tokenizer.batch_encode(
        texts,
        truncation=TruncationStrategy.ONLY_SECOND,
        return_special_tokens_mask=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_token_type_ids=False,
        add_special_tokens=False)

    # Extract relevant data
    tokenids_batch = tokenized_docs_batch["input_ids"]
    offsets_batch = []
    for o in tokenized_docs_batch["offset_mapping"]:
        offsets_batch.append(np.array([x[0] for x in o]))

    start_of_words_batch = []
    for input_ids in tokenized_docs_batch["input_ids"]:
        start_of_words_batch.append([1] * len(input_ids))

    for i_doc, d in enumerate(pre_baskets):
        document_text = d["context"]
        # Tokenize questions one by one
        for i_q, q in enumerate(d["qas"]):
            question_text = q["question"]
            tokenized_q = tokenizer.encode(question_text,
                                           return_special_tokens_mask=True,
                                           return_attention_mask=True,
                                           return_offsets_mapping=True,
                                           return_token_type_ids=False,
                                           add_special_tokens=False)

            # Extract relevant data
            question_tokenids = tokenized_q["input_ids"]

            # Fake offset_mapping
            question_offsets = [(i, i + 1)
                                for i in range(len(question_tokenids))]

            # question start_of_words_batch
            # Fake question_sow
            question_sow = [1] * len(question_tokenids)
            # question_sow = _get_start_of_word_QA(tokenized_q.encodings[0].words)

            # Document.id
            external_id = q["id"]
            # The internal_id depends on unique ids created for each process before forking
            # i_q is always set to 0
            internal_id = f"{indices[i_doc]}-{i_q}"
            raw = {
                "document_text": document_text,
                "document_tokens": tokenids_batch[i_doc],
                "document_offsets": offsets_batch[i_doc],
                "document_start_of_word": start_of_words_batch[i_doc],
                "question_text": question_text,
                "question_tokens": question_tokenids,
                "question_offsets": question_offsets,
                "question_start_of_word": question_sow,
                "answers": q["answers"],
            }
            # TODO add only during debug mode (need to create debug mode)
            # raw["document_tokens_strings"] = tokenized_docs_batch.encodings[i_doc].tokens
            # raw["question_tokens_strings"] = tokenized_q.encodings[0].tokens
            raw["document_tokens_strings"] = tokenizer.convert_ids_to_tokens(
                tokenized_docs_batch["input_ids"][i_doc])
            raw["question_tokens_strings"] = tokenizer.convert_ids_to_tokens(
                question_tokenids)

            baskets.append(
                SampleBasket(raw=raw,
                             id_internal=internal_id,
                             id_external=external_id,
                             samples=None))
    return baskets


def _get_start_of_word_QA(word_ids):
    words = np.array(word_ids)
    start_of_word_single = [1] + list(np.ediff1d(words))
    return start_of_word_single
