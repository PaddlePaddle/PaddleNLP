#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Gets and stores vocabulary for the ATIS data."""

from . import snippets
from .vocabulary import Vocabulary, UNK_TOK, DEL_TOK, EOS_TOK

INPUT_FN_TYPES = [UNK_TOK, DEL_TOK, EOS_TOK]
OUTPUT_FN_TYPES = [UNK_TOK, EOS_TOK]

MIN_INPUT_OCCUR = 1
MIN_OUTPUT_OCCUR = 1


class ATISVocabulary():
    """ Stores the vocabulary for the ATIS data.

    Attributes:
        raw_vocab (`Vocabulary`): Vocabulary object.
        tokens (`set`): Set of all of the strings in the vocabulary.
        inorder_tokens (`list`): List of all tokens, with a strict and
            unchanging order.
    """

    def __init__(self,
                 token_sequences,
                 filename,
                 params,
                 is_input='input',
                 min_occur=1,
                 anonymizer=None,
                 skip=None):

        if is_input == 'input':
            functional_types = INPUT_FN_TYPES
        elif is_input == 'output':
            functional_types = OUTPUT_FN_TYPES
        elif is_input == 'schema':
            functional_types = [UNK_TOK]
        else:
            functional_types = []

        self.raw_vocab = Vocabulary(
            token_sequences,
            filename,
            functional_types=functional_types,
            min_occur=min_occur,
            ignore_fn=lambda x: snippets.is_snippet(x) or
            (anonymizer and anonymizer.is_anon_tok(x)) or (skip and x in skip))
        self.tokens = set(self.raw_vocab.token_to_id.keys())
        self.inorder_tokens = self.raw_vocab.id_to_token

        assert len(self.inorder_tokens) == len(self.raw_vocab)

    def __len__(self):
        return len(self.raw_vocab)

    def token_to_id(self, token):
        """ Maps from a token to a unique ID.

        Args:
            token (`str`): The token to look up.

        Returns:
            `int`: Uniquely identifying the token.
        """
        return self.raw_vocab.token_to_id[token]

    def id_to_token(self, identifier):
        """ Maps from a unique integer to an identifier.

        Args:
            identifier (`int`): The unique ID.

        Returns:
            `str`: Representing the token.
        """
        return self.raw_vocab.id_to_token[identifier]
