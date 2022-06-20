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
"""Contains class and methods for storing and computing a vocabulary from text."""

import operator
import os
import pickle

# Special sequencing tokens.
UNK_TOK = "_UNK"  # Replaces out-of-vocabulary words.
EOS_TOK = "_EOS"  # Appended to the end of a sequence to indicate its end.
DEL_TOK = ";"


class Vocabulary:
    """Vocabulary class: stores information about words in a corpus.

    Attributes:
        functional_types (`list`): Functional vocabulary words, such as EOS.
        max_size (`int`): The maximum size of vocabulary to keep.
        min_occur (`int`): The minimum number of times a word should occur to keep it.
        id_to_token (`list`): Ordered list of word types.
        token_to_id (`dict`): Maps from each unique word type to its index.
    """

    def get_vocab(self, sequences, ignore_fn):
        """Gets vocabulary from a list of sequences.

        Args:
            sequences (`list`): Sequences from which to compute the vocabulary.
            ignore_fn (`function`): Function used to tell whether to ignore a
                token during computation of the vocabulary.

        Returns:
            `list`: List of the unique word types in the vocabulary.
        """
        type_counts = {}

        for sequence in sequences:
            for token in sequence:
                if not ignore_fn(token):
                    if token not in type_counts:
                        type_counts[token] = 0
                    type_counts[token] += 1

        # Create sorted list of tokens, by their counts. Reverse so it is in order of
        # most frequent to least frequent.
        sorted_type_counts = sorted(sorted(type_counts.items()),
                                    key=operator.itemgetter(1))[::-1]

        sorted_types = [
            typecount[0] for typecount in sorted_type_counts
            if typecount[1] >= self.min_occur
        ]

        # Append the necessary functional tokens.
        sorted_types = self.functional_types + sorted_types

        # Cut off if vocab_size is set (nonnegative)
        if self.max_size >= 0:
            vocab = sorted_types[:max(self.max_size, len(sorted_types))]
        else:
            vocab = sorted_types

        return vocab

    def __init__(self,
                 sequences,
                 filename,
                 functional_types=None,
                 max_size=-1,
                 min_occur=0,
                 ignore_fn=lambda x: False):
        self.functional_types = functional_types
        self.max_size = max_size
        self.min_occur = min_occur

        vocab = self.get_vocab(sequences, ignore_fn)

        self.id_to_token = []
        self.token_to_id = {}

        for i, word_type in enumerate(vocab):
            self.id_to_token.append(word_type)
            self.token_to_id[word_type] = i

        # Load the previous vocab, if it exists.
        if os.path.exists(filename):
            infile = open(filename, 'rb')
            loaded_vocab = pickle.load(infile)
            infile.close()

            print("Loaded vocabulary from " + str(filename))
            if loaded_vocab.id_to_token != self.id_to_token \
                or loaded_vocab.token_to_id != self.token_to_id:
                print(
                    "Loaded vocabulary is different than generated vocabulary.")
        else:
            print("Writing vocabulary to " + str(filename))
            outfile = open(filename, 'wb')
            pickle.dump(self, outfile)
            outfile.close()

    def __len__(self):
        return len(self.id_to_token)
