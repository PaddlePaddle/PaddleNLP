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
""" Contains the Snippet class and methods for handling snippets."""

SNIPPET_PREFIX = "SNIPPET_"


def is_snippet(token):
    """ Determines whether a token is a snippet or not.

    Args:
        token (`str`): The token to check.

    Returns:
        `bool`: Indicating whether it's a snippet.
    """
    return token.startswith(SNIPPET_PREFIX)


def expand_snippets(sequence, snippets):
    """ Given a sequence and a list of snippets, expand the snippets in the sequence.

    Args:
        sequence (`list`): Query containing snippet references.
        snippets (`list`): List of available snippets.

    Returns:
        `list`: The expanded sequence list.
    """
    snippet_id_to_snippet = {}
    for snippet in snippets:
        assert snippet.name not in snippet_id_to_snippet
        snippet_id_to_snippet[snippet.name] = snippet
    expanded_seq = []
    for token in sequence:
        if token in snippet_id_to_snippet:
            expanded_seq.extend(snippet_id_to_snippet[token].sequence)
        else:
            assert not is_snippet(token)
            expanded_seq.append(token)

    return expanded_seq


def snippet_index(token):
    """ Returns the index of a snippet.

    Args:
        token (`str`): The snippet to check.

    Returns:
        `int`: The index of the snippet.
    """
    assert is_snippet(token)
    return int(token.split("_")[-1])


class Snippet():
    """ Contains a snippet. """

    def __init__(self, sequence, startpos, sql, age=0):
        self.sequence = sequence
        self.startpos = startpos
        self.sql = sql

        # TODO: age vs. index?
        self.age = age
        self.index = 0

        self.name = ""
        self.embedding = None

        self.endpos = self.startpos + len(self.sequence)
        assert self.endpos < len(self.sql), "End position of snippet is " + str(
            self.endpos) + " which is greater than length of SQL (" + str(
                len(self.sql)) + ")"
        assert self.sequence == self.sql[self.startpos:self.endpos], \
            "Value of snippet (" + " ".join(self.sequence) + ") " \
            "is not the same as SQL at the same positions (" \
            + " ".join(self.sql[self.startpos:self.endpos]) + ")"

    def __str__(self):
        return self.name + "\t" + \
            str(self.age) + "\t" + " ".join(self.sequence)

    def __len__(self):
        return len(self.sequence)

    def increase_age(self):
        """ Ages a snippet by one. """
        self.index += 1

    def assign_id(self, number):
        """ Assigns the name of the snippet to be the prefix + the number. """
        self.name = SNIPPET_PREFIX + str(number)

    def set_embedding(self, embedding):
        """ Sets the embedding of the snippet. """
        self.embedding = embedding
