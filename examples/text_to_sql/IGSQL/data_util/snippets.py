""" Contains the Snippet class and methods for handling snippets.

Attributes:
    SNIPPET_PREFIX: string prefix for snippets.
"""

SNIPPET_PREFIX = "SNIPPET_"


def is_snippet(token):
    """ Determines whether a token is a snippet or not.

    Inputs:
        token (str): The token to check.

    Returns:
        bool, indicating whether it's a snippet.
    """
    return token.startswith(SNIPPET_PREFIX)


def expand_snippets(sequence, snippets):
    """ Given a sequence and a list of snippets, expand the snippets in the sequence.

    Inputs:
        sequence (list of str): Query containing snippet references.
        snippets (list of Snippet): List of available snippets.

    return list of str representing the expanded sequence
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

    Inputs:
        token (str): The snippet to check.

    Returns:
        integer, the index of the snippet.
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
        """ Sets the embedding of the snippet.

        Inputs:
            embedding (dy.Expression)

        """
        self.embedding = embedding
