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

import copy
import random

import pymysql
import signal
import sqlparse
from sqlparse import tokens as token_types
from sqlparse import sql as sql_types

from . import util
from .snippets import Snippet

interesting_selects = ["DISTINCT", "MAX", "MIN", "count"]
ignored_subtrees = [["1", "=", "1"]]


def strip_whitespace_front(token_list):
    """Strips whitespace and punctuation from the front of a SQL token list.

    Args:
        token_list(`list`): the token list.

    Outputs:
        `list`: New token list.
    """
    new_token_list = []
    found_valid = False

    for token in token_list:
        if not (token.is_whitespace
                or token.ttype == token_types.Punctuation) or found_valid:
            found_valid = True
            new_token_list.append(token)

    return new_token_list


def strip_whitespace(token_list):
    """ Strips whitespace from a token list.
    
    Args:
        token_list(`list`): the token list.

    Returns:
        `list`: New token list with no whitespace/punctuation surrounding.
    """
    subtokens = strip_whitespace_front(token_list)
    subtokens = strip_whitespace_front(subtokens[::-1])[::-1]
    return subtokens


def token_list_to_seq(token_list):
    """Converts a Token list to a sequence of strings, stripping out surrounding
    punctuation and all whitespace.

    Args:
        token_list(`list`): the list of tokens.

    Outputs:
        `list`: sequence of strings
    """
    subtokens = strip_whitespace(token_list)

    seq = []
    flat = sqlparse.sql.TokenList(subtokens).flatten()
    for i, token in enumerate(flat):
        strip_token = str(token).strip()
        if len(strip_token) > 0:
            seq.append(strip_token)
    if len(seq) > 0:
        if seq[0] == "(" and seq[-1] == ")":
            seq = seq[1:-1]

    return seq


def find_subtrees(sequence,
                  current_subtrees,
                  where_parent=False,
                  keep_conj_subtrees=False):
    """Finds subtrees for a subsequence of SQL.
    
    Args:
      sequence(`list`): Sequence of SQL tokens.
      current_subtrees(`list`): Current list of subtrees.
      where_parent(`bool`, optional): Whether the parent of the current sequence was a where clause
      keep_conj_subtrees('bool', optional): Whether to look for a conjunction in this sequence and
                          keep its arguments
    """

    # If the parent of the subsequence was a WHERE clause, keep everything in the
    # sequence except for the beginning WHERE and any surrounding parentheses.
    if where_parent:
        # Strip out the beginning WHERE, and any punctuation or whitespace at the
        # beginning or end of the token list.
        seq = token_list_to_seq(sequence.tokens[1:])
        if len(seq) > 0 and seq not in current_subtrees:
            current_subtrees.append(seq)

    # If the current sequence has subtokens, i.e. if it's a node that can be
    # expanded, check for a conjunction in its subtrees, and expand its subtrees.
    # Also check for any SELECT statements and keep track of what follows.
    if sequence.is_group:
        if keep_conj_subtrees:
            subtokens = strip_whitespace(sequence.tokens)

            # Check if there is a conjunction in the subsequence. If so, keep the
            # children. Also make sure you don't split where AND is used within a
            # child -- the subtokens sequence won't treat those ANDs differently (a
            # bit hacky but it works)
            has_and = False
            for i, token in enumerate(subtokens):
                if token.value == "OR" or token.value == "AND":
                    has_and = True
                    break

            if has_and:
                and_subtrees = []
                current_subtree = []
                for i, token in enumerate(subtokens):
                    if token.value == "OR" or (
                            token.value == "AND" and i - 4 >= 0
                            and i - 4 < len(subtokens)
                            and subtokens[i - 4].value != "BETWEEN"):
                        and_subtrees.append(current_subtree)
                        current_subtree = []
                    else:
                        current_subtree.append(token)
                and_subtrees.append(current_subtree)

                for subtree in and_subtrees:
                    seq = token_list_to_seq(subtree)
                    if len(seq) > 0 and seq[0] == "WHERE":
                        seq = seq[1:]
                    if seq not in current_subtrees:
                        current_subtrees.append(seq)

        in_select = False
        select_toks = []
        for i, token in enumerate(sequence.tokens):
            # Mark whether this current token is a WHERE.
            is_where = (isinstance(token, sql_types.Where))

            # If you are in a SELECT, start recording what follows until you hit a
            # FROM
            if token.value == "SELECT":
                in_select = True
            elif in_select:
                select_toks.append(token)
                if token.value == "FROM":
                    in_select = False

                    seq = []
                    if len(sequence.tokens) > i + 2:
                        seq = token_list_to_seq(select_toks +
                                                [sequence.tokens[i + 2]])

                    if seq not in current_subtrees and len(
                            seq) > 0 and seq[0] in interesting_selects:
                        current_subtrees.append(seq)

                    select_toks = []

            # Recursively find subtrees in the children of the node.
            find_subtrees(token, current_subtrees, is_where, where_parent
                          or keep_conj_subtrees)


def get_subtrees(sql, oldsnippets=[]):
    parsed = sqlparse.parse(" ".join(sql))[0]

    subtrees = []
    find_subtrees(parsed, subtrees)

    final_subtrees = []
    for subtree in subtrees:
        if subtree not in ignored_subtrees:
            final_version = []
            keep = True

            parens_counts = 0
            for i, token in enumerate(subtree):
                if token == ".":
                    newtoken = final_version[-1] + "." + subtree[i + 1]
                    final_version = final_version[:-1] + [newtoken]
                    keep = False
                elif keep:
                    final_version.append(token)
                else:
                    keep = True

                if token == "(":
                    parens_counts -= 1
                elif token == ")":
                    parens_counts += 1

            if parens_counts == 0:
                final_subtrees.append(final_version)

    snippets = []
    sql = [str(tok) for tok in sql]
    for subtree in final_subtrees:
        startpos = -1
        for i in range(len(sql) - len(subtree) + 1):
            if sql[i:i + len(subtree)] == subtree:
                startpos = i
        if startpos >= 0 and startpos + len(subtree) < len(sql):
            age = 0
            for prevsnippet in oldsnippets:
                if prevsnippet.sequence == subtree:
                    age = prevsnippet.age + 1
            snippet = Snippet(subtree, startpos, sql, age=age)
            snippets.append(snippet)

    return snippets


def get_subtrees_simple(sql, oldsnippets=[]):
    sql_string = " ".join(sql)
    format_sql = sqlparse.format(sql_string, reindent=True)

    # get subtrees
    subtrees = []
    for sub_sql in format_sql.split('\n'):
        sub_sql = sub_sql.replace('(',
                                  ' ( ').replace(')',
                                                 ' ) ').replace(',', ' , ')

        subtree = sub_sql.strip().split()
        if len(subtree) > 1:
            subtrees.append(subtree)

    final_subtrees = subtrees

    snippets = []
    sql = [str(tok) for tok in sql]
    for subtree in final_subtrees:
        startpos = -1
        for i in range(len(sql) - len(subtree) + 1):
            if sql[i:i + len(subtree)] == subtree:
                startpos = i

        if startpos >= 0 and startpos + len(subtree) <= len(sql):
            age = 0
            for prevsnippet in oldsnippets:
                if prevsnippet.sequence == subtree:
                    age = prevsnippet.age + 1
            new_sql = sql + [';']
            snippet = Snippet(subtree, startpos, new_sql, age=age)
            snippets.append(snippet)

    return snippets


conjunctions = {"AND", "OR", "WHERE"}


def get_all_in_parens(sequence):
    if sequence[-1] == ";":
        sequence = sequence[:-1]

    if not "(" in sequence:
        return []

    if sequence[0] == "(" and sequence[-1] == ")":
        in_parens = sequence[1:-1]
        return [in_parens] + get_all_in_parens(in_parens)
    else:
        paren_subseqs = []
        current_seq = []
        num_parens = 0
        in_parens = False
        for token in sequence:
            if in_parens:
                current_seq.append(token)
                if token == ")":
                    num_parens -= 1
                    if num_parens == 0:
                        in_parens = False
                        paren_subseqs.append(current_seq)
                        current_seq = []
            elif token == "(":
                in_parens = True
                current_seq.append(token)
            if token == "(":
                num_parens += 1

        all_subseqs = []
        for subseq in paren_subseqs:
            all_subseqs.extend(get_all_in_parens(subseq))
        return all_subseqs


def split_by_conj(sequence):
    num_parens = 0
    current_seq = []
    subsequences = []

    for token in sequence:
        if num_parens == 0:
            if token in conjunctions:
                subsequences.append(current_seq)
                current_seq = []
                break
        current_seq.append(token)
        if token == "(":
            num_parens += 1
        elif token == ")":
            num_parens -= 1

        assert num_parens >= 0

    return subsequences


def get_sql_snippets(sequence):
    # First, get all subsequences of the sequence that are surrounded by
    # parentheses.
    all_in_parens = get_all_in_parens(sequence)
    all_subseq = []

    # Then for each one, split the sequence on conjunctions (AND/OR).
    for seq in all_in_parens:
        subsequences = split_by_conj(seq)
        all_subseq.append(seq)
        all_subseq.extend(subsequences)

    # Finally, also get "interesting" selects

    for i, seq in enumerate(all_subseq):
        print(str(i) + "\t" + " ".join(seq))
    exit()


def add_snippets_to_query(snippets, ignored_entities, query, prob_align=1.):
    query_copy = copy.copy(query)

    # Replace the longest snippets first, so sort by length descending.
    sorted_snippets = sorted(snippets, key=lambda s: len(s.sequence))[::-1]

    for snippet in sorted_snippets:
        ignore = False
        snippet_seq = snippet.sequence

        # If it contains an ignored entity, then don't use it.
        for entity in ignored_entities:
            ignore = ignore or util.subsequence(entity, snippet_seq)

        # No NL entities found in snippet, then see if snippet is a substring of
        # the gold sequence
        if not ignore:
            snippet_length = len(snippet_seq)

            # Iterate through gold sequence to see if it's a subsequence.
            for start_idx in range(len(query_copy) - snippet_length + 1):
                if query_copy[start_idx:start_idx +
                              snippet_length] == snippet_seq:
                    align = random.random() < prob_align

                    if align:
                        prev_length = len(query_copy)

                        # At the start position of the snippet, replace with an
                        # identifier.
                        query_copy[start_idx] = snippet.name

                        # Then cut out the indices which were collapsed into
                        # the snippet.
                        query_copy = query_copy[:start_idx + 1] + \
                            query_copy[start_idx + snippet_length:]

                        # Make sure the length is as expected
                        assert len(query_copy) == prev_length - \
                            (snippet_length - 1)

    return query_copy


def execution_results(query, username, password, timeout=3):
    connection = pymysql.connect(user=username, password=password)

    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException

    signal.signal(signal.SIGALRM, timeout_handler)

    syntactic = True
    semantic = True

    table = []

    with connection.cursor() as cursor:
        signal.alarm(timeout)
        try:
            cursor.execute("SET sql_mode='IGNORE_SPACE';")
            cursor.execute("use atis3;")
            cursor.execute(query)
            table = cursor.fetchall()
            cursor.close()
        except TimeoutException:
            signal.alarm(0)
            cursor.close()
        except pymysql.err.ProgrammingError:
            syntactic = False
            semantic = False
            cursor.close()
        except pymysql.err.InternalError:
            semantic = False
            cursor.close()
        except Exception as e:
            signal.alarm(0)
        signal.alarm(0)
        cursor.close()
    signal.alarm(0)

    connection.close()

    return (syntactic, semantic, sorted(table))


def executable(query, username, password, timeout=2):
    return execution_results(query, username, password, timeout)[1]


def fix_parentheses(sequence):
    num_left = sequence.count("(")
    num_right = sequence.count(")")

    if num_right < num_left:
        fixed_sequence = sequence[:-1] + \
            [")" for _ in range(num_left - num_right)] + [sequence[-1]]
        return fixed_sequence

    return sequence
