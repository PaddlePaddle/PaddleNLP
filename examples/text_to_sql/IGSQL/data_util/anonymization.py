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
"""Code for identifying and anonymizing entities in NL and SQL."""

import copy
import json

from . import util

ENTITY_NAME = "ENTITY"
CONSTANT_NAME = "CONSTANT"
TIME_NAME = "TIME"
SEPARATOR = "#"


def timeval(string):
    """Returns the numeric version of a time.

    Args:
        string (`str`): String representing a time.

    Returns:
        `str`: String representing the absolute time.
    """
    if string.endswith("am") or string.endswith("pm") and string[:-2].isdigit():
        numval = int(string[:-2])
        if len(string) == 3 or len(string) == 4:
            numval *= 100
        if string.endswith("pm"):
            numval += 1200
        return str(numval)
    return ""


def is_time(string):
    """Returns whether a string represents a time.

    Args:
        string (str): String to check.

    Returns:
        `bool`: Whether the string represents a time.
    """
    if string.endswith("am") or string.endswith("pm"):
        if string[:-2].isdigit():
            return True

    return False


def deanonymize(sequence, ent_dict, key):
    """Deanonymizes a sequence.

    Args:
        sequence (`list`): List of tokens to deanonymize.
        ent_dict (`dict`): Maps from tokens to the entity dictionary.
        key (`str`): The key to use, in this case either natural language or SQL.

    Returns:
        `list`: Deanonymized sequence of tokens.
    """
    new_sequence = []
    for token in sequence:
        if token in ent_dict:
            new_sequence.extend(ent_dict[token][key])
        else:
            new_sequence.append(token)

    return new_sequence


class Anonymizer:
    """Anonymization class for keeping track of entities in this domain and
       scripts for anonymizing/deanonymizing.

    Attributes:
        anonymization_map (`list`): Containing entities from
            the anonymization file.
        entity_types (`list`): All entities in the anonymization file.
        keys (`set`): Possible keys (types of text handled); in this case it should be
            one for natural language and another for SQL.
        entity_set (`set`): entity_types as a set.
    """

    def __init__(self, filename):
        self.anonymization_map = []
        self.entity_types = []
        self.keys = set()

        pairs = [json.loads(line) for line in open(filename).readlines()]
        for pair in pairs:
            for key in pair:
                if key != "type":
                    self.keys.add(key)
            self.anonymization_map.append(pair)
            if pair["type"] not in self.entity_types:
                self.entity_types.append(pair["type"])

        self.entity_types.append(ENTITY_NAME)
        self.entity_types.append(CONSTANT_NAME)
        self.entity_types.append(TIME_NAME)

        self.entity_set = set(self.entity_types)

    def get_entity_type_from_token(self, token):
        """Gets the type of an entity given an anonymized token.

        Args:
            token (`str`): The entity token.

        Returns:
            `str`: representing the type of the entity.
        """
        # these are in the pattern NAME:#, so just strip the thing after the
        # colon
        colon_loc = token.index(SEPARATOR)
        entity_type = token[:colon_loc]
        assert entity_type in self.entity_set

        return entity_type

    def is_anon_tok(self, token):
        """Returns whether a token is an anonymized token or not.

        Args:
            token (`str`): The token to check.

        Returns:
            `bool`: whether the token is an anonymized token.
        """
        return token.split(SEPARATOR)[0] in self.entity_set

    def get_anon_id(self, token):
        """Gets the entity index (unique ID) for a token.

        Args:
            token (`str`): The token to get the index from.

        Returns:
            `int`: the token ID if it is an anonymized token; otherwise -1.
        """
        if self.is_anon_tok(token):
            return self.entity_types.index(token.split(SEPARATOR)[0])
        else:
            return -1

    def anonymize(self,
                  sequence,
                  tok_to_entity_dict,
                  key,
                  add_new_anon_toks=False):
        """Anonymizes a sequence.

        Args:
            sequence (`list`): Sequence to anonymize.
            tok_to_entity_dict (`dict`): Existing dictionary mapping from anonymized
                tokens to entities.
            key (`str`): Which kind of text this is (natural language or SQL)
            add_new_anon_toks (`bool`): Whether to add new entities to tok_to_entity_dict.

        Returns:
            `list`: The anonymized sequence.
        """
        # Sort the token-tok-entity dict by the length of the modality.
        sorted_dict = sorted(tok_to_entity_dict.items(),
                             key=lambda k: len(k[1][key]))[::-1]

        anonymized_sequence = copy.deepcopy(sequence)

        if add_new_anon_toks:
            type_counts = {}
            for entity_type in self.entity_types:
                type_counts[entity_type] = 0
            for token in tok_to_entity_dict:
                entity_type = self.get_entity_type_from_token(token)
                type_counts[entity_type] += 1

        # First find occurrences of things in the anonymization dictionary.
        for token, modalities in sorted_dict:
            our_modality = modalities[key]

            # Check if this key's version of the anonymized thing is in our
            # sequence.
            while util.subsequence(our_modality, anonymized_sequence):
                found = False
                for startidx in range(
                        len(anonymized_sequence) - len(our_modality) + 1):
                    if anonymized_sequence[startidx:startidx +
                                           len(our_modality)] == our_modality:
                        anonymized_sequence = anonymized_sequence[:startidx] + [
                            token
                        ] + anonymized_sequence[startidx + len(our_modality):]
                        found = True
                        break
                assert found, "Thought " \
                    + str(our_modality) + " was in [" \
                    + str(anonymized_sequence) + "] but could not find it"

        # Now add new keys if they are present.
        if add_new_anon_toks:

            # For every span in the sequence, check whether it is in the anon map
            # for this modality
            sorted_anon_map = sorted(self.anonymization_map,
                                     key=lambda k: len(k[key]))[::-1]

            for pair in sorted_anon_map:
                our_modality = pair[key]

                token_type = pair["type"]
                new_token = token_type + SEPARATOR + \
                    str(type_counts[token_type])

                while util.subsequence(our_modality, anonymized_sequence):
                    found = False
                    for startidx in range(
                            len(anonymized_sequence) - len(our_modality) + 1):
                        if anonymized_sequence[startidx:startidx + \
                            len(our_modality)] == our_modality:
                            if new_token not in tok_to_entity_dict:
                                type_counts[token_type] += 1
                                tok_to_entity_dict[new_token] = pair

                            anonymized_sequence = anonymized_sequence[:startidx] + [
                                new_token
                            ] + anonymized_sequence[startidx +
                                                    len(our_modality):]
                            found = True
                            break
                    assert found, "Thought " \
                        + str(our_modality) + " was in [" \
                        + str(anonymized_sequence) + "] but could not find it"

            # Also replace integers with constants
            for index, token in enumerate(anonymized_sequence):
                if token.isdigit() or is_time(token):
                    if token.isdigit():
                        entity_type = CONSTANT_NAME
                        value = new_token
                    if is_time(token):
                        entity_type = TIME_NAME
                        value = timeval(token)

                    # First try to find the constant in the entity dictionary already,
                    # and get the name if it's found.
                    new_token = ""
                    new_dict = {}
                    found = False
                    for entity, value in tok_to_entity_dict.items():
                        if value[key][0] == token:
                            new_token = entity
                            new_dict = value
                            found = True
                            break

                    if not found:
                        new_token = entity_type + SEPARATOR + \
                            str(type_counts[entity_type])
                        new_dict = {}
                        for tempkey in self.keys:
                            new_dict[tempkey] = [token]

                        tok_to_entity_dict[new_token] = new_dict
                        type_counts[entity_type] += 1

                    anonymized_sequence[index] = new_token

        return anonymized_sequence
