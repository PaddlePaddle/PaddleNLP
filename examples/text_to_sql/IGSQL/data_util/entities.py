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
""" Classes for keeping track of the entities in a natural language string. """

import json


class NLtoSQLDict:
    """
    Entity dict file should contain, on each line, a JSON dictionary with
    "input" and "output" keys specifying the string for the input and output
    pairs. The idea is that the existence of the key in an input sequence
    likely corresponds to the existence of the value in the output sequence.

    The entity_dict should map keys (input strings) to a list of values (output
    strings) where this property holds. This allows keys to map to multiple
    output strings (e.g. for times).
    """

    def __init__(self, entity_dict_filename):
        self.entity_dict = {}

        pairs = [
            json.loads(line) for line in open(entity_dict_filename).readlines()
        ]
        for pair in pairs:
            input_seq = pair["input"]
            output_seq = pair["output"]
            if input_seq not in self.entity_dict:
                self.entity_dict[input_seq] = []
            self.entity_dict[input_seq].append(output_seq)

    def get_sql_entities(self, tokenized_nl_string):
        """
        Gets the output-side entities which correspond to the input entities in
        the input sequence.

        Args:
           tokenized_nl_string (`list`): list of tokens in the input string.

        Outputs:
           `list`: The output strings.
        """
        assert len(tokenized_nl_string) > 0
        flat_input_string = " ".join(tokenized_nl_string)
        entities = []

        # See if any input strings are in our input sequence, and add the
        # corresponding output strings if so.
        for entry, values in self.entity_dict.items():
            in_middle = " " + entry + " " in flat_input_string

            leftspace = " " + entry
            at_end = leftspace in flat_input_string and flat_input_string.endswith(
                leftspace)

            rightspace = entry + " "
            at_beginning = rightspace in flat_input_string and flat_input_string.startswith(
                rightspace)
            if in_middle or at_end or at_beginning:
                for out_string in values:
                    entities.append(out_string)

        # Also add any integers in the input string (these aren't in the entity)
        # dict.
        for token in tokenized_nl_string:
            if token.isnumeric():
                entities.append(token)

        return entities
