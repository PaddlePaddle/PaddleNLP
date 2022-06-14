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
""" Contains the Utterance class. """

from . import sql_util
from . import tokenizers

ANON_INPUT_KEY = "cleaned_nl"
OUTPUT_KEY = "sql"


class Utterance:

    def process_input_seq(self, anonymize, anonymizer, anon_tok_to_ent):
        assert not anon_tok_to_ent or anonymize
        assert not anonymize or anonymizer

        if anonymize:
            assert anonymizer

            self.input_seq_to_use = anonymizer.anonymize(
                self.original_input_seq,
                anon_tok_to_ent,
                ANON_INPUT_KEY,
                add_new_anon_toks=True)
        else:
            self.input_seq_to_use = self.original_input_seq

    def process_gold_seq(self, output_sequences, nl_to_sql_dict,
                         available_snippets, anonymize, anonymizer,
                         anon_tok_to_ent):
        # Get entities in the input sequence:
        #    anonymized entity types
        #    othe recognized entities (this includes "flight")
        entities_in_input = [[tok] for tok in self.input_seq_to_use
                             if tok in anon_tok_to_ent]
        entities_in_input.extend(
            nl_to_sql_dict.get_sql_entities(self.input_seq_to_use))

        # Get the shortest gold query (this is what we use to train)
        shortest_gold_and_results = min(output_sequences,
                                        key=lambda x: len(x[0]))

        # Tokenize and anonymize it if necessary.
        self.original_gold_query = shortest_gold_and_results[0]
        self.gold_sql_results = shortest_gold_and_results[1]

        self.contained_entities = entities_in_input

        # Keep track of all gold queries and the resulting tables so that we can
        # give credit if it predicts a different correct sequence.
        self.all_gold_queries = output_sequences

        self.anonymized_gold_query = self.original_gold_query
        if anonymize:
            self.anonymized_gold_query = anonymizer.anonymize(
                self.original_gold_query,
                anon_tok_to_ent,
                OUTPUT_KEY,
                add_new_anon_toks=False)

        # Add snippets to it.
        self.gold_query_to_use = sql_util.add_snippets_to_query(
            available_snippets, entities_in_input, self.anonymized_gold_query)

    def __init__(self,
                 example,
                 available_snippets,
                 nl_to_sql_dict,
                 params,
                 anon_tok_to_ent={},
                 anonymizer=None):
        # Get output and input sequences from the dictionary representation.
        output_sequences = example[OUTPUT_KEY]
        self.original_input_seq = tokenizers.nl_tokenize(
            example[params.input_key])
        self.available_snippets = available_snippets
        self.keep = False

        if len(output_sequences) > 0 and len(self.original_input_seq) > 0:
            # Only keep this example if there is at least one output sequence.
            self.keep = True
        if len(output_sequences) == 0 or len(self.original_input_seq) == 0:
            return

        # Process the input sequence
        self.process_input_seq(params.anonymize, anonymizer, anon_tok_to_ent)

        # Process the gold sequence
        self.process_gold_seq(output_sequences, nl_to_sql_dict,
                              self.available_snippets, params.anonymize,
                              anonymizer, anon_tok_to_ent)

    def __str__(self):
        string = "Original input: " + " ".join(self.original_input_seq) + "\n"
        string += "Modified input: " + " ".join(self.input_seq_to_use) + "\n"
        string += "Original output: " + " ".join(
            self.original_gold_query) + "\n"
        string += "Modified output: " + " ".join(self.gold_query_to_use) + "\n"
        string += "Snippets:\n"
        for snippet in self.available_snippets:
            string += str(snippet) + "\n"
        return string

    def length_valid(self, input_limit, output_limit):
        return (len(self.input_seq_to_use) < input_limit \
            and len(self.gold_query_to_use) < output_limit)
