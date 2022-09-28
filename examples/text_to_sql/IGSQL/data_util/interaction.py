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
""" Contains the class for an interaction in ATIS. """

import paddle

from . import anonymization as anon
from . import sql_util
from .snippets import expand_snippets
from .utterance import Utterance, OUTPUT_KEY, ANON_INPUT_KEY


class Schema:

    def __init__(self, table_schema, simple=False):
        if simple:
            self.helper1(table_schema)
        else:
            self.helper2(table_schema)

    def helper1(self, table_schema):
        self.table_schema = table_schema
        column_names = table_schema['column_names']
        column_names_original = table_schema['column_names_original']
        table_names = table_schema['table_names']
        table_names_original = table_schema['table_names_original']
        assert len(column_names) == len(column_names_original) and len(
            table_names) == len(table_names_original)

        column_keep_index = []

        self.column_names_surface_form = []
        self.column_names_surface_form_to_id = {}
        for i, (table_id, column_name) in enumerate(column_names_original):
            column_name_surface_form = column_name
            column_name_surface_form = column_name_surface_form.lower()
            if column_name_surface_form not in self.column_names_surface_form_to_id:
                self.column_names_surface_form.append(column_name_surface_form)
                self.column_names_surface_form_to_id[
                    column_name_surface_form] = len(
                        self.column_names_surface_form) - 1
                column_keep_index.append(i)

        column_keep_index_2 = []
        for i, table_name in enumerate(table_names_original):
            column_name_surface_form = table_name.lower()
            if column_name_surface_form not in self.column_names_surface_form_to_id:
                self.column_names_surface_form.append(column_name_surface_form)
                self.column_names_surface_form_to_id[
                    column_name_surface_form] = len(
                        self.column_names_surface_form) - 1
                column_keep_index_2.append(i)

        self.column_names_embedder_input = []
        self.column_names_embedder_input_to_id = {}
        for i, (table_id, column_name) in enumerate(column_names):
            column_name_embedder_input = column_name
            if i in column_keep_index:
                self.column_names_embedder_input.append(
                    column_name_embedder_input)
                self.column_names_embedder_input_to_id[
                    column_name_embedder_input] = len(
                        self.column_names_embedder_input) - 1

        for i, table_name in enumerate(table_names):
            column_name_embedder_input = table_name
            if i in column_keep_index_2:
                self.column_names_embedder_input.append(
                    column_name_embedder_input)
                self.column_names_embedder_input_to_id[
                    column_name_embedder_input] = len(
                        self.column_names_embedder_input) - 1

        max_id_1 = max(v
                       for k, v in self.column_names_surface_form_to_id.items())
        max_id_2 = max(
            v for k, v in self.column_names_embedder_input_to_id.items())
        assert (len(self.column_names_surface_form) - 1) == max_id_2 == max_id_1

        self.num_col = len(self.column_names_surface_form)

    def helper2(self, table_schema):
        self.table_schema = table_schema
        column_names = table_schema['column_names']
        column_names_original = table_schema['column_names_original']
        table_names = table_schema['table_names']
        table_names_original = table_schema['table_names_original']
        assert len(column_names) == len(column_names_original) and len(
            table_names) == len(table_names_original)

        column_keep_index = []

        self.column_names_surface_form = []
        self.column_names_surface_form_to_id = {}
        for i, (table_id, column_name) in enumerate(column_names_original):
            if table_id >= 0:
                table_name = table_names_original[table_id]
                column_name_surface_form = '{}.{}'.format(
                    table_name, column_name)
            else:
                column_name_surface_form = column_name
            column_name_surface_form = column_name_surface_form.lower()
            if column_name_surface_form not in self.column_names_surface_form_to_id:
                self.column_names_surface_form.append(column_name_surface_form)
                self.column_names_surface_form_to_id[
                    column_name_surface_form] = len(
                        self.column_names_surface_form) - 1
                column_keep_index.append(i)

        start_i = len(self.column_names_surface_form_to_id)
        for i, table_name in enumerate(table_names_original):
            column_name_surface_form = '{}.*'.format(table_name.lower())
            self.column_names_surface_form.append(column_name_surface_form)
            self.column_names_surface_form_to_id[
                column_name_surface_form] = i + start_i

        self.column_names_embedder_input = []
        self.column_names_embedder_input_to_id = {}
        for i, (table_id, column_name) in enumerate(column_names):
            if table_id >= 0:
                table_name = table_names[table_id]
                column_name_embedder_input = table_name + ' . ' + column_name
            else:
                column_name_embedder_input = column_name
            if i in column_keep_index:
                self.column_names_embedder_input.append(
                    column_name_embedder_input)
                self.column_names_embedder_input_to_id[
                    column_name_embedder_input] = len(
                        self.column_names_embedder_input) - 1

        start_i = len(self.column_names_embedder_input_to_id)
        for i, table_name in enumerate(table_names):
            column_name_embedder_input = table_name + ' . *'
            self.column_names_embedder_input.append(column_name_embedder_input)
            self.column_names_embedder_input_to_id[
                column_name_embedder_input] = i + start_i

        assert len(self.column_names_surface_form) == len(
            self.column_names_surface_form_to_id) == len(
                self.column_names_embedder_input) == len(
                    self.column_names_embedder_input_to_id)

        max_id_1 = max(v
                       for k, v in self.column_names_surface_form_to_id.items())
        max_id_2 = max(
            v for k, v in self.column_names_embedder_input_to_id.items())
        assert (len(self.column_names_surface_form) - 1) == max_id_2 == max_id_1

        self.num_col = len(self.column_names_surface_form)

    def __len__(self):
        return self.num_col

    def in_vocabulary(self, column_name, surface_form=False):
        if surface_form:
            return column_name in self.column_names_surface_form_to_id
        else:
            return column_name in self.column_names_embedder_input_to_id

    def column_name_embedder_bow(self,
                                 column_name,
                                 surface_form=False,
                                 column_name_token_embedder=None):
        assert self.in_vocabulary(column_name, surface_form)
        if surface_form:
            column_name_id = self.column_names_surface_form_to_id[column_name]
            column_name_embedder_input = self.column_names_embedder_input[
                column_name_id]
        else:
            column_name_embedder_input = column_name

        column_name_embeddings = [
            column_name_token_embedder(token)
            for token in column_name_embedder_input.split()
        ]
        column_name_embeddings = paddle.stack(column_name_embeddings, axis=0)
        return paddle.mean(column_name_embeddings, axis=0)

    def set_column_name_embeddings(self, column_name_embeddings):
        self.column_name_embeddings = column_name_embeddings
        assert len(self.column_name_embeddings) == self.num_col

    def column_name_embedder(self, column_name, surface_form=False):
        assert self.in_vocabulary(column_name, surface_form)
        if surface_form:
            column_name_id = self.column_names_surface_form_to_id[column_name]
        else:
            column_name_id = self.column_names_embedder_input_to_id[column_name]

        return self.column_name_embeddings[column_name_id]


class Interaction:

    def __init__(self, utterances, schema, snippets, anon_tok_to_ent,
                 identifier, params):
        self.utterances = utterances
        self.schema = schema
        self.snippets = snippets
        self.anon_tok_to_ent = anon_tok_to_ent
        self.identifier = identifier

        # Ensure that each utterance's input and output sequences, when remapped
        # without anonymization or snippets, are the same as the original
        # version.
        for i, utterance in enumerate(self.utterances):
            deanon_input = self.deanonymize(utterance.input_seq_to_use,
                                            ANON_INPUT_KEY)
            assert deanon_input == utterance.original_input_seq, "Anonymized sequence [" \
                + " ".join(utterance.input_seq_to_use) + "] is not the same as [" \
                + " ".join(utterance.original_input_seq) + "] when deanonymized (is [" \
                + " ".join(deanon_input) + "] instead)"
            desnippet_gold = self.expand_snippets(utterance.gold_query_to_use)
            deanon_gold = self.deanonymize(desnippet_gold, OUTPUT_KEY)
            assert deanon_gold == utterance.original_gold_query, \
                "Anonymized and/or snippet'd query " \
                + " ".join(utterance.gold_query_to_use) + " is not the same as " \
                + " ".join(utterance.original_gold_query)

    def __str__(self):
        string = "Utterances:\n"
        for utterance in self.utterances:
            string += str(utterance) + "\n"
        string += "Anonymization dictionary:\n"
        for ent_tok, deanon in self.anon_tok_to_ent.items():
            string += ent_tok + "\t" + str(deanon) + "\n"

        return string

    def __len__(self):
        return len(self.utterances)

    def deanonymize(self, sequence, key):
        """ Deanonymizes a predicted query or an input utterance.

        Args:
            sequence (`list`): The sequence to deanonymize.
            key (`str`): The key in the anonymization table, e.g. NL or SQL.
        """
        return anon.deanonymize(sequence, self.anon_tok_to_ent, key)

    def expand_snippets(self, sequence):
        """ Expands snippets for a sequence.

        Args:
            sequence (`list`): A SQL query.

        """
        return expand_snippets(sequence, self.snippets)

    def input_seqs(self):
        in_seqs = []
        for utterance in self.utterances:
            in_seqs.append(utterance.input_seq_to_use)
        return in_seqs

    def output_seqs(self):
        out_seqs = []
        for utterance in self.utterances:
            out_seqs.append(utterance.gold_query_to_use)
        return out_seqs


def load_function(parameters, nl_to_sql_dict, anonymizer, database_schema=None):

    def fn(interaction_example):
        keep = False

        raw_utterances = interaction_example["interaction"]

        if "database_id" in interaction_example:
            database_id = interaction_example["database_id"]
            interaction_id = interaction_example["interaction_id"]
            identifier = database_id + '/' + str(interaction_id)
        else:
            identifier = interaction_example["id"]

        schema = None
        if database_schema:
            if 'removefrom' not in parameters.data_directory:
                schema = Schema(database_schema[database_id], simple=True)
            else:
                schema = Schema(database_schema[database_id])

        snippet_bank = []

        utterance_examples = []

        anon_tok_to_ent = {}

        for utterance in raw_utterances:
            available_snippets = [
                snippet for snippet in snippet_bank if snippet.index <= 1
            ]

            proc_utterance = Utterance(utterance, available_snippets,
                                       nl_to_sql_dict, parameters,
                                       anon_tok_to_ent, anonymizer)
            keep_utterance = proc_utterance.keep

            if schema:
                assert keep_utterance

            if keep_utterance:
                keep = True
                utterance_examples.append(proc_utterance)

                # Update the snippet bank, and age each snippet in it.
                if parameters.use_snippets:
                    if 'atis' in parameters.data_directory:
                        snippets = sql_util.get_subtrees(
                            proc_utterance.anonymized_gold_query,
                            proc_utterance.available_snippets)
                    else:
                        snippets = sql_util.get_subtrees_simple(
                            proc_utterance.anonymized_gold_query,
                            proc_utterance.available_snippets)

                    for snippet in snippets:
                        snippet.assign_id(len(snippet_bank))
                        snippet_bank.append(snippet)

                for snippet in snippet_bank:
                    snippet.increase_age()

        interaction = Interaction(utterance_examples, schema, snippet_bank,
                                  anon_tok_to_ent, identifier, parameters)

        return interaction, keep

    return fn
