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

from . import snippets as snip
from . import sql_util
from . import vocabulary as vocab


class UtteranceItem():

    def __init__(self, interaction, index):
        self.interaction = interaction
        self.utterance_index = index

    def __str__(self):
        return str(self.interaction.utterances[self.utterance_index])

    def histories(self, maximum):
        if maximum > 0:
            history_seqs = []
            for utterance in self.interaction.utterances[:self.utterance_index]:
                history_seqs.append(utterance.input_seq_to_use)

            if len(history_seqs) > maximum:
                history_seqs = history_seqs[-maximum:]

            return history_seqs
        return []

    def input_sequence(self):
        return self.interaction.utterances[
            self.utterance_index].input_seq_to_use

    def previous_query(self):
        if self.utterance_index == 0:
            return []
        return self.interaction.utterances[self.utterance_index -
                                           1].anonymized_gold_query

    def anonymized_gold_query(self):
        return self.interaction.utterances[
            self.utterance_index].anonymized_gold_query

    def snippets(self):
        return self.interaction.utterances[
            self.utterance_index].available_snippets

    def original_gold_query(self):
        return self.interaction.utterances[
            self.utterance_index].original_gold_query

    def contained_entities(self):
        return self.interaction.utterances[
            self.utterance_index].contained_entities

    def original_gold_queries(self):
        return [
            q[0] for q in self.interaction.utterances[
                self.utterance_index].all_gold_queries
        ]

    def gold_tables(self):
        return [
            q[1] for q in self.interaction.utterances[
                self.utterance_index].all_gold_queries
        ]

    def gold_query(self):
        return self.interaction.utterances[
            self.utterance_index].gold_query_to_use + [vocab.EOS_TOK]

    def gold_edit_sequence(self):
        return self.interaction.utterances[
            self.utterance_index].gold_edit_sequence

    def gold_table(self):
        return self.interaction.utterances[
            self.utterance_index].gold_sql_results

    def all_snippets(self):
        return self.interaction.snippets

    def within_limits(self,
                      max_input_length=float('inf'),
                      max_output_length=float('inf')):
        return self.interaction.utterances[self.utterance_index].length_valid(
            max_input_length, max_output_length)

    def expand_snippets(self, sequence):
        # Remove the EOS
        if sequence[-1] == vocab.EOS_TOK:
            sequence = sequence[:-1]

        # First remove the snippets
        no_snippets_sequence = self.interaction.expand_snippets(sequence)
        no_snippets_sequence = sql_util.fix_parentheses(no_snippets_sequence)
        return no_snippets_sequence

    def flatten_sequence(self, sequence):
        # Remove the EOS
        if sequence[-1] == vocab.EOS_TOK:
            sequence = sequence[:-1]

        # First remove the snippets
        no_snippets_sequence = self.interaction.expand_snippets(sequence)

        # Deanonymize
        deanon_sequence = self.interaction.deanonymize(no_snippets_sequence,
                                                       "sql")
        return deanon_sequence


class UtteranceBatch():

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def start(self):
        self.index = 0

    def next(self):
        item = self.items[self.index]
        self.index += 1
        return item

    def done(self):
        return self.index >= len(self.items)


class PredUtteranceItem():

    def __init__(self, input_sequence, interaction_item, previous_query, index,
                 available_snippets):
        self.input_seq_to_use = input_sequence
        self.interaction_item = interaction_item
        self.index = index
        self.available_snippets = available_snippets
        self.prev_pred_query = previous_query

    def input_sequence(self):
        return self.input_seq_to_use

    def histories(self, maximum):
        if maximum == 0:
            return histories
        histories = []
        for utterance in self.interaction_item.processed_utterances[:self.
                                                                    index]:
            histories.append(utterance.input_sequence())
        if len(histories) > maximum:
            histories = histories[-maximum:]
        return histories

    def snippets(self):
        return self.available_snippets

    def previous_query(self):
        return self.prev_pred_query

    def flatten_sequence(self, sequence):
        return self.interaction_item.flatten_sequence(sequence)

    def remove_snippets(self, sequence):
        return sql_util.fix_parentheses(
            self.interaction_item.expand_snippets(sequence))

    def set_predicted_query(self, query):
        self.anonymized_pred_query = query


class InteractionItem():

    def __init__(self,
                 interaction,
                 max_input_length=float('inf'),
                 max_output_length=float('inf'),
                 nl_to_sql_dict={},
                 maximum_length=float('inf')):
        if maximum_length != float('inf'):
            self.interaction = copy.deepcopy(interaction)
            self.interaction.utterances = self.interaction.utterances[:
                                                                      maximum_length]
        else:
            self.interaction = interaction
        self.processed_utterances = []
        self.snippet_bank = []
        self.identifier = self.interaction.identifier

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.nl_to_sql_dict = nl_to_sql_dict

        self.index = 0

    def __len__(self):
        return len(self.interaction)

    def __str__(self):
        s = "Utterances, gold queries, and predictions:\n"
        for i, utterance in enumerate(self.interaction.utterances):
            s += " ".join(utterance.input_seq_to_use) + "\n"
            pred_utterance = self.processed_utterances[i]
            s += " ".join(pred_utterance.gold_query()) + "\n"
            s += " ".join(pred_utterance.anonymized_query()) + "\n"
            s += "\n"
        s += "Snippets:\n"
        for snippet in self.snippet_bank:
            s += str(snippet) + "\n"

        return s

    def start_interaction(self):
        assert len(self.snippet_bank) == 0
        assert len(self.processed_utterances) == 0
        assert self.index == 0

    def next_utterance(self):
        utterance = self.interaction.utterances[self.index]
        self.index += 1

        available_snippets = self.available_snippets(snippet_keep_age=1)

        return PredUtteranceItem(
            utterance.input_seq_to_use, self,
            self.processed_utterances[-1].anonymized_pred_query
            if len(self.processed_utterances) > 0 else [], self.index - 1,
            available_snippets)

    def done(self):
        return len(self.processed_utterances) == len(self.interaction)

    def finish(self):
        self.snippet_bank = []
        self.processed_utterances = []
        self.index = 0

    def utterance_within_limits(self, utterance_item):
        return utterance_item.within_limits(self.max_input_length,
                                            self.max_output_length)

    def available_snippets(self, snippet_keep_age):
        return [
            snippet for snippet in self.snippet_bank
            if snippet.index <= snippet_keep_age
        ]

    def gold_utterances(self):
        utterances = []
        for i, utterance in enumerate(self.interaction.utterances):
            utterances.append(UtteranceItem(self.interaction, i))
        return utterances

    def get_schema(self):
        return self.interaction.schema

    def add_utterance(self,
                      utterance,
                      predicted_sequence,
                      snippets=None,
                      previous_snippets=[],
                      simple=False):
        if not snippets:
            self.add_snippets(predicted_sequence,
                              previous_snippets=previous_snippets,
                              simple=simple)
        else:
            for snippet in snippets:
                snippet.assign_id(len(self.snippet_bank))
                self.snippet_bank.append(snippet)

            for snippet in self.snippet_bank:
                snippet.increase_age()
        self.processed_utterances.append(utterance)

    def add_snippets(self, sequence, previous_snippets=[], simple=False):
        if sequence:
            if simple:
                snippets = sql_util.get_subtrees_simple(
                    sequence, oldsnippets=previous_snippets)
            else:
                snippets = sql_util.get_subtrees(sequence,
                                                 oldsnippets=previous_snippets)
            for snippet in snippets:
                snippet.assign_id(len(self.snippet_bank))
                self.snippet_bank.append(snippet)

        for snippet in self.snippet_bank:
            snippet.increase_age()

    def expand_snippets(self, sequence):
        return sql_util.fix_parentheses(
            snip.expand_snippets(sequence, self.snippet_bank))

    def remove_snippets(self, sequence):
        if sequence[-1] == vocab.EOS_TOK:
            sequence = sequence[:-1]

        no_snippets_sequence = self.expand_snippets(sequence)
        no_snippets_sequence = sql_util.fix_parentheses(no_snippets_sequence)
        return no_snippets_sequence

    def flatten_sequence(self, sequence, gold_snippets=False):
        if sequence[-1] == vocab.EOS_TOK:
            sequence = sequence[:-1]

        if gold_snippets:
            no_snippets_sequence = self.interaction.expand_snippets(sequence)
        else:
            no_snippets_sequence = self.expand_snippets(sequence)
        no_snippets_sequence = sql_util.fix_parentheses(no_snippets_sequence)

        deanon_sequence = self.interaction.deanonymize(no_snippets_sequence,
                                                       "sql")
        return deanon_sequence

    def gold_query(self, index):
        return self.interaction.utterances[index].gold_query_to_use + [
            vocab.EOS_TOK
        ]

    def original_gold_query(self, index):
        return self.interaction.utterances[index].original_gold_query

    def gold_table(self, index):
        return self.interaction.utterances[index].gold_sql_results


class InteractionBatch():

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def start(self):
        self.timestep = 0
        self.current_interactions = []

    def get_next_utterance_batch(self, snippet_keep_age, use_gold=False):
        items = []
        self.current_interactions = []
        for interaction in self.items:
            if self.timestep < len(interaction):
                utterance_item = interaction.original_utterances(
                    snippet_keep_age, use_gold)[self.timestep]
                self.current_interactions.append(interaction)
                items.append(utterance_item)

        self.timestep += 1
        return UtteranceBatch(items)

    def done(self):
        finished = True
        for interaction in self.items:
            if self.timestep < len(interaction):
                finished = False
                return finished
        return finished
