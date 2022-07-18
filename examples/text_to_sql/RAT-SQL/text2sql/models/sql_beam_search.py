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

import operator

import attr
import networkx as nx

from text2sql.models.beam_search import Hypothesis
from text2sql.models.sql_decoder.decoder import TreeState
from text2sql.dataproc.sql_preproc_v2 import get_field_presence_info


@attr.s
class Hypothesis4Filtering(Hypothesis):
    column_history = attr.ib(factory=list)
    table_history = attr.ib(factory=list)
    key_column_history = attr.ib(factory=list)


def beam_search_with_heuristics(model,
                                inputs,
                                beam_size,
                                max_steps,
                                from_cond=True):
    """
    Find the valid FROM clasue with beam search
    """
    orig_inputs = inputs['orig_inputs'][0]
    #inference_state, next_choices = model.inference(inputs, orig_inputs.db)
    inference_state, next_choices = model(inputs,
                                          db=orig_inputs.db,
                                          is_train=False)
    beam = [Hypothesis4Filtering(inference_state, next_choices)]

    cached_finished_seqs = []  # cache filtered trajectories
    beam_prefix = beam
    while True:
        # search prefixes with beam search
        prefixes2fill_from = []
        for step in range(max_steps):
            if len(prefixes2fill_from) >= beam_size:
                break

            candidates = []
            for hyp in beam_prefix:
                if hyp.inference_state.cur_item.state == hyp.inference_state.State.CHILDREN_APPLY \
                        and hyp.inference_state.cur_item.node_type == "from":
                    # only from not fill, save it and process in the following code
                    prefixes2fill_from.append(hyp)
                else:
                    candidates += [(hyp, choice, choice_score.numpy().item(),
                                    hyp.score + choice_score.numpy().item())
                                   for choice, choice_score in hyp.next_choices]
            candidates.sort(key=operator.itemgetter(3), reverse=True)
            candidates = candidates[:beam_size - len(prefixes2fill_from)]

            # Create the new hypotheses from the expansions
            beam_prefix = []
            for hyp, choice, choice_score, cum_score in candidates:
                inference_state = hyp.inference_state.clone()

                # cache column choice
                column_history = hyp.column_history[:]
                if hyp.inference_state.cur_item.state == hyp.inference_state.State.POINTER_APPLY and \
                        hyp.inference_state.cur_item.node_type == "column":
                    column_history = column_history + [choice]

                # get next choices
                next_choices = inference_state.step(choice)
                assert next_choices is not None
                beam_prefix.append(
                    Hypothesis4Filtering(inference_state, next_choices,
                                         cum_score,
                                         hyp.choice_history + [choice],
                                         hyp.score_history + [choice_score],
                                         column_history))

        prefixes2fill_from.sort(key=operator.attrgetter('score'), reverse=True)
        # assert len(prefixes) == beam_size

        # emuerating
        beam_from = prefixes2fill_from
        max_size = 6
        unfiltered_finished = []
        prefixes_unfinished = []
        for step in range(max_steps):
            if len(unfiltered_finished) + len(prefixes_unfinished) > max_size:
                break

            candidates = []
            for hyp in beam_from:
                if step > 0 and hyp.inference_state.cur_item.state == hyp.inference_state.State.CHILDREN_APPLY \
                        and hyp.inference_state.cur_item.node_type == "from":
                    prefixes_unfinished.append(hyp)
                else:
                    candidates += [(hyp, choice, choice_score.numpy().item(),
                                    hyp.score + choice_score.numpy().item())
                                   for choice, choice_score in hyp.next_choices]
            candidates.sort(key=operator.itemgetter(3), reverse=True)
            candidates = candidates[:max_size - len(prefixes_unfinished)]

            beam_from = []
            for hyp, choice, choice_score, cum_score in candidates:
                inference_state = hyp.inference_state.clone()

                # cache table choice
                table_history = hyp.table_history[:]
                key_column_history = hyp.key_column_history[:]
                if hyp.inference_state.cur_item.state == hyp.inference_state.State.POINTER_APPLY:
                    if hyp.inference_state.cur_item.node_type == "table":
                        table_history = table_history + [choice]
                    elif hyp.inference_state.cur_item.node_type == "column":
                        key_column_history = key_column_history + [choice]

                next_choices = inference_state.step(choice)
                if next_choices is None:
                    unfiltered_finished.append(
                        Hypothesis4Filtering(inference_state, None, cum_score,
                                             hyp.choice_history + [choice],
                                             hyp.score_history + [choice_score],
                                             hyp.column_history, table_history,
                                             key_column_history))
                else:
                    beam_from.append(
                        Hypothesis4Filtering(inference_state, next_choices,
                                             cum_score,
                                             hyp.choice_history + [choice],
                                             hyp.score_history + [choice_score],
                                             hyp.column_history, table_history,
                                             key_column_history))
        # [END] for step in range(max_steps)

        unfiltered_finished.sort(key=operator.attrgetter('score'), reverse=True)

        # filtering
        filtered_finished = []
        for hyp in unfiltered_finished:
            mentioned_column_ids = set(hyp.column_history)
            mentioned_key_column_ids = set(hyp.key_column_history)
            mentioned_table_ids = set(hyp.table_history)

            # duplicate tables
            if len(mentioned_table_ids) != len(hyp.table_history):
                continue

            # the foreign key should be correctly used
            # NOTE: the new version does not predict conditions in FROM clause anymore
            if from_cond:
                covered_tables = set()
                must_include_key_columns = set()
                candidate_table_ids = sorted(mentioned_table_ids)
                start_table_id = candidate_table_ids[0]
                for table_id in candidate_table_ids[1:]:
                    if table_id in covered_tables:
                        continue
                    try:
                        path = nx.shortest_path(
                            orig_inputs.db.foreign_key_graph,
                            source=start_table_id,
                            target=table_id)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        covered_tables.add(table_id)
                        continue

                    for source_table_id, target_table_id in zip(path, path[1:]):
                        if target_table_id in covered_tables:
                            continue
                        if target_table_id not in mentioned_table_ids:
                            continue
                        col1, col2 = orig_inputs.db.foreign_key_graph[
                            source_table_id][target_table_id]['columns']
                        must_include_key_columns.add(col1)
                        must_include_key_columns.add(col2)
                if not must_include_key_columns == mentioned_key_column_ids:
                    continue

            # tables whose columns are mentioned should also exist
            must_table_ids = set()
            for col in mentioned_column_ids:
                tab_ = orig_inputs.db.columns[col].table
                if tab_ is not None:
                    must_table_ids.add(tab_.id)
            if not must_table_ids.issubset(mentioned_table_ids):
                continue

            filtered_finished.append(hyp)

        filtered_finished.sort(key=operator.attrgetter('score'), reverse=True)
        # filtered.sort(key=lambda x: x.score / len(x.choice_history), reverse=True)
        prefixes_unfinished.sort(key=operator.attrgetter('score'), reverse=True)
        # new_prefixes.sort(key=lambda x: x.score / len(x.choice_history), reverse=True)

        prefixes_, filtered_ = merge_beams(prefixes_unfinished,
                                           filtered_finished, beam_size)

        if filtered_:
            cached_finished_seqs = cached_finished_seqs + filtered_
            cached_finished_seqs.sort(key=operator.attrgetter('score'),
                                      reverse=True)

        if prefixes_ and len(prefixes_[0].choice_history) < 200:
            beam_prefix = prefixes_
            for hyp in beam_prefix:
                hyp.table_history = []
                hyp.column_history = []
                hyp.key_column_history = []
        elif cached_finished_seqs:
            return cached_finished_seqs[:beam_size]
        else:
            return unfiltered_finished[:beam_size]


# merge sorted beam
def merge_beams(beam_1, beam_2, beam_size):
    if len(beam_1) == 0 or len(beam_2) == 0:
        return beam_1, beam_2

    annoated_beam_1 = [("beam_1", b) for b in beam_1]
    annoated_beam_2 = [("beam_2", b) for b in beam_2]
    merged_beams = annoated_beam_1 + annoated_beam_2
    merged_beams.sort(key=lambda x: x[1].score, reverse=True)

    ret_beam_1 = []
    ret_beam_2 = []
    for label, beam in merged_beams[:beam_size]:
        if label == "beam_1":
            ret_beam_1.append(beam)
        else:
            assert label == "beam_2"
            ret_beam_2.append(beam)
    return ret_beam_1, ret_beam_2


def beam_search_with_oracle_column(model, inputs, preproc_item, beam_size,
                                   max_steps):
    inference_state, next_choices = model.begin_inference(inputs, preproc_item)
    beam = [Hypothesis(inference_state, next_choices)]
    finished = []
    assert beam_size == 1

    # identify all the cols mentioned in the gold sql
    root_node = preproc_item[1].tree

    col_queue = list(
        reversed([
            val
            for val in model.decoder.ast_wrapper.find_all_descendants_of_type(
                root_node, "column")
        ]))
    tab_queue = list(
        reversed([
            val
            for val in model.decoder.ast_wrapper.find_all_descendants_of_type(
                root_node, "table")
        ]))
    col_queue_copy = col_queue[:]
    tab_queue_copy = tab_queue[:]

    predict_counter = 0

    for step in range(max_steps):
        # Check if all beams are finished
        if len(finished) == beam_size:
            break

        # hijack the next choice using the gold col
        assert len(beam) == 1
        hyp = beam[0]
        if hyp.inference_state.cur_item.state == hyp.inference_state.State.POINTER_APPLY:
            if hyp.inference_state.cur_item.node_type == "column" \
                    and len(col_queue) > 0:
                gold_col = col_queue[0]

                flag = False
                for _choice in hyp.next_choices:
                    if _choice[0] == gold_col:
                        flag = True
                        hyp.next_choices = [_choice]
                        col_queue = col_queue[1:]
                        break
                assert flag
            elif hyp.inference_state.cur_item.node_type == "table" \
                    and len(tab_queue) > 0:
                gold_tab = tab_queue[0]

                flag = False
                for _choice in hyp.next_choices:
                    if _choice[0] == gold_tab:
                        flag = True
                        hyp.next_choices = [_choice]
                        tab_queue = tab_queue[1:]
                        break
                assert flag

        # for debug
        if hyp.inference_state.cur_item.state == hyp.inference_state.State.POINTER_APPLY:
            predict_counter += 1

        # For each hypothesis, get possible expansions
        # Score each expansion
        candidates = []
        for hyp in beam:
            candidates += [(hyp, choice, choice_score.numpy().item(),
                            hyp.score + choice_score.numpy().item())
                           for choice, choice_score in hyp.next_choices]

        # Keep the top K expansions
        candidates.sort(key=operator.itemgetter(3), reverse=True)
        candidates = candidates[:beam_size - len(finished)]

        # Create the new hypotheses from the expansions
        beam = []
        for hyp, choice, choice_score, cum_score in candidates:
            inference_state = hyp.inference_state.clone()
            next_choices = inference_state.step(choice)
            if next_choices is None:
                finished.append(
                    Hypothesis(inference_state, None, cum_score,
                               hyp.choice_history + [choice],
                               hyp.score_history + [choice_score]))
            else:
                beam.append(
                    Hypothesis(inference_state, next_choices, cum_score,
                               hyp.choice_history + [choice],
                               hyp.score_history + [choice_score]))
    if (len(col_queue_copy) + len(tab_queue_copy)) != predict_counter:
        # print("The number of column/tables are not matched")
        pass
    finished.sort(key=operator.attrgetter('score'), reverse=True)
    return finished


def beam_search_with_oracle_sketch(model, inputs, preproc_item, beam_size,
                                   max_steps):
    inference_state, next_choices = model.begin_inference(inputs, preproc_item)
    hyp = Hypothesis(inference_state, next_choices)

    parsed = model.decoder.preproc.grammar.parse(inputs['orig_inputs'][0].code,
                                                 "val")
    if not parsed:
        return []

    queue = [
        TreeState(
            node=preproc_item[1].tree,
            parent_field_type=model.decoder.preproc.grammar.root_type,
        )
    ]

    while queue:
        item = queue.pop()
        node = item.node
        parent_field_type = item.parent_field_type

        if isinstance(node, (list, tuple)):
            node_type = parent_field_type + '*'
            rule = (node_type, len(node))
            if rule not in model.decoder.rules_index:
                return []
            rule_idx = model.decoder.rules_index[rule]
            assert inference_state.cur_item.state == inference_state.State.LIST_LENGTH_APPLY

            if model.decoder.preproc.use_seq_elem_rules and \
                    parent_field_type in model.decoder.ast_wrapper.sum_types:
                parent_field_type += '_seq_elem'

            for i, elem in reversed(list(enumerate(node))):
                queue.append(
                    TreeState(node=elem, parent_field_type=parent_field_type))

            hyp = Hypothesis(inference_state, None, 0,
                             hyp.choice_history + [rule_idx],
                             hyp.score_history + [0])
            continue

        if parent_field_type in model.decoder.preproc.grammar.pointers:
            assert inference_state.cur_item.state == inference_state.State.POINTER_APPLY
            # best_choice = max(next_choices, key=lambda x: x[1])
            # node = best_choice[0] # override the node

            assert isinstance(node, int)
            next_choices = inference_state.step(node)
            hyp = Hypothesis(inference_state, None, 0,
                             hyp.choice_history + [node],
                             hyp.score_history + [0])
            continue

        if parent_field_type in model.decoder.ast_wrapper.primitive_types:
            field_value_split = model.decoder.preproc.grammar.tokenize_field_value(
                node) + ['<EOS>']

            for token in field_value_split:
                next_choices = inference_state.step(token)
            hyp = Hypothesis(inference_state, None, 0,
                             hyp.choice_history + field_value_split,
                             hyp.score_history + [0])
            continue

        type_info = model.decoder.ast_wrapper.singular_types[node['_type']]

        if parent_field_type in model.decoder.preproc.sum_type_constructors:
            # ApplyRule, like expr -> Call
            rule = (parent_field_type, type_info.name)
            rule_idx = model.decoder.rules_index[rule]
            assert inference_state.cur_item.state == inference_state.State.SUM_TYPE_APPLY
            extra_rules = [
                model.decoder.rules_index[parent_field_type, extra_type]
                for extra_type in node.get('_extra_types', [])
            ]
            next_choices = inference_state.step(rule_idx, extra_rules)

            hyp = Hypothesis(inference_state, None, 0,
                             hyp.choice_history + [rule_idx],
                             hyp.score_history + [0])

        if type_info.fields:
            # ApplyRule, like Call -> expr[func] expr*[args] keyword*[keywords]
            # Figure out which rule needs to be applied
            present = get_field_presence_info(model.decoder.ast_wrapper, node,
                                              type_info.fields)
            rule = (node['_type'], tuple(present))
            rule_idx = model.decoder.rules_index[rule]
            next_choices = inference_state.step(rule_idx)

            hyp = Hypothesis(inference_state, None, 0,
                             hyp.choice_history + [rule_idx],
                             hyp.score_history + [0])

        # reversed so that we perform a DFS in left-to-right order
        for field_info in reversed(type_info.fields):
            if field_info.name not in node:
                continue
            queue.append(
                TreeState(node=node[field_info.name],
                          parent_field_type=field_info.type))

    return [hyp]
