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

import enum

import attr
import pyrsistent

from text2sql.utils import nn_utils
from text2sql.dataproc import vocab


@attr.s
class TreeState:
    node = attr.ib()
    parent_field_type = attr.ib()


class TreeTraversal:

    class Handler:
        handlers = {}

        @classmethod
        def register_handler(cls, func_type):
            """register_handler"""
            if func_type in cls.handlers:
                raise RuntimeError(f"{func_type} handler is already registered")

            def inner_func(func):
                """inner_func"""
                cls.handlers[func_type] = func.__name__
                return func

            return inner_func

    @attr.s(frozen=True)
    class QueueItem:
        item_id = attr.ib()
        state = attr.ib()
        node_type = attr.ib()
        parent_action_emb = attr.ib()
        parent_h = attr.ib()  # parent hidden state
        parent_field_name = attr.ib()

        def to_str(self):
            """to_str"""
            return f"<state: {self.state}, node_type: {self.node_type}, parent_field_name: {self.parent_field_name}>"

    class State(enum.Enum):
        """State"""
        SUM_TYPE_INQUIRE = 0
        SUM_TYPE_APPLY = 1
        CHILDREN_INQUIRE = 2
        CHILDREN_APPLY = 3
        LIST_LENGTH_INQUIRE = 4
        LIST_LENGTH_APPLY = 5
        GEN_TOKEN = 6
        POINTER_INQUIRE = 7
        POINTER_APPLY = 8
        NODE_FINISHED = 9

    def __init__(self, model, desc_enc):
        """__init__"""
        if model is None:
            return

        self.model = model
        self.desc_enc = desc_enc

        ##< TODO: model.state_update.set_dropout_masks(batch_size=1)
        self.recurrent_state = nn_utils.lstm_init(None,
                                                  self.model.recurrent_size, 1)
        self.prev_action_emb = model.zero_rule_emb

        root_type = model.preproc.grammar.root_type
        if root_type in model.preproc.ast_wrapper.sum_types:
            initial_state = TreeTraversal.State.SUM_TYPE_INQUIRE
        else:
            initial_state = TreeTraversal.State.CHILDREN_INQUIRE

        self.queue = pyrsistent.pvector()
        self.cur_item = TreeTraversal.QueueItem(
            item_id=0,
            state=initial_state,
            node_type=root_type,
            parent_action_emb=self.model.zero_rule_emb,
            parent_h=self.model.zero_recurrent_emb,
            parent_field_name=None,
        )
        self.next_item_id = 1

        self.update_prev_action_emb = TreeTraversal._update_prev_action_emb_apply_rule

    def clone(self):
        """clone"""
        other = self.__class__(None, None)
        other.model = self.model
        other.desc_enc = self.desc_enc
        other.recurrent_state = self.recurrent_state
        other.prev_action_emb = self.prev_action_emb
        other.queue = self.queue
        other.cur_item = self.cur_item
        other.next_item_id = self.next_item_id
        other.actions = self.actions
        other.update_prev_action_emb = self.update_prev_action_emb
        return other

    def step(self, last_choice, extra_choice_info=None, attention_offset=None):
        """step"""
        while True:
            ##<debug print('2' * 10, 'in one step...', 'last_choice:', last_choice, 'cur_item:', self.cur_item.to_str())
            self.update_using_last_choice(last_choice, extra_choice_info,
                                          attention_offset)

            handler_name = TreeTraversal.Handler.handlers[self.cur_item.state]
            handler = getattr(self, handler_name)
            choices, continued = handler(last_choice)
            if continued:
                last_choice = choices
                continue
            else:
                ##<debug print('>>>', '2' * 10, 'one step finished.')
                return choices

    def update_using_last_choice(self, last_choice, extra_choice_info,
                                 attention_offset):
        """update_using_last_choice"""
        if last_choice is None:
            return
        self.update_prev_action_emb(self, last_choice, extra_choice_info)

    @classmethod
    def _update_prev_action_emb_apply_rule(cls, self, last_choice,
                                           extra_choice_info):
        """_update_prev_action_emb_apply_rule"""
        rule_idx = self.model._tensor([last_choice])
        self.prev_action_emb = self.model.rule_embedding(rule_idx)

    @classmethod
    def _update_prev_action_emb_gen_token(cls, self, last_choice,
                                          extra_choice_info):
        """_update_prev_action_emb_gen_token"""
        # token_idx shape: batch (=1), LongTensor
        token_idx = self.model._index(self.model.terminal_vocab, last_choice)
        # action_emb shape: batch (=1) x emb_size
        self.prev_action_emb = self.model.terminal_embedding(token_idx)

    @classmethod
    def _update_prev_action_emb_pointer(cls, self, last_choice,
                                        extra_choice_info):
        """_update_prev_action_emb_pointer"""
        # TODO batching
        self.prev_action_emb = self.model.pointer_action_emb_proj[
            self.cur_item.node_type](self.desc_enc.pointer_memories[
                self.cur_item.node_type][:, last_choice])

    def pop(self):
        """pop"""
        ##<debug print('v' * 66)
        ##<debug print([x.to_str() for x in self.queue])
        if self.queue:
            self.cur_item = self.queue[-1]
            self.queue = self.queue.delete(-1)
            return True
        return False

    @Handler.register_handler(State.SUM_TYPE_INQUIRE)
    def process_sum_inquire(self, last_choice):
        """process_sum_inquire"""
        # 1. ApplyRule, like expr -> Call
        # a. Ask which one to choose
        output, self.recurrent_state, rule_logits = self.model.apply_rule(
            self.cur_item.node_type,
            self.recurrent_state,
            self.prev_action_emb,
            self.cur_item.parent_h,
            self.cur_item.parent_action_emb,
            self.desc_enc,
        )
        self.cur_item = attr.evolve(self.cur_item,
                                    state=TreeTraversal.State.SUM_TYPE_APPLY,
                                    parent_h=output)

        self.update_prev_action_emb = TreeTraversal._update_prev_action_emb_apply_rule
        choices = self.rule_choice(self.cur_item.node_type, rule_logits)
        return choices, False

    @Handler.register_handler(State.SUM_TYPE_APPLY)
    def process_sum_apply(self, last_choice):
        """process_sum_apply"""
        # b. Add action, prepare for #2
        sum_type, singular_type = self.model.preproc.all_rules[last_choice]
        assert sum_type == self.cur_item.node_type

        self.cur_item = attr.evolve(
            self.cur_item,
            node_type=singular_type,
            parent_action_emb=self.prev_action_emb,
            state=TreeTraversal.State.CHILDREN_INQUIRE,
        )
        return None, True

    @Handler.register_handler(State.CHILDREN_INQUIRE)
    def process_children_inquire(self, last_choice):
        """process_children_inquire"""
        # 2. ApplyRule, like Call -> expr[func] expr*[args] keyword*[keywords]
        # Check if we have no children
        type_info = self.model.ast_wrapper.singular_types[
            self.cur_item.node_type]
        if not type_info.fields:
            if self.pop():
                last_choice = None
                return last_choice, True
            else:
                return None, False

        # a. Ask about presence
        output, self.recurrent_state, rule_logits = self.model.apply_rule(
            self.cur_item.node_type,
            self.recurrent_state,
            self.prev_action_emb,
            self.cur_item.parent_h,
            self.cur_item.parent_action_emb,
            self.desc_enc,
        )
        self.cur_item = attr.evolve(self.cur_item,
                                    state=TreeTraversal.State.CHILDREN_APPLY,
                                    parent_h=output)

        self.update_prev_action_emb = TreeTraversal._update_prev_action_emb_apply_rule
        choices = self.rule_choice(self.cur_item.node_type, rule_logits)
        return choices, False

    @Handler.register_handler(State.CHILDREN_APPLY)
    def process_children_apply(self, last_choice):
        """process_children_apply"""
        # b. Create the children
        node_type, children_presence = self.model.preproc.all_rules[last_choice]
        assert node_type == self.cur_item.node_type

        self.queue = self.queue.append(
            TreeTraversal.QueueItem(
                item_id=self.cur_item.item_id,
                state=TreeTraversal.State.NODE_FINISHED,
                node_type=None,
                parent_action_emb=None,
                parent_h=None,
                parent_field_name=None,
            ))
        for field_info, present in reversed(
                list(
                    zip(
                        self.model.ast_wrapper.singular_types[node_type].fields,
                        children_presence,
                    ))):
            if not present:
                continue

            # seq field: LIST_LENGTH_INQUIRE x
            # sum type: SUM_TYPE_INQUIRE x
            # product type:
            #   no children: not possible
            #   children: CHILDREN_INQUIRE
            # constructor type: not possible x
            # builtin type: GEN_TOKEN x
            child_type = field_type = field_info.type
            if field_info.seq:
                child_state = TreeTraversal.State.LIST_LENGTH_INQUIRE
            elif field_type in self.model.ast_wrapper.sum_types:
                child_state = TreeTraversal.State.SUM_TYPE_INQUIRE
            elif field_type in self.model.ast_wrapper.product_types:
                assert self.model.ast_wrapper.product_types[field_type].fields
                child_state = TreeTraversal.State.CHILDREN_INQUIRE
            elif field_type in self.model.preproc.grammar.pointers:
                child_state = TreeTraversal.State.POINTER_INQUIRE
            elif field_type in self.model.ast_wrapper.primitive_types:
                child_state = TreeTraversal.State.GEN_TOKEN
                child_type = present
            else:
                raise ValueError(f"Unable to handle field type {field_type}")

            self.queue = self.queue.append(
                TreeTraversal.QueueItem(
                    item_id=self.next_item_id,
                    state=child_state,
                    node_type=child_type,
                    parent_action_emb=self.prev_action_emb,
                    parent_h=self.cur_item.parent_h,
                    parent_field_name=field_info.name,
                ))
            self.next_item_id += 1

        # pop queue 的最后一个元素，并赋值给 self.cur_item
        advanced = self.pop()
        assert advanced
        last_choice = None
        return last_choice, True

    @Handler.register_handler(State.LIST_LENGTH_INQUIRE)
    def process_list_length_inquire(self, last_choice):
        """process_list_length_inquire"""
        list_type = self.cur_item.node_type + "*"
        output, self.recurrent_state, rule_logits = self.model.apply_rule(
            list_type,
            self.recurrent_state,
            self.prev_action_emb,
            self.cur_item.parent_h,
            self.cur_item.parent_action_emb,
            self.desc_enc,
        )
        self.cur_item = attr.evolve(self.cur_item,
                                    state=TreeTraversal.State.LIST_LENGTH_APPLY,
                                    parent_h=output)

        self.update_prev_action_emb = TreeTraversal._update_prev_action_emb_apply_rule
        choices = self.rule_choice(list_type, rule_logits)
        return choices, False

    @Handler.register_handler(State.LIST_LENGTH_APPLY)
    def process_list_length_apply(self, last_choice):
        """process_list_length_apply"""
        list_type, num_children = self.model.preproc.all_rules[last_choice]
        elem_type = self.cur_item.node_type
        assert list_type == elem_type + "*"

        child_node_type = elem_type
        if elem_type in self.model.ast_wrapper.sum_types:
            child_state = TreeTraversal.State.SUM_TYPE_INQUIRE
            if self.model.preproc.use_seq_elem_rules:
                child_node_type = elem_type + "_seq_elem"
        elif elem_type in self.model.ast_wrapper.product_types:
            child_state = TreeTraversal.State.CHILDREN_INQUIRE
        elif elem_type == "identifier":
            child_state = TreeTraversal.State.GEN_TOKEN
            child_node_type = "str"
        elif elem_type in self.model.ast_wrapper.primitive_types:
            raise ValueError("sequential builtin types not supported")
        else:
            raise ValueError(f"Unable to handle seq field type {elem_type}")

        for i in range(num_children):
            self.queue = self.queue.append(
                TreeTraversal.QueueItem(
                    item_id=self.next_item_id,
                    state=child_state,
                    node_type=child_node_type,
                    parent_action_emb=self.prev_action_emb,
                    parent_h=self.cur_item.parent_h,
                    parent_field_name=self.cur_item.parent_field_name,
                ))
            self.next_item_id += 1

        advanced = self.pop()
        assert advanced
        last_choice = None
        return last_choice, True

    @Handler.register_handler(State.GEN_TOKEN)
    def process_gen_token(self, last_choice):
        """process_gen_token"""
        if last_choice == vocab.EOS:
            if self.pop():
                last_choice = None
                return last_choice, True
            else:
                return None, False

        self.recurrent_state, output, gen_logodds = self.model.gen_token(
            self.cur_item.node_type,
            self.recurrent_state,
            self.prev_action_emb,
            self.cur_item.parent_h,
            self.cur_item.parent_action_emb,
            self.desc_enc,
        )
        self.update_prev_action_emb = TreeTraversal._update_prev_action_emb_gen_token
        choices = self.token_choice(output, gen_logodds)
        return choices, False

    @Handler.register_handler(State.POINTER_INQUIRE)
    def process_pointer_inquire(self, last_choice):
        """process_pointer_inquire"""
        # a. Ask which one to choose
        output, self.recurrent_state, logits, attention_logits = self.model.compute_pointer_with_align(
            self.cur_item.node_type,
            self.recurrent_state,
            self.prev_action_emb,
            self.cur_item.parent_h,
            self.cur_item.parent_action_emb,
            self.desc_enc,
        )
        self.cur_item = attr.evolve(self.cur_item,
                                    state=TreeTraversal.State.POINTER_APPLY,
                                    parent_h=output)

        self.update_prev_action_emb = TreeTraversal._update_prev_action_emb_pointer
        choices = self.pointer_choice(self.cur_item.node_type, logits,
                                      attention_logits)
        return choices, False

    @Handler.register_handler(State.POINTER_APPLY)
    def process_pointer_apply(self, last_choice):
        """process_pointer_apply"""
        if self.pop():
            last_choice = None
            return last_choice, True
        else:
            return None, False

    @Handler.register_handler(State.NODE_FINISHED)
    def process_node_finished(self, last_choice):
        """process_node_finished"""
        if self.pop():
            last_choice = None
            return last_choice, True
        else:
            return None, False

    def rule_choice(self, node_type, rule_logits):
        """rule_choice"""
        raise NotImplementedError

    def token_choice(self, output, gen_logodds):
        """token_choice"""
        raise NotImplementedError

    def pointer_choice(self, node_type, logits, attention_logits):
        """pointer_choice"""
        raise NotImplementedError
