# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import math
import paddle
import paddle.nn as nn

from components import HandshakingKernel


class HandshakingTaggingScheme(object):

    def __init__(self, rel2id, max_seq_len, entity_type2id):
        super().__init__()
        self.rel2id = rel2id
        self.id2rel = {ind: rel for rel, ind in rel2id.items()}

        self.separator = "\u2E80"
        self.link_types = {
            "SH2OH",  # subject head to object head
            "OH2SH",  # object head to subject head
            "ST2OT",  # subject tail to object tail
            "OT2ST",  # object tail to subject tail
        }
        self.tags = {
            self.separator.join([rel, lt])
            for rel in self.rel2id.keys() for lt in self.link_types
        }

        self.ent2id = entity_type2id
        self.id2ent = {ind: ent for ent, ind in self.ent2id.items()}
        self.tags |= {
            self.separator.join([ent, "EH2ET"])
            for ent in self.ent2id.keys()
        }  # EH2ET: entity head to entity tail

        self.tags = sorted(self.tags)

        self.tag2id = {t: idx for idx, t in enumerate(self.tags)}
        self.id2tag = {idx: t for t, idx in self.tag2id.items()}
        self.matrix_size = max_seq_len

        # map
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.shaking_idx2matrix_idx = [
            (ind, end_ind) for ind in range(self.matrix_size)
            for end_ind in list(range(self.matrix_size))[ind:]
        ]

        self.matrix_idx2shaking_idx = [[0 for i in range(self.matrix_size)]
                                       for j in range(self.matrix_size)]
        for shaking_ind, matrix_ind in enumerate(self.shaking_idx2matrix_idx):
            self.matrix_idx2shaking_idx[matrix_ind[0]][
                matrix_ind[1]] = shaking_ind

    def get_tag_size(self):
        return len(self.tag2id)

    def get_spots(self, sample):
        '''
        matrix_spots: [(tok_pos1, tok_pos2, tag_id), ]
        '''
        matrix_spots = []
        spot_memory_set = set()

        def add_spot(spot):
            memory = "{},{},{}".format(*spot)
            if memory not in spot_memory_set:
                matrix_spots.append(spot)
                spot_memory_set.add(memory)


#         # if entity_list exist, need to distinguish entity types
#         if self.ent2id is not None and "entity_list" in sample:

        for ent in sample["entity_list"]:
            add_spot((ent["tok_span"][0], ent["tok_span"][1] - 1,
                      self.tag2id[self.separator.join([ent["type"], "EH2ET"])]))

        for rel in sample["relation_list"]:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]
            rel = rel["predicate"]
            #             if self.ent2id is None: # set all entities to default type
            #                 add_spot((subj_tok_span[0], subj_tok_span[1] - 1, self.tag2id[self.separator.join(["DEFAULT", "EH2ET"])]))
            #                 add_spot((obj_tok_span[0], obj_tok_span[1] - 1, self.tag2id[self.separator.join(["DEFAULT", "EH2ET"])]))
            if subj_tok_span[0] <= obj_tok_span[0]:
                add_spot((subj_tok_span[0], obj_tok_span[0],
                          self.tag2id[self.separator.join([rel, "SH2OH"])]))
            else:
                add_spot((obj_tok_span[0], subj_tok_span[0],
                          self.tag2id[self.separator.join([rel, "OH2SH"])]))
            if subj_tok_span[1] <= obj_tok_span[1]:
                add_spot((subj_tok_span[1] - 1, obj_tok_span[1] - 1,
                          self.tag2id[self.separator.join([rel, "ST2OT"])]))
            else:
                add_spot((obj_tok_span[1] - 1, subj_tok_span[1] - 1,
                          self.tag2id[self.separator.join([rel, "OT2ST"])]))
        return matrix_spots

    def spots2shaking_tag(self, spots):
        '''
        convert spots to matrix tag
        spots: [(start_ind, end_ind, tag_id), ]
        return: 
            shaking_tag: (shaking_seq_len, tag_size)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_tag = paddle.zeros(shaking_seq_len, len(self.tag2id)).long()
        for sp in spots:
            shaking_idx = self.matrix_idx2shaking_idx[sp[0]][sp[1]]
            shaking_tag[shaking_idx][sp[2]] = 1
        return shaking_tag

    def spots2shaking_tag4batch(self, batch_spots):
        '''
        batch_spots: a batch of spots, [spots1, spots2, ...]
            spots: [(start_ind, end_ind, tag_id), ]
        return: 
            batch_shaking_tag: (batch_size, shaking_seq_len, tag_size)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_tag = paddle.zeros(len(batch_spots), shaking_seq_len,
                                         len(self.tag2id)).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_idx = self.matrix_idx2shaking_idx[sp[0]][sp[1]]
                batch_shaking_tag[batch_id][shaking_idx][sp[2]] = 1
        return batch_shaking_tag

    def get_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, tag_id)
        spots: [(start_ind, end_ind, tag_id), ]
        '''
        spots = []
        nonzero_points = paddle.nonzero(shaking_tag, as_tuple=False)
        for point in nonzero_points:
            shaking_idx, tag_idx = point[0].item(), point[1].item()
            pos1, pos2 = self.shaking_idx2matrix_idx[shaking_idx]
            spot = (pos1, pos2, tag_idx)
            spots.append(spot)
        return spots

    def decode_rel(self,
                   text,
                   shaking_tag,
                   tok2char_span,
                   tok_offset=0,
                   char_offset=0):
        '''
        shaking_tag: (shaking_seq_len, tag_id_num)
        '''
        rel_list = []
        matrix_spots = self.get_spots_fr_shaking_tag(shaking_tag)

        # entity
        head_ind2entities = {}
        ent_list = []
        for sp in matrix_spots:
            tag = self.id2tag[sp[2]]
            ent_type, link_type = tag.split(self.separator)
            if link_type != "EH2ET" or sp[0] > sp[
                    1]:  # for an entity, the start position can not be larger than the end pos.
                continue

            char_span_list = tok2char_span[sp[0]:sp[1] + 1]
            char_sp = [char_span_list[0][0], char_span_list[-1][1]]
            ent_text = text[char_sp[0]:char_sp[1]]
            entity = {
                "type": ent_type,
                "text": ent_text,
                "tok_span": [sp[0], sp[1] + 1],
                "char_span": char_sp,
            }
            head_key = str(sp[0])  # take ent_head_pos as the key to entity list
            if head_key not in head_ind2entities:
                head_ind2entities[head_key] = []
            head_ind2entities[head_key].append(entity)
            ent_list.append(entity)

        # tail link
        tail_link_memory_set = set()
        for sp in matrix_spots:
            tag = self.id2tag[sp[2]]
            rel, link_type = tag.split(self.separator)
            if link_type == "ST2OT":
                tail_link_memory = self.separator.join(
                    [rel, str(sp[0]), str(sp[1])])
                tail_link_memory_set.add(tail_link_memory)
            elif link_type == "OT2ST":
                tail_link_memory = self.separator.join(
                    [rel, str(sp[1]), str(sp[0])])
                tail_link_memory_set.add(tail_link_memory)

        # head link
        for sp in matrix_spots:
            tag = self.id2tag[sp[2]]
            rel, link_type = tag.split(self.separator)

            if link_type == "SH2OH":
                subj_head_key, obj_head_key = str(sp[0]), str(sp[1])
            elif link_type == "OH2SH":
                subj_head_key, obj_head_key = str(sp[1]), str(sp[0])
            else:
                continue

            if subj_head_key not in head_ind2entities or obj_head_key not in head_ind2entities:
                # no entity start with subj_head_key and obj_head_key
                continue

            subj_list = head_ind2entities[
                subj_head_key]  # all entities start with this subject head
            obj_list = head_ind2entities[
                obj_head_key]  # all entities start with this object head

            # go over all subj-obj pair to check whether the tail link exists
            for subj in subj_list:
                for obj in obj_list:
                    tail_link_memory = self.separator.join([
                        rel,
                        str(subj["tok_span"][1] - 1),
                        str(obj["tok_span"][1] - 1)
                    ])
                    if tail_link_memory not in tail_link_memory_set:
                        # no such relation
                        continue
                    rel_list.append({
                        "subject":
                        subj["text"],
                        "object":
                        obj["text"],
                        "subj_tok_span": [
                            subj["tok_span"][0] + tok_offset,
                            subj["tok_span"][1] + tok_offset
                        ],
                        "obj_tok_span": [
                            obj["tok_span"][0] + tok_offset,
                            obj["tok_span"][1] + tok_offset
                        ],
                        "subj_char_span": [
                            subj["char_span"][0] + char_offset,
                            subj["char_span"][1] + char_offset
                        ],
                        "obj_char_span": [
                            obj["char_span"][0] + char_offset,
                            obj["char_span"][1] + char_offset
                        ],
                        "predicate":
                        rel,
                    })
            # recover the positons in the original text
            for ent in ent_list:
                ent["char_span"] = [
                    ent["char_span"][0] + char_offset,
                    ent["char_span"][1] + char_offset
                ]
                ent["tok_span"] = [
                    ent["tok_span"][0] + tok_offset,
                    ent["tok_span"][1] + tok_offset
                ]

        return rel_list, ent_list

    def trans2ee(self, rel_list, ent_list):
        sepatator = "_"  # \u2E80
        trigger_set, arg_iden_set, arg_class_set = set(), set(), set()
        trigger_offset2vote = {}
        trigger_offset2trigger_text = {}
        trigger_offset2trigger_char_span = {}
        # get candidate trigger types from relation
        for rel in rel_list:
            trigger_offset = rel["obj_tok_span"]
            trigger_offset_str = "{},{}".format(trigger_offset[0],
                                                trigger_offset[1])
            trigger_offset2trigger_text[trigger_offset_str] = rel["object"]
            trigger_offset2trigger_char_span[trigger_offset_str] = rel[
                "obj_char_span"]
            _, event_type = rel["predicate"].split(sepatator)

            if trigger_offset_str not in trigger_offset2vote:
                trigger_offset2vote[trigger_offset_str] = {}
            trigger_offset2vote[trigger_offset_str][
                event_type] = trigger_offset2vote[trigger_offset_str].get(
                    event_type, 0) + 1

        # get candidate trigger types from entity types
        for ent in ent_list:
            t1, t2 = ent["type"].split(sepatator)
            assert t1 == "Trigger" or t1 == "Argument"
            if t1 == "Trigger":  # trigger
                event_type = t2
                trigger_span = ent["tok_span"]
                trigger_offset_str = "{},{}".format(trigger_span[0],
                                                    trigger_span[1])
                trigger_offset2trigger_text[trigger_offset_str] = ent["text"]
                trigger_offset2trigger_char_span[trigger_offset_str] = ent[
                    "char_span"]
                if trigger_offset_str not in trigger_offset2vote:
                    trigger_offset2vote[trigger_offset_str] = {}
                trigger_offset2vote[trigger_offset_str][
                    event_type] = trigger_offset2vote[trigger_offset_str].get(
                        event_type,
                        0) + 1.1  # if even, entity type makes the call

        # voting
        tirigger_offset2event = {}
        for trigger_offet_str, event_type2score in trigger_offset2vote.items():
            event_type = sorted(event_type2score.items(),
                                key=lambda x: x[1],
                                reverse=True)[0][0]
            tirigger_offset2event[
                trigger_offet_str] = event_type  # final event type

        # generate event list
        trigger_offset2arguments = {}
        for rel in rel_list:
            trigger_offset = rel["obj_tok_span"]
            argument_role, event_type = rel["predicate"].split(sepatator)
            trigger_offset_str = "{},{}".format(trigger_offset[0],
                                                trigger_offset[1])
            if tirigger_offset2event[
                    trigger_offset_str] != event_type:  # filter false relations
                #                 set_trace()
                continue

            # append arguments
            if trigger_offset_str not in trigger_offset2arguments:
                trigger_offset2arguments[trigger_offset_str] = []
            trigger_offset2arguments[trigger_offset_str].append({
                "text":
                rel["subject"],
                "type":
                argument_role,
                "char_span":
                rel["subj_char_span"],
                "tok_span":
                rel["subj_tok_span"],
            })
        event_list = []
        for trigger_offset_str, event_type in tirigger_offset2event.items():
            arguments = trigger_offset2arguments[trigger_offset_str] if trigger_offset_str in trigger_offset2arguments else []
            event = {
                "trigger":
                trigger_offset2trigger_text[trigger_offset_str],
                "trigger_char_span":
                trigger_offset2trigger_char_span[trigger_offset_str],
                "trigger_tok_span":
                trigger_offset_str.split(","),
                "trigger_type":
                event_type,
                "argument_list":
                arguments,
            }
            event_list.append(event)
        return event_list


class TPLinkerPlus(nn.Layer):

    def __init__(self,
                 encoder,
                 tag_size,
                 shaking_type="cln_plus",
                 inner_enc_type="lstm",
                 tok_pair_sample_rate=1):
        super().__init__()
        self.encoder = encoder
        self.tok_pair_sample_rate = tok_pair_sample_rate

        shaking_hidden_size = encoder.config["hidden_size"]

        self.fc = nn.Linear(shaking_hidden_size, tag_size)

        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(shaking_hidden_size,
                                                    shaking_type,
                                                    inner_enc_type)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
        context_outputs = self.encoder(input_ids, attention_mask,
                                       token_type_ids)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        seq_len = last_hidden_state.size()[1]
        # shaking_hiddens: (batch_size, shaking_seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(last_hidden_state)

        sampled_tok_pair_indices = None
        if self.training:
            # randomly sample segments of token pairs
            shaking_seq_len = shaking_hiddens.size()[1]
            segment_len = int(shaking_seq_len * self.tok_pair_sample_rate)
            seg_num = math.ceil(shaking_seq_len // segment_len)
            start_ind = paddle.randint(seg_num, []) * segment_len
            end_ind = min(start_ind + segment_len, shaking_seq_len)
            # sampled_tok_pair_indices: (batch_size, ~segment_len) ~end_ind - start_ind <= segment_len
            sampled_tok_pair_indices = paddle.arange(
                start_ind, end_ind)[None, :].repeat(shaking_hiddens.size()[0],
                                                    1)
            sampled_tok_pair_indices = sampled_tok_pair_indices.to(
                shaking_hiddens.device)

            # sampled_tok_pair_indices will tell model what token pairs should be fed into fcs
            # shaking_hiddens: (batch_size, ~segment_len, hidden_size)
            shaking_hiddens = shaking_hiddens.gather(
                1, sampled_tok_pair_indices[:, :, None].repeat(
                    1, 1,
                    shaking_hiddens.size()[-1]))

        # outputs: (batch_size, segment_len, tag_size) or (batch_size, shaking_seq_len, tag_size)
        outputs = self.fc(shaking_hiddens)

        return outputs, sampled_tok_pair_indices
