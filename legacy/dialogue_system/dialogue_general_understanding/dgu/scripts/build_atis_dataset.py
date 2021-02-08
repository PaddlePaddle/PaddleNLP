# -*- coding: utf-8 -*-
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved. 
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
"""build swda train dev test dataset"""

import json
import sys
import csv
import os
import io
import re


class ATIS(object):
    """
    nlu dataset atis data process
    """

    def __init__(self):
        """
        init instance
        """
        self.slot_id = 2
        self.slot_dict = {"PAD": 0, "O": 1}
        self.intent_id = 0
        self.intent_dict = dict()
        self.src_dir = "../../data/input/data/atis/source_data"
        self.out_slot_dir = "../../data/input/data/atis/atis_slot"
        self.out_intent_dir = "../../data/input/data/atis/atis_intent"
        self.map_tag_slot = "../../data/input/data/atis/atis_slot/map_tag_slot_id.txt"
        self.map_tag_intent = "../../data/input/data/atis/atis_intent/map_tag_intent_id.txt"

    def _load_file(self, data_type):
        """
        load dataset filename
        """
        slot_stat = os.path.exists(self.out_slot_dir)
        if not slot_stat:
            os.makedirs(self.out_slot_dir)
        intent_stat = os.path.exists(self.out_intent_dir)
        if not intent_stat:
            os.makedirs(self.out_intent_dir)
        src_examples = []
        json_file = os.path.join(self.src_dir, "%s.json" % data_type)
        load_f = io.open(json_file, 'r', encoding="utf8")
        json_dict = json.load(load_f)
        examples = json_dict['rasa_nlu_data']['common_examples']
        for example in examples:
            text = example.get('text')
            intent = example.get('intent')
            entities = example.get('entities')
            src_examples.append((text, intent, entities))
        return src_examples

    def _parser_intent_data(self, examples, data_type):
        """
        parser intent dataset
        """
        out_filename = "%s/%s.txt" % (self.out_intent_dir, data_type)
        fw = io.open(out_filename, 'w', encoding="utf8")
        for example in examples:
            if example[1] not in self.intent_dict:
                self.intent_dict[example[1]] = self.intent_id
                self.intent_id += 1
            fw.write(u"%s\t%s\n" %
                     (self.intent_dict[example[1]], example[0].lower()))

        fw = io.open(self.map_tag_intent, 'w', encoding="utf8")
        for tag in self.intent_dict:
            fw.write(u"%s\t%s\n" % (tag, self.intent_dict[tag]))

    def _parser_slot_data(self, examples, data_type):
        """
        parser slot dataset
        """
        out_filename = "%s/%s.txt" % (self.out_slot_dir, data_type)
        fw = io.open(out_filename, 'w', encoding="utf8")
        for example in examples:
            tags = []
            text = example[0]
            entities = example[2]
            if not entities:
                tags = [str(self.slot_dict['O'])] * len(text.strip().split())
                continue
            for i in range(len(entities)):
                enty = entities[i]
                start = enty['start']
                value_num = len(enty['value'].split())
                tags_slot = []
                for j in range(value_num):
                    if j == 0:
                        bround_tag = "B"
                    else:
                        bround_tag = "I"
                    tag = "%s-%s" % (bround_tag, enty['entity'])
                    if tag not in self.slot_dict:
                        self.slot_dict[tag] = self.slot_id
                        self.slot_id += 1
                    tags_slot.append(str(self.slot_dict[tag]))
                if i == 0:
                    if start not in [0, 1]:
                        prefix_num = len(text[:start].strip().split())
                        tags.extend([str(self.slot_dict['O'])] * prefix_num)
                    tags.extend(tags_slot)
                else:
                    prefix_num = len(text[entities[i - 1]['end']:start].strip()
                                     .split())
                    tags.extend([str(self.slot_dict['O'])] * prefix_num)
                    tags.extend(tags_slot)
            if entities[-1]['end'] < len(text):
                suffix_num = len(text[entities[-1]['end']:].strip().split())
                tags.extend([str(self.slot_dict['O'])] * suffix_num)
            fw.write(u"%s\t%s\n" %
                     (text.encode('utf8'), " ".join(tags).encode('utf8')))

        fw = io.open(self.map_tag_slot, 'w', encoding="utf8")
        for slot in self.slot_dict:
            fw.write(u"%s\t%s\n" % (slot, self.slot_dict[slot]))

    def get_train_dataset(self):
        """
        parser train dataset and print train.txt
        """
        train_examples = self._load_file("train")
        self._parser_intent_data(train_examples, "train")
        self._parser_slot_data(train_examples, "train")

    def get_test_dataset(self):
        """
        parser test dataset and print test.txt
        """
        test_examples = self._load_file("test")
        self._parser_intent_data(test_examples, "test")
        self._parser_slot_data(test_examples, "test")

    def main(self):
        """
        run data process
        """
        self.get_train_dataset()
        self.get_test_dataset()


if __name__ == "__main__":
    atis_inst = ATIS()
    atis_inst.main()
