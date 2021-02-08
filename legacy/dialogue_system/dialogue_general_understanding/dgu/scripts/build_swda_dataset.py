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

import sys
import csv
import os
import io
import re

import commonlib


class SWDA(object):
    """
    dialogue act dataset swda data process
    """

    def __init__(self):
        """
        init instance
        """
        self.tag_id = 0
        self.map_tag_dict = dict()
        self.out_dir = "../../data/input/data/swda"
        self.data_list = "./conf/swda.conf"
        self.map_tag = "../../data/input/data/swda/map_tag_id.txt"
        self.src_dir = "../../data/input/data/swda/source_data/swda"
        self._load_file()

    def _load_file(self):
        """
        load dataset filename
        """
        self.data_dict = commonlib.load_dict(self.data_list)
        self.file_dict = {}
        child_dir = commonlib.get_dir_list(self.src_dir)
        for chd in child_dir:
            file_list, file_path = commonlib.get_file_list(chd)
            for i in range(len(file_list)):
                name = file_list[i]
                keyword = "sw%s" % name.split('.')[0].split('_')[-1]
                self.file_dict[keyword] = file_path[i]

    def _parser_dataset(self, data_type):
        """
        parser train dev test dataset
        """
        out_filename = "%s/%s.txt" % (self.out_dir, data_type)
        fw = io.open(out_filename, 'w', encoding='utf8')
        for name in self.data_dict[data_type]:
            file_path = self.file_dict[name]
            fr = io.open(file_path, 'r', encoding="utf8")
            idx = 0
            row = csv.reader(fr, delimiter=',')
            for r in row:
                if idx == 0:
                    idx += 1
                    continue
                out = self._parser_utterence(r)
                fw.write(u"%s\n" % out)

    def _clean_text(self, text):
        """
        text cleaning for dialogue act dataset
        """
        if text.startswith('<') and text.endswith('>.'):
            return text
        if "[" in text or "]" in text:
            stat = True
        else:
            stat = False
        group = re.findall("\[.*?\+.*?\]", text)
        while group and stat:
            for elem in group:
                elem_src = elem
                elem = re.sub('\+', '', elem.lstrip('[').rstrip(']'))
                text = text.replace(elem_src, elem)
            if "[" in text or "]" in text:
                stat = True
            else:
                stat = False
            group = re.findall("\[.*?\+.*?\]", text)
        if "{" in text or "}" in text:
            stat = True
        else:
            stat = False
        group = re.findall("{[A-Z].*?}", text)
        while group and stat:
            child_group = re.findall("{[A-Z]*(.*?)}", text)
            for i in range(len(group)):
                text = text.replace(group[i], child_group[i])
            if "{" in text or "}" in text:
                stat = True
            else:
                stat = False
            group = re.findall("{[A-Z].*?}", text)
        if "(" in text or ")" in text:
            stat = True
        else:
            stat = False
        group = re.findall("\(\(.*?\)\)", text)
        while group and stat:
            for elem in group:
                if elem:
                    elem_clean = re.sub("\(|\)", "", elem)
                    text = text.replace(elem, elem_clean)
                else:
                    text = text.replace(elem, "mumblex")
            if "(" in text or ")" in text:
                stat = True
            else:
                stat = False
            group = re.findall("\(\((.*?)\)\)", text)

        group = re.findall("\<.*?\>", text)
        if group:
            for elem in group:
                text = text.replace(elem, "")

        text = re.sub(r" \'s", "\'s", text)
        text = re.sub(r" n\'t", "n\'t", text)
        text = re.sub(r" \'t", "\'t", text)
        text = re.sub(" +", " ", text)
        text = text.rstrip('\/').strip().strip('-')
        text = re.sub("\[|\]|\+|\>|\<|\{|\}", "", text)
        return text.strip().lower()

    def _map_tag(self, da_tag):
        """
        map tag to 42 classes
        """
        curr_da_tags = []
        curr_das = re.split(r"\s*[,;]\s*", da_tag)
        for curr_da in curr_das:
            if curr_da == "qy_d" or curr_da == "qw^d" or curr_da == "b^m":
                pass
            elif curr_da == "nn^e":
                curr_da = "ng"
            elif curr_da == "ny^e":
                curr_da = "na"
            else:
                curr_da = re.sub(r'(.)\^.*', r'\1', curr_da)
                curr_da = re.sub(r'[\(\)@*]', '', curr_da)
                tag = curr_da
                if tag in ('qr', 'qy'):
                    tag = 'qy'
                elif tag in ('fe', 'ba'):
                    tag = 'ba'
                elif tag in ('oo', 'co', 'cc'):
                    tag = 'oo_co_cc'
                elif tag in ('fx', 'sv'):
                    tag = 'sv'
                elif tag in ('aap', 'am'):
                    tag = 'aap_am'
                elif tag in ('arp', 'nd'):
                    tag = 'arp_nd'
                elif tag in ('fo', 'o', 'fw', '"', 'by', 'bc'):
                    tag = 'fo_o_fw_"_by_bc'
                curr_da = tag
            curr_da_tags.append(curr_da)
        if curr_da_tags[0] not in self.map_tag_dict:
            self.map_tag_dict[curr_da_tags[0]] = self.tag_id
            self.tag_id += 1
        return self.map_tag_dict[curr_da_tags[0]]

    def _parser_utterence(self, line):
        """
        parser one turn dialogue
        """
        conversation_no = line[2]
        act_tag = line[4]
        caller = line[5]
        text = line[8]
        text = self._clean_text(text)
        act_tag = self._map_tag(act_tag)

        out = "%s\t%s\t%s\t%s" % (conversation_no, act_tag, caller, text)
        return out

    def get_train_dataset(self):
        """
        parser train dataset and print train.txt
        """
        self._parser_dataset("train")

    def get_dev_dataset(self):
        """
        parser dev dataset and print dev.txt
        """
        self._parser_dataset("dev")

    def get_test_dataset(self):
        """
        parser test dataset and print test.txt
        """
        self._parser_dataset("test")

    def get_labels(self):
        """
        get tag and map ids file
        """
        fw = io.open(self.map_tag, 'w', encoding='utf8')
        for elem in self.map_tag_dict:
            fw.write(u"%s\t%s\n" % (elem, self.map_tag_dict[elem]))

    def main(self):
        """
        run data process
        """
        self.get_train_dataset()
        self.get_dev_dataset()
        self.get_test_dataset()
        self.get_labels()


if __name__ == "__main__":
    swda_inst = SWDA()
    swda_inst.main()
