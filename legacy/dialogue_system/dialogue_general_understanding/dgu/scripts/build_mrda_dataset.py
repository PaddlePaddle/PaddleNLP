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
"""build mrda train dev test dataset"""

import sys
import csv
import os
import io
import re

import commonlib


class MRDA(object):
    """
    dialogue act dataset mrda data process
    """

    def __init__(self):
        """
        init instance
        """
        self.tag_id = 0
        self.map_tag_dict = dict()
        self.out_dir = "../../data/input/data/mrda"
        self.data_list = "./conf/mrda.conf"
        self.map_tag = "../../data/input/data/mrda/map_tag_id.txt"
        self.voc_map_tag = "../../data/input/data/mrda/source_data/icsi_mrda+hs_corpus_050512/classmaps/map_01b_expanded_w_split"
        self.src_dir = "../../data/input/data/mrda/source_data/icsi_mrda+hs_corpus_050512/data"
        self._load_file()
        self.tag_dict = commonlib.load_voc(self.voc_map_tag)

    def _load_file(self):
        """
        load dataset filename
        """
        self.dadb_dict = {}
        self.trans_dict = {}
        self.data_dict = commonlib.load_dict(self.data_list)
        file_list, file_path = commonlib.get_file_list(self.src_dir)
        for i in range(len(file_list)):
            name = file_list[i]
            keyword = name.split('.')[0]
            if 'dadb' in name:
                self.dadb_dict[keyword] = file_path[i]
            if 'trans' in name:
                self.trans_dict[keyword] = file_path[i]

    def load_dadb(self, data_type):
        """
        load dadb dataset
        """
        dadb_dict = {}
        conv_id_list = []
        dadb_list = self.data_dict[data_type]
        for dadb_key in dadb_list:
            dadb_file = self.dadb_dict[dadb_key]
            fr = io.open(dadb_file, 'r', encoding="utf8")
            row = csv.reader(fr, delimiter=',')
            for line in row:
                elems = line
                conv_id = elems[2]
                conv_id_list.append(conv_id)
                if len(elems) != 14:
                    continue
                error_code = elems[3]
                da_tag = elems[-9]
                da_ori_tag = elems[-6]
                dadb_dict[conv_id] = (error_code, da_ori_tag, da_tag)
        return dadb_dict, conv_id_list

    def load_trans(self, data_type):
        """load trans data"""
        trans_dict = {}
        trans_list = self.data_dict[data_type]
        for trans_key in trans_list:
            trans_file = self.trans_dict[trans_key]
            fr = io.open(trans_file, 'r', encoding="utf8")
            row = csv.reader(fr, delimiter=',')
            for line in row:
                elems = line
                if len(elems) != 3:
                    continue
                conv_id = elems[0]
                text = elems[1]
                text_process = elems[2]
                trans_dict[conv_id] = (text, text_process)
        return trans_dict

    def _parser_dataset(self, data_type):
        """
        parser train dev test dataset
        """
        out_filename = "%s/%s.txt" % (self.out_dir, data_type)
        dadb_dict, conv_id_list = self.load_dadb(data_type)
        trans_dict = self.load_trans(data_type)
        fw = io.open(out_filename, 'w', encoding="utf8")
        for elem in conv_id_list:
            v_dadb = dadb_dict[elem]
            v_trans = trans_dict[elem]
            da_tag = v_dadb[2]
            if da_tag not in self.tag_dict:
                continue
            tag = self.tag_dict[da_tag]
            if tag == "Z":
                continue
            if tag not in self.map_tag_dict:
                self.map_tag_dict[tag] = self.tag_id
                self.tag_id += 1
            caller = elem.split('_')[0].split('-')[-1]
            conv_no = elem.split('_')[0].split('-')[0]
            out = "%s\t%s\t%s\t%s" % (conv_no, self.map_tag_dict[tag], caller,
                                      v_trans[0])
            fw.write(u"%s\n" % out)

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
        fw = io.open(self.map_tag, 'w', encoding="utf8")
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
    mrda_inst = MRDA()
    mrda_inst.main()
