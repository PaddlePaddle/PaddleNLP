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
"""common function"""
import sys
import io
import os


def get_file_list(dir_name):
    """
    get file list in directory
    """
    file_list = list()
    file_path = list()
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            file_list.append(file)
            file_path.append(os.path.join(root, file))
    return file_list, file_path


def get_dir_list(dir_name):
    """
    get directory names
    """
    child_dir = []
    dir_list = os.listdir(dir_name)
    for cur_file in dir_list:
        path = os.path.join(dir_name, cur_file)
        if not os.path.isdir(path):
            continue
        child_dir.append(path)
    return child_dir


def load_dict(conf):
    """
    load swda dataset config
    """
    conf_dict = dict()
    fr = io.open(conf, 'r', encoding="utf8")
    for line in fr:
        line = line.strip()
        elems = line.split('\t')
        if elems[0] not in conf_dict:
            conf_dict[elems[0]] = []
        conf_dict[elems[0]].append(elems[1])
    return conf_dict


def load_voc(conf):
    """
    load map dict
    """
    map_dict = {}
    fr = io.open(conf, 'r', encoding="utf8")
    for line in fr:
        line = line.strip()
        elems = line.split('\t')
        map_dict[elems[0]] = elems[1]
    return map_dict
