#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""

from __future__ import unicode_literals
import sys
import os
import random
import paddle
import logging
import paddle.fluid as fluid
import numpy as np
import collections
import six
import codecs
try:
    import configparser as cp
except ImportError:
    import ConfigParser as cp

random_seed = 7
logger = logging.getLogger()
format = "%(asctime)s - %(name)s - %(levelname)s -%(filename)s-%(lineno)4d -%(message)s"
# format = "%(levelname)8s: %(asctime)s: %(filename)s:%(lineno)4d %(message)s"
logging.basicConfig(format=format)
logger.setLevel(logging.INFO)
logger = logging.getLogger('Paddle-DDC')


def str2bool(v):
    """[ because argparse does not support to parse "true, False" as python
     boolean directly]
    Arguments:
        v {[type]} -- [description]
    Returns:
        [type] -- [description]
    """
    return v.lower() in ("true", "t", "1")


def to_lodtensor(data, place):
    """
    convert ot LODtensor
    """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


class ArgumentGroup(object):
    """[ArgumentGroup]
    
    Arguments:
        object {[type]} -- [description]
    """

    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, **kwargs):
        """[add_arg]
        
        Arguments:
            name {[type]} -- [description]
            type {[type]} -- [description]
            default {[type]} -- [description]
            help {[type]} -- [description]
        """
        type = str2bool if type == bool else type
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)


class DataReader(object):
    """[get data generator for dataset]
    
    Arguments:
        object {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    def __init__(self, char_vocab, intent_dict, max_len):
        self._char_vocab = char_vocab
        self._intent_dict = intent_dict
        self._oov_id = 0
        self.intent_size = len(intent_dict)
        self.all_data = []
        self.max_len = max_len
        self.padding_id = 0

    def _get_num_examples(self):
        return len(self.all_data)

    def prepare_data(self, data_path, batch_size, mode):
        """
        prepare data
        """
        # print word_dict_path
        # assert os.path.exists(
        #     word_dict_path), "The given word dictionary dose not exist."
        assert os.path.exists(data_path), "The given data file does not exist."
        if mode == "train":
            train_reader = fluid.io.batch(
                fluid.io.shuffle(
                    self.data_reader(
                        data_path, self.max_len, shuffle=True),
                    buf_size=batch_size * 100),
                batch_size)
            return train_reader
        else:
            test_reader = fluid.io.batch(
                self.data_reader(data_path, self.max_len), batch_size)
            return test_reader

    def data_reader(self, file_path, max_len, shuffle=False):
        """
        Convert query into id list
        use fixed voc
        """

        for line in codecs.open(file_path, "r", encoding="utf8"):
            line = line.strip()
            if isinstance(line, six.binary_type):
                line = line.decode("utf8", errors="ignore")
            query, intent = line.split("\t")
            char_id_list = list(map(lambda x: 0 if x not in self._char_vocab else int(self._char_vocab[x]), \
                            list(query)))
            if len(char_id_list) < max_len:
                char_id_list.extend([self.padding_id] *
                                    (max_len - len(char_id_list)))
            char_id_list = char_id_list[:max_len]
            intent_id_list = [self.padding_id] * self.intent_size
            for item in intent.split('\2'):
                intent_id_list[int(self._intent_dict[item])] = 1
            self.all_data.append([char_id_list, intent_id_list])
        if shuffle:
            random.seed(random_seed)
            random.shuffle(self.all_data)

        def reader():
            """
            reader
            """
            for char_id_list, intent_id_list in self.all_data:
                # print char_id_list, intent_id
                yield char_id_list, intent_id_list

        return reader


class DataProcesser(object):
    """[file process methods]
    
    Arguments:
        object {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    @staticmethod
    def read_dict(filename):
        """
        read_dict: key\2value
        """
        res_dict = {}
        for line in codecs.open(filename, encoding="utf8"):
            try:
                if isinstance(line, six.binary_type):
                    line = line.strip().decode("utf8")
                line = line.strip()
                key, value = line.strip().split("\2")
                res_dict[key] = value
            except Exception as err:
                logger.error(str(err))
                logger.error("read dict[%s] failed" % filename)
        return res_dict

    @staticmethod
    def build_dict(filename, save_dir, min_num_char=2, min_num_intent=2):
        """[build_dict  from file]
        
        Arguments:
            filename {[type]} -- [description]
            save_dir {[type]} -- [description]
        
        Keyword Arguments:
            min_num_char {int} -- [description] (default: {2})
            min_num_intent {int} -- [description] (default: {2})
        """
        char_dict = {}
        intent_dict = {}
        # readfile
        for line in codecs.open(filename):
            line = line.strip()
            if isinstance(line, six.binary_type):
                line = line.strip().decode("utf8", errors="ignore")
            query, intents = line.split("\t")
            # read query
            for char_item in list(query):
                if char_item not in char_dict:
                    char_dict[char_item] = 0
                char_dict[char_item] += 1
            # read intents
            for intent in intents.split('\002'):
                if intent not in intent_dict:
                    intent_dict[intent] = 0
                intent_dict[intent] += 1
        #   save char dict
        with codecs.open(
                "%s/char.dict" % save_dir, "w", encoding="utf8") as f_out:
            f_out.write("PAD\0020\n")
            f_out.write("OOV\0021\n")
            char_id = 2
            for key, value in char_dict.items():
                if value >= min_num_char:
                    if isinstance(key, six.binary_type):
                        key = key.encode("utf8")
                    f_out.write("%s\002%d\n" % (key, char_id))
                    char_id += 1
        #   save intent dict
        with codecs.open(
                "%s/domain.dict" % save_dir, "w", encoding="utf8") as f_out:
            f_out.write("SYS_OTHER\0020\n")
            intent_id = 1
            for key, value in intent_dict.items():
                if value >= min_num_intent and key != u'SYS_OTHER':
                    if isinstance(key, six.binary_type):
                        key = key.encode("utf8")
                    f_out.write("%s\002%d\n" % (key, intent_id))
                    intent_id += 1


class ConfigReader(object):
    """[read model config file]
    
    Arguments:
        object {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    @staticmethod
    def read_conf(conf_file):
        """[read_conf]
        
        Arguments:
            conf_file {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        flow_data = collections.defaultdict(lambda: {})
        class2key = set(["model"])
        param_conf = cp.ConfigParser()
        param_conf.read(conf_file)
        for section in param_conf.sections():
            if section not in class2key:
                continue
            for option in param_conf.items(section):
                flow_data[section][option[0]] = eval(option[1])
        return flow_data


def init_checkpoint(exe, init_checkpoint_path, main_program):
    """
    Init CheckPoint
    """
    fluid.load(main_program, init_checkpoint_path, exe)
    print("Load model from {}".format(init_checkpoint_path))


def print_arguments(args):
    """
    Print Arguments
    """
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def check_version(version='1.6.0'):
    """
    Log error and exit when the installed version of paddlepaddle is
    not satisfied.
    """
    err = "PaddlePaddle version 1.6 or higher is required, " \
          "or a suitable develop version is satisfied as well. \n" \
          "Please make sure the version is good with your code." \

    try:
        fluid.require_version(version)
    except Exception as e:
        logger.error(err)
        sys.exit(1)
