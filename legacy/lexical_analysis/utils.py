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
util tools
"""
from __future__ import print_function
import os
import sys
import numpy as np
import paddle.fluid as fluid
import yaml
import io


def str2bool(v):
    """
    argparse does not support True or False in python
    """
    return v.lower() in ("true", "t", "1")


class ArgumentGroup(object):
    """
    Put arguments to one group
    """

    def __init__(self, parser, title, des):
        """none"""
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, **kwargs):
        """ Add argument """
        type = str2bool if type == bool else type
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)


def load_yaml(parser, file_name, **kwargs):
    with io.open(file_name, 'r', encoding='utf8') as f:
        args = yaml.load(f)
        for title in args:
            group = parser.add_argument_group(title=title, description='')
            for name in args[title]:
                _type = type(args[title][name]['val'])
                _type = str2bool if _type == bool else _type
                group.add_argument(
                    "--" + name,
                    default=args[title][name]['val'],
                    type=_type,
                    help=args[title][name]['meaning'] +
                    ' Default: %(default)s.',
                    **kwargs)


def print_arguments(args):
    """none"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def to_str(string, encoding="utf-8"):
    """convert to str for print"""
    if sys.version_info.major == 3:
        if isinstance(string, bytes):
            return string.decode(encoding)
    elif sys.version_info.major == 2:
        if isinstance(string, unicode):
            if os.name == 'nt':
                return string
            else:
                return string.encode(encoding)
    return string


def to_lodtensor(data, place):
    """
    Convert data in list into lodtensor.
    """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.Tensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def parse_result(words, crf_decode, dataset):
    """ parse result """
    offset_list = (crf_decode.lod())[0]
    words = np.array(words)
    crf_decode = np.array(crf_decode)
    batch_size = len(offset_list) - 1

    batch_out = []
    for sent_index in range(batch_size):
        begin, end = offset_list[sent_index], offset_list[sent_index + 1]
        sent = [dataset.id2word_dict[str(id[0])] for id in words[begin:end]]
        tags = [
            dataset.id2label_dict[str(id[0])] for id in crf_decode[begin:end]
        ]

        sent_out = []
        tags_out = []
        parital_word = ""
        for ind, tag in enumerate(tags):
            # for the first word
            if parital_word == "":
                parital_word = sent[ind]
                tags_out.append(tag.split('-')[0])
                continue

            # for the beginning of word
            if tag.endswith("-B") or (tag == "O" and tags[ind - 1] != "O"):
                sent_out.append(parital_word)
                tags_out.append(tag.split('-')[0])
                parital_word = sent[ind]
                continue

            parital_word += sent[ind]

        # append the last word, except for len(tags)=0
        if len(sent_out) < len(tags_out):
            sent_out.append(parital_word)

        batch_out.append([sent_out, tags_out])
    return batch_out


def parse_padding_result(words, crf_decode, seq_lens, dataset):
    """ parse padding result """
    words = np.squeeze(words)
    batch_size = len(seq_lens)

    batch_out = []
    for sent_index in range(batch_size):

        sent = [
            dataset.id2word_dict[str(id)]
            for id in words[sent_index][1:seq_lens[sent_index] - 1]
        ]
        tags = [
            dataset.id2label_dict[str(id)]
            for id in crf_decode[sent_index][1:seq_lens[sent_index] - 1]
        ]

        sent_out = []
        tags_out = []
        parital_word = ""
        for ind, tag in enumerate(tags):
            # for the first word
            if parital_word == "":
                parital_word = sent[ind]
                tags_out.append(tag.split('-')[0])
                continue

            # for the beginning of word
            if tag.endswith("-B") or (tag == "O" and tags[ind - 1] != "O"):
                sent_out.append(parital_word)
                tags_out.append(tag.split('-')[0])
                parital_word = sent[ind]
                continue

            parital_word += sent[ind]

        # append the last word, except for len(tags)=0
        if len(sent_out) < len(tags_out):
            sent_out.append(parital_word)

        batch_out.append([sent_out, tags_out])
    return batch_out


def init_checkpoint(exe, init_checkpoint_path, main_program):
    """
    Init CheckPoint
    """
    assert os.path.exists(
        init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path
    
    try:
        checkpoint_path = os.path.join(init_checkpoint_path, "checkpoint")
        fluid.load(main_program, checkpoint_path, exe)
    except:
        fluid.load(main_program, init_checkpoint_path, exe)
    print("Load model from {}".format(init_checkpoint_path))

def init_pretraining_params(exe,
                            pretraining_params_path,
                            main_program,
                            use_fp16=False):
    """load params of pretrained model, NOT including moment, learning_rate"""
    assert os.path.exists(pretraining_params_path
                          ), "[%s] cann't be found." % pretraining_params_path

    fluid.load(main_program, pretraining_params_path, exe)
    print("Load pretraining parameters from {}.".format(
        pretraining_params_path))
