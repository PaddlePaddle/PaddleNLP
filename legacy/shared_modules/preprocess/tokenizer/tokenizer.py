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
This module provides wordseg tools
"""
from __future__ import print_function
import numpy as np
import reader
import paddle.fluid as fluid
import paddle
import argparse
import time
import sys
import io

if sys.version_info > (3, ):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
else:
    reload(sys)
    sys.setdefaultencoding("utf8")


def parse_args():
    """
    Arguments Parse
    """
    parser = argparse.ArgumentParser("Run inference.")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=5,
        help='The size of a batch. (default: %(default)d)')
    parser.add_argument(
        '--model_path',
        type=str,
        default='./conf/model',
        help='A path to the model. (default: %(default)s)')
    parser.add_argument(
        '--test_data_dir',
        type=str,
        default='./data/test_data',
        help='A directory with test data files. (default: %(default)s)')
    parser.add_argument(
        "--word_dict_path",
        type=str,
        default="./conf/word.dic",
        help="The path of the word dictionary. (default: %(default)s)")
    parser.add_argument(
        "--label_dict_path",
        type=str,
        default="./conf/tag.dic",
        help="The path of the label dictionary. (default: %(default)s)")
    parser.add_argument(
        "--word_rep_dict_path",
        type=str,
        default="./conf/q2b.dic",
        help="The path of the word replacement Dictionary. (default: %(default)s)"
    )
    args = parser.parse_args()
    return args


def print_arguments(args):
    """
    Print the arguments
    """
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def get_real_tag(origin_tag):
    """
    Get real tag
    """
    if origin_tag == "O":
        return "O"
    return origin_tag[0:len(origin_tag) - 2]


def to_lodtensor(data, place):
    """
    Trans to lodtensor
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


def infer(args):
    """
    Tokenize
    """
    id2word_dict = reader.load_dict(args.word_dict_path)
    word2id_dict = reader.load_reverse_dict(args.word_dict_path)

    id2label_dict = reader.load_dict(args.label_dict_path)
    label2id_dict = reader.load_reverse_dict(args.label_dict_path)
    q2b_dict = reader.load_dict(args.word_rep_dict_path)
    test_data = paddle.batch(
        reader.test_reader(args.test_data_dir, word2id_dict, label2id_dict,
                           q2b_dict),
        batch_size=args.batch_size)
    place = fluid.CPUPlace()
    #place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    #print("aaa")
    inference_scope = fluid.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(args.model_path, exe)
        #print('bbb')
        for data in test_data():
            full_out_str = ""
            word_idx = to_lodtensor([x[0] for x in data], place)
            #print(word_idx)
            word_list = [x[1] for x in data]
            (crf_decode, ) = exe.run(inference_program,
                                     feed={"word": word_idx},
                                     fetch_list=fetch_targets,
                                     return_numpy=False)
            lod_info = (crf_decode.lod())[0]
            np_data = np.array(crf_decode)
            assert len(data) == len(lod_info) - 1
            for sen_index in range(len(data)):
                assert len(data[sen_index][0]) == lod_info[
                    sen_index + 1] - lod_info[sen_index]
                word_index = 0
                outstr = ""
                cur_full_word = ""
                cur_full_tag = ""
                words = word_list[sen_index]
                for tag_index in range(lod_info[sen_index],
                                       lod_info[sen_index + 1]):
                    cur_word = words[word_index]
                    cur_tag = id2label_dict[str(np_data[tag_index][0])]
                    if cur_tag.endswith("-B") or cur_tag.endswith("O"):
                        if len(cur_full_word) != 0:
                            outstr += cur_full_word + " "
                        cur_full_word = cur_word
                        cur_full_tag = get_real_tag(cur_tag)
                    else:
                        cur_full_word += cur_word
                    word_index += 1
                #print(cur_full_word)
                #outstr += cur_full_word + "/" + cur_full_tag + " "    
                outstr += cur_full_word + ' '
                outstr = outstr.strip()
                full_out_str += outstr + "\n"
            print(full_out_str.strip())
            #print(full_out_str.strip(), file=sys.stdout)


if __name__ == "__main__":
    args = parse_args()
    #print_arguments(args)
    infer(args)
