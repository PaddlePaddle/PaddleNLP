# -*- coding: utf-8 -*-
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
evaluate wordseg for LAC and other open-source wordseg tools
"""
from __future__ import print_function
from __future__ import division

import sys
import os
import io


def to_unicode(string):
    """ string compatibility for python2 & python3 """
    if sys.version_info.major == 2 and isinstance(string, str):
        return string.decode("utf-8")
    else:
        return string


def to_set(words):
    """ cut list to set of (string, off) """
    off = 0
    s = set()
    for w in words:
        if w:
            s.add((off, w))
        off += len(w)
    return s


def cal_fscore(standard, result, split_delim=" "):
    """ caculate fscore for wordseg
    Param: standard, list of str, ground-truth labels , e.g. ["a b c", "d ef g"]
    Param: result, list of str, predicted result, e.g. ["ab c", "d e fg"]
    """
    assert len(standard) == len(result)
    std, rst, cor = 0, 0, 0
    for s, r in zip(standard, result):
        s = to_set(s.rstrip().split(split_delim))
        r = to_set(r.rstrip().split(split_delim))
        std += len(s)
        rst += len(r)
        cor += len(s & r)
    p = 1.0 * cor / rst
    r = 1.0 * cor / std
    f = 2 * p * r / (p + r)

    print("std, rst, cor = %d, %d, %d" % (std, rst, cor))
    print("precision = %.5f, recall = %.5f, f1 = %.5f" % (p, r, f))
    #print("| | %.5f | %.5f | %.5f |" % (p, r, f))
    print("")

    return p, r, f


def load_testdata(datapath="./data/test_data/test_part"):
    """none"""
    sentences = []
    sent_seg_list = []
    for line in io.open(datapath, 'r', encoding='utf8'):
        sent, label = line.strip().split("\t")
        sentences.append(sent)

        sent = to_unicode(sent)
        label = label.split(" ")
        assert len(sent) == len(label)

        # parse segment
        words = []
        current_word = ""
        for w, l in zip(sent, label):
            if l.endswith("-B"):
                if current_word != "":
                    words.append(current_word)
                current_word = w
            elif l.endswith("-I"):
                current_word += w
            elif l.endswith("-O"):
                if current_word != "":
                    words.append(current_word)
                words.append(w)
                current_word = ""
            else:
                raise ValueError("wrong label: " + l)
        if current_word != "":
            words.append(current_word)
        sent_seg = " ".join(words)
        sent_seg_list.append(sent_seg)
    print("got %d lines" % (len(sent_seg_list)))
    return sent_seg_list, sentences


def get_lac_result():
    """
    get LAC predicted result by:
        `sh run.sh | tail -n 100 > result.txt`
    """
    sent_seg_list = []
    for line in io.open("./result.txt", 'r', encoding='utf8'):
        line = line.strip().split(" ")
        words = [pair.split("/")[0] for pair in line]
        labels = [pair.split("/")[1] for pair in line]
        sent_seg = " ".join(words)
        sent_seg = to_unicode(sent_seg)
        sent_seg_list.append(sent_seg)
    return sent_seg_list


def get_jieba_result(sentences):
    """
    Ref to: https://github.com/fxsjy/jieba
    Install by `pip install jieba`
    """
    import jieba
    preds = []
    for sentence in sentences:
        sent_seg = " ".join(jieba.lcut(sentence))
        sent_seg = to_unicode(sent_seg)
        preds.append(sent_seg)
    return preds


def get_thulac_result(sentences):
    """
    Ref to: http://thulac.thunlp.org/
    Install by: `pip install thulac`
    """
    import thulac
    preds = []
    lac = thulac.thulac(seg_only=True)
    for sentence in sentences:
        sent_seg = lac.cut(sentence, text=True)
        sent_seg = to_unicode(sent_seg)
        preds.append(sent_seg)
    return preds


def get_pkuseg_result(sentences):
    """
    Ref to: https://github.com/lancopku/pkuseg-python
    Install by: `pip3 install pkuseg`
    You should noticed that pkuseg-python only support python3
    """
    import pkuseg
    seg = pkuseg.pkuseg()
    preds = []
    for sentence in sentences:
        sent_seg = " ".join(seg.cut(sentence))
        sent_seg = to_unicode(sent_seg)
        preds.append(sent_seg)
    return preds


def get_hanlp_result(sentences):
    """
    Ref to: https://github.com/hankcs/pyhanlp
    Install by: pip install pyhanlp
        (Before using pyhanlp, you need to download the model manully.)
    """
    from pyhanlp import HanLP
    preds = []
    for sentence in sentences:
        arraylist = HanLP.segment(sentence)
        sent_seg = " ".join(
            [term.toString().split("/")[0] for term in arraylist])
        sent_seg = to_unicode(sent_seg)
        preds.append(sent_seg)
    return preds


def get_nlpir_result(sentences):
    """
    Ref to: https://github.com/tsroten/pynlpir
    Install by `pip install pynlpir`
    Run `pynlpir update` to update License
    """
    import pynlpir
    pynlpir.open()
    preds = []
    for sentence in sentences:
        sent_seg = " ".join(pynlpir.segment(sentence, pos_tagging=False))
        sent_seg = to_unicode(sent_seg)
        preds.append(sent_seg)
    return preds


def get_ltp_result(sentences):
    """
    Ref to: https://github.com/HIT-SCIR/pyltp
        1. Install by `pip install pyltp`
        2. Download models from http://ltp.ai/download.html
    """
    from pyltp import Segmentor
    segmentor = Segmentor()
    model_path = "./ltp_data_v3.4.0/cws.model"
    if not os.path.exists(model_path):
        raise IOError("LTP Model do not exist! Download it first!")
    segmentor.load(model_path)
    preds = []
    for sentence in sentences:
        sent_seg = " ".join(segmentor.segment(sentence))
        sent_seg = to_unicode(sent_seg)
        preds.append(sent_seg)
    segmentor.release()

    return preds


def print_array(array):
    """print some case"""
    for i in [1, 10, 20, 30, 40]:
        print("case " + str(i) + ": \t" + array[i])


def evaluate_all():
    """none"""
    standard, sentences = load_testdata()
    print_array(standard)

    # evaluate lac
    preds = get_lac_result()
    print("lac result:")
    print_array(preds)
    cal_fscore(standard=standard, result=preds)

    # evaluate jieba
    preds = get_jieba_result(sentences)
    print("jieba result")
    print_array(preds)
    cal_fscore(standard=standard, result=preds)

    # evaluate thulac
    preds = get_thulac_result(sentences)
    print("thulac result")
    print_array(preds)
    cal_fscore(standard=standard, result=preds)

    # evaluate pkuseg, but pyuseg only support python3
    if sys.version_info.major == 3:
        preds = get_pkuseg_result(sentences)
        print("pkuseg result")
        print_array(preds)
        cal_fscore(standard=standard, result=preds)

    # evaluate HanLP
    preds = get_hanlp_result(sentences)
    print("HanLP result")
    print_array(preds)
    cal_fscore(standard=standard, result=preds)

    # evaluate NLPIR
    preds = get_nlpir_result(sentences)
    print("NLPIR result")
    print_array(preds)
    cal_fscore(standard=standard, result=preds)

    # evaluate LTP
    preds = get_ltp_result(sentences)
    print("LTP result")
    print_array(preds)
    cal_fscore(standard=standard, result=preds)


if __name__ == "__main__":
    evaluate_all()
