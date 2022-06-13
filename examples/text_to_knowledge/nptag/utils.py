# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import json
from collections import OrderedDict
from typing import List

import numpy as np


def construct_dict_map(tokenizer, name_dict_path):
    """Construct dict map"""
    with open(name_dict_path, encoding="utf-8") as fp:
        name_dict = json.load(fp)
    cls_vocabs = OrderedDict()
    bk_tree = BurkhardKellerTree()
    for k in name_dict:
        bk_tree.add(k)
        for c in k:
            if c not in cls_vocabs:
                cls_vocabs[c] = len(cls_vocabs)
    cls_vocabs["[PAD]"] = len(cls_vocabs)
    id_vocabs = dict(zip(cls_vocabs.values(), cls_vocabs.keys()))
    vocab_ids = tokenizer.vocab.to_indices(list(cls_vocabs.keys()))
    return name_dict, bk_tree, id_vocabs, vocab_ids


def decode(pred_ids, id_vocabs):
    tokens = [id_vocabs[i] for i in pred_ids]
    valid_token = []
    for token in tokens:
        if token == "[PAD]":
            break
        valid_token.append(token)
    return "".join(valid_token)


def search(scores_can, pred_ids_can, depth, path, score):
    if depth >= 5:
        return [(path, score)]
    res = []
    for i in range(len(pred_ids_can[0])):
        tmp_res = search(scores_can, pred_ids_can, depth + 1,
                         path + [pred_ids_can[depth][i]],
                         score + scores_can[depth][i])
        res.extend(tmp_res)
    return res


def find_topk(a, k, axis=-1, largest=True, sorted=True):
    if axis is None:
        axis_size = a.size
    else:
        axis_size = a.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(a)
    if largest:
        index_array = np.argpartition(a, axis_size - k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k) - 1, axis=axis)
    else:
        index_array = np.argpartition(a, k - 1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)
    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
        sorted_topk_values = np.take_along_axis(topk_values,
                                                sorted_indices_in_topk,
                                                axis=axis)
        sorted_topk_indices = np.take_along_axis(topk_indices,
                                                 sorted_indices_in_topk,
                                                 axis=axis)
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices


def levenstein_distance(s1: str, s2: str) -> int:
    """Calculate minimal Levenstein distance between s1 and s2.

    Args:
        s1 (str): string
        s2 (str): string

    Returns:
        int: the minimal distance.
    """
    m, n = len(s1) + 1, len(s2) + 1

    # Initialize
    dp = [[0] * n for i in range(m)]
    dp[0][0] = 0
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + 1
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + 1

    for i in range(1, m):
        for j in range(1, n):
            if s1[i - 1] != s2[j - 1]:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
            else:
                dp[i][j] = dp[i - 1][j - 1]
    return dp[m - 1][n - 1]


class BurkhardKellerNode(object):
    """Node implementatation for BK-Tree. A BK-Tree node stores the information of current word, and its approximate words calculated by levenstein distance.

    Args:
        word (str): word of current node.
    """

    def __init__(self, word: str):
        self.word = word
        self.next = {}


class BurkhardKellerTree(object):
    """Implementataion of BK-Tree
    """

    def __init__(self):
        self.root = None
        self.nodes = {}

    def __add(self, cur_node: BurkhardKellerNode, word: str):
        """Insert a word into current tree. If tree is empty, set this word to root.

        Args:
            word (str): word to be inserted.
        """
        if self.root is None:
            self.root = BurkhardKellerNode(word)
            return
        if word in self.nodes:
            return
        dist = levenstein_distance(word, cur_node.word)
        if dist not in cur_node.next:
            self.nodes[word] = cur_node.next[dist] = BurkhardKellerNode(word)
        else:
            self.__add(cur_node.next[dist], word)

    def add(self, word: str):
        """Insert a word into current tree. If tree is empty, set this word to root.

        Args:
            word (str): word to be inserted.
        """
        return self.__add(self.root, word)

    def __search_similar_word(self,
                              cur_node: BurkhardKellerNode,
                              s: str,
                              threshold: int = 2) -> List[str]:
        res = []
        if cur_node is None:
            return res
        dist = levenstein_distance(cur_node.word, s)
        if dist <= threshold:
            res.append((cur_node.word, dist))
        start = max(dist - threshold, 1)
        while start < dist + threshold:
            tmp_res = self.__search_similar_word(cur_node.next.get(start, None),
                                                 s)[:]
            res.extend(tmp_res)
            start += 1
        return res

    def search_similar_word(self, word: str) -> List[str]:
        """Search the most similar (minimal levenstain distance) word between `s`.

        Args:
            s (str): target word

        Returns:
            List[str]: similar words.
        """
        res = self.__search_similar_word(self.root, word)

        def max_prefix(s1: str, s2: str) -> int:
            res = 0
            length = min(len(s1), len(s2))
            for i in range(length):
                if s1[i] == s2[i]:
                    res += 1
                else:
                    break
            return res

        res.sort(key=lambda d: (d[1], -max_prefix(d[0], word)))
        return res
