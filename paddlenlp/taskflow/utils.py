# coding:utf-8
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import contextlib
import copy
import csv
import json
import math
import os
import pickle
import re
import traceback
import warnings
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from datetime import datetime
from functools import cmp_to_key
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn.functional as F
import six
from paddle.dataset.common import md5file
from PIL import Image

from ..transformers.tokenizer_utils_base import PaddingStrategy, PretrainedTokenizerBase
from ..utils.downloader import DownloaderCheck, get_path_from_url
from ..utils.image_utils import (
    Bbox,
    DecodeImage,
    NormalizeImage,
    PadBatch,
    Permute,
    ResizeImage,
    check,
    img2base64,
    two_dimension_sort_layout,
)
from ..utils.log import logger

DOC_FORMAT = r"""
    Examples:
        .. code-block:: python
              """
DOWNLOAD_CHECK = False


def download_file(save_dir, filename, url, md5=None):
    """
    Download the file from the url to specified directory.
    Check md5 value when the file is exists, if the md5 value is the same as the existed file, just use
    the older file, if not, will download the file from the url.

    Args:
        save_dir(string): The specified directory saving the file.
        filename(string): The specified filename saving the file.
        url(string): The url downling the file.
        md5(string, optional): The md5 value that checking the version downloaded.
    """
    fullname = os.path.join(save_dir, filename)
    if os.path.exists(fullname):
        if md5 and (not md5file(fullname) == md5):
            logger.info("Updating {} from {}".format(filename, url))
            logger.disable()
            get_path_from_url(url, save_dir, md5)
    else:
        logger.info("Downloading {} from {}".format(filename, url))
        logger.disable()
        get_path_from_url(url, save_dir, md5)
    logger.enable()
    return fullname


def download_check(task):
    """
    Check the resource status in the specified task.

    Args:
        task(string): The name of specified task.
    """
    logger.disable()
    global DOWNLOAD_CHECK
    if not DOWNLOAD_CHECK:
        DOWNLOAD_CHECK = True
        checker = DownloaderCheck(task)
        checker.start()
        checker.join()
    logger.enable()


def add_docstrings(*docstr):
    """
    The function that add the doc string to doc of class.
    """

    def docstring_decorator(fn):
        fn.__doc__ = fn.__doc__ + "".join(DOC_FORMAT) + "".join(docstr)
        return fn

    return docstring_decorator


@contextlib.contextmanager
def static_mode_guard():
    paddle.enable_static()
    yield
    paddle.disable_static()


@contextlib.contextmanager
def dygraph_mode_guard():
    paddle.disable_static()
    yield


def cut_chinese_sent(para):
    """
    Cut the Chinese sentences more precisely, reference to "https://blog.csdn.net/blmoistawinde/article/details/82379256".
    """
    para = re.sub(r"([。！？\?])([^”’])", r"\1\n\2", para)
    para = re.sub(r"(\.{6})([^”’])", r"\1\n\2", para)
    para = re.sub(r"(\…{2})([^”’])", r"\1\n\2", para)
    para = re.sub(r"([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)
    para = para.rstrip()
    return para.split("\n")


class TermTreeNode(object):
    """Defination of term node. All members are protected, to keep rigorism of data struct.

    Args:
        sid (str): term id of node.
        term (str): term, common name of this term.
        base (str): `cb` indicates concept base, `eb` indicates entity base.
        term_type (Optional[str], optional): type of this term, constructs hirechical of `term` node. Defaults to None.
        hyper (Optional[str], optional): parent type of a `type` node. Defaults to None.
        node_type (str, optional): type statement of node, `type` or `term`. Defaults to "term".
        alias (Optional[List[str]], optional): alias of this term. Defaults to None.
        alias_ext (Optional[List[str]], optional): extended alias of this term, CANNOT be used in matching.
            Defaults to None.
        sub_type (Optional[List[str]], optional): grouped by some term. Defaults to None.
        sub_term (Optional[List[str]], optional): some lower term. Defaults to None.
        data (Optional[Dict[str, Any]], optional): to sore full imformation of a term. Defaults to None.

    """

    def __init__(
        self,
        sid: str,
        term: str,
        base: str,
        node_type: str = "term",
        term_type: Optional[str] = None,
        hyper: Optional[str] = None,
        level: Optional[int] = None,
        alias: Optional[List[str]] = None,
        alias_ext: Optional[List[str]] = None,
        sub_type: Optional[List[str]] = None,
        sub_term: Optional[List[str]] = None,
        data: Optional[Dict[str, Any]] = None,
    ):
        self._sid = sid
        self._term = term
        self._base = base
        self._term_type = term_type
        self._hyper = hyper
        self._sub_term = sub_term if sub_term is not None else []
        self._sub_type = sub_type if sub_type is not None else []
        self._alias = alias if alias is not None else []
        self._alias_ext = alias_ext if alias_ext is not None else []
        self._data = data
        self._level = level
        self._node_type = node_type
        self._sons = set()

    def __str__(self):
        if self._data is not None:
            return json.dumps(self._data, ensure_ascii=False)
        else:
            res = {
                "termid": self._sid,
                "term": self._term,
                "src": self._base,
                "alias": self._alias,
                "alias_ext": self._alias_ext,
                "termtype": self._term_type,
                "subterms": self._sub_term,
                "subtype": self._sub_type,
                "links": [],
            }
            return json.dumps(res, ensure_ascii=False)

    @property
    def sid(self):
        return self._sid

    @property
    def term(self):
        return self._term

    @property
    def base(self):
        return self._base

    @property
    def alias(self):
        return self._alias

    @property
    def alias_ext(self):
        return self._alias_ext

    @property
    def termtype(self):
        return self._term_type

    @property
    def subtype(self):
        return self._sub_type

    @property
    def subterm(self):
        return self._sub_term

    @property
    def hyper(self):
        return self._hyper

    @property
    def level(self):
        return self._level

    @property
    def sons(self):
        return self._sons

    @property
    def node_type(self):
        return self._node_type

    def add_son(self, son_name):
        self._sons.add(son_name)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Build a node from dictionary data.

        Args:
            data (Dict[str, Any]): Dictionary data contain all k-v data.

        Returns:
            [type]: TermTree node object.
        """
        return cls(
            sid=data["termid"],
            term=data["term"],
            base=data["src"],
            term_type=data["termtype"],
            sub_type=data["subtype"],
            sub_term=data["subterms"],
            alias=data["alias"],
            alias_ext=data["alias_ext"],
            data=data,
        )

    @classmethod
    def from_json(cls, json_str: str):
        """Build a node from JSON string.

        Args:
            json_str (str): JSON string formatted by TermTree data.

        Returns:
            [type]: TermTree node object.
        """
        dict_data = json.loads(json_str)
        return cls.from_dict(dict_data)


class TermTree(object):
    """TermTree class."""

    def __init__(self):
        self._nodes: Dict[str, TermTreeNode] = {}
        self._root = TermTreeNode(sid="root", term="root", base="cb", node_type="root", level=0)
        self._nodes["root"] = self.root
        self._index = {}

    def __build_sons(self):
        for node in self._nodes:
            self.__build_son(self._nodes[node])

    def __getitem__(self, item):
        return self._nodes[item]

    def __contains__(self, item):
        return item in self._nodes

    def __iter__(self):
        return self._nodes.__iter__()

    @property
    def root(self):
        return self._root

    def __load_type(self, file_path: str):
        with open(file_path, "rt", newline="", encoding="utf8") as csvfile:
            file_handler = csv.DictReader(csvfile, delimiter="\t")
            for row in file_handler:
                if row["type-1"] not in self:
                    self.add_type(type_name=row["type-1"], hyper_type="root")
                if row["type-2"] != "" and row["type-2"] not in self:
                    self.add_type(type_name=row["type-2"], hyper_type=row["type-1"])
                if row["type-3"] != "" and row["type-3"] not in self:
                    self.add_type(type_name=row["type-3"], hyper_type=row["type-2"])

    def __judge_term_node(self, node: TermTreeNode) -> bool:
        if node.termtype not in self:
            raise ValueError(f"Term type of new node {node.termtype} does not exists.")
        if node.sid in self:
            warnings.warn(f"{node.sid} exists, will be replaced by new node.")

    def add_term(
        self,
        term: Optional[str] = None,
        base: Optional[str] = None,
        term_type: Optional[str] = None,
        sub_type: Optional[List[str]] = None,
        sub_term: Optional[List[str]] = None,
        alias: Optional[List[str]] = None,
        alias_ext: Optional[List[str]] = None,
        data: Optional[Dict[str, Any]] = None,
    ):
        """Add a term into TermTree.

        Args:
            term (str): common name of name.
            base (str): term is concept or entity.
            term_type (str): term type of this term
            sub_type (Optional[List[str]], optional): sub type of this term, must exists in TermTree. Defaults to None.
            sub_terms (Optional[List[str]], optional): sub terms of this term. Defaults to None.
            alias (Optional[List[str]], optional): alias of this term. Defaults to None.
            alias_ext (Optional[List[str]], optional): . Defaults to None.
            data (Optional[Dict[str, Any]], optional): [description]. Defaults to None.
        """
        if data is not None:
            new_node = TermTreeNode.from_dict(data)
        else:
            new_node = TermTreeNode(
                sid=f"{term_type}_{base}_{term}",
                term=term,
                base=base,
                term_type=term_type,
                sub_term=sub_term,
                sub_type=sub_type,
                alias=alias,
                alias_ext=alias_ext,
                node_type="term",
            )
        self.__judge_term_node(new_node)
        self._nodes[new_node.sid] = new_node
        self.__build_index(new_node)

    def add_type(self, type_name, hyper_type):
        if type_name in self._nodes:
            raise ValueError(f"Term Type {type_name} exists.")
        if hyper_type not in self._nodes:
            raise ValueError(f"Hyper type {hyper_type} does not exist, please add it first.")
        if self._nodes[hyper_type].level == 3:
            raise ValueError(
                "Term type schema must be 3-LEVEL, 3rd level type node should not be a parent of type node."
            )
        self._nodes[type_name] = TermTreeNode(
            sid=type_name,
            term=type_name,
            base=None,
            hyper=hyper_type,
            node_type="type",
            level=self._nodes[hyper_type].level + 1,
        )
        self.__build_index(self._nodes[type_name])

    def __load_file(self, file_path: str):
        with open(file_path, encoding="utf-8") as fp:
            for line in fp:
                data = json.loads(line)
                self.add_term(data=data)

    def __build_son(self, node: TermTreeNode):
        """Build sons of a node

        Args:
            node (TermTreeNode): son node.
        """
        type_node = None
        if node.termtype is not None:
            type_node = self._nodes[node.termtype]
        elif node.hyper is not None:
            type_node = self._nodes[node.hyper]
        if type_node is not None:
            type_node.add_son(node.sid)
        for sub_type in node.subtype:
            sub_type_node = self._nodes[sub_type]
            sub_type_node.add_son(node.sid)

    def build_son(self, node: str):
        self.__build_son(self[node])

    def __build_index(self, node: TermTreeNode):
        if node.term not in self._index:
            self._index[node.term] = []
        self._index[node.term].append(node.sid)
        for alia in node.alias:
            if alia not in self._index:
                self._index[alia] = []
            self._index[alia].append(node.sid)

    def __judge_hyper(self, source_id, target_id) -> bool:
        queue = [source_id]
        visited_node = {source_id}
        while len(queue) > 0:
            cur_id = queue.pop(0)
            if cur_id == target_id:
                return True
            cur_node = self._nodes[cur_id]
            edge = []
            if cur_node.hyper is not None:
                edge.append(cur_node.hyper)
            if cur_node.termtype is not None:
                edge.append(cur_node.termtype)
            edge.extend(cur_node.subtype)
            for next_id in edge:
                if next_id not in visited_node:
                    queue.append(next_id)
                    visited_node.add(next_id)
        return False

    def find_term(self, term: str, term_type: Optional[str] = None) -> Tuple[bool, Union[List[str], None]]:
        """Find a term in Term Tree. If term not exists, return None.
        If `term_type` is not None, will find term with this type.

        Args:
            term (str): term to look up.
            term_type (Optional[str], optional): find term in this term_type. Defaults to None.

        Returns:
            Union[None, List[str]]: [description]
        """
        if term not in self._index:
            return False, None
        else:
            if term_type is None:
                return True, self._index[term]
            else:
                out = []
                for term_id in self._index[term]:
                    if self.__judge_hyper(term_id, term_type) is True:
                        out.append(term_id)
                if len(out) > 0:
                    return True, out
                else:
                    return False, None

    def build_from_dir(self, term_schema_path, term_data_path, linking=True):
        """Build TermTree from a directory which should contain type schema and term data.

        Args:
            dir ([type]): [description]
        """
        self.__load_type(term_schema_path)
        if linking:
            self.__load_file(term_data_path)
            self.__build_sons()

    @classmethod
    def from_dir(cls, term_schema_path, term_data_path, linking) -> "TermTree":
        """Build TermTree from a directory which should contain type schema and term data.

        Args:
            source_dir ([type]): [description]

        Returns:
            TermTree: [description]
        """
        term_tree = cls()
        term_tree.build_from_dir(term_schema_path, term_data_path, linking)
        return term_tree

    def __dfs(self, cur_id: str, depth: int, path: Dict[str, str], writer: csv.DictWriter):
        cur_node = self._nodes[cur_id]
        if cur_node.node_type == "term":
            return
        if depth > 0:
            path[f"type-{depth}"] = cur_id
        if path["type-1"] != "":
            writer.writerow(path)
        for son in cur_node.sons:
            self.__dfs(son, depth + 1, path, writer)
        if depth > 0:
            path[f"type-{depth}"] = ""

    def save(self, save_dir):
        """Save term tree to directory `save_dir`

        Args:
            save_dir ([type]): Directory.
        """
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir, exist_ok=True)
        out_path = {}
        for i in range(1, 3):
            out_path[f"type-{i}"] = ""
        with open(f"{save_dir}/termtree_type.csv", "wt", encoding="utf-8", newline="") as fp:
            fieldnames = ["type-1", "type-2", "type-3"]
            csv_writer = csv.DictWriter(fp, delimiter="\t", fieldnames=fieldnames)
            csv_writer.writeheader()
            self.__dfs("root", 0, out_path, csv_writer)
        with open(f"{save_dir}/termtree_data", "w", encoding="utf-8", newline="") as fp:
            for nid in self:
                node = self[nid]
                if node.node_type == "term":
                    print(node, file=fp)


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
    """Implementataion of BK-Tree"""

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

    def __search_similar_word(self, cur_node: BurkhardKellerNode, s: str, threshold: int = 2) -> List[str]:
        res = []
        if cur_node is None:
            return res
        dist = levenstein_distance(cur_node.word, s)
        if dist <= threshold:
            res.append((cur_node.word, dist))
        start = max(dist - threshold, 1)
        while start < dist + threshold:
            tmp_res = self.__search_similar_word(cur_node.next.get(start, None), s)[:]
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


class TriedTree(object):
    """Implementataion of TriedTree"""

    def __init__(self):
        self.tree = {}

    def add_word(self, word):
        """add single word into TriedTree"""
        self.tree[word] = len(word)
        for i in range(1, len(word)):
            wfrag = word[:i]
            self.tree[wfrag] = self.tree.get(wfrag, None)

    def search(self, content):
        """Backward maximum matching

        Args:
            content (str): string to be searched
        Returns:
            List[Tuple]: list of maximum matching words, each element represents
                the starting and ending position of the matching string.
        """
        result = []
        length = len(content)
        for start in range(length):
            for end in range(start + 1, length + 1):
                pos = self.tree.get(content[start:end], -1)
                if pos == -1:
                    break
                if pos and (len(result) == 0 or end > result[-1][1]):
                    result.append((start, end))
        return result


class Customization(object):
    """
    User intervention based on Aho-Corasick automaton
    """

    def __init__(self):
        self.dictitem = {}
        self.ac = None

    def load_customization(self, filename, sep=None):
        """Load the custom vocab"""
        self.ac = TriedTree()
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                if sep is None:
                    words = line.strip().split()

                if len(words) == 0:
                    continue

                phrase = ""
                tags = []
                offset = []
                for word in words:
                    if word.rfind("/") < 1:
                        phrase += word
                        tags.append("")
                    else:
                        phrase += word[: word.rfind("/")]
                        tags.append(word[word.rfind("/") + 1 :])
                    offset.append(len(phrase))

                if len(phrase) < 2 and tags[0] == "":
                    continue

                self.dictitem[phrase] = (tags, offset)
                self.ac.add_word(phrase)

    def parse_customization(self, query, lac_tags, prefix=False):
        """Use custom vocab to modify the lac results"""
        if not self.ac:
            logger.warning("customization dict is not load")
            return
        ac_res = self.ac.search(query)

        for begin, end in ac_res:
            phrase = query[begin:end]
            index = begin

            tags, offsets = self.dictitem[phrase]

            if prefix:
                for tag, offset in zip(tags, offsets):
                    while index < begin + offset:
                        if len(tag) == 0:
                            lac_tags[index] = "I" + lac_tags[index][1:]
                        else:
                            lac_tags[index] = "I-" + tag
                        index += 1
                lac_tags[begin] = "B" + lac_tags[begin][1:]
                for offset in offsets:
                    index = begin + offset
                    if index < len(lac_tags):
                        lac_tags[index] = "B" + lac_tags[index][1:]
            else:
                for tag, offset in zip(tags, offsets):
                    while index < begin + offset:
                        if len(tag) == 0:
                            lac_tags[index] = lac_tags[index][:-1] + "I"
                        else:
                            lac_tags[index] = tag + "-I"
                        index += 1
                lac_tags[begin] = lac_tags[begin][:-1] + "B"
                for offset in offsets:
                    index = begin + offset
                    if index < len(lac_tags):
                        lac_tags[index] = lac_tags[index][:-1] + "B"


class SchemaTree(object):
    """
    Implementataion of SchemaTree
    """

    def __init__(self, name="root", children=None):
        self.name = name
        self.children = []
        self.prefix = None
        self.parent_relations = None
        self.parent = None
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(node, SchemaTree), "The children of a node should be an instacne of SchemaTree."
        self.children.append(node)


def get_id_and_prob(span_set, offset_mapping):
    """
    Return text id and probability of predicted spans

    Args:
        span_set (set): set of predicted spans.
        offset_mapping (list[int]): list of pair preserving the
                index of start and end char in original text pair (prompt + text) for each token.
    Returns:
        sentence_id (list[tuple]): index of start and end char in original text.
        prob (list[float]): probabilities of predicted spans.
    """
    prompt_end_token_id = offset_mapping[1:].index([0, 0])
    bias = offset_mapping[prompt_end_token_id][1] + 1
    for idx in range(1, prompt_end_token_id + 1):
        offset_mapping[idx][0] -= bias
        offset_mapping[idx][1] -= bias

    sentence_id = []
    prob = []
    for start, end in span_set:
        prob.append(start[1] * end[1])
        start_id = offset_mapping[start[0]][0]
        end_id = offset_mapping[end[0]][1]
        sentence_id.append((start_id, end_id))
    return sentence_id, prob


def dbc2sbc(s):
    rs = ""
    for char in s:
        code = ord(char)
        if code == 0x3000:
            code = 0x0020
        else:
            code -= 0xFEE0
        if not (0x0021 <= code and code <= 0x7E):
            rs += char
            continue
        rs += chr(code)
    return rs


class WordTagRelationExtractor(object):
    """Implement of information extractor."""

    _chain_items = {"和", "与", "兼", "及", "以及", "还有", "并"}
    _all_items = None
    _jux_buf = []

    def __init__(self, schema):
        self._schema = schema

    @property
    def schema(self):
        return self._schema

    @classmethod
    def from_dict(cls, config_dict):
        """Make an instance from a configuration dictionary.

        Args:
            config_dict (Dict[str, Any]): configuration dict.
        """
        res = {}

        for i, trip_config in enumerate(config_dict):
            head_role_type = trip_config["head_role"]
            if head_role_type not in res:
                res[head_role_type] = {"trigger": {}, "g_t_map": {}, "rel_group": {}, "trig_word": {}}
            group_name = trip_config["group"]
            if "rel_group" in trip_config:
                res[head_role_type]["rel_group"][group_name] = trip_config["rel_group"]
            if group_name not in res[head_role_type]["trig_word"]:
                res[head_role_type]["trig_word"][group_name] = set()
            for trig_word in trip_config["trig_word"]:
                res[head_role_type]["trigger"][trig_word] = {
                    "trigger_type": trip_config["trig_type"],
                    "group_name": group_name,
                    "rev_flag": trip_config["reverse"],
                }
                res[head_role_type]["trig_word"][group_name].add(trig_word)
            res[head_role_type]["g_t_map"][group_name] = trip_config["tail_role"]

        return cls(res)

    @classmethod
    def from_json(cls, json_str):
        """Implement an instance from JSON str."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    @classmethod
    def from_pkl(cls, pkl_path):
        """Implement an instance from a serialized pickle package."""
        with open(pkl_path, "rb") as fp:
            schema = pickle.load(fp)
        return cls(schema)

    @classmethod
    def from_config(cls, config_path):
        """Implement an instance from a configuration file."""
        with open(config_path, encoding="utf-8") as fp:
            config_json = json.load(fp)
        return cls.from_dict(config_json)

    def add_schema_from_dict(self, config_dict):
        """Add the schema from the dict."""
        for i, trip_config in enumerate(config_dict):
            head_role_type = trip_config["head_role"]
            if head_role_type not in self._schema:
                self._schema[head_role_type] = {"trigger": {}, "g_t_map": {}, "rel_group": {}, "trig_word": {}}
            group_name = trip_config["group"]
            if "rel_group" in self._schema:
                self._schema[head_role_type]["rel_group"][group_name] = trip_config["rel_group"]
            if group_name not in self._schema[head_role_type]["trig_word"]:
                self._schema[head_role_type]["trig_word"][group_name] = set()
            for trig_word in trip_config["trig_word"]:
                self._schema[head_role_type]["trigger"][trig_word] = {
                    "trigger_type": trip_config["trig_type"],
                    "group_name": group_name,
                    "rev_flag": trip_config["reverse"],
                }
                self._schema[head_role_type]["trig_word"][group_name].add(trig_word)
            self._schema[head_role_type]["g_t_map"][group_name] = trip_config["tail_role"]

    def _judge_jux(self, wordtag_item):
        """Judge whether `wordtag_item` is a relevance componet between two juxtaposed items.

        Args:
            wordtag_item (dict): input item.

        Returns:
            bool: [description]
        """
        if wordtag_item["item"] in {"、", " ", "《", "》", "/"}:
            return True
        if wordtag_item["item"] in self._chain_items and wordtag_item["wordtag_label"] == "连词":
            return True
        return False

    def _search_jux(self, cur_item, cur_pos=0, jux_type=None, jux_word=None, status_flag=None, search_list=None):
        """Find juxtaposed items with `cur_item` at `cur_pos` in `self._all_items`.

        Args:
            cur_item (Dict[str, Any]): the item current viewing.
            cur_pos (int, optional): current position of viewing item. Defaults to 0.
            jux_type (Set[str], optional):  wordtag labels that can be considered as juxtaposed item. Defaults to None.
            jux_word (Set[str], optional):  words that can be considered as juxtaposed item. Defaults to None.
            status_flag (bool, optional): if True, on the juxtaposed item, or on chain item. Defaults to None.

        Returns:
            int: end postion of juxtable items.
        """
        if search_list is None:
            search_list = self._all_items

        if jux_type is None and jux_word is None:
            raise ValueError("`jux_type` and `jux_word` are both None.")

        if status_flag is True:
            self._jux_buf.append(cur_item)

        if cur_pos >= len(search_list) - 1:
            return cur_pos

        next_item = search_list[cur_pos + 1]

        if self._judge_jux(next_item) is True:
            return self._search_jux(
                cur_item=next_item,
                cur_pos=cur_pos + 1,
                jux_type=jux_type,
                jux_word=jux_word,
                status_flag=False,
                search_list=search_list,
            )

        next_flag = True
        if jux_type is not None:
            next_flag = next_flag and self._match_item(next_item, jux_type)
        if jux_word is not None:
            next_flag = next_flag and (next_item["item"] in jux_word)
        if next_flag is True:
            return self._search_jux(
                cur_item=next_item, cur_pos=cur_pos + 1, jux_type=jux_type, jux_word=jux_word, status_flag=True
            )
        if next_flag is not True:
            while self._judge_jux(search_list[cur_pos]) is True:
                cur_pos -= 1
        return cur_pos

    @staticmethod
    def _match_item(item, type_can):
        match_key = item["wordtag_label"].split("_")[0]
        return match_key in type_can or item["wordtag_label"] in type_can

    def _trig_handler(self, cur_item, head_conf):
        """Whether current item is a trigger, if True, return corresponding flag and configuration.

        Args:
            cur_item (Dict[str, Any]): current viewing ite,
            st_conf (Dict[str, Any]): config

        Returns:
            Tuple[str, Union[None, dict]]: [description]
        """
        trigger_conf = head_conf["trigger"]
        if cur_item["item"] in trigger_conf:
            # find a trigger, then judge whether it is a tail-trigger or a rel trigger.
            if trigger_conf[cur_item["item"]]["trigger_type"] == "role":
                # find a tail-trigger, then judge wordtag label.
                group_name = trigger_conf[cur_item["item"]]["group_name"]
                for tail_conf in head_conf["g_t_map"][group_name]:
                    if self._match_item(cur_item, tail_conf["main"]) is True:
                        return "trig_t", tail_conf
                else:
                    return "un_trig", None
            else:
                return "trig_g", None
        else:
            return "un_trig", None

    def _find_tail(self, search_range, sg_conf, head_hype):
        """Find tail role in `search_range`

        Args:
            search_range (List[int]): index range of `self._all_items`, items to be checked.
            sg_conf (Dict[str, Any]): configuration of group.
            head_type (str): wordtag label of head role item.
        """
        for i in search_range:
            item = self._all_items[i]
            if item["item"] in {"，", "？", "、", "。", "；"}:
                return -2, None
            for j, tail_conf in enumerate(sg_conf):
                flag = self._match_item(item, tail_conf["main"])
                if flag is True:
                    return i, tail_conf
                if item["wordtag_label"].startswith(head_hype):
                    return -1, None

        return -2, None

    def _find_supp(self, search_range, search_type):
        res = []
        for i in search_range:
            item = self._all_items[i]
            if item["item"] == "，":
                break
            if any(item["wordtag_label"].startswith(sup_t) for sup_t in search_type):
                res.append(item)
        return res if len(res) > 0 else None

    def _make_output(self, head_item, tail_item, group, source, support=None, trig_word=None, **kwargs):
        """Make formatted outputs of mined results.

        Args:
            head_item (Dict[str, Any]): [description]
            head_index (int): [description]
            tail_item (List[Dict[str, Any]]): [description]
            tail_indices (List[int]): [description]
            group (str): [description]
            source (str): [description]
            support (List[Dict[str, Any]], optional): [description]. Defaults to None.
            support_indices (List[int], optional): [description]. Defaults to None.
            trig_word (List[str], optional): [description]. Defaults to None.
            trig_indices (List[int], optional): [description]. Defaults to None.
        """
        res = {
            "HEAD_ROLE": {
                "item": head_item["item"],
                "type": head_item["wordtag_label"],
                "offset": head_item["offset"],
            },
            "TAIL_ROLE": [
                {"item": ti["item"], "offset": ti["offset"], "type": ti["wordtag_label"]} for ti in tail_item
            ],
            "GROUP": group,
            "SRC": source,
        }
        if support is not None:
            res["SUPPORT"] = [
                {
                    "item": si["item"],
                    "offset": si["offset"],
                    "type": si["wordtag_label"],
                }
                for si in support
            ]
        if trig_word is not None:
            res["TRIG"] = [
                {
                    "item": ti["item"],
                    "offset": ti["offset"],
                }
                for ti in trig_word
            ]
        return res

    def _reverse(self, res, group_name=None):
        ret = []
        for rev_head in res["TAIL_ROLE"]:
            rev_tmp = {
                "HEAD_ROLE": rev_head,
                "TAIL_ROLE": [res["HEAD_ROLE"]],
                "GROUP": group_name if group_name is not None else res["GROUP"],
            }
            if "SUPPORT" in res:
                rev_tmp["SUPPORT"] = res["SUPPORT"]
            if "TRIG" in res:
                rev_tmp["TRIG"] = res["TRIG"]
            rev_tmp["SRC"] = "REVERSE" if group_name is not None else res["SRC"]
            ret.append(rev_tmp)
        return ret

    def extract_spo(self, all_items):
        """Pipeline of mining procedure.

        Args:
            all_items ([type]): [description]
        """
        self._all_items = all_items

        res_cand = []

        # Match head role, and consider it as central, search others.
        for i, head_cand in enumerate(self._all_items):
            last_end = i
            try:
                datetime.strptime(head_cand["item"], "%Y年%m月%d日")
                head_cand["wordtag_label"] = "时间类_具体时间"
            except ValueError:
                pass

            if head_cand["wordtag_label"] in self._schema:
                head_conf = self._schema[head_cand["wordtag_label"]]
                head_type = head_cand["wordtag_label"]
            else:
                match_key = head_cand["wordtag_label"].split("_")[0]
                if match_key in self._schema:
                    head_conf = self._schema[match_key]
                    head_type = match_key
                else:
                    continue

            trig_status = "un_trig"

            # Consider `head_cand` as a start item, find trigger words behind.
            # We suppose that minning strategy is directed, so only search items behinds head.
            # If need, we can reverse constructed triples.
            j = i + 1
            while j < len(self._all_items):
                cur_item = all_items[j]
                cur_pos = j
                j += 1

                trig_status, trig_conf = self._trig_handler(cur_item, self._schema[head_type])

                # Find a tail role, generate corresponding triple.
                if trig_status == "trig_t":
                    trig_status = "un_trig"
                    tail_flag = True
                    for k in range(i + 1, j):
                        if self._all_items[k]["wordtag_label"] == head_cand["wordtag_label"]:
                            tail_flag = False
                            break
                    if tail_flag is False:
                        continue

                    group_name = head_conf["trigger"][cur_item["item"]]["group_name"]
                    del self._jux_buf[:]
                    idx = self._search_jux(
                        cur_item=cur_item, cur_pos=cur_pos, jux_type=trig_conf["main"], status_flag=True
                    )
                    supports = self._find_supp(search_range=range(j - 1, i, -1), search_type=trig_conf["support"])

                    tmp = self._make_output(
                        head_item=head_cand,
                        tail_item=self._jux_buf[:],
                        group=group_name,
                        support=supports,
                        source="TAIL",
                    )

                    # Reverse triple if group has relative.
                    if (
                        group_name in head_conf.get("rel_group", {})
                        or head_conf["trigger"][cur_item["item"]]["rev_flag"] is True
                    ):
                        rev_tmp = self._reverse(tmp, head_conf.get("rel_group", {}).get(group_name, None))
                        res_cand.extend(rev_tmp[:])
                    if head_conf["trigger"][cur_item["item"]]["rev_flag"] is False:
                        res_cand.append(tmp.copy())

                    j = idx + 1
                    last_end = idx
                    continue

                # Find a group trigger word, look for tail role items of current head role and group argument.
                # Searching range is items behind group trigger and items between head rold and group trigger word.
                if trig_status == "trig_g":
                    trig_status = "un_trig"
                    group_name = head_conf["trigger"][cur_item["item"]]["group_name"]

                    del self._jux_buf[:]
                    g_start_idx = j - 1
                    g_idx = self._search_jux(
                        cur_item=cur_item,
                        cur_pos=cur_pos,
                        jux_word=head_conf["trig_word"][group_name],
                        status_flag=True,
                    )

                    g_trig_words = self._jux_buf[:]
                    j = g_idx + 1

                    # Search right.
                    if j < len(self._all_items) - 1:
                        tail_idx, tail_conf = self._find_tail(
                            range(g_idx + 1, len(self._all_items)), head_conf["g_t_map"][group_name], head_type
                        )

                        if tail_idx > 0:
                            # Find a tail.
                            tail_item = self._all_items[tail_idx]
                            del self._jux_buf[:]
                            idx = self._search_jux(
                                cur_item=tail_item, cur_pos=tail_idx, status_flag=True, jux_type=tail_conf["main"]
                            )
                            tail_cand = self._jux_buf[:]
                            supports = self._find_supp(range(tail_idx - 1, i, -1), tail_conf["support"])

                            tmp = self._make_output(
                                head_item=head_cand,
                                tail_item=tail_cand,
                                group=group_name,
                                source="HGT",
                                support=supports,
                                trig_word=g_trig_words,
                            )

                            if (
                                group_name in head_conf.get("rel_group", {})
                                or head_conf["trigger"][cur_item["item"]]["rev_flag"] is True
                            ):
                                rev_tmp = self._reverse(tmp, head_conf.get("rel_group", {}).get(group_name, None))
                                res_cand.extend(rev_tmp[:])
                            if head_conf["trigger"][cur_item["item"]]["rev_flag"] is False:
                                res_cand.append(tmp.copy())

                            j = idx + 1
                            last_end = idx
                            continue

                    # Search left
                    if g_idx - i > len(g_trig_words):
                        tail_idx, tail_conf = self._find_tail(
                            range(g_start_idx, last_end, -1), head_conf["g_t_map"][group_name], head_type
                        )
                        tail_item = self._all_items[tail_idx]
                        if tail_idx > 0:
                            del self._jux_buf[:]
                            _ = self._search_jux(
                                cur_item=tail_item,
                                cur_pos=0,
                                jux_type=tail_conf["main"],
                                status_flag=True,
                                search_list=self._all_items[i + 1 : tail_idx][::-1],
                            )
                            tail_cand = self._jux_buf[:]
                            supports = self._find_supp(range(g_idx - 1, last_end, -1), tail_conf["support"])
                            last_end = g_idx

                            tmp = self._make_output(
                                head_item=head_cand,
                                tail_item=tail_cand,
                                group=group_name,
                                trig_word=g_trig_words,
                                source="HTG",
                                support=supports,
                            )

                            if (
                                group_name in head_conf.get("rel_group", {})
                                or head_conf["trigger"][cur_item["item"]]["rev_flag"] is True
                            ):
                                rev_tmp = self._reverse(tmp, head_conf.get("rel_group", {}).get(group_name, None))
                                res_cand.extend(rev_tmp[:])
                            if head_conf["trigger"][cur_item["item"]]["rev_flag"] is False:
                                res_cand.append(tmp.copy())
                            continue
        return res_cand


@dataclass
class DataCollatorGP:
    tokenizer: PretrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    label_maps: Optional[dict] = None
    task_type: Optional[str] = None

    def __call__(self, features: List[Dict[str, Union[List[int], paddle.Tensor]]]) -> Dict[str, paddle.Tensor]:
        new_features = [{k: v for k, v in f.items() if k not in ["offset_mapping", "text"]} for f in features]

        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
        )

        batch = [paddle.to_tensor(batch[k]) for k in batch.keys()]
        batch.append([feature["offset_mapping"] for feature in features])
        batch.append([feature["text"] for feature in features])
        return batch


def gp_decode(batch_outputs, offset_mappings, texts, label_maps, task_type="relation_extraction"):
    if task_type == "entity_extraction":
        batch_ent_results = []
        for entity_output, offset_mapping, text in zip(batch_outputs[0].numpy(), offset_mappings, texts):
            entity_output[:, [0, -1]] -= np.inf
            entity_output[:, :, [0, -1]] -= np.inf
            entity_probs = F.softmax(paddle.to_tensor(entity_output), axis=1).numpy()
            ent_list = []
            for l, start, end in zip(*np.where(entity_output > 0.0)):
                ent_prob = entity_probs[l, start, end]
                start, end = (offset_mapping[start][0], offset_mapping[end][-1])
                ent = {
                    "text": text[start:end],
                    "type": label_maps["id2entity"][str(l)],
                    "start_index": start,
                    "probability": ent_prob,
                }
                ent_list.append(ent)
            batch_ent_results.append(ent_list)
        return batch_ent_results
    else:
        batch_ent_results = []
        batch_rel_results = []
        for entity_output, head_output, tail_output, offset_mapping, text in zip(
            batch_outputs[0].numpy(),
            batch_outputs[1].numpy(),
            batch_outputs[2].numpy(),
            offset_mappings,
            texts,
        ):
            entity_output[:, [0, -1]] -= np.inf
            entity_output[:, :, [0, -1]] -= np.inf
            entity_probs = F.softmax(paddle.to_tensor(entity_output), axis=1).numpy()
            head_probs = F.softmax(paddle.to_tensor(head_output), axis=1).numpy()
            tail_probs = F.softmax(paddle.to_tensor(tail_output), axis=1).numpy()

            ents = set()
            ent_list = []
            for l, start, end in zip(*np.where(entity_output > 0.0)):
                ent_prob = entity_probs[l, start, end]
                ents.add((start, end))
                start, end = (offset_mapping[start][0], offset_mapping[end][-1])
                ent = {
                    "text": text[start:end],
                    "type": label_maps["id2entity"][str(l)],
                    "start_index": start,
                    "probability": ent_prob,
                }
                ent_list.append(ent)
            batch_ent_results.append(ent_list)

            rel_list = []
            for sh, st in ents:
                for oh, ot in ents:
                    p1s = np.where(head_output[:, sh, oh] > 0.0)[0]
                    p2s = np.where(tail_output[:, st, ot] > 0.0)[0]
                    ps = set(p1s) & set(p2s)
                    for p in ps:
                        rel_prob = head_probs[p, sh, oh] * tail_probs[p, st, ot]
                        if task_type == "relation_extraction":
                            rel = {
                                "subject": text[offset_mapping[sh][0] : offset_mapping[st][1]],
                                "predicate": label_maps["id2relation"][str(p)],
                                "object": text[offset_mapping[oh][0] : offset_mapping[ot][1]],
                                "subject_start_index": offset_mapping[sh][0],
                                "object_start_index": offset_mapping[oh][0],
                                "probability": rel_prob,
                            }
                        else:
                            rel = {
                                "aspect": text[offset_mapping[sh][0] : offset_mapping[st][1]],
                                "sentiment": label_maps["id2relation"][str(p)],
                                "opinion": text[offset_mapping[oh][0] : offset_mapping[ot][1]],
                                "aspect_start_index": offset_mapping[sh][0],
                                "opinion_start_index": offset_mapping[oh][0],
                                "probability": rel_prob,
                            }
                        rel_list.append(rel)
            batch_rel_results.append(rel_list)
        return (batch_ent_results, batch_rel_results)


DocSpan = namedtuple("DocSpan", ["start", "length"])

Example = namedtuple(
    "Example",
    [
        "keys",
        "key_labels",
        "doc_tokens",
        "text",
        "qas_id",
        "model_type",
        "seq_labels",
        "ori_boxes",
        "boxes",
        "segment_ids",
        "symbol_ids",
        "im_base64",
        "image_rois",
    ],
)

Feature = namedtuple(
    "Feature",
    [
        "unique_id",
        "example_index",
        "qas_id",
        "doc_span_index",
        "tokens",
        "token_to_orig_map",
        "token_is_max_context",
        "token_ids",
        "position_ids",
        "text_type_ids",
        "text_symbol_ids",
        "overlaps",
        "key_labels",
        "seq_labels",
        "se_seq_labels",
        "bio_seq_labels",
        "bioes_seq_labels",
        "keys",
        "model_type",
        "doc_tokens",
        "doc_labels",
        "text",
        "boxes",
        "segment_ids",
        "im_base64",
        "image_rois",
    ],
)


class Compose(object):
    """compose"""

    def __init__(self, transforms, ctx=None):
        """init"""
        self.transforms = transforms
        self.ctx = ctx

    def __call__(self, data):
        """call"""
        ctx = self.ctx if self.ctx else {}
        for f in self.transforms:
            try:
                data = f(data, ctx)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning("fail to map op [{}] with error: {} and stack:\n{}".format(f, e, str(stack_info)))
                raise e
        return data


def batch_arrange(batch_samples, fields):
    def _segm(samples):
        """"""
        assert "gt_poly" in samples
        segms = samples["gt_poly"]
        if "is_crowd" in samples:
            is_crowd = samples["is_crowd"]
            if len(segms) != 0:
                assert len(segms) == is_crowd.shape[0]

        gt_masks = []
        valid = True
        for i in range(len(segms)):
            segm = segms[i]
            gt_segm = []
            if "is_crowd" in samples and is_crowd[i]:
                gt_segm.append([[0, 0]])
            else:
                for poly in segm:
                    if len(poly) == 0:
                        valid = False
                        break
                    gt_segm.append(np.array(poly).reshape(-1, 2))
            if (not valid) or len(gt_segm) == 0:
                break
            gt_masks.append(gt_segm)
        return gt_masks

    def im_shape(samples, dim=3):
        # hard code
        assert "h" in samples
        assert "w" in samples
        if dim == 3:  # RCNN, ..
            return np.array((samples["h"], samples["w"], 1), dtype=np.float32)
        else:  # YOLOv3, ..
            return np.array((samples["h"], samples["w"]), dtype=np.int32)

    arrange_batch = []
    for samples in batch_samples:
        one_ins = ()
        for i, field in enumerate(fields):
            if field == "gt_mask":
                one_ins += (_segm(samples),)
            elif field == "im_shape":
                one_ins += (im_shape(samples),)
            elif field == "im_size":
                one_ins += (im_shape(samples, 2),)
            else:
                if field == "is_difficult":
                    field = "difficult"
                assert field in samples, "{} not in samples".format(field)
                one_ins += (samples[field],)
        arrange_batch.append(one_ins)
    return arrange_batch


class ProcessReader(object):
    """
    Args:
        dataset (DataSet): DataSet object
        sample_transforms (list of BaseOperator): a list of sample transforms
            operators.
        batch_transforms (list of BaseOperator): a list of batch transforms
            operators.
        batch_size (int): batch size.
        shuffle (bool): whether shuffle dataset or not. Default False.
        drop_last (bool): whether drop last batch or not. Default False.
        drop_empty (bool): whether drop sample when it's gt is empty or not.
            Default True.
        mixup_epoch (int): mixup epoc number. Default is -1, meaning
            not use mixup.
        cutmix_epoch (int): cutmix epoc number. Default is -1, meaning
            not use cutmix.
        class_aware_sampling (bool): whether use class-aware sampling or not.
            Default False.
        worker_num (int): number of working threads/processes.
            Default -1, meaning not use multi-threads/multi-processes.
        use_process (bool): whether use multi-processes or not.
            It only works when worker_num > 1. Default False.
        bufsize (int): buffer size for multi-threads/multi-processes,
            please note, one instance in buffer is one batch data.
        memsize (str): size of shared memory used in result queue when
            use_process is true. Default 3G.
        inputs_def (dict): network input definition use to get input fields,
            which is used to determine the order of returned data.
        devices_num (int): number of devices.
        num_trainers (int): number of trainers. Default 1.
    """

    def __init__(
        self,
        dataset=None,
        sample_transforms=None,
        batch_transforms=None,
        batch_size=None,
        shuffle=False,
        drop_last=False,
        drop_empty=True,
        mixup_epoch=-1,
        cutmix_epoch=-1,
        class_aware_sampling=False,
        use_process=False,
        use_fine_grained_loss=False,
        num_classes=80,
        bufsize=-1,
        memsize="3G",
        inputs_def=None,
        devices_num=1,
        num_trainers=1,
    ):
        """"""
        self._fields = copy.deepcopy(inputs_def["fields"]) if inputs_def else None

        # transform
        self._sample_transforms = Compose(sample_transforms, {"fields": self._fields})
        self._batch_transforms = None

        if batch_transforms:
            batch_transforms = [bt for bt in batch_transforms]
            self._batch_transforms = Compose(batch_transforms, {"fields": self._fields})

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._drop_empty = drop_empty

        # sampling
        self._mixup_epoch = mixup_epoch // num_trainers
        self._cutmix_epoch = cutmix_epoch // num_trainers
        self._class_aware_sampling = class_aware_sampling

        self._indexes = None
        self._pos = -1
        self._epoch = -1
        self._curr_iter = 0

    def process(self, dataset):
        """process"""
        batch = self._load_batch(dataset)
        res = self.worker(self._drop_empty, batch)
        return res

    def _load_batch(self, dataset):
        batch = []
        for data in dataset:
            sample = copy.deepcopy(data)
            batch.append(sample)
        return batch

    def worker(self, drop_empty=True, batch_samples=None):
        """
        sample transform and batch transform.
        """
        batch = []
        for sample in batch_samples:
            sample = self._sample_transforms(sample)
            batch.append(sample)
        if len(batch) > 0 and self._batch_transforms:
            batch = self._batch_transforms(batch)
        if len(batch) > 0 and self._fields:
            batch = batch_arrange(batch, self._fields)
        return batch


def pad_batch_data(
    insts,
    pad_idx=0,
    max_seq_len=None,
    return_pos=False,
    return_input_mask=False,
    return_max_len=False,
    return_num_token=False,
    return_seq_lens=False,
    pad_2d_pos_ids=False,
    pad_segment_id=False,
    select=False,
    extract=False,
):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts) if max_seq_len is None else max_seq_len
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.
    if pad_2d_pos_ids:
        boxes = [x + [[0, 0, 0, 0]] * (max_len - len(x)) for x in insts]
        boxes = np.array(boxes, dtype="int64")
        return boxes

    inst_data = np.array([inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, max_len, 1])]

    # position data
    if return_pos:
        inst_pos = np.array([list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst)) for inst in insts])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] * (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape([-1, 1])]

    return return_list if len(return_list) > 1 else return_list[0]


class ImageReader(object):
    def __init__(
        self,
        super_rel_pos,
        tokenizer,
        max_key_len=16,
        max_seq_len=512,
        image_size=1024,
        block_w=7,
        block_h=7,
        im_npos=224,
    ):
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()

        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.mask_id = self.vocab["[MASK]"]
        self.pad = "[PAD]"
        self.cls = "[CLS]"
        self.sep = "[SEP]"
        self.mask = "[MASK]"

        self.super_rel_pos = super_rel_pos
        self.max_key_len = max_key_len
        self.max_seq_len = max_seq_len
        self.doc_stride = 128
        self.unique_id = 10000000

        self.examples = {}
        self.features = {}

        self.image_size = image_size
        self.block_w = block_w
        self.block_h = block_h
        self.im_npos = im_npos
        self.image_rois = []
        cut_width, cut_height = int(self.image_size / self.block_w), int(self.image_size / self.block_h)
        for idh in range(self.block_h):
            for idw in range(self.block_w):
                self.image_rois.append([idw * cut_width, idh * cut_height, cut_width, cut_height])

        sample_trans = [
            DecodeImage(),
            ResizeImage(target_size=self.im_npos, interp=1),
            NormalizeImage(
                is_channel_first=False,
                mean=[103.530, 116.280, 123.675],
                std=[57.375, 57.120, 58.395],
            ),
            Permute(to_bgr=False),
        ]

        batch_trans = [PadBatch(pad_to_stride=32, use_padded_im_info=True)]

        inputs_def = {
            "fields": ["image", "im_info", "im_id", "gt_bbox"],
        }
        self.data_loader = ProcessReader(
            sample_transforms=sample_trans,
            batch_transforms=batch_trans,
            shuffle=False,
            drop_empty=True,
            inputs_def=inputs_def,
        )

    def ppocr2example(self, ocr_res, img_path, querys):
        examples = []
        segments = []
        for rst in ocr_res:
            left = min(rst[0][0][0], rst[0][3][0])
            top = min(rst[0][0][-1], rst[0][1][-1])
            width = max(rst[0][1][0], rst[0][2][0]) - min(rst[0][0][0], rst[0][3][0])
            height = max(rst[0][2][-1], rst[0][3][-1]) - min(rst[0][0][-1], rst[0][1][-1])
            segments.append({"bbox": Bbox(*[left, top, width, height]), "text": rst[-1][0]})
        segments.sort(key=cmp_to_key(two_dimension_sort_layout))
        # 2. im_base64
        img_base64 = img2base64(img_path)
        # 3. doc_tokens, doc_boxes, segment_ids
        doc_tokens = []
        doc_boxes = []
        ori_boxes = []
        doc_segment_ids = []

        im_w_box = max([seg["bbox"].left + seg["bbox"].width for seg in segments]) + 20
        im_h_box = max([seg["bbox"].top + seg["bbox"].height for seg in segments]) + 20
        img = Image.open(img_path)
        im_w, im_h = img.size
        im_w, im_h = max(im_w, im_w_box), max(im_h, im_h_box)

        scale_x = self.image_size / im_w
        scale_y = self.image_size / im_h
        for segment_id, segment in enumerate(segments):
            bbox = segment["bbox"]  # x, y, w, h
            x1, y1, w, h = bbox.left, bbox.top, bbox.width, bbox.height
            sc_w = int(min(w * scale_x, self.image_size - 1))
            sc_h = int(min(h * scale_y, self.image_size - 1))
            sc_y1 = int(max(0, min(y1 * scale_y, self.image_size - h - 1)))
            sc_x1 = int(max(0, min(x1 * scale_x, self.image_size - w - 1)))
            if w < 0:
                raise ValueError("Incorrect bbox, please check the input word boxes.")
            ori_bbox = Bbox(*[x1, y1, w, h])
            sc_bbox = Bbox(*[sc_x1, sc_y1, sc_w, sc_h])
            text = segment["text"]
            char_num = []
            eng_word = ""
            for char in text:
                if not check(char) and not eng_word:
                    doc_tokens.append([char])
                    doc_segment_ids.append([segment_id])
                    char_num.append(2)
                elif not check(char) and eng_word:
                    doc_tokens.append([eng_word])
                    doc_segment_ids.append([segment_id])
                    char_num.append(len(eng_word))
                    eng_word = ""
                    doc_tokens.append([char])
                    doc_segment_ids.append([segment_id])
                    char_num.append(2)
                else:
                    eng_word += char
            if eng_word:
                doc_tokens.append([eng_word])
                doc_segment_ids.append([segment_id])
                char_num.append(len(eng_word))
            ori_char_width = round(ori_bbox.width / sum(char_num), 1)
            sc_char_width = round(sc_bbox.width / sum(char_num), 1)
            for chr_idx in range(len(char_num)):
                if chr_idx == 0:
                    doc_boxes.append(
                        [Bbox(*[sc_bbox.left, sc_bbox.top, (sc_char_width * char_num[chr_idx]), sc_bbox.height])]
                    )
                    ori_boxes.append(
                        [Bbox(*[ori_bbox.left, ori_bbox.top, (ori_char_width * char_num[chr_idx]), ori_bbox.height])]
                    )
                else:
                    doc_boxes.append(
                        [
                            Bbox(
                                *[
                                    sc_bbox.left + (sc_char_width * sum(char_num[:chr_idx])),
                                    sc_bbox.top,
                                    (sc_char_width * char_num[chr_idx]),
                                    sc_bbox.height,
                                ]
                            )
                        ]
                    )
                    ori_boxes.append(
                        [
                            Bbox(
                                *[
                                    ori_bbox.left + (ori_char_width * sum(char_num[:chr_idx])),
                                    ori_bbox.top,
                                    (ori_char_width * char_num[chr_idx]),
                                    ori_bbox.height,
                                ]
                            )
                        ]
                    )

        qas_id = 0
        for query in querys:
            example = Example(
                keys=[query],
                key_labels=[0],
                doc_tokens=doc_tokens,
                seq_labels=[0 for one in doc_tokens],
                text="",
                qas_id="0_" + str(qas_id),
                model_type=None,
                ori_boxes=ori_boxes,
                boxes=doc_boxes,
                segment_ids=doc_segment_ids,
                symbol_ids=None,
                image_rois=self.image_rois,
                im_base64=img_base64,
            )

            examples.append(example)
            qas_id += 1
        return examples

    def box2example(self, ocr_res, img_path, querys):
        """
        ocr_res = [[word_str, [x1, y1, x2, y2]], [word_str, [x1, y1, x2, y2]], ...]
        """
        examples = []
        doc_boxes = []
        ori_boxes = []
        boxes = [x[1] for x in ocr_res]
        im_w_box = max([b[2] for b in boxes]) + 20
        im_h_box = max([b[3] for b in boxes]) + 20
        img = Image.open(img_path)
        im_w, im_h = img.size
        im_w, im_h = max(im_w, im_w_box), max(im_h, im_h_box)

        scale_x = self.image_size / im_w
        scale_y = self.image_size / im_h
        for box in boxes:
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Invalid bbox format")
            w = max(x1, x2) - min(x1, x2)
            h = max(y1, y2) - min(y1, y2)
            ori_boxes.append([Bbox(*[x1, y1, w, h])])
            w = int(min(w * scale_x, self.image_size - 1))
            h = int(min(h * scale_y, self.image_size - 1))
            x1 = int(max(0, min(x1 * scale_x, self.image_size - w - 1)))
            y1 = int(max(0, min(y1 * scale_y, self.image_size - h - 1)))
            if w < 0:
                raise ValueError("Invalid bbox format")
            doc_boxes.append([Bbox(*[x1, y1, w, h])])

        img_base64 = img2base64(img_path)

        doc_tokens = [[x[0]] for x in ocr_res]
        doc_segment_ids = [[0]] * len(doc_tokens)

        qas_id = 0
        for query in querys:
            example = Example(
                keys=[query],
                key_labels=[0],
                doc_tokens=doc_tokens,
                seq_labels=[0 for one in doc_tokens],
                text="",
                qas_id=str(qas_id),
                model_type=None,
                ori_boxes=ori_boxes,
                boxes=doc_boxes,
                segment_ids=doc_segment_ids,
                symbol_ids=None,
                image_rois=self.image_rois,
                im_base64=img_base64,
            )

            if not (len(example.doc_tokens) == len(example.boxes) == len(example.segment_ids)):
                raise ValueError(
                    "Incorrect word_boxes, the format should be `List[str, Tuple[float, float, float, float]]`"
                )

            examples.append(example)
            qas_id += 1

        return examples

    def example2feature(self, example, tokenizer, max_line_id=128):
        features = []
        all_doc_tokens = []
        tok_to_orig_index = []
        boxes = []
        segment_ids = []
        all_doc_labels = []

        query_tokens = tokenizer.tokenize("&" + str(example.keys[0]))[1:][: self.max_key_len]

        for i, (token_list, box_list, seg_list, l) in enumerate(
            zip(example.doc_tokens, example.boxes, example.segment_ids, example.seq_labels)
        ):
            assert len(token_list) == len(box_list) == len(seg_list)
            for idt, (token, box, seg) in enumerate(zip(token_list, box_list, seg_list)):
                sub_tokens = tokenizer.tokenize("&" + token)[1:]
                for ii, sub_token in enumerate(sub_tokens):
                    width_split = box.width / len(sub_tokens)
                    boxes.append([box.left + ii * width_split, box.top, width_split, box.height])
                    segment_ids.append(seg)
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)
                    all_doc_labels.extend([0])

        max_tokens_for_doc = self.max_seq_len - len(query_tokens) - 4
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, self.doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            labels = []
            feature_segment_ids = []
            feature_boxes = []
            token_to_orig_map = {}
            token_is_max_context = {}
            text_type_ids = []
            tokens.append(self.cls)
            feature_boxes.append([0, 0, 0, 0])
            labels.append(0)
            text_type_ids.append(0)
            feature_segment_ids.append(max_line_id - 1)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                is_max_context = self._check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])

                feature_boxes.append(boxes[split_token_index])
                feature_segment_ids.append(segment_ids[split_token_index])
                text_type_ids.append(0)
                labels.append(all_doc_labels[split_token_index])

            tokens.append(self.sep)
            feature_boxes.append([0, 0, 0, 0])
            text_type_ids.append(0)
            feature_segment_ids.append(max_line_id - 1)
            labels.append(0)
            for token in query_tokens:
                tokens.append(token)
                feature_boxes.append([0, 0, 0, 0])
                feature_segment_ids.append(max_line_id - 1)
                text_type_ids.append(1)
                labels.append(0)

            tokens = tokens + [self.sep]
            feature_boxes.extend([[0, 0, 0, 0]])
            feature_segment_ids = feature_segment_ids + [max_line_id - 1]
            text_type_ids = text_type_ids + [1]
            labels.append(0)

            position_ids = list(range(len(tokens)))
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            feature_segment_ids = [x % max_line_id for x in feature_segment_ids]

            feature = Feature(
                unique_id=self.unique_id,
                example_index=0,
                qas_id=example.qas_id,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                token_ids=token_ids,
                position_ids=position_ids,
                text_type_ids=text_type_ids,
                text_symbol_ids=None,
                overlaps=None,
                keys=example.keys,
                seq_labels=labels,
                se_seq_labels=None,
                bio_seq_labels=None,
                bioes_seq_labels=None,
                key_labels=example.key_labels,
                model_type=example.model_type,
                doc_tokens=example.doc_tokens,
                doc_labels=example.seq_labels,
                text=example.text,
                boxes=feature_boxes,
                segment_ids=feature_segment_ids,
                im_base64=example.im_base64,
                image_rois=example.image_rois,
            )
            features.append(feature)
            self.unique_id += 1
        return features

    def _pad_batch_records(self, batch_records, max_line_id=128, phase="infer"):
        """pad batch records"""
        return_list = []
        batch_token_ids = []
        batch_sent_ids = []
        batch_pos_ids = []
        batch_2d_pos_ids = []
        batch_segment_ids = []
        batch_labels = []
        batch_unique_id = []
        batch_image_base64 = []
        batch_image_rois = []

        for i in range(len(batch_records)):
            batch_token_ids.append(batch_records[i].token_ids)
            batch_sent_ids.append(batch_records[i].text_type_ids)
            batch_segment_ids.append(batch_records[i].segment_ids)
            batch_labels.append(batch_records[i].seq_labels)
            batch_unique_id.append(batch_records[i].unique_id)
            batch_pos_ids.append(batch_records[i].position_ids)
            batch_2d_pos_ids.append(batch_records[i].boxes)
            batch_image_base64.append(batch_records[i].im_base64)
            batch_image_rois.append(batch_records[i].image_rois)

        padded_token_ids, _ = pad_batch_data(batch_token_ids, pad_idx=self.pad_id, return_input_mask=True)
        padded_sent_ids = pad_batch_data(batch_sent_ids, pad_idx=self.pad_id)
        padded_pos_ids = pad_batch_data(batch_pos_ids, pad_idx=self.pad_id)
        new_padded_pos_ids = []
        for idp, pos_ids in enumerate(padded_pos_ids):
            new_padded_pos_ids.append(
                np.concatenate((pos_ids, np.array([[x] for x in range(self.block_w * self.block_h)])), axis=0)
            )
        padded_pos_ids = np.array(new_padded_pos_ids)
        padded_2d_pos_ids = pad_batch_data(batch_2d_pos_ids, pad_2d_pos_ids=True, select=False, extract=True)
        new_padded_2d_pos_ids = []
        for pos_ids_2d, batch_record in zip(padded_2d_pos_ids, batch_records):
            new_padded_2d_pos_ids.append(np.concatenate((pos_ids_2d, np.array(batch_record.image_rois)), axis=0))
        padded_2d_pos_ids = np.array(new_padded_2d_pos_ids)
        padded_segment_ids = pad_batch_data(batch_segment_ids, pad_idx=max_line_id - 1)

        input_mask_mat = self._build_input_mask(
            np.array([list(x) + [[-1] for _ in range(self.block_w * self.block_h)] for x in padded_token_ids])
        )
        super_rel_pos = self._build_rel_pos(
            np.array([list(x) + [[-1] for _ in range(self.block_w * self.block_h)] for x in padded_token_ids])
        )

        unique_id = np.array(batch_unique_id).astype("float32").reshape([-1, 1])

        bsz, seq_len, _ = padded_token_ids.shape
        task_ids = np.ones((bsz, seq_len, 1)).astype("int64")
        for b in range(bsz):
            if np.sum(padded_2d_pos_ids[b]) > 0:
                task_ids[b, :, :] = 0
            else:
                task_ids[b, :, :] = 1

        coco_data = self.generate_coco_data(
            [""] * len(batch_image_base64),
            batch_image_base64,
            [self.image_size] * len(batch_image_base64),
            [self.image_size] * len(batch_image_base64),
            batch_image_rois,
        )

        image_data = self.im_make_batch(
            self.data_loader.process(coco_data),
            self.block_w * self.block_h,
            len(batch_image_base64),
        )

        return_list = [
            padded_token_ids,
            padded_sent_ids,
            padded_pos_ids,
            padded_2d_pos_ids,
            padded_segment_ids,
            task_ids,
            input_mask_mat,
            super_rel_pos,
            unique_id,
            image_data,
        ]
        return return_list

    def data_generator(self, ocr_res, img_path, querys, batch_size, ocr_type="ppocr", phase="infer"):
        if ocr_type == "ppocr":
            self.examples[phase] = self.ppocr2example(ocr_res, img_path, querys)
        elif ocr_type == "word_boxes":
            self.examples[phase] = self.box2example(ocr_res, img_path, querys)
        self.features[phase] = sum([self.example2feature(e, self.tokenizer) for e in self.examples[phase]], [])
        for batch_data in self._prepare_batch_data(self.features[phase], batch_size, phase=phase):
            yield self._pad_batch_records(batch_data)

    def _prepare_batch_data(self, features, batch_size, phase=None):
        """generate batch records"""
        batch_records = []
        for feature in features:
            to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(feature)
            else:
                yield batch_records
                batch_records = [feature]

        if phase == "infer" and batch_records:
            yield batch_records

    def _build_input_mask(self, padded_token_ids):
        """build_input_mask"""
        bsz, seq_len, _ = padded_token_ids.shape
        return np.ones((bsz, seq_len, seq_len)).astype("float32")

    def _build_rel_pos(self, padded_token_ids):
        """build relative position"""
        bsz, seq_len, _ = padded_token_ids.shape
        rel_pos = np.zeros((bsz, seq_len, seq_len)).astype("int64")
        return rel_pos

    def generate_coco_data(
        self,
        batch_image_path,
        batch_image_base64,
        batch_scaled_width,
        batch_scaled_height,
        batch_rois,
    ):
        """generator coco data"""

        def transform(dataset):
            roidbs = []
            for i in dataset:
                rvl_rec = {
                    "im_file": i[0],
                    "im_id": np.array([i[1]]),
                    "h": i[2],
                    "w": i[3],
                    "gt_bbox": i[4],
                    "cover_box": i[5],
                    "im_base64": i[6],
                }

                roidbs.append(rvl_rec)
            return roidbs

        result = []
        for image_path, im_base64, width, height, roi in zip(
            batch_image_path,
            batch_image_base64,
            batch_scaled_width,
            batch_scaled_height,
            batch_rois,
        ):
            result.append((image_path, 0, height, width, roi, None, im_base64))
        return transform(result)

    def im_make_batch(self, dataset, image_boxes_nums, bsize):
        """make image batch"""
        img_batch = np.array([i[0] for i in dataset], "float32")
        return img_batch

    def BIO2SPAN(self, BIO):
        start_label, end_label = [], []
        for seq in BIO:
            first_one = True
            start_pos = [1 if x == 2 else 0 for x in seq]
            if sum(start_pos) == 0 and sum(seq) != 0:
                start_pos = []
                for idp, p in enumerate(seq):
                    if p == 1 and first_one:
                        start_pos.append(1)
                        first_one = False
                    else:
                        start_pos.append(0)

            start_label.append(start_pos)

            end_tmp = []
            for index, s in enumerate(seq):
                if s == -100 or s == 0:
                    end_tmp.append(s)
                elif s == 2 and index + 1 < len(seq) and (seq[index + 1] == 0 or seq[index + 1] == 2):
                    end_tmp.append(1)
                elif s == 2 and index + 1 < len(seq) and seq[index + 1] != 0:
                    end_tmp.append(0)
                elif s == 2 and index + 1 == len(seq):
                    end_tmp.append(1)
                elif s == 1 and (index + 1 == len(seq) or seq[index + 1] != 1):
                    end_tmp.append(1)
                else:
                    end_tmp.append(0)
            end_label.append(end_tmp)

        return start_label, end_label

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index
        return cur_span_index == best_span_index


def get_doc_pred(result, ans_pos, example, tokenizer, feature, do_lower_case, all_key_probs, example_index):
    def _compute_softmax(scores):
        """Compute softmax probability over raw logits."""
        if len(scores) == 0:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs

    preds = []
    for start_index, end_index in ans_pos:
        # process data
        tok_tokens = feature.tokens[start_index : end_index + 1]
        tok_text = " ".join(tok_tokens)
        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")
        tok_text = tok_text.strip()
        tok_text = "".join(tok_text.split())

        orig_doc_start = feature.token_to_orig_map[start_index]
        orig_doc_end = feature.token_to_orig_map[end_index]
        orig_tokens = example.doc_tokens[orig_doc_start : orig_doc_end + 1]

        # Clean whitespace
        orig_text = "".join(["".join(x) for x in orig_tokens])
        final_text = get_final_text(tok_text, orig_text, tokenizer, do_lower_case)

        probs = []
        for idx, logit in enumerate(result.seq_logits[start_index : end_index + 1]):
            if idx == 0:
                # -1 is for B in  OIB or I in OI
                probs.append(_compute_softmax(logit)[-1])
            else:
                # 1 is for I in OIB or I in OI
                probs.append(_compute_softmax(logit)[1])
        avg_prob = sum(probs) / len(probs)
        preds.append({"value": final_text, "prob": round(avg_prob, 2), "start": orig_doc_start, "end": orig_doc_end})
    return preds


def get_final_text(pred_text, orig_text, tokenizer, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text


def find_bio_pos(label):
    """find answer position from BIO label"""
    e = []
    cand_ans = []
    last_l = None
    for idx, l in enumerate(label):
        if l == "O":
            if e:
                cand_ans.append([e[0], e[-1]])
            e = []
        elif l.startswith("B"):
            if last_l == "O" or last_l is None:
                if len(e) != 0:
                    e = []
                e.append(idx)
            else:  # I B
                if e:
                    cand_ans.append([e[0], e[-1]])
                    e = []
                e.append(idx)
        elif l.startswith("I"):
            if len(e) == 0:
                continue
            else:
                e.append(idx)
        last_l = l
    if e:
        cand_ans.append([e[0], e[-1]])
    return cand_ans


def viterbi_decode(logits):
    np_logits = np.array(logits)  # shape: L * D
    length, dim = np_logits.shape
    f = np.zeros(np_logits.shape)
    path = [["" for i in range(dim)] for j in range(length)]
    label_scheme = "OIB"
    # oib label 0:O, 1:I, 2:B
    # illegal matrix: [O, I ,B, start, end] * [O, I, B, start, end]
    illegal = np.array([[0, -1, 0, -1, 0], [0, 0, 0, -1, 0], [0, 0, 0, 0, 0], [0, -1, 0, 0, 0], [-1, -1, -1, -1, -1]])
    illegal = illegal * 1000

    f[0, :] = np_logits[0, :] + illegal[3, :3]
    path[0] = [label_scheme[i] for i in range(dim)]

    for step in range(1, length):
        last_s = f[step - 1, :]
        for d in range(dim):
            cand_score = illegal[:3, d] + last_s + np_logits[step, d]
            f[step, d] = np.max(cand_score)
            path[step][d] = path[step - 1][np.argmax(cand_score)] + label_scheme[d]
    final_path = path[-1][np.argmax(f[-1, :])]
    return final_path


def find_answer_pos(logits, feature):
    start_index = -1
    end_index = -1
    ans = []
    cand_ans = []

    best_path = viterbi_decode(logits)
    cand_ans = find_bio_pos(best_path)

    for start_index, end_index in cand_ans:
        is_valid = True
        if start_index not in feature.token_to_orig_map:
            is_valid = False
        if end_index not in feature.token_to_orig_map:
            is_valid = False
        if not feature.token_is_max_context.get(start_index, False):
            is_valid = False
        if end_index < start_index:
            is_valid = False
        if is_valid:
            ans.append([start_index, end_index])

    return ans


def calEuclidean(x_list, y_list):
    """
    Calculate euclidean distance
    """
    if x_list is None or y_list is None:
        return None
    else:
        dist = np.sqrt(np.square(x_list[0] - y_list[0]) + np.square(x_list[1] - y_list[1]))
        return dist


def longestCommonSequence(question_tokens, context_tokens):
    """
    Longest common sequence
    """
    max_index = -1
    max_len = 0
    m, n = len(question_tokens), len(context_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if question_tokens[i - 1].lower() == context_tokens[j - 1][0].lower():
                dp[i][j] = 1 + dp[i - 1][j - 1]
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    max_index = j - 1
    return max_index, max_len


def sort_res(prompt, ans_list, context, boxes, lang="en"):
    if len(ans_list) == 1:
        return ans_list
    else:
        ans_val = []
        for ans in ans_list:
            ans_val.append(ans["value"])
        if len(set(ans_val)) == len(ans_val):
            sorted_ans_list = sorted(ans_list, key=lambda x: x["prob"], reverse=True)
            return sorted_ans_list
        else:
            if lang == "en":
                clean_prompt = [word for word in prompt.split(" ")]
            else:
                clean_prompt = [word for word in prompt]

            max_index, max_len = longestCommonSequence(clean_prompt, context)
            if max_index == -1:
                sorted_ans_list = sorted(ans_list, key=lambda x: x["prob"], reverse=True)
                return sorted_ans_list
            else:
                prompt_center = []
                for idx in range(max_index - max_len + 1, max_index + 1):
                    box = boxes[idx][0]
                    x = box.left + box.width / 2
                    y = box.top + box.height / 2
                    prompt_center.append([x, y])

                ans_center = []
                ans_prob = []
                for ans in ans_list:
                    ans_prob.append(ans["prob"])
                    cent_list = []
                    for idx in range(ans["start"], ans["end"] + 1):
                        box = boxes[idx][0]
                        x = box.left + box.width / 2
                        y = box.top + box.height / 2
                        cent_list.append([x, y])
                    ans_center.append(cent_list)

                ans_odist = []
                for ans_c in ans_center:
                    odist = 0
                    for a_c in ans_c:
                        for p_c in prompt_center:
                            odist += calEuclidean(a_c, p_c)
                    odist /= len(ans_c)
                    ans_odist.append(odist * (-1))

                ans_score = np.sum([ans_prob, ans_odist], axis=0).tolist()
                sorted_ans_list = sorted(ans_list, key=lambda x: ans_score[ans_list.index(x)], reverse=True)
                return sorted_ans_list
