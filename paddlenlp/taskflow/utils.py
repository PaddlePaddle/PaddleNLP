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

import os
import re
import csv
from datetime import datetime
import json
import pickle
import warnings
import contextlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
from paddle.dataset.common import md5file
from ..utils.log import logger
from ..utils.downloader import get_path_from_url, DownloaderCheck

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
    Check the resource statuc in the specified task.

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
    para = re.sub(r'([。！？\?])([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\.{6})([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\…{2})([^”’])', r'\1\n\2', para)
    para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
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

    def __init__(self,
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
                 data: Optional[Dict[str, Any]] = None):
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
                "links": []
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
        return cls(sid=data["termid"],
                   term=data["term"],
                   base=data["src"],
                   term_type=data["termtype"],
                   sub_type=data["subtype"],
                   sub_term=data["subterms"],
                   alias=data["alias"],
                   alias_ext=data["alias_ext"],
                   data=data)

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
    """TermTree class.
    """

    def __init__(self):
        self._nodes: Dict[str, TermTreeNode] = {}
        self._root = TermTreeNode(sid="root",
                                  term="root",
                                  base="cb",
                                  node_type="root",
                                  level=0)
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
                    self.add_type(type_name=row["type-2"],
                                  hyper_type=row["type-1"])
                if row["type-3"] != "" and row["type-3"] not in self:
                    self.add_type(type_name=row["type-3"],
                                  hyper_type=row["type-2"])

    def __judge_term_node(self, node: TermTreeNode) -> bool:
        if node.termtype not in self:
            raise ValueError(
                f"Term type of new node {node.termtype} does not exists.")
        if node.sid in self:
            warnings.warn(f"{node.sid} exists, will be replaced by new node.")

    def add_term(self,
                 term: Optional[str] = None,
                 base: Optional[str] = None,
                 term_type: Optional[str] = None,
                 sub_type: Optional[List[str]] = None,
                 sub_term: Optional[List[str]] = None,
                 alias: Optional[List[str]] = None,
                 alias_ext: Optional[List[str]] = None,
                 data: Optional[Dict[str, Any]] = None):
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
            new_node = TermTreeNode(sid=f"{term_type}_{base}_{term}",
                                    term=term,
                                    base=base,
                                    term_type=term_type,
                                    sub_term=sub_term,
                                    sub_type=sub_type,
                                    alias=alias,
                                    alias_ext=alias_ext,
                                    node_type="term")
        self.__judge_term_node(new_node)
        self._nodes[new_node.sid] = new_node
        self.__build_index(new_node)

    def add_type(self, type_name, hyper_type):
        if type_name in self._nodes:
            raise ValueError(f"Term Type {type_name} exists.")
        if hyper_type not in self._nodes:
            raise ValueError(
                f"Hyper type {hyper_type} does not exist, please add it first.")
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
            level=self._nodes[hyper_type].level + 1)
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

    def find_term(
            self,
            term: str,
            term_type: Optional[str] = None
    ) -> Tuple[bool, Union[List[str], None]]:
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

    def __dfs(self, cur_id: str, depth: int, path: Dict[str, str],
              writer: csv.DictWriter):
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
        with open(f"{save_dir}/termtree_type.csv",
                  "wt",
                  encoding="utf-8",
                  newline="") as fp:
            fieldnames = ["type-1", "type-2", "type-3"]
            csv_writer = csv.DictWriter(fp,
                                        delimiter="\t",
                                        fieldnames=fieldnames)
            csv_writer.writeheader()
            self.__dfs("root", 0, out_path, csv_writer)
        with open(f"{save_dir}/termtree_data",
                  "w",
                  encoding="utf-8",
                  newline="") as fp:
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


class TriedTree(object):
    """Implementataion of TriedTree
    """

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
        with open(filename, 'r', encoding='utf8') as f:
            for line in f:
                if sep == None:
                    words = line.strip().split()
                else:
                    sep = strdecode(sep)
                    words = line.strip().split(sep)

                if len(words) == 0:
                    continue

                phrase = ""
                tags = []
                offset = []
                for word in words:
                    if word.rfind('/') < 1:
                        phrase += word
                        tags.append('')
                    else:
                        phrase += word[:word.rfind('/')]
                        tags.append(word[word.rfind('/') + 1:])
                    offset.append(len(phrase))

                if len(phrase) < 2 and tags[0] == '':
                    continue

                self.dictitem[phrase] = (tags, offset)
                self.ac.add_word(phrase)

    def parse_customization(self, query, lac_tags, prefix=False):
        """Use custom vocab to modify the lac results"""
        if not self.ac:
            logging.warning("customization dict is not load")
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

    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        self.prefix = None
        self.parent_relations = None
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(
            node, SchemaTree
        ), "The children of a node should be an instacne of SchemaTree."
        self.children.append(node)


def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
    """
    get idx of the last dim in prob arraies, which is greater than a limitation
    input: [[0.1, 0.1, 0.2, 0.5, 0.1, 0.3], [0.7, 0.6, 0.1, 0.1, 0.1, 0.1]]
        0.4
    output: [[3], [0, 1]]
    """
    probs = np.array(probs)
    dim_len = len(probs.shape)
    if dim_len > 1:
        result = []
        for p in probs:
            result.append(get_bool_ids_greater_than(p, limit, return_prob))
        return result
    else:
        result = []
        for i, p in enumerate(probs):
            if p > limit:
                if return_prob:
                    result.append((i, p))
                else:
                    result.append(i)
        return result


def get_span(start_ids, end_ids, with_prob=False):
    """
    every id can only be used once
    get span set from position start and end list
    input: [1, 2, 10] [4, 12]
    output: set((2, 4), (10, 12))
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            if start_ids[start_pointer][0] == end_ids[end_pointer][0]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                end_pointer += 1
                continue
            if start_ids[start_pointer][0] < end_ids[end_pointer][0]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                continue
            if start_ids[start_pointer][0] > end_ids[end_pointer][0]:
                end_pointer += 1
                continue
        else:
            if start_ids[start_pointer] == end_ids[end_pointer]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                end_pointer += 1
                continue
            if start_ids[start_pointer] < end_ids[end_pointer]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                continue
            if start_ids[start_pointer] > end_ids[end_pointer]:
                end_pointer += 1
                continue
    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result


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
    for index in range(1, prompt_end_token_id + 1):
        offset_mapping[index][0] -= bias
        offset_mapping[index][1] -= bias

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
            code -= 0xfee0
        if not (0x0021 <= code and code <= 0x7e):
            rs += char
            continue
        rs += chr(code)
    return rs


class WordTagRelationExtractor(object):
    """Implement of information extractor.
    """
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
                res[head_role_type] = {
                    "trigger": {},
                    "g_t_map": {},
                    "rel_group": {},
                    "trig_word": {}
                }
            group_name = trip_config["group"]
            if "rel_group" in trip_config:
                res[head_role_type]["rel_group"][group_name] = trip_config[
                    "rel_group"]
            if group_name not in res[head_role_type]["trig_word"]:
                res[head_role_type]["trig_word"][group_name] = set()
            for trig_word in trip_config["trig_word"]:
                res[head_role_type]["trigger"][trig_word] = {
                    "trigger_type": trip_config["trig_type"],
                    "group_name": group_name,
                    "rev_flag": trip_config["reverse"]
                }
                res[head_role_type]["trig_word"][group_name].add(trig_word)
            res[head_role_type]["g_t_map"][group_name] = trip_config[
                "tail_role"]

        return cls(res)

    @classmethod
    def from_json(cls, json_str):
        """Implement an instance from JSON str.
        """
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    @classmethod
    def from_pkl(cls, pkl_path):
        """Implement an instance from a serialized pickle package.
        """
        with open(pkl_path, "rb") as fp:
            schema = pickle.load(fp)
        return cls(schema)

    @classmethod
    def from_config(cls, config_path):
        """Implement an instance from a configuration file.
        """
        with open(config_path, encoding="utf-8") as fp:
            config_json = json.load(fp)
        return cls.from_dict(config_json)

    def add_schema_from_dict(self, config_dict):
        """Add the schema from the dict.
        """
        for i, trip_config in enumerate(config_dict):
            head_role_type = trip_config["head_role"]
            if head_role_type not in self._schema:
                self._schema[head_role_type] = {
                    "trigger": {},
                    "g_t_map": {},
                    "rel_group": {},
                    "trig_word": {}
                }
            group_name = trip_config["group"]
            if "rel_group" in self._schema:
                self._schema[head_role_type]["rel_group"][
                    group_name] = trip_config["rel_group"]
            if group_name not in self._schema[head_role_type]["trig_word"]:
                self._schema[head_role_type]["trig_word"][group_name] = set()
            for trig_word in trip_config["trig_word"]:
                self._schema[head_role_type]["trigger"][trig_word] = {
                    "trigger_type": trip_config["trig_type"],
                    "group_name": group_name,
                    "rev_flag": trip_config["reverse"]
                }
                self._schema[head_role_type]["trig_word"][group_name].add(
                    trig_word)
            self._schema[head_role_type]["g_t_map"][group_name] = trip_config[
                "tail_role"]

    def _judge_jux(self, wordtag_item):
        """Judge whether `wordtag_item` is a relevance componet between two juxtaposed items.

        Args:
            wordtag_item (dict): input item.

        Returns:
            bool: [description]
        """
        if wordtag_item["item"] in {"、", " ", "《", "》", "/"}:
            return True
        if wordtag_item["item"] in self._chain_items and wordtag_item[
                "wordtag_label"] == "连词":
            return True
        return False

    def _search_jux(self,
                    cur_item,
                    cur_pos=0,
                    jux_type=None,
                    jux_word=None,
                    status_flag=None,
                    search_list=None):
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
            return self._search_jux(cur_item=next_item,
                                    cur_pos=cur_pos + 1,
                                    jux_type=jux_type,
                                    jux_word=jux_word,
                                    status_flag=False,
                                    search_list=search_list)

        next_flag = True
        if jux_type is not None:
            next_flag = next_flag and self._match_item(next_item, jux_type)
        if jux_word is not None:
            next_flag = next_flag and (next_item["item"] in jux_word)
        if next_flag is True:
            return self._search_jux(cur_item=next_item,
                                    cur_pos=cur_pos + 1,
                                    jux_type=jux_type,
                                    jux_word=jux_word,
                                    status_flag=True)
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
            if any(item["wordtag_label"].startswith(sup_t)
                   for sup_t in search_type):
                res.append(item)
        return res if len(res) > 0 else None

    def _make_output(self,
                     head_item,
                     tail_item,
                     group,
                     source,
                     support=None,
                     trig_word=None,
                     **kwargs):
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
            "TAIL_ROLE": [{
                "item": ti["item"],
                "offset": ti["offset"],
                "type": ti["wordtag_label"]
            } for ti in tail_item],
            "GROUP":
            group,
            "SRC":
            source,
        }
        if support is not None:
            res["SUPPORT"] = [{
                "item": si["item"],
                "offset": si["offset"],
                "type": si["wordtag_label"],
            } for si in support]
        if trig_word is not None:
            res["TRIG"] = [{
                "item": ti["item"],
                "offset": ti["offset"],
            } for ti in trig_word]
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

                trig_status, trig_conf = self._trig_handler(
                    cur_item, self._schema[head_type])

                # Find a tail role, generate corresponding triple.
                if trig_status == "trig_t":
                    trig_status = "un_trig"
                    tail_flag = True
                    for k in range(i + 1, j):
                        if self._all_items[k]["wordtag_label"] == head_cand[
                                "wordtag_label"]:
                            tail_flag = False
                            break
                    if tail_flag is False:
                        continue

                    group_name = head_conf["trigger"][
                        cur_item["item"]]["group_name"]
                    del self._jux_buf[:]
                    idx = self._search_jux(cur_item=cur_item,
                                           cur_pos=cur_pos,
                                           jux_type=trig_conf["main"],
                                           status_flag=True)
                    supports = self._find_supp(search_range=range(j - 1, i, -1),
                                               search_type=trig_conf["support"])

                    tmp = self._make_output(head_item=head_cand,
                                            tail_item=self._jux_buf[:],
                                            group=group_name,
                                            support=supports,
                                            source="TAIL")

                    # Reverse triple if group has relative.
                    if (group_name in head_conf.get("rel_group", {}) or
                            head_conf["trigger"][cur_item["item"]]["rev_flag"]
                            is True):
                        rev_tmp = self._reverse(
                            tmp,
                            head_conf.get("rel_group",
                                          {}).get(group_name, None))
                        res_cand.extend(rev_tmp[:])
                    if head_conf["trigger"][
                            cur_item["item"]]["rev_flag"] is False:
                        res_cand.append(tmp.copy())

                    j = idx + 1
                    last_end = idx
                    continue

                # Find a group trigger word, look for tail role items of current head role and group argument.
                # Searching range is items behind group trigger and items between head rold and group trigger word.
                if trig_status == "trig_g":
                    trig_status = "un_trig"
                    group_name = head_conf["trigger"][
                        cur_item["item"]]["group_name"]

                    del self._jux_buf[:]
                    g_start_idx = j - 1
                    g_idx = self._search_jux(
                        cur_item=cur_item,
                        cur_pos=cur_pos,
                        jux_word=head_conf["trig_word"][group_name],
                        status_flag=True)

                    g_trig_words = self._jux_buf[:]
                    j = g_idx + 1

                    # Search right.
                    if j < len(self._all_items) - 1:
                        tail_idx, tail_conf = self._find_tail(
                            range(g_idx + 1, len(self._all_items)),
                            head_conf["g_t_map"][group_name], head_type)

                        if tail_idx > 0:
                            # Find a tail.
                            tail_item = self._all_items[tail_idx]
                            del self._jux_buf[:]
                            idx = self._search_jux(cur_item=tail_item,
                                                   cur_pos=tail_idx,
                                                   status_flag=True,
                                                   jux_type=tail_conf["main"])
                            tail_cand = self._jux_buf[:]
                            supports = self._find_supp(
                                range(tail_idx - 1, i, -1),
                                tail_conf["support"])

                            tmp = self._make_output(head_item=head_cand,
                                                    tail_item=tail_cand,
                                                    group=group_name,
                                                    source="HGT",
                                                    support=supports,
                                                    trig_word=g_trig_words)

                            if (group_name in head_conf.get("rel_group", {})
                                    or head_conf["trigger"][
                                        cur_item["item"]]["rev_flag"] is True):
                                rev_tmp = self._reverse(
                                    tmp,
                                    head_conf.get("rel_group",
                                                  {}).get(group_name, None))
                                res_cand.extend(rev_tmp[:])
                            if head_conf["trigger"][
                                    cur_item["item"]]["rev_flag"] is False:
                                res_cand.append(tmp.copy())

                            j = idx + 1
                            last_end = idx
                            continue

                    # Search left
                    if g_idx - i > len(g_trig_words):
                        tail_idx, tail_conf = self._find_tail(
                            range(g_start_idx, last_end, -1),
                            head_conf["g_t_map"][group_name], head_type)
                        tail_item = self._all_items[tail_idx]
                        if tail_idx > 0:
                            del self._jux_buf[:]
                            _ = self._search_jux(
                                cur_item=tail_item,
                                cur_pos=0,
                                jux_type=tail_conf["main"],
                                status_flag=True,
                                search_list=self._all_items[i +
                                                            1:tail_idx][::-1])
                            tail_cand = self._jux_buf[:]
                            supports = self._find_supp(
                                range(g_idx - 1, last_end, -1),
                                tail_conf["support"])
                            last_end = g_idx

                            tmp = self._make_output(head_item=head_cand,
                                                    tail_item=tail_cand,
                                                    group=group_name,
                                                    trig_word=g_trig_words,
                                                    source="HTG",
                                                    support=supports)

                            if (group_name in head_conf.get("rel_group", {})
                                    or head_conf["trigger"][
                                        cur_item["item"]]["rev_flag"] is True):
                                rev_tmp = self._reverse(
                                    tmp,
                                    head_conf.get("rel_group",
                                                  {}).get(group_name, None))
                                res_cand.extend(rev_tmp[:])
                            if head_conf["trigger"][
                                    cur_item["item"]]["rev_flag"] is False:
                                res_cand.append(tmp.copy())
                            continue
        return res_cand
