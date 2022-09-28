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

import os
import json
import csv
from typing import Any, Dict, List, Optional, Tuple, Union


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
    def from_dir(cls,
                 term_schema_path,
                 term_data_path,
                 linking=True) -> "TermTree":
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
