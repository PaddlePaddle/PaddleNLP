# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import io
import os
import sys
import argparse
import logging
import multiprocessing
from functools import partial
from io import open

import numpy as np
import yaml
import tqdm
from easydict import EasyDict as edict
import pgl
from pgl.graph_kernel import alias_sample_build_table
from pgl.utils.logger import log
from paddlenlp.transformers import ErnieTinyTokenizer, ErnieTokenizer

TOKENIZER_CLASSES = {
    "ernie-tiny": ErnieTinyTokenizer,
    "ernie-1.0": ErnieTokenizer,
}


def term2id(string, tokenizer, max_seqlen):
    #string = string.split("\t")[1]
    tokens = tokenizer._tokenize(string)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = ids[:max_seqlen - 1]
    ids = ids + [tokenizer.sep_token_id]
    ids = ids + [tokenizer.pad_token_id] * (max_seqlen - len(ids))
    return ids


def load_graph(config, str2id, term_file, terms, item_distribution):
    edges = []
    with io.open(config.graph_data, encoding=config.encoding) as f:
        for idx, line in enumerate(f):
            if idx % 100000 == 0:
                log.info("%s readed %s lines" % (config.graph_data, idx))
            slots = []
            for col_idx, col in enumerate(line.strip("\n").split("\t")):
                s = col[:config.max_seqlen]
                if s not in str2id:
                    str2id[s] = len(str2id)
                    term_file.write(str(col_idx) + "\t" + col + "\n")
                    item_distribution.append(0)
                slots.append(str2id[s])

            src = slots[0]
            dst = slots[1]
            edges.append((src, dst))
            edges.append((dst, src))
            item_distribution[dst] += 1
    edges = np.array(edges, dtype="int64")
    return edges


def load_link_prediction_train_data(config, str2id, term_file, terms,
                                    item_distribution):
    train_data = []
    neg_samples = []
    with io.open(config.train_data, encoding=config.encoding) as f:
        for idx, line in enumerate(f):
            if idx % 100000 == 0:
                log.info("%s readed %s lines" % (config.train_data, idx))
            slots = []
            for col_idx, col in enumerate(line.strip("\n").split("\t")):
                s = col[:config.max_seqlen]
                if s not in str2id:
                    str2id[s] = len(str2id)
                    term_file.write(str(col_idx) + "\t" + col + "\n")
                    item_distribution.append(0)
                slots.append(str2id[s])

            src = slots[0]
            dst = slots[1]
            neg_samples.append(slots[2:])
            train_data.append((src, dst))
    train_data = np.array(train_data, dtype="int64")
    np.save(os.path.join(config.graph_work_path, "train_data.npy"), train_data)
    if len(neg_samples) != 0:
        np.save(os.path.join(config.graph_work_path, "neg_samples.npy"),
                np.array(neg_samples))


def dump_graph(config):
    if not os.path.exists(config.graph_work_path):
        os.makedirs(config.graph_work_path)
    str2id = dict()
    term_file = io.open(os.path.join(config.graph_work_path, "terms.txt"),
                        "w",
                        encoding=config.encoding)
    terms = []
    item_distribution = []

    edges = load_graph(config, str2id, term_file, terms, item_distribution)
    if config.task == "link_prediction":
        load_link_prediction_train_data(config, str2id, term_file, terms,
                                        item_distribution)
    else:
        raise ValueError

    term_file.close()
    num_nodes = len(str2id)
    str2id.clear()

    log.info("Building graph...")
    graph = pgl.graph.Graph(num_nodes=num_nodes, edges=edges)
    indegree = graph.indegree()
    graph.indegree()
    graph.outdegree()
    graph.dump(config.graph_work_path)

    # dump alias sample table
    item_distribution = np.array(item_distribution)
    item_distribution = np.sqrt(item_distribution)
    distribution = 1. * item_distribution / item_distribution.sum()
    alias, events = alias_sample_build_table(distribution)
    np.save(os.path.join(config.graph_work_path, "alias.npy"), alias)
    np.save(os.path.join(config.graph_work_path, "events.npy"), events)
    log.info("End Build Graph")


def dump_node_feat(config):
    log.info("Dump node feat starting...")
    id2str = [
        line.strip("\n").split("\t")[-1]
        for line in io.open(os.path.join(config.graph_work_path, "terms.txt"),
                            encoding=config.encoding)
    ]
    # pool = multiprocessing.Pool()

    tokenizer_class = TOKENIZER_CLASSES[config.model_name_or_path]
    tokenizer = tokenizer_class.from_pretrained(config.model_name_or_path)
    fn = partial(term2id, tokenizer=tokenizer, max_seqlen=config.max_seqlen)
    term_ids = [fn(x) for x in id2str]

    np.save(os.path.join(config.graph_work_path, "term_ids.npy"),
            np.array(term_ids, np.uint16))
    log.info("Dump node feat done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    dump_graph(config)
    dump_node_feat(config)
