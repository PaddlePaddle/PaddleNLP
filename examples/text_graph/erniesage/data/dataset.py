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

import os

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, IterableDataset
from paddlenlp.utils.log import logger
import pgl
from pgl import Graph
from pgl.sampling import graphsage_sample

__all__ = [
    "TrainData",
    "PredictData",
    "batch_fn",
]


class TrainData(Dataset):

    def __init__(self, graph_work_path):
        trainer_id = paddle.distributed.get_rank()
        trainer_count = paddle.distributed.get_world_size()
        print("trainer_id: %s, trainer_count: %s." %
              (trainer_id, trainer_count))

        edges = np.load(os.path.join(graph_work_path, "train_data.npy"),
                        allow_pickle=True)
        # edges is bidirectional.
        train_src = edges[trainer_id::trainer_count, 0]
        train_dst = edges[trainer_id::trainer_count, 1]
        returns = {"train_data": [train_src, train_dst]}

        if os.path.exists(os.path.join(graph_work_path, "neg_samples.npy")):
            neg_samples = np.load(os.path.join(graph_work_path,
                                               "neg_samples.npy"),
                                  allow_pickle=True)
            if neg_samples.size != 0:
                train_negs = neg_samples[trainer_id::trainer_count]
                returns["train_data"].append(train_negs)
        print("Load train_data done.")
        self.data = returns

    def __getitem__(self, index):
        return [data[index] for data in self.data["train_data"]]

    def __len__(self):
        return len(self.data["train_data"][0])


class PredictData(Dataset):

    def __init__(self, num_nodes):
        trainer_id = paddle.distributed.get_rank()
        trainer_count = paddle.distributed.get_world_size()
        self.data = np.arange(trainer_id, num_nodes, trainer_count)

    def __getitem__(self, index):
        return [self.data[index], self.data[index]]

    def __len__(self):
        return len(self.data)


def batch_fn(batch_ex, samples, base_graph, term_ids):
    batch_src = []
    batch_dst = []
    batch_neg = []
    for batch in batch_ex:
        batch_src.append(batch[0])
        batch_dst.append(batch[1])
        if len(batch) == 3:  # default neg samples
            batch_neg.append(batch[2])

    batch_src = np.array(batch_src, dtype="int64")
    batch_dst = np.array(batch_dst, dtype="int64")
    if len(batch_neg) > 0:
        batch_neg = np.unique(np.concatenate(batch_neg))
    else:
        batch_neg = batch_dst

    nodes = np.unique(np.concatenate([batch_src, batch_dst, batch_neg], 0))
    subgraphs = graphsage_sample(base_graph, nodes, samples)

    subgraph, sample_index, node_index = subgraphs[0]
    from_reindex = {int(x): i for i, x in enumerate(sample_index)}

    term_ids = term_ids[sample_index].astype(np.int64)

    sub_src_idx = pgl.graph_kernel.map_nodes(batch_src, from_reindex)
    sub_dst_idx = pgl.graph_kernel.map_nodes(batch_dst, from_reindex)
    sub_neg_idx = pgl.graph_kernel.map_nodes(batch_neg, from_reindex)

    user_index = np.array(sub_src_idx, dtype="int64")
    pos_item_index = np.array(sub_dst_idx, dtype="int64")
    neg_item_index = np.array(sub_neg_idx, dtype="int64")

    user_real_index = np.array(batch_src, dtype="int64")
    pos_item_real_index = np.array(batch_dst, dtype="int64")

    return np.array([subgraph.num_nodes], dtype="int32"), \
        subgraph.edges.astype("int32"), \
        term_ids, user_index, pos_item_index, neg_item_index, \
        user_real_index, pos_item_real_index
