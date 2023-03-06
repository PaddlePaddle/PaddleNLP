# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from sklearn.metrics import pairwise_distances

__all__ = ["MRR"]


class MRR:
    """
    MRR - Mean Reciprocal Rank, is a popular metric for recommend system
    and other retrival task. The higher mrr is, the better performance of
    model in retrival task.

    Args:
        distance: which algorithm to use to get distance of embeddings, for example: "cosine", "euclidean"

    """

    def __init__(self, distance="cosine"):
        super().__init__()
        self.distance = distance

    def reset_distance(self, distance):
        """
        change the algorithm of calculating distance, need to be supported of sklearn.metrics.pairwise_distance
        """
        self.distance = distance

    def compute_matrix_mrr(self, labels, embeddings):
        """
        A function which can calculate the distance of one embedding to other embeddings
        in the matrix, and then it can find the most similar embedding's index to calculate
        the mrr metric for this one embedding. After getting all the embeddings' mrr metric,
        a mean pool is used to get the final mrr metric for input matrix.

        Param:
          - labels(np.array): label matrix, shape=[size, ]
          - embeddings(np.array): embedding matrix, shape=[size, emb_dim]

        Return:
            mrr metric for input embedding matrix.
        """
        matrix_size = labels.shape[0]
        if labels.shape[0] != embeddings.shape[0]:
            raise Exception("label and embedding matrix must have same size at dim=0 !")
        row_mrr = []  # mrr metric for each embedding of matrix
        for i in range(0, matrix_size):
            emb, label = embeddings[i, :], labels[i]
            dists = pairwise_distances(emb.reshape(1, -1), embeddings, metric=self.distance).reshape(-1)
            ranks_ids = np.argsort(dists)[1:]
            ranks = (labels[ranks_ids] == label).astype(int)
            ranks_nonzero_ids = ranks.nonzero()[0]
            row_mrr.append(1.0 / (1 + ranks_nonzero_ids[0]) if ranks_nonzero_ids.size else 0.0)
        mrr = np.mean(row_mrr)  # user mean value as final mrr metric for the matrix.
        return mrr
