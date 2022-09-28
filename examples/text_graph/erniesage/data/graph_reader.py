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

import paddle
import copy
import pgl
from paddle.io import DataLoader

__all__ = ["GraphDataLoader"]


class GraphDataLoader(object):

    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=True,
                 num_workers=1,
                 collate_fn=None,
                 **kwargs):
        self.loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn,
                                 **kwargs)

    def __iter__(self):
        func = self.__callback__()
        for data in self.loader():
            yield func(data)

    def __call__(self):
        return self.__iter__()

    def __callback__(self):
        """ callback function, for recontruct a dict or graph.
        """

        def construct(tensors):
            """ tensor list to ([graph_tensor, graph_tensor, ...], 
            other tensor) 
            """
            graph_num = 1
            start_len = 0
            data = []
            graph_list = []
            for graph in range(graph_num):
                graph_list.append(
                    pgl.Graph(num_nodes=tensors[start_len],
                              edges=tensors[start_len + 1]))
                start_len += 2

            for i in range(start_len, len(tensors)):
                data.append(tensors[i])
            return graph_list, data

        return construct
