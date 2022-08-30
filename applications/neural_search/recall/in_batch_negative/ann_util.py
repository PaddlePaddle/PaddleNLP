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

# coding=UTF-8

import numpy as np
import hnswlib
from paddlenlp.utils.log import logger


def build_index(args, data_loader, model):

    index = hnswlib.Index(
        space='ip',
        dim=args.output_emb_size if args.output_emb_size > 0 else 768)

    # Initializing index
    # max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
    # during insertion of an element.
    # The capacity can be increased by saving/loading the index, see below.
    #
    # ef_construction - controls index search speed/build speed tradeoff
    #
    # M - is tightly connected with internal dimensionality of the data. Strongly affects memory consumption (~M)
    # Higher M leads to higher accuracy/run_time at fixed ef/efConstruction
    index.init_index(max_elements=args.hnsw_max_elements,
                     ef_construction=args.hnsw_ef,
                     M=args.hnsw_m)

    # Controlling the recall by setting ef:
    # higher ef leads to better accuracy, but slower search
    index.set_ef(args.hnsw_ef)

    # Set number of threads used during batch search/construction
    # By default using all available cores
    index.set_num_threads(16)

    logger.info("start build index..........")

    all_embeddings = []

    for text_embeddings in model.get_semantic_embedding(data_loader):
        all_embeddings.append(text_embeddings.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    index.add_items(all_embeddings)

    logger.info("Total index number:{}".format(index.get_current_count()))

    return index
