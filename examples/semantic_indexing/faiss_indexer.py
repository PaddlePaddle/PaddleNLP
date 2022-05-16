#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 FAISS-based index components for dense retriver
"""

import os
import time
import logging
import pickle
from typing import List, Tuple, Iterator

import faiss
import numpy as np

logger = logging.getLogger()


class DenseIndexer(object):

    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def index_data(self, vector_files: List[str]):
        start_time = time.time()
        buffer = []
        for i, item in enumerate(iterate_encoded_files(vector_files)):
            db_id, doc_vector = item
            buffer.append((db_id, doc_vector))
            if 0 < self.buffer_size == len(buffer):
                # indexing in batches is beneficial for many faiss index types
                self._index_batch(buffer)
                logger.info('data indexed %d, used_time: %f sec.',
                            len(self.index_id_to_db_id), time.time() - start_time)
                buffer = []
        self._index_batch(buffer)

        indexed_cnt = len(self.index_id_to_db_id)
        logger.info('Total data indexed %d', indexed_cnt)
        logger.info('Data indexing completed.')

    def _index_batch(self, data: List[Tuple[object, np.array]]):
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        raise NotImplementedError

    def serialize(self, file: str):
        logger.info('Serializing index to %s', file)

        if os.path.isdir(file):
            index_file = os.path.join(file, "index.dpr")
            meta_file = os.path.join(file, "index_meta.dpr")
        else:
            index_file = file + '.index.dpr'
            meta_file = file + '.index_meta.dpr'

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, file: str):
        logger.info('Loading index from %s', file)

        if os.path.isdir(file):
            index_file = os.path.join(file, "index.dpr")
            meta_file = os.path.join(file, "index_meta.dpr")
        else:
            index_file = file + '.index.dpr'
            meta_file = file + '.index_meta.dpr'

        self.index = faiss.read_index(index_file)
        logger.info('Loaded index of type %s and size %d', type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert len(
            self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'

    def _update_id_mapping(self, db_ids: List):
        self.index_id_to_db_id.extend(db_ids)


class DenseFlatIndexer(DenseIndexer):

    def __init__(self, vector_sz: int, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)
        self.index = faiss.IndexFlatIP(vector_sz)

    def _index_batch(self, data: List[Tuple[object, np.array]]):
        db_ids = [t[0] for t in data]
        vectors = [np.reshape(t[1], (1, -1)) for t in data]
        vectors = np.concatenate(vectors, axis=0)
        self._update_id_mapping(db_ids)
        self.index.add(vectors)

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        scores, indexes = self.index.search(query_vectors, top_docs)
        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result


class DenseHNSWFlatIndexer(DenseIndexer):
    """
     Efficient index for retrieval. Note: default settings are for hugh accuracy but also high RAM usage
    """

    def __init__(self, vector_sz: int, buffer_size: int = 50000, store_n: int = 512
                 , ef_search: int = 128, ef_construction: int = 200):
        super(DenseHNSWFlatIndexer, self).__init__(buffer_size=buffer_size)

        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        index = faiss.IndexHNSWFlat(vector_sz + 1, store_n)
        index.hnsw.efSearch = ef_search
        index.hnsw.efConstruction = ef_construction
        self.index = index
        self.phi = None

    def index_data(self, vector_files: List[str]):
        self._set_phi(vector_files)

        super(DenseHNSWFlatIndexer, self).index_data(vector_files)

    def _set_phi(self, vector_files: List[str]):
        """
        Calculates the max norm from the whole data and assign it to self.phi: necessary to transform IP -> L2 space
        :param vector_files: file names to get passages vectors from
        :return:
        """
        phi = 0
        for i, item in enumerate(iterate_encoded_files(vector_files)):
            id, doc_vector = item
            norms = (doc_vector ** 2).sum()
            phi = max(phi, norms)
        logger.info('HNSWF DotProduct -> L2 space phi={}'.format(phi))
        self.phi = phi

    def _index_batch(self, data: List[Tuple[object, np.array]]):
        # max norm is required before putting all vectors in the index to convert inner product similarity to L2
        if self.phi is None:
            raise RuntimeError('Max norm needs to be calculated from all data at once,'
                               'results will be unpredictable otherwise.'
                               'Run `_set_phi()` before calling this method.')

        db_ids = [t[0] for t in data]
        vectors = [np.reshape(t[1], (1, -1)) for t in data]

        norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
        aux_dims = [np.sqrt(self.phi - norm) for norm in norms]
        hnsw_vectors = [np.hstack((doc_vector, aux_dims[i].reshape(-1, 1))) for i, doc_vector in
                        enumerate(vectors)]
        hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)

        self._update_id_mapping(db_ids)
        self.index.add(hnsw_vectors)

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:

        aux_dim = np.zeros(len(query_vectors), dtype='float32')
        query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
        logger.info('query_hnsw_vectors %s', query_nhsw_vectors.shape)
        scores, indexes = self.index.search(query_nhsw_vectors, top_docs)
        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result

    def deserialize_from(self, file: str):
        super(DenseHNSWFlatIndexer, self).deserialize_from(file)
        # to trigger warning on subsequent indexing
        self.phi = None


def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for i, file in enumerate(vector_files):
        logger.info('Reading file %s', file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
                yield db_id, doc_vector