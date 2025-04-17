# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import math
import os
import pickle
from typing import Dict, List, Sequence

import faiss
import numpy as np
from tqdm import tqdm

from .utils import encode, load_data, load_model


class Indexer:
    def __init__(self, model_name_or_path: str = "BAAI/bge-large-en-v1.5", url=None):
        self.model_name = model_name_or_path
        self.model, self.tokenizer = load_model(model_name_or_path=model_name_or_path)

    def __call__(self, file_path: str) -> Dict:
        data = load_data(file_path=file_path, mode="Indexing")
        return self.index_doc(data=data, file_path=file_path)

    def get_corpus_embedding(self, corpus: Sequence[str], batch_size: int = 128) -> np.ndarray:
        """
        Generate embeddings for a corpus of text by processing it in batches.

        Args:
            corpus (Sequence[str]): A sequence of text strings to be embedded.
            batch_size (int, optional): The number of text strings to process in each batch. Defaults to 128.

        Returns:
            np.ndarray: A numpy array containing the embeddings for the entire corpus.
        """
        logging.info("Getting embedding")
        for i in tqdm(range(math.ceil(len(corpus) / batch_size))):
            corpus_embeddings = encode(
                sentences=corpus[i * batch_size : (i + 1) * batch_size],
                model=self.model,
                tokenizer=self.tokenizer,
                convert_to_numpy=True,
            )
            if i == 0:
                corpus_embeddings_list = corpus_embeddings
            else:
                corpus_embeddings_list = np.concatenate((corpus_embeddings_list, corpus_embeddings), axis=0)
        return corpus_embeddings_list

    def build_engine(self, corpus_embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """
        Build a FAISS index using the provided corpus embeddings.

        Args:
            corpus_embeddings (np.ndarray): A numpy array containing the embeddings of the corpus.

        Returns:
            faiss.IndexFlatIP: A FAISS index built using the inner product metric.
        """
        embedding_dim = corpus_embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(corpus_embeddings.astype("float32"))
        return index

    def index_doc(
        self,
        file_path: str,
        data: Dict,
        save: bool = True,
        save_dir: str = "data/index",
    ) -> Dict:
        """
        Index a document by extracting its content, generating embeddings, and optionally saving the results.

        Args:
            file_path (str): The path to the file being indexed.
            data (Dict): A dictionary containing the document's data, including nodes with content and summary.
            save (bool, optional): Whether to save the generated embeddings and metadata. Defaults to True.
            save_dir (str, optional): The directory where the saved files will be stored. Defaults to "data/index".

        Returns:
            Dict: The original data dictionary, potentially modified during the indexing process.
        """
        corpus: List[str] = []
        info: List[dict] = []
        for node in data["nodes"]:
            if "summary" in node and len(node["summary"]) > 0:
                content = node["summary"]
            else:
                content = node["content"]
            corpus.append(content)
            level = node["level"] if "level" in node else 0
            info.append({"content": content, "source": data["source"], "level": level})
        corpus_embeddings: np.ndarray = self.get_corpus_embedding(corpus)

        filename = os.path.splitext(os.path.basename(file_path))[0]
        if save:
            save_dir = os.path.join(save_dir, self.model_name)
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"{filename}.npy"), corpus_embeddings)
            with open(os.path.join(save_dir, f"{filename}.pkl"), "wb") as f:
                pickle.dump(info, f)
            logging.info(f"Index cache and Corpus saved to {save_dir}/{filename}.*")

        return data

    def _search(
        self,
        index: faiss.IndexFlatIP,
        query_embeddings: np.ndarray,
        top_k: int,
        batch_size: int = 4000,
    ) -> List[Dict]:
        """
        retrieves the top_k hits for each query embedding
        Calling index.search()
        """
        hits = []
        for i in range(math.ceil(len(query_embeddings) / batch_size)):
            q_emb_matrix = query_embeddings[i * batch_size : (i + 1) * batch_size]
            res_dist, res_p_id = index.search(q_emb_matrix.astype("float32"), top_k)
            assert len(res_dist) == len(q_emb_matrix)
            assert len(res_p_id) == len(q_emb_matrix)

            for i in range(len(q_emb_matrix)):
                passages = []
                assert len(res_p_id[i]) == len(res_dist[i])
                for j in range(min(top_k, len(res_p_id[i]))):
                    pid = res_p_id[i][j]
                    score = res_dist[i][j]
                    passages.append({"corpus_id": int(pid), "score": float(score)})
                hits.append(passages)
        return hits
