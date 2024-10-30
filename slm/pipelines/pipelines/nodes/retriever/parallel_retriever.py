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

import logging

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import multiprocessing
import time
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Optional, Union

import numpy as np
from pipelines.document_stores import BaseDocumentStore
from pipelines.nodes.retriever.base import BaseRetriever
from pipelines.schema import ContentTypes, Document
from tqdm.auto import tqdm
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

logger = logging.getLogger(__name__)


class TritonRunner:
    DEFAULT_MAX_RESP_WAIT_S = 120

    def __init__(
        self, server_url: str, model_name: str, model_version: str, verbose=False, resp_wait_s: Optional[float] = None
    ):
        """
        :param server_url: The port of server
        :param model_name: The model name needs to match the name in config.txt
        :param model_version: Model version number
        :param resp_wait_s: the response waiting time
        """
        self._server_url = server_url
        self._model_name = model_name
        self._model_version = model_version
        self._verbose = verbose
        self._response_wait_t = self.DEFAULT_MAX_RESP_WAIT_S if resp_wait_s is None else resp_wait_s

        self._client = InferenceServerClient(self._server_url, verbose=self._verbose)
        error = self._verify_triton_state(self._client)
        if error:
            raise RuntimeError(f"Could not communicate to Triton Server: {error}")
        model_metadata = self._client.get_model_metadata(self._model_name, self._model_version)
        self._inputs = {tm["name"]: tm for tm in model_metadata["inputs"]}
        self._input_names = list(self._inputs)
        self._outputs = {tm["name"]: tm for tm in model_metadata["outputs"]}
        self._output_names = list(self._outputs)
        self._outputs_req = [InferRequestedOutput(name) for name in self._outputs]

    def Run_docs(self, documents, embed_title=False):
        documents = deepcopy(documents)
        docs = [Document.from_dict(doc) for doc in documents]
        dicts = [
            {
                "passages": [
                    {
                        "title": d.meta["name"] if d.meta and "name" in d.meta else "",
                        "text": d.content,
                        "label": d.meta["label"] if d.meta and "label" in d.meta else "positive",
                        "external_id": d.id,
                    }
                ]
            }
            for d in docs
        ]
        datasets = []
        for passages in dicts:
            for item in passages["passages"]:
                if embed_title:
                    datasets.append(item["title"] + item["text"])
                else:
                    datasets.append(item["text"])
        infer_inputs = []
        for idx, data in enumerate([datasets]):
            data = np.array([[x.encode("utf-8")] for x in data], dtype=np.object_)
            infer_input = InferInput(self._input_names[idx], [len(data), 1], "BYTES")
            infer_input.set_data_from_numpy(data)
            infer_inputs.append(infer_input)
        try:
            results = self._client.infer(
                model_name=self._model_name,
                model_version=self._model_version,
                inputs=infer_inputs,
                outputs=self._outputs_req,
            )
        except Exception as e:
            logger.error("InferenceServerClient infer error {}".format(e))
        results = {name: results.as_numpy(name) for name in self._output_names}
        for doc, emb in zip(documents, results["embedding"]):
            doc["embedding"] = emb
        return documents

    def Run_query(self, query):
        """
        Args:
            inputs: list, Each value corresponds to an input name of self._input_names
        Returns:
            results: dict, {name : numpy.array}
        """
        infer_inputs = []
        for idx, data in enumerate([query]):
            data = np.array([[x.encode("utf-8")] for x in data], dtype=np.object_)
            infer_input = InferInput(self._input_names[idx], [len(data), 1], "BYTES")
            infer_input.set_data_from_numpy(data)
            infer_inputs.append(infer_input)
        try:
            results = self._client.infer(
                model_name=self._model_name,
                model_version=self._model_version,
                inputs=infer_inputs,
                outputs=self._outputs_req,
            )
        except Exception as e:
            logger.error("InferenceServerClient infer error {}".format(e))
        results = {name: results.as_numpy(name) for name in self._output_names}
        return results["embedding"]

    def _verify_triton_state(self, triton_client):
        if not triton_client.is_server_live():
            return f"Triton server {self._server_url} is not live"
        elif not triton_client.is_server_ready():
            return f"Triton server {self._server_url} is not ready"
        elif not triton_client.is_model_ready(self._model_name, self._model_version):
            return f"Model {self._model_name}:{self._model_version} is not ready"
        return None


def run_main_doc(item, url="0.0.0.0:8082", model_name="m3e", model_version="1"):
    runner = TritonRunner(url, model_name, model_version)
    return runner.Run_docs(item)


def run_main_query(query, url="0.0.0.0:8082", model_name="m3e", model_version="1"):
    runner = TritonRunner(url, model_name, model_version)
    return runner.Run_query(query)


def embeddings_multi_doc(data, batch_size=32, num_process=10, url="0.0.0.0:8082", model_name="m3e", model_version="1"):
    workers = len(data) // batch_size + 1
    offset = [i * batch_size for i in range(workers)]
    if offset[-1] != len(data):
        offset += [len(data)]
    data_index = zip(offset, offset[1:])
    data_list = [data[start:end] for start, end in data_index]
    func = partial(run_main_doc, url=url, model_name=model_name, model_version=model_version)
    pool = Pool(processes=min(num_process, multiprocessing.cpu_count()))
    result = pool.map(func, data_list)
    pool.close()  # close the process pool and no longer accept new processes
    pool.join()
    return result


class ParallelRetriever(BaseRetriever):
    def __init__(
        self,
        document_store: BaseDocumentStore,
        model_version: Optional[str] = None,
        output_emb_size: Optional[int] = None,
        reinitialize: bool = False,
        share_parameters: bool = False,
        max_seq_len_query: int = 64,
        max_seq_len_passage: int = 384,
        top_k: int = 10,
        use_gpu: bool = True,
        batch_size: int = 16,
        embed_title: bool = True,
        similarity_function: str = "dot_product",
        progress_bar: bool = True,
        mode: Literal["snippets", "raw_documents", "preprocessed_documents"] = "preprocessed_documents",
        url="0.0.0.0:8082",
        num_process=10,
        **kwargs
    ):
        """
        :param url: the port of the HTTP service
        :param num_process: the number of processes
        """
        self.set_config(
            document_store=document_store,
            model_version=model_version,
            max_seq_len_query=max_seq_len_query,
            max_seq_len_passage=max_seq_len_passage,
            top_k=top_k,
            use_gpu=use_gpu,
            batch_size=batch_size,
            embed_title=embed_title,
            reinitialize=reinitialize,
            share_parameters=share_parameters,
            output_emb_size=output_emb_size,
            similarity_function=similarity_function,
            progress_bar=progress_bar,
        )

        self.document_store = document_store
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.top_k = top_k
        self.embed_title = embed_title
        self.mode = mode
        self.url = url
        self.num_process = num_process
        self.model_name = kwargs.get("model_name", "m3e")

        if document_store is None:
            logger.warning("DensePassageRetriever initialized without a document store. ")
        elif document_store.similarity != "dot_product":
            logger.warning(
                f"You are using a Dense Passage Retriever model with the {document_store.similarity} function. "
                "We recommend you use dot_product instead. "
                "This can be set when initializing the DocumentStore"
            )

    def retrieve(
        self,
        query: str,
        query_type: Optional[ContentTypes] = None,
        filters: dict = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """
        if top_k is None:
            top_k = self.top_k
        if not self.document_store:
            logger.error("Cannot perform retrieve() since DensePassageRetriever initialized with document_store=None")
            return []
        if index is None:
            index = self.document_store.index

        query_emb = self.embed_queries(texts=[query], **kwargs)
        documents = self.document_store.query_by_embedding(
            query_emb=query_emb[0], top_k=top_k, filters=filters, index=index, headers=headers, return_embedding=False
        )
        return documents

    def retrieve_batch(
        self,
        queries: List[str],
        queries_type: Optional[ContentTypes] = None,
        filters: Optional[
            Union[
                Dict[str, Union[Dict, List, str, int, float, bool]],
                List[Dict[str, Union[Dict, List, str, int, float, bool]]],
            ]
        ] = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        scale_score: bool = None,
        **kwargs,
    ) -> List[List[Document]]:
        if top_k is None:
            top_k = self.top_k
        if batch_size is None:
            batch_size = self.batch_size

        if isinstance(filters, list):
            if len(filters) != len(queries):
                raise Exception(
                    "Number of filters does not match number of queries. Please provide as many filters"
                    " as queries or a single filter that will be applied to each query."
                )
        else:
            filters = [filters] * len(queries) if filters is not None else [{}] * len(queries)
        if index is None:
            index = self.document_store.index
        if not self.document_store:
            logger.error(
                "Cannot perform retrieve_batch() since DensePassageRetriever initialized with document_store=None"
            )
            return [[] * len(queries)]  # type: ignore
        documents = []
        query_embs: List[np.ndarray] = []
        for batch in self._get_batches(queries=queries, batch_size=batch_size):
            query_embs.extend(self.embed_queries(texts=batch, **kwargs))
        for query_emb, cur_filters in tqdm(
            zip(query_embs, filters), total=len(query_embs), disable=not self.progress_bar, desc="Querying"
        ):
            cur_docs = self.document_store.query_by_embedding(
                query_emb=query_emb,
                top_k=top_k,
                filters=cur_filters,
                index=index,
                headers=headers,
                return_embedding=False,
            )
            documents.append(cur_docs)

        return documents

    def run_indexing(self, documents: List[dict], **kwargs):
        time1 = time.time()
        documents_list = embeddings_multi_doc(
            documents,
            batch_size=self.batch_size,
            num_process=self.num_process,
            url=self.url,
            model_name=self.model_name,
        )
        documents = []
        for i in documents_list:
            documents.extend(i)
        output = {"documents": documents}
        time2 = time.time()
        logger.info(f"The time cost of create docs: {time2-time1:.3f} s")
        return output, "output_1"

    def embed_queries(self, texts: List[str], **kwargs) -> List[np.ndarray]:
        return run_main_query(texts, self.url)

    def embed_documents(self, docs: List[Document], **kwargs):
        return run_main_query([d.content for d in docs], self.url)
