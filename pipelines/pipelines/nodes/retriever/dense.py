# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 deepset GmbH. All Rights Reserved.
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

from typing import List, Dict, Union, Optional

import logging
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

import paddle
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieDualEncoder, AutoTokenizer

from pipelines.schema import Document
from pipelines.document_stores import BaseDocumentStore
from pipelines.nodes.retriever.base import BaseRetriever
from pipelines.data_handler.processor import TextSimilarityProcessor
from pipelines.utils.common_utils import initialize_device_settings

logger = logging.getLogger(__name__)


class DensePassageRetriever(BaseRetriever):
    """
    Retriever that uses a bi-encoder (one transformer for query, one transformer for passage).    
    """

    def __init__(
        self,
        document_store: BaseDocumentStore,
        query_embedding_model: Union[
            Path, str] = "rocketqa-zh-dureader-query-encoder",
        passage_embedding_model: Union[
            Path, str] = "rocketqa-zh-dureader-para-encoder",
        model_version: Optional[str] = None,
        max_seq_len_query: int = 64,
        max_seq_len_passage: int = 256,
        top_k: int = 10,
        use_gpu: bool = True,
        batch_size: int = 16,
        embed_title: bool = True,
        similarity_function: str = "dot_product",
        progress_bar: bool = True,
    ):
        """
        Init the Retriever incl. the two encoder models from a local or remote model checkpoint.

        **Example:**

                ```python
                |    # remote model from FAIR
                |    DensePassageRetriever(document_store=your_doc_store,
                |                          query_embedding_model="rocketqa-zh-dureader-query-encoder",
                |                          passage_embedding_model="rocketqa-zh-dureader-para-encoder")
                |    # or from local path
                |    DensePassageRetriever(document_store=your_doc_store,
                |                          query_embedding_model="model_directory/question-encoder",
                |                          passage_embedding_model="model_directory/context-encoder")
                ```

        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param query_embedding_model: Local path or remote name of question encoder checkpoint. The format equals the
                                      one used by paddlenlp transformers' models
                                      Currently available remote names: ``"rocketqa-zh-dureader-query-encoder"``
        :param passage_embedding_model: Local path or remote name of passage encoder checkpoint. The format equals the
                                        one used by paddlenlp transformers' models
                                        Currently available remote names: ``"rocketqa-zh-dureader-para-encoder"``
        :param max_seq_len_query: Longest length of each query sequence. Maximum number of tokens for the query text. Longer ones will be cut down."
        :param max_seq_len_passage: Longest length of each passage/context sequence. Maximum number of tokens for the passage text. Longer ones will be cut down."
        :param top_k: How many documents to return per query.
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param batch_size: Number of questions or passages to encode at once. In case of multiple gpus, this will be the total batch size.
        :param embed_title: Whether to concatenate title and passage to a text pair that is then used to create the embedding.
                            This is the approach used in the original paper and is likely to improve performance if your
                            titles contain meaningful information for retrieval (topic, entities etc.) .
                            The title is expected to be present in doc.meta["name"] and can be supplied in the documents
                            before writing them to the DocumentStore like this:
                            {"text": "my text", "meta": {"name": "my title"}}.
        :param similarity_function: Which function to apply for calculating the similarity of query and passage embeddings during training.
                                    Options: `dot_product` (Default) or `cosine`
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        """
        # save init parameters to enable export of component config as YAML
        self.set_config(
            document_store=document_store,
            query_embedding_model=query_embedding_model,
            passage_embedding_model=passage_embedding_model,
            model_version=model_version,
            max_seq_len_query=max_seq_len_query,
            max_seq_len_passage=max_seq_len_passage,
            top_k=top_k,
            use_gpu=use_gpu,
            batch_size=batch_size,
            embed_title=embed_title,
            similarity_function=similarity_function,
            progress_bar=progress_bar,
        )

        self.devices, _ = initialize_device_settings(use_cuda=use_gpu,
                                                     multi_gpu=True)

        if batch_size < len(self.devices):
            logger.warning(
                "Batch size is less than the number of devices. All gpus will not be utilized."
            )

        self.document_store = document_store
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.top_k = top_k

        if document_store is None:
            logger.warning(
                "DensePassageRetriever initialized without a document store. ")
        elif document_store.similarity != "dot_product":
            logger.warning(
                f"You are using a Dense Passage Retriever model with the {document_store.similarity} function. "
                "We recommend you use dot_product instead. "
                "This can be set when initializing the DocumentStore")

        # Init & Load Encoders
        #self.query_encoder = ErnieDualEncoder.from_pretrained(query_embedding_model)
        self.ernie_dual_encoder = ErnieDualEncoder(query_embedding_model,
                                                   passage_embedding_model)
        self.query_tokenizer = AutoTokenizer.from_pretrained(
            query_embedding_model)
        self.passage_tokenizer = AutoTokenizer.from_pretrained(
            passage_embedding_model)

        self.processor = TextSimilarityProcessor(
            query_tokenizer=self.query_tokenizer,
            passage_tokenizer=self.passage_tokenizer,
            max_seq_len_passage=max_seq_len_passage,
            max_seq_len_query=max_seq_len_query,
            label_list=["hard_negative", "positive"],
            metric="text_similarity_metric",
            embed_title=embed_title,
            num_hard_negatives=0,
            num_positives=1,
        )

    def retrieve(
        self,
        query: str,
        filters: dict = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
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
            logger.error(
                "Cannot perform retrieve() since DensePassageRetriever initialized with document_store=None"
            )
            return []
        if index is None:
            index = self.document_store.index

        query_emb = self.embed_queries(texts=[query])
        documents = self.document_store.query_by_embedding(
            query_emb=query_emb[0],
            top_k=top_k,
            filters=filters,
            index=index,
            headers=headers,
            return_embedding=False)
        return documents

    def _get_predictions(self, dicts):
        """
        Feed a preprocessed dataset to the model and get the actual predictions (forward pass + formatting).

        :param dicts: list of dictionaries
        examples:[{'query': "where is florida?"}, {'query': "who wrote lord of the rings?"}, ...]
                [{'passages': [{
                    "title": 'Big Little Lies (TV series)',
                    "text": 'series garnered several accolades. It received..',
                    "label": 'positive',
                    "external_id": '18768923'},
                    {"title": 'Framlingham Castle',
                    "text": 'Castle on the Hill "Castle on the Hill" is a song by English..',
                    "label": 'positive',
                    "external_id": '19930582'}, ...]
        :return: dictionary of embeddings for "passages" and "query"
        """

        dataset, tensor_names, _, baskets = self.processor.dataset_from_dicts(
            dicts, indices=[i for i in range(len(dicts))], return_baskets=True)

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self.passage_tokenizer.pad_token_id
                ),  # input_ids
            Pad(axis=0, pad_val=self.passage_tokenizer.pad_token_type_id
                ),  # token_type_ids
        ): [data for data in fn(samples)]

        batch_sampler = paddle.io.BatchSampler(dataset,
                                               batch_size=self.batch_size,
                                               shuffle=False)

        data_loader = paddle.io.DataLoader(dataset=dataset,
                                           batch_sampler=batch_sampler,
                                           collate_fn=batchify_fn,
                                           return_list=True)

        all_embeddings = {"query": [], "passages": []}

        # Todo(tianxin04): ErnieDualEncoder subclass nn.Module,
        self.ernie_dual_encoder.eval()

        # When running evaluations etc., we don't want a progress bar for every single query
        if len(dataset) == 1:
            disable_tqdm = True
        else:
            disable_tqdm = not self.progress_bar

        with tqdm(
                total=len(data_loader) * self.batch_size,
                unit=" Docs",
                desc=f"Create embeddings",
                position=1,
                leave=False,
                disable=disable_tqdm,
        ) as progress_bar:
            for batch in data_loader:
                input_ids, token_type_ids = batch
                #input_ids, token_type_ids, label_ids = batch
                with paddle.no_grad():
                    cls_embeddings = self.ernie_dual_encoder.get_pooled_embedding(
                        input_ids=input_ids, token_type_ids=token_type_ids)
                    if "query" in dicts[0]:
                        all_embeddings["query"].append(
                            cls_embeddings.cpu().numpy())
                    if "passages" in dicts[0]:
                        all_embeddings["passages"].append(
                            cls_embeddings.cpu().numpy())
                progress_bar.update(self.batch_size)

        if all_embeddings["passages"]:
            all_embeddings["passages"] = np.concatenate(
                all_embeddings["passages"])
        if all_embeddings["query"]:
            all_embeddings["query"] = np.concatenate(all_embeddings["query"])
        return all_embeddings

    def embed_queries(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create embeddings for a list of queries using the query encoder

        :param texts: Queries to embed
        :return: Embeddings, one per input queries
        """
        queries = [{"query": q} for q in texts]
        result = self._get_predictions(queries)["query"]
        return result

    def embed_documents(self, docs: List[Document]) -> List[np.ndarray]:
        """
        Create embeddings for a list of documents using the passage encoder

        :param docs: List of Document objects used to represent documents / passages in a standardized way within pipelines.
        :return: Embeddings of documents / passages shape (batch_size, embedding_dim)
        """
        if self.processor.num_hard_negatives != 0:
            logger.warning(
                f"'num_hard_negatives' is set to {self.processor.num_hard_negatives}, but inference does "
                f"not require any hard negatives. Setting num_hard_negatives to 0."
            )
            self.processor.num_hard_negatives = 0

        passages = [{
            "passages": [{
                "title":
                d.meta["name"] if d.meta and "name" in d.meta else "",
                "text":
                d.content,
                "label":
                d.meta["label"] if d.meta and "label" in d.meta else "positive",
                "external_id":
                d.id,
            }]
        } for d in docs]
        embeddings = self._get_predictions(passages)["passages"]

        return embeddings
