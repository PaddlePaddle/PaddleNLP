# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

from pipelines.nodes.ranker import BaseRanker
from pipelines.schema import Document
from pipelines.utils.common_utils import initialize_device_settings
from tqdm import tqdm

from paddlenlp import Taskflow

logger = logging.getLogger(__name__)


class ErnieRanker(BaseRanker):
    """
    Re-Ranking can be used on top of a retriever to boost the performance for document search. This is particularly useful if the retriever has a high recall but is bad in sorting the documents by relevance.

    Usage example:
    ...
    retriever = ElasticsearchRetriever(document_store=document_store)
    ranker = SentenceTransformersRanker(model_name_or_path="rocketqa-zh-dureader-cross-encoder")
    p = Pipeline()
    p.add_node(component=retriever, name="ESRetriever", inputs=["Query"])
    p.add_node(component=ranker, name="Ranker", inputs=["ESRetriever"])
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        top_k: int = 10,
        use_gpu: bool = True,
        max_seq_len: int = 512,
        progress_bar: bool = True,
        batch_size: int = 1000,
        reinitialize: bool = False,
        embed_title: bool = False,
        use_en: bool = False,
    ):
        """
        :param model_name_or_path: Directory of a saved model or the name of a public model e.g.
        'rocketqa-zh-dureader-cross-encoder'.
        :param top_k: The maximum number of documents to return
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        """

        # save init parameters to enable export of component config as YAML
        self.set_config(
            model_name_or_path=model_name_or_path,
            top_k=top_k,
            use_en=use_en,
        )

        self.top_k = top_k
        # Parameter to control the use of English Cross Encoder Model
        self.use_en = use_en

        self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=True)

        logger.info("Loading Parameters from:{}".format(model_name_or_path))
        self.embed_title = embed_title
        self.progress_bar = progress_bar
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        # self.transformer_model = ErnieCrossEncoder(model_name_or_path, reinitialize=reinitialize)
        self.transformer_model = Taskflow(
            "text_similarity", model=model_name_or_path, batch_size=self.batch_size, device_id=0 if use_gpu else -1
        )

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """
        Use loaded ranker model to re-rank the supplied list of Document.

        Returns list of Document sorted by (desc.) similarity with the query.

        :param query: Query string
        :param documents: List of Document to be re-ranked
        :param top_k: The maximum number of documents to return
        :return: List of Document
        """
        if top_k is None:
            top_k = self.top_k
        datasets = []
        for doc in documents:
            if self.embed_title:
                datasets.append([query, doc.meta["name"] + doc.content])
            else:
                datasets.append([query, doc.content])
        outputs = self.transformer_model(datasets)
        similarity_scores = [item["similarity"] for item in outputs]

        for doc, rank_score in zip(documents, similarity_scores):
            doc.rank_score = rank_score
            doc.score = rank_score

        sorted_scores_and_documents = sorted(
            zip(similarity_scores, documents),
            key=lambda similarity_document_tuple: similarity_document_tuple[0],
            reverse=True,
        )

        # Rank documents according to scores
        sorted_documents = [doc for _, doc in sorted_scores_and_documents]
        return sorted_documents[:top_k]

    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Union[List[Document], List[List[Document]]]:
        """
        Use loaded ranker model to re-rank the supplied lists of Documents

        Returns lists of Documents sorted by (desc.) similarity with the corresponding queries.

        :param queries: Single query string or list of queries
        :param documents: Single list of Documents or list of lists of Documents to be reranked.
        :param top_k: The maximum number of documents to return per Document list.
        :param batch_size: Number of Documents to process at a time.
        """
        if top_k is None:
            top_k = self.top_k

        if batch_size is None:
            batch_size = self.batch_size

        number_of_docs, all_queries, all_docs, single_list_of_docs = self._preprocess_batch_queries_and_docs(
            queries=queries, documents=documents
        )
        batches = self._get_batches(all_queries=all_queries, all_docs=all_docs, batch_size=batch_size)
        pb = tqdm(total=len(all_docs), disable=not self.progress_bar, desc="Ranking")

        preds = []
        for cur_queries, cur_docs in batches:
            datasets = []
            for query, doc in zip(cur_queries, cur_docs):
                if self.embed_title:
                    datasets.append([query, doc.meta["name"] + doc.content])
                else:
                    datasets.append([query, doc.content])
            outputs = self.transformer_model(datasets)
            similarity_scores = [item["similarity"] for item in outputs]
            preds.extend(similarity_scores)

            for doc, rank_score in zip(cur_docs, similarity_scores):
                doc.rank_score = rank_score
                doc.score = rank_score
            pb.update(len(cur_docs))
        pb.close()
        if single_list_of_docs:
            sorted_scores_and_documents = sorted(
                zip(preds, documents),
                key=lambda similarity_document_tuple: similarity_document_tuple[0],
                reverse=True,
            )
            sorted_documents = [doc for _, doc in sorted_scores_and_documents]
            return sorted_documents[:top_k]
        else:
            grouped_predictions = []
            left_idx = 0
            right_idx = 0
            for number in number_of_docs:
                right_idx = left_idx + number
                grouped_predictions.append(preds[left_idx:right_idx])
                left_idx = right_idx
            result = []
            for pred_group, doc_group in zip(grouped_predictions, documents):
                sorted_scores_and_documents = sorted(
                    zip(pred_group, doc_group),
                    key=lambda similarity_document_tuple: similarity_document_tuple[0],
                    reverse=True,
                )
                sorted_documents = [doc for _, doc in sorted_scores_and_documents]
                result.append(sorted_documents[:top_k])
            return result

    def _preprocess_batch_queries_and_docs(
        self, queries: List[str], documents: Union[List[Document], List[List[Document]]]
    ) -> Tuple[List[int], List[str], List[Document], bool]:
        number_of_docs = []
        all_queries = []
        all_docs: List[Document] = []
        single_list_of_docs = False

        # Docs case 1: single list of Documents -> rerank single list of Documents based on single query
        if len(documents) > 0 and isinstance(documents[0], Document):
            if len(queries) != 1:
                raise Exception("Number of queries must be 1 if a single list of Documents is provided.")
            query = queries[0]
            number_of_docs = [len(documents)]
            all_queries = [query] * len(documents)
            all_docs = documents  # type: ignore
            single_list_of_docs = True

        # Docs case 2: list of lists of Documents -> rerank each list of Documents based on corresponding query
        # If queries contains a single query, apply it to each list of Documents
        if len(documents) > 0 and isinstance(documents[0], list):
            if len(queries) == 1:
                queries = queries * len(documents)
            if len(queries) != len(documents):
                raise Exception("Number of queries must be equal to number of provided Document lists.")
            for query, cur_docs in zip(queries, documents):
                if not isinstance(cur_docs, list):
                    raise Exception(f"cur_docs was of type {type(cur_docs)}, but expected a list of Documents.")
                number_of_docs.append(len(cur_docs))
                all_queries.extend([query] * len(cur_docs))
                all_docs.extend(cur_docs)

        return number_of_docs, all_queries, all_docs, single_list_of_docs

    @staticmethod
    def _get_batches(
        all_queries: List[str], all_docs: List[Document], batch_size: Optional[int]
    ) -> Iterator[Tuple[List[str], List[Document]]]:
        if batch_size is None:
            yield all_queries, all_docs
            return
        else:
            for index in range(0, len(all_queries), batch_size):
                yield all_queries[index : index + batch_size], all_docs[index : index + batch_size]
