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

from typing import List, Optional, Union
import logging
from pathlib import Path

import paddle
from paddlenlp.transformers import ErnieCrossEncoder, AutoTokenizer

from pipelines.schema import Document
from pipelines.nodes.ranker import BaseRanker
from pipelines.utils.common_utils import initialize_device_settings

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
        )

        self.top_k = top_k

        self.devices, _ = initialize_device_settings(use_cuda=use_gpu,
                                                     multi_gpu=True)

        self.transformer_model = ErnieCrossEncoder(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.transformer_model.eval()

        if len(self.devices) > 1:
            self.model = paddle.DataParallel(self.transformer_model)

    def predict_batch(self,
                      query_doc_list: List[dict],
                      top_k: int = None,
                      batch_size: int = None):
        """
        Use loaded Ranker model to, for a list of queries, rank each query's supplied list of Document.

        Returns list of dictionary of query and list of document sorted by (desc.) similarity with query

        :param query_doc_list: List of dictionaries containing queries with their retrieved documents
        :param top_k: The maximum number of answers to return for each query
        :param batch_size: Number of samples the model receives in one batch for inference
        :return: List of dictionaries containing query and ranked list of Document
        """
        raise NotImplementedError

    def predict(self,
                query: str,
                documents: List[Document],
                top_k: Optional[int] = None) -> List[Document]:
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

        features = self.tokenizer([query for doc in documents],
                                  [doc.content for doc in documents],
                                  max_seq_len=256,
                                  pad_to_max_seq_len=True,
                                  truncation_strategy="longest_first")

        tensors = {k: paddle.to_tensor(v) for (k, v) in features.items()}

        with paddle.no_grad():
            similarity_scores = self.transformer_model.matching(
                **tensors).numpy()

        for doc, rank_score in zip(documents, similarity_scores):
            doc.rank_score = rank_score
            doc.score = rank_score

        sorted_scores_and_documents = sorted(
            zip(similarity_scores, documents),
            key=lambda similarity_document_tuple: similarity_document_tuple[0],
            reverse=True,
        )

        # rank documents according to scores
        sorted_documents = [doc for _, doc in sorted_scores_and_documents]
        return sorted_documents[:top_k]
