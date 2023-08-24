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

import json
import logging
from pathlib import Path
from typing import Optional, Union

import requests

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
from tqdm.auto import tqdm

from pipelines.document_stores import BaseDocumentStore
from pipelines.nodes.retriever.dense import DensePassageRetriever

logger = logging.getLogger(__name__)


class ErnieRetriever(DensePassageRetriever):

    """
    Retriever that uses a bi-encoder (query model for query, passage model for passage).
    """

    def __init__(
        self,
        document_store: BaseDocumentStore,
        query_embedding_model: Union[Path, str] = "ernie-embedding-v1",
        passage_embedding_model: Union[Path, str] = "ernie-embedding-v1",
        max_seq_len_query: int = 64,
        max_seq_len_passage: int = 384,
        top_k: int = 10,
        batch_size: int = 16,
        embed_title: bool = True,
        similarity_function: str = "dot_product",
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        progress_bar: bool = True,
        mode: Literal["snippets", "raw_documents", "preprocessed_documents"] = "preprocessed_documents",
        **kwargs
    ):

        """
        Init the Retriever incl. the two encoder models from a local or remote model checkpoint.
        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param query_embedding_model: Local path or remote name of question encoder checkpoint. The format equals the
                                      one used by paddlenlp transformers' models
                                      Currently available remote names: ``"ernie-embedding-v1"``
        :param passage_embedding_model: Local path or remote name of passage encoder checkpoint. The format equals the
                                        one used by paddlenlp transformers' models
                                        Currently available remote names: ``"ernie-embedding-v1"``
        :param max_seq_len_query: Longest length of each query sequence. Maximum number of tokens for the query text. Longer ones will be cut down."
        :param max_seq_len_passage: Longest length of each passage/context sequence. Maximum number of tokens for the passage text. Longer ones will be cut down."
        :param top_k: How many documents to return per query.
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
        if api_key is None or secret_key is None:
            raise Exception(
                "Please apply api_key and secret_key from https://cloud.baidu.com/doc/WENXINWORKSHOP/s/alj562vvu"
            )
        self.api_key = api_key
        self.secret_key = secret_key
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.document_store = document_store
        self.top_k = top_k
        self.embed_title = embed_title
        self.token = self._apply_token(self.api_key, self.secret_key)

    def _apply_token(self, api_key, secret_key):
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        payload = ""
        token_host = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
        response = requests.request("POST", token_host, headers=headers, data=payload)
        if response:
            res = response.json()
        else:
            raise RuntimeError("Request access token error.")

        return res["access_token"]

    def predict(self, data, api_key=None, secret_key=None):
        if api_key is not None and secret_key is not None:
            self.token = self._apply_token(api_key, secret_key)
        payload = json.dumps(
            {
                "input": data,
            }
        )
        headers = {"Content-Type": "application/json"}
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token={}".format(
            self.token
        )
        try:

            response = requests.request("POST", url, headers=headers, data=payload)
        except Exception as e:
            logger.error(e)
        response_json = json.loads(response.text)
        embedding_data = response_json["data"]
        embeddings = [item["embedding"] for item in embedding_data]
        return embeddings

    def _get_predictions(self, dicts, **kwargs):
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
        datasets = []
        if "passages" in dicts[0]:
            # dicts is a list of passages
            for passages in dicts:
                for item in passages["passages"]:
                    if self.embed_title:
                        datasets.append(item["title"] + item["text"])
                    else:
                        datasets.append(item["text"])
        elif "query" in dicts[0]:
            # dicts is a list of passages
            for passages in dicts:
                datasets.append(passages["query"])

        all_embeddings = {"query": [], "passages": []}

        # When running evaluations etc., we don't want a progress bar for every single query
        if len(datasets) == 1:
            disable_tqdm = True
        else:
            disable_tqdm = not self.progress_bar
        with tqdm(
            total=len(datasets) // self.batch_size,
            unit=" Docs",
            desc="Create embeddings",
            position=1,
            leave=False,
            disable=disable_tqdm,
        ) as progress_bar:
            for i in range(0, len(datasets), self.batch_size):

                if "query" in dicts[0]:
                    batch_embeddings = self.predict(datasets[i : i + self.batch_size], **kwargs)
                    all_embeddings["query"].append(batch_embeddings)
                if "passages" in dicts[0]:
                    batch_embeddings = self.predict(datasets[i : i + self.batch_size], **kwargs)
                    all_embeddings["passages"].append(batch_embeddings)
                progress_bar.update(self.batch_size)

        if all_embeddings["passages"]:
            all_embeddings["passages"] = np.concatenate(all_embeddings["passages"])
        if all_embeddings["query"]:
            all_embeddings["query"] = np.concatenate(all_embeddings["query"])
        return all_embeddings
