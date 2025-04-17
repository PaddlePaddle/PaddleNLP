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
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import requests
from pipelines.nodes.retriever.base_embedding_encoder import _BaseEmbeddingEncoder
from pipelines.schema import Document
from tqdm import tqdm

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pipelines.nodes.retriever import EmbeddingRetriever


class ErnieEmbeddingEncoder(_BaseEmbeddingEncoder):
    def __init__(self, retriever: "EmbeddingRetriever"):
        self.api_key = retriever.api_key
        self.secret_key = retriever.secret_key
        self.batch_size = min(16, retriever.batch_size)
        self.progress_bar = retriever.progress_bar
        self.token = self._apply_token(self.api_key, self.secret_key)
        self._setup_encoding_models(retriever.embedding_model, retriever.max_seq_len)

    def _setup_encoding_models(self, model_name: str, max_seq_len: int):
        """
        Setup the encoding models for the retriever.
        """
        # new generation of embedding models (December 2022), we need to specify the full name
        if model_name.startswith("ernie"):
            self.query_encoder_model = model_name
            self.doc_encoder_model = model_name
            self.max_seq_len = min(384, max_seq_len)

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

    def _ensure_text_limit(self, text: str) -> str:
        """
        Ensure that length of the text is within the maximum length of the model.
        OpenAI v1 embedding models have a limit of 2046 tokens, and v2 models have a limit of 8191 tokens.
        """
        n_tokens = len(text)
        if n_tokens <= self.max_seq_len:
            return text

        logger.warning(
            "The prompt has been truncated from %s tokens to %s tokens to fit within the max token limit."
            " Reduce the length of the prompt to prevent it from being cut off.",
            n_tokens,
            self.max_seq_len,
        )

        tokenized_payload = text[: self.max_seq_len]

        return tokenized_payload

    def embed(self, model: str, text: List[str]) -> np.ndarray:
        generated_embeddings: List[Any] = []
        headers: Dict[str, str] = {"Content-Type": "application/json"}

        payload = json.dumps(
            {
                "input": text,
            }
        )
        headers = {"Content-Type": "application/json"}
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token={}".format(
            self.token
        )
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            response_json = json.loads(response.text)
            embedding_data = response_json["data"]
        except Exception as e:
            logger.error(e)
            logger.error(response_json)

        generated_embeddings = [item["embedding"] for item in embedding_data]

        return generated_embeddings

    def embed_batch(self, model: str, text: List[str]) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(
            range(0, len(text), self.batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = text[i : i + self.batch_size]
            batch_limited = [self._ensure_text_limit(content) for content in batch]
            generated_embeddings = self.embed(model, batch_limited)
            all_embeddings.append(generated_embeddings)
        return np.concatenate(all_embeddings)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        return self.embed_batch(self.query_encoder_model, queries)

    def embed_documents(self, docs: List[Document]) -> np.ndarray:
        return self.embed_batch(self.doc_encoder_model, [d.content for d in docs])
