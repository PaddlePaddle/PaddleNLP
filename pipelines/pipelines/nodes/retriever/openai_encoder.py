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
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Any, Dict, Iterator, List

import numpy as np
import openai
import tiktoken
from tqdm import tqdm

from pipelines.nodes.retriever.base_embedding_encoder import _BaseEmbeddingEncoder
from pipelines.schema import Document

if TYPE_CHECKING:
    from pipelines.nodes.retriever import EmbeddingRetriever

logger = logging.getLogger(__name__)

OPENAI_TIMEOUT = 30


def load_openai_tokenizer(tokenizer_name: str):
    """Load either the tokenizer from tiktoken (if the library is available) or fallback to the GPT2TokenizerFast
    from the transformers library.

    :param tokenizer_name: The name of the tokenizer to load.
    """

    logger.debug("Using tiktoken %s tokenizer", tokenizer_name)
    return tiktoken.get_encoding(tokenizer_name)


class OpenAIEmbeddingEncoder(_BaseEmbeddingEncoder):
    def __init__(self, retriever: "EmbeddingRetriever"):
        # See https://platform.openai.com/docs/guides/embeddings and
        # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/embeddings?tabs=console for more details
        self.using_azure = (
            retriever.azure_deployment_name is not None
            and retriever.azure_base_url is not None
            and retriever.api_version is not None
        )

        if self.using_azure:
            self.url = f"{retriever.azure_base_url}/openai/deployments/{retriever.azure_deployment_name}/embeddings?api-version={retriever.api_version}"
            openai.api_type = "azure"
            openai.api_base = "https://zeyuchen.openai.azure.com/"
            openai.api_version = "2023-07-01-preview"
            openai.api_key = "b0ac6ac2582948ba832f1ea4e1d8767a"
            # deployment_id = 'text-embedding-ada'
        else:
            self.url = f"{retriever.api_base}/embeddings"

        self.api_key = retriever.api_key
        self.openai_organization = retriever.openai_organization
        self.batch_size = min(64, retriever.batch_size)
        self.progress_bar = retriever.progress_bar
        self.azure_deployment_name = retriever.azure_deployment_name

        model_class: str = next(
            (m for m in ["ada", "babbage", "davinci", "curie"] if m in retriever.embedding_model), "babbage"
        )

        tokenizer = self._setup_encoding_models(model_class, retriever.embedding_model, retriever.max_seq_len)
        self._tokenizer = load_openai_tokenizer(tokenizer_name=tokenizer)

    def _setup_encoding_models(self, model_class: str, model_name: str, max_seq_len: int):
        """
        Setup the encoding models for the retriever.
        """

        tokenizer_name = "gpt2"
        # new generation of embedding models (December 2022), we need to specify the full name
        if model_name.endswith("-002"):
            self.query_encoder_model = model_name
            self.doc_encoder_model = model_name
            self.max_seq_len = min(8191, max_seq_len)
            try:
                tokenizer_name = tiktoken.encoding_name_for_model(model_name)
            except KeyError:
                tokenizer_name = "cl100k_base"
        else:
            self.query_encoder_model = f"text-search-{model_class}-query-001"
            self.doc_encoder_model = f"text-search-{model_class}-doc-001"
            self.max_seq_len = min(2046, max_seq_len)

        return tokenizer_name

    def _ensure_text_limit(self, text: str) -> str:
        """
        Ensure that length of the text is within the maximum length of the model.
        OpenAI v1 embedding models have a limit of 2046 tokens, and v2 models have a limit of 8191 tokens.
        """
        n_tokens = len(self._tokenizer.encode(text))
        if n_tokens <= self.max_seq_len:
            return text

        logger.warning(
            "The prompt has been truncated from %s tokens to %s tokens to fit within the max token limit."
            " Reduce the length of the prompt to prevent it from being cut off.",
            n_tokens,
            self.max_seq_len,
        )

        tokenized_payload = self._tokenizer.encode(text)
        decoded_string = self._tokenizer.decode(tokenized_payload[: self.max_seq_len])

        return decoded_string

    def embed(self, model: str, text: List[str]) -> np.ndarray:
        if self.api_key is None:
            raise ValueError(
                f"{'Azure ' if self.using_azure else ''}OpenAI API key is not set. You can set it via the `api_key` parameter of the EmbeddingRetriever."
            )

        generated_embeddings: List[Any] = []

        def azure_get_embedding(input: str):
            deployment_id = "text-embedding-ada"
            res = openai.Embedding.create(deployment_id=deployment_id, input=[input])
            return res["data"]

        if self.using_azure:
            thread_count = cpu_count() if len(text) > cpu_count() else len(text)
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                results: Iterator[Dict[str, Any]] = executor.map(azure_get_embedding, text)
                generated_embeddings.extend(results)
        # else:
        #     payload: Dict[str, Union[List[str], str]] = {"model": model, "input": text}
        #     headers["Authorization"] = f"Bearer {self.api_key}"
        #     if self.openai_organization:
        #         headers["OpenAI-Organization"] = self.openai_organization

        #     res = openai_request(url=self.url, headers=headers, payload=payload, timeout=OPENAI_TIMEOUT)

        #     unordered_embeddings = [(ans["index"], ans["embedding"]) for ans in res["data"]]
        #     ordered_embeddings = sorted(unordered_embeddings, key=lambda x: x[0])

        # generated_embeddings = [emb[1] for emb in ordered_embeddings]
        # breakpoint()
        generated_embeddings = [item["embedding"] for item in generated_embeddings[0]]
        return np.array(generated_embeddings)

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
