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

from typing import Dict, List, Union

import numpy as np
import paddle
from tqdm import tqdm


class EncodeModel:
    def __init__(
        self,
        model,
        tokenizer,
        pooling_method: str = "last",
        query_instruction: str = None,
        document_instruction: str = None,
        eval_batch_size: int = 64,
        max_seq_length: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.pooling_method = pooling_method
        self.query_instruction = query_instruction
        self.document_instruction = document_instruction
        self.eval_batch_size = eval_batch_size
        self.max_seq_length = max_seq_length

        if paddle.device.is_compiled_with_cuda():
            self.device = paddle.device.set_device("gpu")
        else:
            self.device = paddle.device.set_device("cpu")
        self.model = self.model.to(self.device)

        num_gpus = paddle.device.cuda.device_count()
        if num_gpus > 1:
            raise NotImplementedError("Multi-GPU is not supported yet.")

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """
        This function will be used to encode queries for retrieval task
        if there is a instruction for queries, we will add it to the query text
        """
        if self.query_instruction is not None:
            input_texts = [f"{self.query_instruction}{query}" for query in queries]
        else:
            input_texts = queries
        return self.encode(input_texts)

    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        """
        This function will be used to encode corpus for retrieval task
        if there is a instruction for docs, we will add it to the doc text
        """
        if isinstance(corpus[0], dict):
            if self.document_instruction is not None:
                input_texts = [
                    "{}{} {}".format(self.document_instruction, doc.get("title", ""), doc["text"]).strip()
                    for doc in corpus
                ]
            else:
                input_texts = ["{} {}".format(doc.get("title", ""), doc["text"]).strip() for doc in corpus]
        else:
            if self.document_instruction is not None:
                input_texts = [f"{self.document_instruction}{doc}" for doc in corpus]
            else:
                input_texts = corpus
        return self.encode(input_texts)

    @paddle.no_grad()
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        self.model.eval()
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), self.eval_batch_size), desc="Batches"):
            sentences_batch = sentences[start_index : start_index + self.eval_batch_size]

            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors="pd",
                max_length=self.max_seq_length,
                return_attention_mask=True,
            )
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            last_hidden_state = outputs.hidden_states[-1]

            if self.pooling_method == "last":
                if self.tokenizer.padding_side == "right":
                    sequence_lengths = inputs.attention_mask.sum(axis=1)
                    last_token_indices = sequence_lengths - 1
                    embeddings = last_hidden_state[paddle.arange(last_hidden_state.shape[0]), last_token_indices]
                elif self.tokenizer.padding_side == "left":
                    embeddings = last_hidden_state[:, -1]
                else:
                    raise NotImplementedError(f"Padding side {self.tokenizer.padding_side} not supported.")
            elif self.pooling_method == "cls":
                embeddings = last_hidden_state[:, 1]
            elif self.pooling_method == "mean":
                s = paddle.sum(last_hidden_state * inputs.attention_mask.unsqueeze(-1), axis=1)
                d = inputs.attention_mask.sum(axis=1, keepdim=True)
                embeddings = s / d
            else:
                raise NotImplementedError(f"Pooling method {self.pooling_method} not supported.")

            embeddings = paddle.nn.functional.normalize(embeddings, p=2, axis=-1)

            all_embeddings.append(embeddings.cpu().numpy().astype("float32"))

        return np.concatenate(all_embeddings, axis=0)
