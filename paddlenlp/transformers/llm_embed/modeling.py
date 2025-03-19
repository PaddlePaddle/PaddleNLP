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

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
from tqdm import tqdm

from ...utils.log import logger
from .. import AutoConfig, AutoModel, AutoTokenizer, PretrainedModel
from ..model_outputs import ModelOutput


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[paddle.Tensor] = None
    p_reps: Optional[paddle.Tensor] = None
    loss: Optional[paddle.Tensor] = None
    scores: Optional[paddle.Tensor] = None


__all__ = ["BiEncoderModel"]


class BiEncoderModel(PretrainedModel):
    def __init__(
        self,
        model_name_or_path: str = None,
        corpus_model_name_or_path: str = None,
        query_model_name_or_path: str = None,
        dtype: str = "float16",
        normalized: bool = False,
        sentence_pooling_method: str = "cls",
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        use_inbatch_neg: bool = True,
        margin: float = 0.3,
        matryoshka_dims: Optional[List[int]] = None,
        matryoshka_loss_weights: Optional[List[float]] = None,
        query_instruction: Optional[str] = None,
        document_instruction: Optional[str] = None,
        eval_batch_size: int = 8,
        tokenizer=None,
        max_seq_length: int = 4096,
        model_flag: str = None,
    ):
        super().__init__()

        # Load Model
        self.model = None
        self.model_config = None
        self.corpus_model = None
        self.query_model = None
        if model_name_or_path is not None:
            self.model = AutoModel.from_pretrained(model_name_or_path, dtype=dtype, convert_from_torch=True)
            self.model_config = AutoConfig.from_pretrained(model_name_or_path)
        if corpus_model_name_or_path is not None:
            self.corpus_model = AutoModel.from_pretrained(
                corpus_model_name_or_path, dtype=dtype, convert_from_torch=True
            )
        if query_model_name_or_path is not None:
            self.query_model = AutoModel.from_pretrained(
                query_model_name_or_path, dtype=dtype, convert_from_torch=True
            )
        if self.corpus_model is None:
            self.corpus_model = self.model
        if self.query_model is None:
            self.query_model = self.model
        assert self.corpus_model is not None and self.query_model is not None

        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

        self.normalized = normalized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model_config
        self.margin = margin
        self.matryoshka_dims = matryoshka_dims

        self.query_instruction = query_instruction
        self.document_instruction = document_instruction
        self.eval_batch_size = eval_batch_size
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        if self.matryoshka_dims:
            self.matryoshka_loss_weights = (
                matryoshka_loss_weights if matryoshka_loss_weights else [1] * len(self.matryoshka_dims)
            )
        else:
            self.matryoshka_loss_weights = None

        if not normalized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError("Distributed training has not been initialized for representation all gather.")
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.model_flag = model_flag

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == "mean":
            s = paddle.sum(hidden_state * mask.unsqueeze(-1).float(), axis=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == "cls":
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == "last":
            # return hidden_state[:, -1] # this is for padding side is left
            sequence_lengths = mask.sum(axis=1)
            last_token_indices = sequence_lengths - 1
            embeddings = hidden_state[paddle.arange(hidden_state.shape[0]), last_token_indices]
            return embeddings
        else:
            raise ValueError(f"Invalid sentence pooling method: {self.sentence_pooling_method}")

    def get_model_config(
        self,
    ):
        return self.model_config.to_dict()

    def encode(self, features, model: AutoModel):
        psg_out = model(**features, return_dict=True, output_hidden_states=True)
        p_reps = self.sentence_embedding(psg_out.hidden_states[-1], features["attention_mask"])
        return p_reps

    def compute_similarity(self, q_reps, p_reps):
        # q_reps [batch_size, embedding_dim]
        # p_reps [batch_size, embedding_dim]
        return paddle.matmul(q_reps, p_reps.transpose([1, 0]))

    def hard_negative_loss(self, q_reps, p_reps):
        scores = self.compute_similarity(q_reps, p_reps)
        scores = scores / self.temperature
        scores = scores.reshape([q_reps.shape[0], -1])

        target = paddle.arange(scores.shape[0], dtype="int64")
        target = target * (p_reps.shape[0] // q_reps.shape[0])
        loss = self.compute_loss(scores, target)
        return scores, loss

    def in_batch_negative_loss(self, q_reps, p_reps):
        # In batch negatives
        scores = self.compute_similarity(q_reps, p_reps)
        # Substract margin from all positive samples cosine_sim()
        margin_diag = paddle.full(shape=[q_reps.shape[0]], fill_value=self.margin, dtype=q_reps.dtype)
        scores = scores - paddle.diag(margin_diag)
        # Scale cosine to ease training converge
        scores = scores / self.temperature
        target = paddle.arange(0, q_reps.shape[0], dtype="int64")
        loss = self.compute_loss(scores, target)
        return scores, loss

    def forward(
        self,
        query: Dict[str, paddle.Tensor] = None,
        passage: Dict[str, paddle.Tensor] = None,
        teacher_score: paddle.Tensor = None,
    ):
        q_reps = self.encode(query, self.query_model)
        p_reps = self.encode(passage, self.corpus_model)

        # For non-matryoshka loss, we normalize the representations
        if not self.matryoshka_dims:
            if self.normalized:
                q_reps = paddle.nn.functional.normalize(q_reps, axis=-1)
                p_reps = paddle.nn.functional.normalize(p_reps, axis=-1)

        if self.training:
            # Cross device negatives
            if self.negatives_cross_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            if self.matryoshka_dims:
                loss = 0.0
                scores = 0.0
                for loss_weight, dim in zip(self.matryoshka_loss_weights, self.matryoshka_dims):
                    reduced_q = q_reps[:, :dim]
                    reduced_d = p_reps[:, :dim]
                    if self.normalized:
                        reduced_q = paddle.nn.functional.normalize(reduced_q, axis=-1)
                        reduced_d = paddle.nn.functional.normalize(reduced_d, axis=-1)

                    if self.use_inbatch_neg:
                        dim_score, dim_loss = self.in_batch_negative_loss(reduced_q, reduced_d)
                    else:
                        dim_score, dim_loss = self.hard_negative_loss(reduced_q, reduced_d)
                    scores += dim_score
                    loss += loss_weight * dim_loss

            elif self.use_inbatch_neg:
                scores, loss = self.in_batch_negative_loss(q_reps, p_reps)
            else:
                scores, loss = self.hard_negative_loss(q_reps, p_reps)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[paddle.Tensor]):
        if t is None:
            return None

        all_tensors = [paddle.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = paddle.concat(all_tensors, axis=0)

        return all_tensors

    def save_pretrained(self, output_dir: str, **kwargs):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)

    @paddle.no_grad()
    def encode_sentences(
        self, sentences: List[str], model: AutoModel, tokenizer: AutoTokenizer, titles: List[str] = None, **kwargs
    ) -> np.ndarray:
        model.eval()
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), self.eval_batch_size), desc="Batches"):
            sentences_batch = sentences[start_index : start_index + self.eval_batch_size]
            if titles:
                titles_batch = titles[start_index : start_index + self.eval_batch_size]
                assert len(sentences_batch) == len(titles_batch)
                inputs = tokenizer(
                    titles_batch,
                    sentences_batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pd",
                    max_length=self.max_seq_length,
                    return_attention_mask=True,
                )
            else:
                inputs = tokenizer(
                    sentences_batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pd",
                    max_length=self.max_seq_length,
                    return_attention_mask=True,
                )
            outputs = model(
                **inputs,  # 注意 bert 类型有 token_type_ids
                return_dict=True,
                output_hidden_states=True,
            )
            last_hidden_state = outputs.hidden_states[-1]

            if self.sentence_pooling_method == "last":
                if tokenizer.padding_side == "right":
                    sequence_lengths = inputs.attention_mask.sum(axis=1)
                    last_token_indices = sequence_lengths - 1
                    embeddings = last_hidden_state[paddle.arange(last_hidden_state.shape[0]), last_token_indices]
                elif tokenizer.padding_side == "left":
                    embeddings = last_hidden_state[:, -1]
                else:
                    raise NotImplementedError(f"Padding side {tokenizer.padding_side} not supported.")
            elif self.sentence_pooling_method == "cls":
                embeddings = last_hidden_state[:, 0]
            elif self.sentence_pooling_method == "mean":
                inputs.attention_mask = paddle.cast(
                    inputs.attention_mask, dtype="float32"
                )  # float cannot * int64, maybe paddle's bug
                s = paddle.sum(last_hidden_state * inputs.attention_mask.unsqueeze(-1), axis=1)
                d = inputs.attention_mask.sum(axis=1, keepdim=True)
                embeddings = s / d
            elif self.sentence_pooling_method == "last_8":
                last_8_embeddings = last_hidden_state[:, -8:, :]
                embeddings = paddle.mean(last_8_embeddings, axis=1)
            else:
                raise NotImplementedError(f"Pooling method {self.pooling_method} not supported.")

            if self.normalized:
                embeddings = paddle.nn.functional.normalize(embeddings, p=2, axis=-1)

            all_embeddings.append(embeddings.cpu().numpy().astype("float32"))

        return np.concatenate(all_embeddings, axis=0)

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """
        This function will be used to encode queries for retrieval task
        if there is a instruction for queries, we will add it to the query text
        """
        if self.query_instruction is not None:
            input_texts = [f"{self.query_instruction}{query}" for query in queries]
        else:
            input_texts = queries

        if self.model_flag == "llara":
            input_texts = self.preprocess_sentences_for_llara(input_texts, query_or_doc="query")
        if self.model_flag == "bge-en-icl":
            input_texts = self.preprocess_sentences_for_bge_en_icl(input_texts, query_or_doc="query")

        encode_results = self.encode_sentences(sentences=input_texts, model=self.query_model, tokenizer=self.tokenizer)
        return encode_results

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
        input_titles = None

        if self.model_flag == "llara":
            input_texts = self.preprocess_sentences_for_llara(input_texts, query_or_doc="doc")
        if "RocketQA" in self.model_flag:
            if isinstance(corpus[0], dict):
                input_texts = [doc["text"] for doc in corpus]
                input_titles = [doc.get("title", "") for doc in corpus]

        encode_results = self.encode_sentences(
            sentences=input_texts, titles=input_titles, model=self.corpus_model, tokenizer=self.tokenizer
        )
        return encode_results

    def preprocess_sentences_for_bge_en_icl(self, sentences: List[str], query_or_doc: str, **kwargs) -> List[str]:
        if query_or_doc == "query":
            query_suffix = "\n<response> "
        else:
            raise ValueError(f"Invalid query_or_doc: {query_or_doc}")

        input_texts = []
        for query in sentences:
            new_query = f"{query}{query_suffix}"
            input_length = len(self.tokenizer(new_query)["input_ids"])
            if input_length > self.max_seq_length:
                cur_len = 0
                add_len = 1
                while add_len < len(query):
                    add_len *= 2
                while add_len > 1:
                    add_len //= 2
                    assert isinstance(cur_len, int) and isinstance(
                        add_len, int
                    ), f"cur_len={cur_len} add_len={add_len}"
                    new_query = f"{query[:cur_len+add_len]}{query_suffix}"
                    input_length = len(self.tokenizer(new_query)["input_ids"])
                    if input_length <= self.max_seq_length:
                        cur_len += add_len
                new_query = f"{query[:cur_len]}{query_suffix}"
            input_texts.append(new_query)

        return input_texts

    def preprocess_sentences_for_llara(self, sentences: List[str], query_or_doc: str, **kwargs) -> List[str]:

        prefix = '"'
        if query_or_doc == "query":
            suffix = '", predict the following passage within eight words: <s9><s10><s11><s12><s13><s14><s15><s16>'
        elif query_or_doc == "doc":
            suffix = '", summarize the above passage within eight words: <s1><s2><s3><s4><s5><s6><s7><s8>'
        else:
            raise ValueError(f"Invalid query_or_doc: {query_or_doc}")

        sentences_after_process = []
        for sentence in sentences:
            inputs = self.tokenizer(
                sentence,
                return_tensors=None,
                max_length=self.max_seq_length - 20,
                truncation=True,
                add_special_tokens=False,
            )
            sentences_after_process.append(self.tokenizer.decode(inputs["input_ids"], skip_special_tokens=True))

        sentences_after_process = [prefix + " " + sentence + " " + suffix for sentence in sentences_after_process]

        return sentences_after_process
