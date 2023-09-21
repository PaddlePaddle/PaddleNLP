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
from dataclasses import dataclass
from typing import Dict, Optional

import paddle
import paddle.distributed as dist
import paddle.nn as nn

from paddlenlp.transformers import AutoModel
from paddlenlp.transformers.model_outputs import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[paddle.Tensor] = None
    p_reps: Optional[paddle.Tensor] = None
    loss: Optional[paddle.Tensor] = None
    scores: Optional[paddle.Tensor] = None


class BiEncoderModel(nn.Layer):
    TRANSFORMER_CLS = AutoModel

    def __init__(
        self,
        model_name: str = None,
        normalized: bool = False,
        sentence_pooling_method: str = "cls",
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

        self.normalized = normalized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        if not normalized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError("Distributed training has not been initialized for representation all gather.")
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def sentence_embedding(self, hidden_state, mask):

        if self.sentence_pooling_method == "weighted_mean":
            # Use weighted mean to compute similarity for decoder only LLMs
            # refer to https://github.com/Muennighoff/sgpt/blob/9728de441b1dd2e638a8a64e1c83f77716f47d9a/biencoder/beir/beir_dense_retriever.py#L258
            weights = (
                paddle.arange(start=1, end=hidden_state.shape[1] + 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(hidden_state.shape)
            )
            # [batch_size, seq_len] -> [batch_size, seq_len, higgen_dim]
            input_mask_expanded = mask.unsqueeze(-1).expand(hidden_state.shape)
            # bs, seq_len, hidden_dim -> bs, hidden_dim
            sum_embeddings = paddle.sum(hidden_state * input_mask_expanded * weights, axis=1)
            sum_mask = paddle.sum(input_mask_expanded * weights, axis=1)
            embedding = sum_embeddings / sum_mask
            return embedding

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features["attention_mask"])
        if self.normalized:
            p_reps = paddle.nn.functional.normalize(p_reps, axis=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.shape) == 2:
            return paddle.matmul(q_reps, p_reps.transpose([1, 0]))
        return paddle.matmul(q_reps, p_reps.transpose([-2, -1]))

    def forward(
        self,
        query: Dict[str, paddle.Tensor] = None,
        passage: Dict[str, paddle.Tensor] = None,
        teacher_score: paddle.Tensor = None,
    ):
        q_reps = self.encode(query)
        p_reps = self.encode(passage)

        if self.training:
            if self.negatives_cross_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores / self.temperature
            scores = scores.reshape([q_reps.shape[0], -1])

            target = paddle.arange(scores.shape[0], dtype="int64")
            target = target * (p_reps.shape[0] // q_reps.shape[0])
            loss = self.compute_loss(scores, target)

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

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
