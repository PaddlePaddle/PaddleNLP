# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import paddle
import paddle.nn.functional as F
from base_model import SemanticIndexBase


class SemanticIndexCacheNeg(SemanticIndexBase):
    def __init__(self, pretrained_model, dropout=None, margin=0.3, scale=30, output_emb_size=None):
        super().__init__(pretrained_model, dropout, output_emb_size)
        self.margin = margin
        # Used scaling cosine similarity to ease converge
        self.scale = scale

    def get_pooled_embedding_with_no_grad(
        self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None
    ):
        if self.use_fp16:
            if attention_mask is None:
                attention_mask = paddle.unsqueeze(
                    (input_ids == self.ptm.pad_token_id).astype(self.ptm.pooler.dense.weight.dtype) * -1e4, axis=[1, 2]
                )

            with paddle.no_grad():
                embedding_output = self.ptm.embeddings(
                    input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids
                )

            embedding_output = paddle.cast(embedding_output, "float16")
            attention_mask = paddle.cast(attention_mask, "float16")

            with paddle.no_grad():
                encoder_outputs = self.ptm.encoder(embedding_output, attention_mask)
            if self.use_fp16:
                encoder_outputs = paddle.cast(encoder_outputs, "float32")
            cls_embedding = self.ptm.pooler(encoder_outputs)
        else:
            _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)

        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)
        return cls_embedding

    def forward(
        self,
        query_input_ids,
        title_input_ids,
        query_token_type_ids=None,
        query_position_ids=None,
        query_attention_mask=None,
        title_token_type_ids=None,
        title_position_ids=None,
        title_attention_mask=None,
    ):

        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask
        )

        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids, title_token_type_ids, title_position_ids, title_attention_mask
        )

        cosine_sim = paddle.matmul(query_cls_embedding, title_cls_embedding, transpose_y=True)

        # subtract margin from all positive samples cosine_sim()
        margin_diag = paddle.full(
            shape=[query_cls_embedding.shape[0]], fill_value=self.margin, dtype=paddle.get_default_dtype()
        )

        cosine_sim = cosine_sim - paddle.diag(margin_diag)

        # scale cosine to ease training converge
        cosine_sim *= self.scale

        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype="int64")
        labels = paddle.reshape(labels, shape=[-1, 1])

        return cosine_sim, labels, query_cls_embedding, title_cls_embedding
