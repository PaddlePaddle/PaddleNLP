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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class BiEncoder(nn.Layer):
    """dual-encoder model

    Attributes:
        state: for question or for context
        question_encoder: used to code the problem
        context_encoder: used to code the context

    """

    def __init__(self, question_encoder, context_encoder, state=None):
        super(BiEncoder, self).__init__()
        self.state = state
        if self.state is None:
            self.question_encoder = question_encoder
            self.context_encoder = context_encoder
        elif self.state == "FORQUESTION":
            self.question_encoder = question_encoder
        elif self.state == "FORCONTEXT":
            self.context_encoder = context_encoder

    def get_question_pooled_embedding(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):

        _, cls_embedding = self.question_encoder(input_ids, token_type_ids, position_ids, attention_mask)
        """cls_embedding = self.emb_reduce_linear(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)"""

        return cls_embedding

    def get_context_pooled_embedding(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):

        _, cls_embedding = self.context_encoder(input_ids, token_type_ids, position_ids, attention_mask)
        """cls_embedding = self.emb_reduce_linear(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)"""

        return cls_embedding

    def forward(
        self,
        question_id,
        question_segments,
        question_attn_mask,
        context_ids,
        context_segments,
        context_attn_mask,
    ):

        question_pooled_out = self.get_question_pooled_embedding(question_id, question_segments, question_attn_mask)
        context_pooled_out = self.get_context_pooled_embedding(context_ids, context_segments, context_attn_mask)

        return question_pooled_out, context_pooled_out


class BiEncoderNllLoss(object):
    """
    calculate the nll loss for dual-encoder model
    """

    def calc(self, q_vectors, ctx_vectors, positive_idx_per_question, loss_scale=None):

        scorces = paddle.matmul(q_vectors, paddle.transpose(ctx_vectors, [1, 0]))

        # if len(q_vectors.shape()) > 1:
        q_num = q_vectors.shape[0]
        scores = scorces.reshape([q_num, -1])

        softmax_scorces = F.log_softmax(scores, axis=1)

        loss = F.nll_loss(softmax_scorces, paddle.to_tensor(positive_idx_per_question))

        correct_predictions_count = None

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count
