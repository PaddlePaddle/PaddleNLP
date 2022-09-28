# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import paddle.nn as nn
from paddlenlp.transformers.ernie.modeling import ErniePretrainedModel, ErniePretrainingHeads, ErnieLMPredictionHead


class ErnieForPretraining(ErniePretrainedModel):

    def __init__(self, ernie):
        super(ErnieForPretraining, self).__init__()
        self.ernie = ernie
        self.cls = ErniePretrainingHeads(
            self.ernie.config["hidden_size"],
            self.ernie.config["vocab_size"],
            self.ernie.config["hidden_act"],
            embedding_weights=self.ernie.embeddings.word_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None):
        with paddle.static.amp.fp16_guard():
            outputs = self.ernie(input_ids,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids,
                                 attention_mask=attention_mask)

            sequence_output, pooled_output = outputs[:2]

            max_len = input_ids.shape[1]
            new_masked_positions = masked_positions

            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, new_masked_positions)

            return prediction_scores

    def predict(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None):

        prediction_logits = self.forward(input_ids=input_ids,
                                         token_type_ids=token_type_ids,
                                         position_ids=position_ids,
                                         attention_mask=attention_mask,
                                         masked_positions=masked_positions)

        softmax_fn = paddle.nn.Softmax()
        return softmax_fn(prediction_logits)


class ErnieMLMCriterion(paddle.nn.Layer):

    def __init__(self):
        super(ErnieMLMCriterion, self).__init__()

    def forward(self, prediction_scores, masked_lm_labels, masked_lm_scale=1.0):
        masked_lm_labels = paddle.reshape(masked_lm_labels, shape=[-1, 1])

        with paddle.static.amp.fp16_guard():
            masked_lm_loss = paddle.nn.functional.softmax_with_cross_entropy(
                prediction_scores, masked_lm_labels, ignore_index=-1)
            masked_lm_loss = masked_lm_loss / masked_lm_scale
            return paddle.mean(masked_lm_loss)
