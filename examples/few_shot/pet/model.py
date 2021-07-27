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
from paddlenlp.transformers.albert.modeling import AlbertPretrainedModel, AlbertMLMHead, AlbertForMaskedLM

# Logan: No AlbertPretrainingHeads


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
            outputs = self.ernie(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask)

            sequence_output, pooled_output = outputs[:2]
            # print("Ernie sequence_outputs shape", sequence_output.shape)
            # Ernie sequence_outputs shape [32, 40, 768]
            max_len = input_ids.shape[1]
            new_masked_positions = masked_positions
            # # masked_positions: [bs, label_length]
            # for bs_index, mask_pos in enumerate(masked_positions.numpy()):
            #     for pos in mask_pos:
            #         new_masked_positions.append(bs_index * max_len + pos)

            # # new_masked_positions: [bs * label_length, 1]
            # new_masked_positions = np.array(new_masked_positions).astype(
            #     'int32')
            # new_masked_positions = paddle.to_tensor(new_masked_positions, stop_gradient=False)

            # print('Ernie sequence_output', sequence_output.shape, sequence_output)
            # print('Ernie new_masked_positions', new_masked_positions.shape, new_masked_positions)
            # Ernie sequence_output and new_masked_positions shapes [32, 40, 768] [64]

            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, new_masked_positions)
            # print("Done LMHead, no problem; prediction_scores shape", prediction_scores.shape)
            # prediction_scores shape [64, 18000]

            return prediction_scores

    def predict(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None):

        prediction_logits = self.forward(
            input_ids=input_ids,
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


class AlbertLMPredictionHead(nn.Layer):
    r"""
    Bert Model with a `language modeling` head on top.
    """

    def __init__(self,
                 embedding_size,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):

        super(AlbertLMPredictionHead, self).__init__()
        self.transform = nn.Linear(hidden_size, embedding_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.decoder_weight = self.create_parameter(
            shape=[hidden_size, vocab_size],
            dtype=self.transform.weight.dtype,
            is_bias=False) if embedding_weights is None else embedding_weights
        self.decoder_bias = self.create_parameter(
            shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(hidden_states,
                                           [-1, hidden_states.shape[-1]])

            hidden_states = paddle.tensor.gather(hidden_states,
                                                 masked_positions)

        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)

        hidden_states = self.activation(hidden_states)

        hidden_states = self.layer_norm(hidden_states)

        hidden_states = paddle.tensor.matmul(
            hidden_states, self.decoder_weight, transpose_y=True
        ) + self.decoder_bias  # this is different for Albert

        return hidden_states


class AlbertForPretraining(AlbertForMaskedLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None):
        with paddle.static.amp.fp16_guard():
            prediction_scores = super().forward(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask, )

            masked_number = masked_positions.shape[1]
            vocab_size = prediction_scores.shape[-1]

            if masked_number not in [1, 2, 3, 4]:
                raise ValueError("Illegal masked_number:{}".format(
                    masked_number))

            # Masked positions: [bs, label_length]
            if masked_number in [1, 2, 3, 4]:
                # CHID task: masked positions of each example is different
                max_len = input_ids.shape[1]
                new_masked_positions = []
                # masked_positions: [bs, label_length]
                for bs_index, mask_pos in enumerate(masked_positions.numpy()):
                    for pos in mask_pos:
                        new_masked_positions.append(bs_index * max_len + pos)

                # new_masked_positions: [bs * label_length, 1]
                new_masked_positions = np.array(new_masked_positions).astype(
                    'int32')
                new_masked_positions = paddle.to_tensor(
                    new_masked_positions, stop_gradient=False)

                prediction_scores = paddle.reshape(prediction_scores,
                                                   [-1, vocab_size])
                filtered_scores = paddle.gather(
                    prediction_scores, new_masked_positions, axis=0)
            else:
                pass
            return filtered_scores

    def predict(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None):

        prediction_logits = self.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            masked_positions=masked_positions)

        softmax_fn = paddle.nn.Softmax()
        return softmax_fn(prediction_logits)


class AlbertMLMCriterion(paddle.nn.Layer):
    def __init__(self):
        super(AlbertMLMCriterion, self).__init__()

    def forward(self, prediction_scores, masked_lm_labels, masked_lm_scale=1.0):

        masked_lm_labels = paddle.reshape(masked_lm_labels, shape=[-1, 1])

        with paddle.static.amp.fp16_guard():
            masked_lm_loss = paddle.nn.functional.softmax_with_cross_entropy(
                prediction_scores, masked_lm_labels, ignore_index=-1)
            masked_lm_loss = masked_lm_loss / masked_lm_scale
            return paddle.mean(masked_lm_loss)