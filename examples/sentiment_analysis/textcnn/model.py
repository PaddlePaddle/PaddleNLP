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

import paddle
import paddle.nn as nn

from paddlenlp.seq2vec import CNNEncoder


class TextCNNModel(nn.Layer):
    """
    This class implements the Text Convolution Neural Network model.
    At a high level, the model starts by embedding the tokens and running them through
    a word embedding. Then, we encode these epresentations with a `CNNEncoder`.
    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filter. The number of times a convolution layer will be used
    is `num_tokens - ngram_size + 1`. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max. 
    Lastly, we take the output of the encoder to create a final representation,
    which is passed through some feed-forward layers to output a logits (`output_layer`).

    """

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 num_filter=128,
                 ngram_filter_sizes=(1, 2, 3),
                 fc_hidden_size=96):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size,
                                     emb_dim,
                                     padding_idx=padding_idx)
        self.encoder = CNNEncoder(emb_dim=emb_dim,
                                  num_filter=num_filter,
                                  ngram_filter_sizes=ngram_filter_sizes)
        self.fc = nn.Linear(self.encoder.get_output_dim(), fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, len(ngram_filter_sizes) * num_filter)
        encoder_out = paddle.tanh(self.encoder(embedded_text))
        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(encoder_out))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        return logits
