#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.nn as nn
from paddlenlp.seq2vec import TCNEncoder


class TCNNetwork(nn.Layer):
    """
    Temporal Convolutional Networks is a simple convolutional architecture. It outperforms canonical recurrent networks
    such as LSTMs in many tasks. See https://arxiv.org/pdf/1803.01271.pdf for more details.

    Args:
        input_size (obj:`int`, required): The number of expected features in the input (the last dimension).
        next_k (obj:`int`, optional): The number of the forecasting time step. Defaults to 1.
        num_channels (obj:`list` or obj:`tuple`, optional): The number of channels in different layer. Defaults to [64,128,256].
        kernel_size (obj:`int`, optional): The kernel size. Defaults to 2.
        dropout (obj:`float`, optional): The dropout probability. Defaults to 0.2.
    """

    def __init__(self,
                 input_size,
                 next_k=1,
                 num_channels=[64, 128, 256],
                 kernel_size=2,
                 dropout=0.2):
        super(TCNNetwork, self).__init__()

        self.last_num_channel = num_channels[-1]

        self.tcn = TCNEncoder(input_size=input_size,
                              num_channels=num_channels,
                              kernel_size=kernel_size,
                              dropout=dropout)

        self.linear = nn.Linear(in_features=self.last_num_channel,
                                out_features=next_k)

    def forward(self, x):
        tcn_out = self.tcn(x)
        y_pred = self.linear(tcn_out)
        return y_pred
