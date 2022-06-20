#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os
import traceback
import logging
import json
import attr

import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F

from text2sql.models import encoder_v2
from text2sql.models.sql_decoder import decoder as decoder_v2


class EncDecModel(nn.Layer):
    """Dygraph version of BoomUp Model"""

    def __init__(self, config, label_encoder, model_version='v2'):
        super(EncDecModel, self).__init__()

        self._config = config
        self._model_version = model_version

        assert model_version in ('v2', ), "model_version only support v2"
        self.encoder = encoder_v2.Text2SQLEncoderV2(config)
        self.decoder = decoder_v2.Text2SQLDecoder(label_encoder,
                                                  dropout=0.2,
                                                  desc_attn='mha',
                                                  use_align_mat=True,
                                                  use_align_loss=True)

    def forward(self, inputs, labels=None, db=None, is_train=True):
        if is_train:
            assert labels is not None, "labels should not be None while training"
            return self._train(inputs, labels)
        else:
            assert db is not None, "db should not be None while inferencing"
            return self._inference(inputs, db)

    def _train(self, inputs, labels):
        enc_results = self.encoder(inputs)
        lst_loss = []
        for orig_inputs, label_info, enc_result in zip(inputs['orig_inputs'],
                                                       labels, enc_results):
            loss = self.decoder.compute_loss(orig_inputs, label_info,
                                             enc_result)
            lst_loss.append(loss)

        return paddle.mean(paddle.stack(lst_loss, axis=0), axis=0)

    def _inference(self, inputs, db):
        enc_state = self.encoder(inputs)
        if self._model_version == 'v1':
            return self.decoder.inference(enc_state[0], db)
        elif self._model_version == 'v2':
            return self.decoder.inference(enc_state[0], db,
                                          inputs['orig_inputs'][0].values)


if __name__ == "__main__":
    """run some simple test cases"""
    pass
