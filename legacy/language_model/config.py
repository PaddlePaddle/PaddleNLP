#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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


class RNNConfig(object):
    def __init__(self, args):
        self.model_type = args.model_type
        self.rnn_model = args.rnn_model

        self.vocab_size = 10000
        if self.model_type == "test":
            self.num_layers = 1
            self.batch_size = 2
            self.hidden_size = 10
            self.num_steps = 3
            self.init_scale = 0.1
            self.max_grad_norm = 5.0
            self.epoch_start_decay = 1
            self.max_epoch = 1
            self.dropout = 0.0
            self.lr_decay = 0.5
            self.base_learning_rate = 1.0
        elif self.model_type == "small":
            self.num_layers = 2
            self.batch_size = 20
            self.hidden_size = 200
            self.num_steps = 20
            self.init_scale = 0.1
            self.max_grad_norm = 5.0
            self.epoch_start_decay = 4
            self.max_epoch = 13
            self.dropout = 0.0
            self.lr_decay = 0.5
            self.base_learning_rate = 1.0
        elif self.model_type == "medium":
            self.num_layers = 2
            self.batch_size = 20
            self.hidden_size = 650
            self.num_steps = 35
            self.init_scale = 0.05
            self.max_grad_norm = 5.0
            self.epoch_start_decay = 6
            self.max_epoch = 39
            self.dropout = 0.5
            self.lr_decay = 0.8
            self.base_learning_rate = 1.0
        elif self.model_type == "large":
            self.num_layers = 2
            self.batch_size = 20
            self.hidden_size = 1500
            self.num_steps = 35
            self.init_scale = 0.04
            self.max_grad_norm = 10.0
            self.epoch_start_decay = 14
            self.max_epoch = 55
            self.dropout = 0.65
            self.lr_decay = 1.0 / 1.15
            self.base_learning_rate = 1.0
        else:
            raise ValueError('Unsupported model_type.')

        if args.rnn_model not in ('static', 'padding', 'cudnn', 'basic_lstm'):
            raise ValueError('Unsupported rnn_model.')

        if args.batch_size > 0:
            self.batch_size = args.batch_size

        if args.max_epoch > 0:
            self.max_epoch = args.max_epoch
