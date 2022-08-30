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

import argparse

import paddle
from paddlenlp.data import Vocab

from model import BoWModel, BiLSTMAttentionModel, CNNModel, LSTMModel, GRUModel, RNNModel, SelfInteractiveAttention

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--vocab_path", type=str, default="./vocab.json", help="The file path to vocabulary.")
parser.add_argument('--network', choices=['bow', 'lstm', 'bilstm', 'gru', 'bigru', 'rnn', 'birnn', 'bilstm_attn', 'cnn'],
    default="bilstm", help="Select which network to train, defaults to bilstm.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--params_path", type=str, default='./checkpoints/final.pdparams', help="The path of model parameter to be loaded.")
parser.add_argument("--output_path", type=str, default='./static_graph_params', help="The path of model parameter in static graph to be saved.")
args = parser.parse_args()
# yapf: enable


def main():
    # Load vocab.
    vocab = Vocab.from_json(args.vocab_path)
    label_map = {0: 'negative', 1: 'positive'}

    # Constructs the newtork.
    network = args.network.lower()
    vocab_size = len(vocab)
    num_classes = len(label_map)
    pad_token_id = vocab.to_indices('[PAD]')
    if network == 'bow':
        model = BoWModel(vocab_size, num_classes, padding_idx=pad_token_id)
    elif network == 'bigru':
        model = GRUModel(vocab_size,
                         num_classes,
                         direction='bidirect',
                         padding_idx=pad_token_id)
    elif network == 'bilstm':
        model = LSTMModel(vocab_size,
                          num_classes,
                          direction='bidirect',
                          padding_idx=pad_token_id)
    elif network == 'bilstm_attn':
        lstm_hidden_size = 196
        attention = SelfInteractiveAttention(hidden_size=2 * lstm_hidden_size)
        model = BiLSTMAttentionModel(attention_layer=attention,
                                     vocab_size=vocab_size,
                                     lstm_hidden_size=lstm_hidden_size,
                                     num_classes=num_classes,
                                     padding_idx=pad_token_id)
    elif network == 'birnn':
        model = RNNModel(vocab_size,
                         num_classes,
                         direction='bidirect',
                         padding_idx=pad_token_id)
    elif network == 'cnn':
        model = CNNModel(vocab_size, num_classes, padding_idx=pad_token_id)
    elif network == 'gru':
        model = GRUModel(vocab_size,
                         num_classes,
                         direction='forward',
                         padding_idx=pad_token_id,
                         pooling_type='max')
    elif network == 'lstm':
        model = LSTMModel(vocab_size,
                          num_classes,
                          direction='forward',
                          padding_idx=pad_token_id,
                          pooling_type='max')
    elif network == 'rnn':
        model = RNNModel(vocab_size,
                         num_classes,
                         direction='forward',
                         padding_idx=pad_token_id,
                         pooling_type='max')
    else:
        raise ValueError(
            "Unknown network: %s, it must be one of bow, lstm, bilstm, cnn, gru, bigru, rnn, birnn and bilstm_attn."
            % network)

    # Load model parameters.
    state_dict = paddle.load(args.params_path)
    model.set_dict(state_dict)
    model.eval()

    inputs = [paddle.static.InputSpec(shape=[None, None], dtype="int64")]
    # Convert to static graph with specific input description
    if args.network in [
            "lstm", "bilstm", "gru", "bigru", "rnn", "birnn", "bilstm_attn"
    ]:
        inputs.append(paddle.static.InputSpec(shape=[None],
                                              dtype="int64"))  # seq_len

    model = paddle.jit.to_static(model, input_spec=inputs)
    # Save in static graph model.
    paddle.jit.save(model, args.output_path)


if __name__ == "__main__":
    main()
