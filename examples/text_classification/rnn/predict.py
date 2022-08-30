# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import argparse

import paddle
import paddle.nn.functional as F
from paddlenlp.data import JiebaTokenizer, Stack, Tuple, Pad, Vocab

from model import BoWModel, BiLSTMAttentionModel, CNNModel, LSTMModel, GRUModel, RNNModel, SelfInteractiveAttention
from utils import preprocess_prediction_data

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--batch_size", type=int, default=1, help="Total examples' number of a batch for training.")
parser.add_argument("--vocab_path", type=str, default="./vocab.json", help="The file path to vocabulary.")
parser.add_argument('--network', choices=['bow', 'lstm', 'bilstm', 'gru', 'bigru', 'rnn', 'birnn', 'bilstm_attn', 'cnn'],
    default="bilstm", help="Select which network to train, defaults to bilstm.")
parser.add_argument("--params_path", type=str, default='./checkpoints/final.pdparams', help="The path of model parameter to be loaded.")
args = parser.parse_args()
# yapf: enable


def predict(model, data, label_map, batch_size=1, pad_token_id=0):
    """
    Predicts the data labels.

    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `se_len`(sequence length).
        label_map(obj:`dict`): The label id (key) to label str (value) map.
        batch_size(obj:`int`, defaults to 1): The number of batch.
        pad_token_id(obj:`int`, optional, defaults to 0): The pad token index.

    Returns:
        results(obj:`dict`): All the predictions labels.
    """

    # Seperates data into some batches.
    batches = [
        data[idx:idx + batch_size] for idx in range(0, len(data), batch_size)
    ]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=pad_token_id),  # input_ids
        Stack(dtype="int64"),  # seq len
    ): [data for data in fn(samples)]

    results = []
    model.eval()
    for batch in batches:
        texts, seq_lens = batchify_fn(batch)
        texts = paddle.to_tensor(texts)
        seq_lens = paddle.to_tensor(seq_lens)
        logits = model(texts, seq_lens)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results


if __name__ == "__main__":
    paddle.set_device(args.device.lower())

    # Loads vocab.
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

    # Loads model parameters.
    state_dict = paddle.load(args.params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % args.params_path)

    # Firstly pre-processing prediction data  and then do predict.
    data = [
        '非常不错，服务很好，位于市中心区，交通方便，不过价格也高！',
        '怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片',
        '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。',
    ]
    tokenizer = JiebaTokenizer(vocab)
    examples = preprocess_prediction_data(data, tokenizer)

    results = predict(model,
                      examples,
                      label_map=label_map,
                      batch_size=args.batch_size,
                      pad_token_id=vocab.token_to_idx.get("[PAD]", 0))
    for idx, text in enumerate(data):
        print('Data: {} \t Label: {}'.format(text, results[idx]))
