# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from functools import partial
import argparse
import sys

sys.path.append("../../..")
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Pad, Stack, Tuple, Vocab
from paddlenlp.datasets import load_dataset

from model import SimNet
from utils import preprocess_data, CharTokenizer

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--batch_size", type=int, default=1, help="Total examples' number of a batch for training.")
parser.add_argument("--vocab_path", type=str, default="./vocab.char", help="The path to vocabulary.")
parser.add_argument('--network', type=str, default="lstm", help="Which network you would like to choose bow, cnn, lstm or gru ?")
parser.add_argument("--params_path", type=str, default='./checkpoints/final.pdparams', help="The path of model parameter to be loaded.")
parser.add_argument("--language", type=str, required=True, help="Language that this model based on")
args = parser.parse_args()
# yapf: enable


def interpret(model, data, label_map, batch_size=1, pad_token_id=0, vocab=None):
    """
    Predicts the data labels.

    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `seq_len`(sequence length).
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
        Pad(axis=0, pad_val=pad_token_id),  # query_ids
        Pad(axis=0, pad_val=pad_token_id),  # title_ids
        Stack(dtype="int64"),  # query_seq_lens
        Stack(dtype="int64"),  # title_seq_lens
    ): [data for data in fn(samples)]

    model.eval()
    results = []
    for batch in batches:
        query_ids, title_ids, query_seq_lens, title_seq_lens = batchify_fn(
            batch)
        query_ids = paddle.to_tensor(query_ids)
        title_ids = paddle.to_tensor(title_ids)
        query_seq_lens = paddle.to_tensor(query_seq_lens)
        title_seq_lens = paddle.to_tensor(title_seq_lens)

        logits, attention, _ = model.forward_interpret(query_ids, title_ids,
                                                       query_seq_lens,
                                                       title_seq_lens)
        query_att = attention[0]
        title_att = attention[1]

        model.clear_gradients()
        for query_id, title_id in zip(query_ids.numpy().tolist(),
                                      title_ids.numpy().tolist()):
            query = [vocab._idx_to_token[idx] for idx in query_id]
            title = [vocab._idx_to_token[idx] for idx in title_id]
        results.append([query_att, query, title_att, title])

        print('query_att: %s' % query_att.shape)
        print('title_att: %s' % title_att.shape)

    return results


if __name__ == "__main__":
    paddle.set_device(args.device + ':2')
    # Loads vocab.
    vocab = Vocab.load_vocabulary(args.vocab_path,
                                  unk_token='[UNK]',
                                  pad_token='[PAD]')
    tokenizer = CharTokenizer(vocab, args.language)
    label_map = {0: 'dissimilar', 1: 'similar'}

    # Constructs the newtork.
    model = SimNet(network=args.network,
                   vocab_size=len(vocab),
                   num_classes=len(label_map))

    # Loads model parameters.
    state_dict = paddle.load(args.params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % args.params_path)

    # Firstly pre-processing prediction data  and then do predict.
    dev_ds, test_ds = load_dataset("lcqmc", splits=["dev", "test"])

    dev_examples = preprocess_data(dev_ds.data, tokenizer, args.language)
    test_examples = preprocess_data(test_ds.data, tokenizer, args.language)
    results = interpret(model,
                        dev_examples,
                        label_map=label_map,
                        batch_size=args.batch_size,
                        pad_token_id=vocab.token_to_idx.get('[PAD]', 0),
                        vocab=vocab)
