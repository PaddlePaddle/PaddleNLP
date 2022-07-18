# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddlenlp.data import JiebaTokenizer, Pad, Vocab

from model import TextCNNModel
from data import preprocess_prediction_data

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--batch_size", type=int, default=1, help="Total examples' number of a batch for training.")
parser.add_argument("--vocab_path", type=str, default="./robot_chat_word_dict.txt", help="The path to vocabulary.")
parser.add_argument("--params_path", type=str, default='./checkpoints/final.pdparams', help="The path of model parameter to be loaded.")
args = parser.parse_args()
# yapf: enable


def predict(model, data, label_map, batch_size=1, pad_token_id=0):
    """
    Predicts the data labels.

    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data (obj:`list`): The processed data whose each element 
            is a `list` object, which contains 
            
            - word_ids(obj:`list[int]`): The list of word ids.
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
    batchify_fn = lambda samples, fn=Pad(axis=0, pad_val=pad_token_id
                                         ): [data for data in fn(samples)]

    results = []
    model.eval()
    for batch in batches:
        texts = paddle.to_tensor(batchify_fn(batch))
        logits = model(texts)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results


if __name__ == "__main__":
    paddle.set_device(args.device)

    # Load vocab.
    vocab = Vocab.load_vocabulary(args.vocab_path,
                                  unk_token='[UNK]',
                                  pad_token='[PAD]')
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

    # Construct the newtork.
    vocab_size = len(vocab)
    num_classes = len(label_map)
    pad_token_id = vocab.to_indices('[PAD]')

    model = TextCNNModel(vocab_size,
                         num_classes,
                         padding_idx=pad_token_id,
                         ngram_filter_sizes=(1, 2, 3))

    # Load model parameters.
    state_dict = paddle.load(args.params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % args.params_path)

    # Firstly pre-processing prediction data  and then do predict.
    data = ['你再骂我我真的不跟你聊了', '你看看我附近有什么好吃的', '我喜欢画画也喜欢唱歌']
    tokenizer = JiebaTokenizer(vocab)
    examples = preprocess_prediction_data(data, tokenizer, pad_token_id)

    results = predict(model,
                      examples,
                      label_map=label_map,
                      batch_size=args.batch_size,
                      pad_token_id=pad_token_id)
    for idx, text in enumerate(data):
        print('Data: {} \t Label: {}'.format(text, results[idx]))
