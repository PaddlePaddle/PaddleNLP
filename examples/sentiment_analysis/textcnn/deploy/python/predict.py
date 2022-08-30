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

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import inference
from paddlenlp.data import JiebaTokenizer, Pad, Vocab

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_file", type=str, required=True,
    default='./static_graph_params.pdmodel', help="The path to model info in static graph.")
parser.add_argument("--params_file", type=str, required=True,
    default='./static_graph_params.pdiparams', help="The path to parameters in static graph.")
parser.add_argument("--vocab_path", type=str, default="./robot_chat_word_dict.txt", help="The path to vocabulary.")
parser.add_argument("--max_seq_length",
    default=128, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=2, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'],
    default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def convert_example(data, tokenizer, pad_token_id=0, max_ngram_filter_size=3):
    """convert_example"""
    input_ids = tokenizer.encode(data)
    seq_len = len(input_ids)
    # Sequence length should larger or equal than the maximum ngram_filter_size in TextCNN model
    if seq_len < max_ngram_filter_size:
        input_ids.extend([pad_token_id] * (max_ngram_filter_size - seq_len))
    input_ids = np.array(input_ids, dtype='int64')
    return input_ids


class Predictor(object):

    def __init__(self, model_file, params_file, device, max_seq_length):
        self.max_seq_length = max_seq_length

        config = paddle.inference.Config(model_file, params_file)
        if device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
        elif device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
        elif device == "xpu":
            # set XPU configs accordingly
            config.enable_xpu(100)
        config.switch_use_feed_fetch_ops(False)
        self.predictor = paddle.inference.create_predictor(config)

        self.input_handles = [
            self.predictor.get_input_handle(name)
            for name in self.predictor.get_input_names()
        ]

        self.output_handle = self.predictor.get_output_handle(
            self.predictor.get_output_names()[0])

    def predict(self, data, tokenizer, label_map, batch_size=1, pad_token_id=0):
        """
        Predicts the data labels.

        Args:
            data (obj:`list(str)`): Data to be predicted.
            tokenizer(obj: paddlenlp.data.JiebaTokenizer): It use jieba to cut the chinese string.
            label_map(obj:`dict`): The label id (key) to label str (value) map.
            batch_size(obj:`int`, defaults to 1): The number of batch.
            pad_token_id(obj:`int`, optional, defaults to 0): The pad token index.

        Returns:
            results(obj:`dict`): All the predictions labels.
        """
        examples = []
        for text in data:
            input_ids = convert_example(text, tokenizer)
            examples.append(input_ids)

        # Seperates data into some batches.
        batches = [
            examples[idx:idx + batch_size]
            for idx in range(0, len(examples), batch_size)
        ]

        batchify_fn = lambda samples, fn=Pad(
            axis=0,
            pad_val=pad_token_id  # input
        ): fn(samples)

        results = []
        for batch in batches:
            input_ids = batchify_fn(batch)
            self.input_handles[0].copy_from_cpu(input_ids)
            self.predictor.run()
            logits = paddle.to_tensor(self.output_handle.copy_to_cpu())
            probs = F.softmax(logits, axis=1)
            idx = paddle.argmax(probs, axis=1).numpy()
            idx = idx.tolist()
            labels = [label_map[i] for i in idx]
            results.extend(labels)
        return results


if __name__ == "__main__":
    # Define predictor to do prediction.
    predictor = Predictor(args.model_file, args.params_file, args.device,
                          args.max_seq_length)

    vocab = Vocab.load_vocabulary(args.vocab_path,
                                  unk_token='[UNK]',
                                  pad_token='[PAD]')
    pad_token_id = vocab.to_indices('[PAD]')
    tokenizer = JiebaTokenizer(vocab)
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

    # Firstly pre-processing prediction data and then do predict.
    data = ['你再骂我我真的不跟你聊了', '你看看我附近有什么好吃的', '我喜欢画画也喜欢唱歌']

    results = predictor.predict(data,
                                tokenizer,
                                label_map,
                                batch_size=args.batch_size,
                                pad_token_id=pad_token_id)
    for idx, text in enumerate(data):
        print('Data: {} \t Label: {}'.format(text, results[idx]))
