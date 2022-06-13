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
import sys

import paddle
import paddle.nn.functional as F
from paddlenlp.data import Tuple, Pad
from paddlenlp.transformers import BertTokenizer

sys.path.append('./')

from data import convert_example

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_file", type=str, required=True, default='./static_graph_params.pdmodel', help="The path to model info in static graph.")
parser.add_argument("--params_file", type=str, required=True, default='./static_graph_params.pdiparams', help="The path to parameters in static graph.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--threshold", default=0.5, type=float, help="The threshold for converting probabilities to labels")
parser.add_argument("--batch_size", default=2, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


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

    def predict(self, data, tokenizer, batch_size=1, threshold=0.5):
        """
        Predicts the data labels.

        Args:
            data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
                A Example object contains `text`(word_ids) and `se_len`(sequence length).
            tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
                which contains most of the methods. Users should refer to the superclass for more information regarding methods.
            batch_size(obj:`int`, defaults to 1): The number of batch.
            threshold(obj:`int`, defaults to 0.5): The threshold for converting probabilities to labels.

        Returns:
            results(obj:`dict`): All the predictions labels.
        """
        examples = []
        for text in data:
            example = {"text": text}
            input_ids, segment_ids = convert_example(
                example,
                tokenizer,
                max_seq_length=self.max_seq_length,
                is_test=True)
            examples.append((input_ids, segment_ids))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        ): fn(samples)

        # Seperates data into some batches.
        batches = [
            examples[idx:idx + batch_size]
            for idx in range(0, len(examples), batch_size)
        ]

        results = []
        for batch in batches:
            input_ids, segment_ids = batchify_fn(batch)
            self.input_handles[0].copy_from_cpu(input_ids)
            self.input_handles[1].copy_from_cpu(segment_ids)
            self.predictor.run()
            logits = paddle.to_tensor(self.output_handle.copy_to_cpu())
            probs = F.sigmoid(logits)
            preds = (probs.numpy() > threshold).astype(int)
            results.extend(preds)
        return results


if __name__ == "__main__":
    # Define predictor to do prediction.
    predictor = Predictor(args.model_file, args.params_file, args.device,
                          args.max_seq_length)

    # Load bert tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    data = [
        "Your bullshit is not welcome here.",
        "Thank you for understanding. I think very highly of you and would not revert without discussion.",
    ]
    label_info = [
        'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
    ]

    results = predictor.predict(data,
                                tokenizer,
                                batch_size=args.batch_size,
                                threshold=args.threshold)
    for idx, text in enumerate(data):
        print('Data: \t {}'.format(text))
        for i, k in enumerate(label_info):
            print('{}: \t {}'.format(k, results[idx][i]))
