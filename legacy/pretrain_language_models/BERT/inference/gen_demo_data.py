#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
sys.path.append("..")
from reader import cls


def main():
    args = parse_args()
    task_name = args.task_name.lower()
    processors = {
        'xnli': cls.XnliProcessor,
        'cola': cls.ColaProcessor,
        'mrpc': cls.MrpcProcessor,
        'mnli': cls.MnliProcessor,
    }

    processor = processors[task_name](data_dir=args.data_path,
                                      vocab_path=args.vocab_path,
                                      max_seq_len=args.max_seq_len,
                                      do_lower_case=args.do_lower_case,
                                      in_tokens=args.in_tokens,
                                      random_seed=args.random_seed)
    example = processor.get_test_examples(args.data_path)[0]
    gen = processor.data_generator(
        args.batch_size, phase='test', epoch=1, shuffle=False)()

    for i, data in enumerate(gen):
        data = data[:4]
        sample = []
        for field in data:
            shape_str = ' '.join(map(str, field.shape))
            data_str = ' '.join(map(str, field.reshape(-1).tolist()))
            sample.append(shape_str + ':' + data_str)
        print(';'.join(sample))


def str2bool(v):
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return v.lower() in ("true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(prog="bert data prepare")
    parser.add_argument(
        "--task_name",
        type=str,
        default='xnli',
        choices=["xnli", "mnli", "cola", "mrpc"],
        help="task name, used to specify data preprocessor")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="batch size, see also --in_tokens")
    parser.add_argument(
        "--in_tokens",
        action='store_true',
        help="if set, batch_size means token number in a batch, otherwise "
        "it means example number in a batch")
    parser.add_argument(
        '--do_lower_case',
        type=str2bool,
        default=True,
        choices=[True, False],
        help="Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.")
    parser.add_argument("--vocab_path", type=str, help="path of vocabulary")
    parser.add_argument("--data_path", type=str, help="path of data to process")
    parser.add_argument(
        "--max_seq_len", type=int, default=128, help="max sequence length")
    parser.add_argument(
        "--random_seed", type=int, default=0, help="random seed")
    return parser.parse_args()


if __name__ == "__main__":
    main()
