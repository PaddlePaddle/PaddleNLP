#Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from ernie_predictor import *
from multiprocessing import cpu_count


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default='tnews',
        type=str,
        help="The name of the task to perform predict, selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default="ernie-3.0-medium-zh",
        type=str,
        help="The directory or name of model.", )
    parser.add_argument(
        "--model_path",
        default='tnews_quant_models/mse4/int8',
        type=str,
        required=True,
        help="The path prefix of inference model to be used.", )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for predict.", )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--perf_warmup_steps",
        default=20,
        type=int,
        help="Warmup steps for performance test.", )
    parser.add_argument(
        "--perf",
        action='store_true',
        help="Whether to test performance.", )
    parser.add_argument(
        "--enable_quantize",
        action='store_true',
        help="Whether to enable quantization for acceleration.", )
    parser.add_argument(
        "--num_threads",
        default=cpu_count(),
        type=int,
        help="num_threads for cpu.", )
    args = parser.parse_args()
    return args


def main():
    paddle.seed(42)
    args = parse_args()

    args.task_name = args.task_name.lower()
    args.device = 'cpu'
    predictor = ErniePredictor(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    dev_ds = load_dataset('clue', args.task_name, splits='dev')

    trans_func = partial(
        convert_example,
        label_list=dev_ds.label_list,
        tokenizer=tokenizer,
        is_test=False)
    dev_ds = dev_ds.map(trans_func, lazy=False)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        Stack(dtype="int64" if dev_ds.label_list else "float32")  # label
    ): fn(samples)
    outputs = predictor.predict(dev_ds, tokenizer, batchify_fn, args)


if __name__ == "__main__":
    main()
