# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import json
import time

import paddle
import paddle.distributed as dist
from gen_utils import create_data_loader, print_args, select_sum, set_seed

from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import UNIMOLMHeadModel, UNIMOTokenizer


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset_name', type=str, default='dureader_qg', help='The name of the dataset to load.')
    parser.add_argument('--model_name_or_path', type=str, default='unimo-text-1.0', help='The path or shortcut name of the pre-trained model.')
    parser.add_argument("--predict_file", type=str, required=False, default=None, help="Predict data path.")
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='The directory where the checkpoints will be saved.')
    parser.add_argument('--logging_steps', type=int, default=100, help='Log every X updates steps.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for initialization.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU/CPU for training.')
    parser.add_argument('--max_seq_len', type=int, default=512, help='The maximum sequence length of training.')
    parser.add_argument('--max_target_len', type=int, default=30, help='The maximum target sequence length of training.')
    parser.add_argument('--max_title_len', type=int, default=30, help='The maximum title sequence length of training.')
    parser.add_argument('--max_dec_len', type=int, default=20, help='The maximum sequence length of decoding.')
    parser.add_argument('--min_dec_len', type=int, default=3, help='The minimal sequence length of decoding.')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='The numbers of returned sequences for one input in generation.')
    parser.add_argument('--decode_strategy', type=str, default='beam_search', help='The decode strategy in generation.')
    parser.add_argument('--top_k', type=int, default=0, help='The number of highest probability vocabulary tokens to keep for top-k sampling.')
    parser.add_argument('--temperature', type=float, default=1.0, help='The value used to module the next token probabilities.')
    parser.add_argument('--top_p', type=float, default=1.0, help='The cumulative probability for top-p sampling.')
    parser.add_argument('--num_beams', type=int, default=6, help='The number of beams for beam search.')
    parser.add_argument('--length_penalty', type=float, default=1.2, help='The exponential penalty to the sequence length for beam search.')
    parser.add_argument('--device', type=str, default='gpu', help='The device to select for training the model.')
    parser.add_argument('--output_path', type=str, default='./predict.txt', help='The file path where the infer result will be saved.')
    parser.add_argument("--do_predict", action='store_true', help="Whether to eval and predict.")
    parser.add_argument("--template", type=int, default=1, help="The template used during training, select from [0, 1, 2, 3, 4].")

    args = parser.parse_args()
    return args
# yapf: enable


def read_file(file):
    with open(file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            line = json.loads(line)
            yield line


def run(args):
    paddle.set_device(args.device)
    world_size = dist.get_world_size()

    if world_size > 1:
        dist.init_parallel_env()
    set_seed(args.seed)

    model = UNIMOLMHeadModel.from_pretrained(args.model_name_or_path)
    tokenizer = UNIMOTokenizer.from_pretrained(args.model_name_or_path)

    if world_size > 1:
        model = paddle.DataParallel(model)

    if args.predict_file:
        dev_ds = load_dataset(read_file, file=args.predict_file, lazy=False)
    else:
        dev_ds = load_dataset(args.dataset_name, splits="dev", data_files=args.predict_file)

    dev_ds, dev_data_loader = create_data_loader(dev_ds, tokenizer, args, "test")

    if args.do_predict:
        model_eval = model._layers if isinstance(model, paddle.DataParallel) else model
        prediction(model_eval, dev_data_loader, args, tokenizer)


@paddle.no_grad()
def prediction(model, data_loader, args, tokenizer):
    print("\nPred begin...")
    model.eval()
    pred_ref = []
    time_begin = time.time()
    total_time = 0.0
    start_time = time.time()
    for step, inputs in enumerate(data_loader, 1):
        input_ids, token_type_ids, position_ids, attention_mask = inputs
        ids, scores = model.generate(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            max_length=args.max_dec_len,
            min_length=args.min_dec_len,
            decode_strategy=args.decode_strategy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            num_return_sequences=args.num_return_sequences,
            bos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.mask_token_id,
        )

        total_time += time.time() - start_time
        if step % args.logging_steps == 0:
            print("step %d - %.3fs/step" % (step, total_time / args.logging_steps))
            total_time = 0.0

        results = select_sum(ids, scores, tokenizer, args.max_dec_len, args.num_return_sequences)
        pred_ref.extend(results)
        start_time = time.time()
    print("Generation cost time:", time.time() - time_begin)

    with open(args.output_path, "w", encoding="utf-8") as fout:
        for ref in pred_ref:
            fout.write(ref + "\n")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    run(args)
