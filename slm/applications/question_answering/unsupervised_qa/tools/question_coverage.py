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
import multiprocessing
import os
import time

from tqdm import tqdm
from tqdm.contrib import tzip

from paddlenlp.metrics import BLEU
from paddlenlp.transformers import BasicTokenizer


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--true_file_path', type=str, default=None, help='the source json file path')
    parser.add_argument('--generate_file_path', type=str, default=None, help='the target json file path')
    parser.add_argument('--num_return_sequences', type=int, default=3, help='the number of return sequences for each input sample, it should be less than num_beams')
    parser.add_argument('--all_sample_num', type=int, default=None, help='the number of valid sample')
    parser.add_argument('--bleu_n_size', type=int, default=4, help='the bleu n size')
    parser.add_argument('--bleu_threshold', type=float, default=0.3, help='the bleu threshold')
    parser.add_argument("--do_log_file", action="store_true", help="is log analysis file")
    parser.add_argument('--log_dir', type=str, default=None, help='the log dir')
    parser.add_argument("--do_multiprocessing", action="store_true", help="is do multiprocessing")
    parser.add_argument("--do_map_async", action="store_true", help="is use map_async or apply_async when do multiprocessing")
    args = parser.parse_args()
    return args
# yapf: enable


def calc_bleu_n(preds, targets, n_size=4):
    assert len(preds) == len(targets), (
        "The length of pred_responses should be equal to the length of "
        "target_responses. But received {} and {}.".format(len(preds), len(targets))
    )
    bleu = BLEU(n_size=n_size)
    tokenizer = BasicTokenizer()

    for pred, target in zip(preds, targets):
        pred_tokens = tokenizer.tokenize(pred)
        target_token = tokenizer.tokenize(target)

        bleu.add_inst(pred_tokens, [target_token])
    return bleu.score()


def worker_apply_async(true_question, generate_question_group, bleu_n_size, bleu_threshold, i):
    first_positive_pair = None
    for generate_question in generate_question_group:
        bleu_score = calc_bleu_n([generate_question], [true_question], bleu_n_size)
        if bleu_score > bleu_threshold:
            first_positive_pair = (generate_question, true_question, i)
    if first_positive_pair:
        return (True, first_positive_pair)
    else:
        return (False, (generate_question_group[0], true_question))


def worker_map_async(args):
    true_question, generate_question_group, bleu_n_size, bleu_threshold, i = args
    first_positive_pair = None
    for generate_question in generate_question_group:
        bleu_score = calc_bleu_n([generate_question], [true_question], bleu_n_size)
        if bleu_score > bleu_threshold:
            first_positive_pair = (generate_question, true_question, i)
    if first_positive_pair:
        return (True, first_positive_pair)
    else:
        return (False, (generate_question_group[0], true_question))


def coverage_rate(
    true_file_path,
    generate_file_path,
    bleu_n_size,
    bleu_threshold,
    num_return_sequences,
    all_sample_num=None,
    is_log_file=False,
    log_dir=None,
    is_multiprocessing=True,
    is_map_async=True,
):
    true_questions = []
    with open(true_file_path, "r", encoding="utf-8") as rf:
        for i, json_line in enumerate(tqdm(rf.readlines())):
            if i >= all_sample_num:
                break
            line_dict = json.loads(json_line)
            true_questions.append(
                line_dict["question"][0] if isinstance(line_dict["question"], list) else line_dict["question"]
            )

    generate_question_groups = []
    with open(generate_file_path, "r", encoding="utf-8") as rf:
        group = []
        for i, json_line in enumerate(tqdm(rf.readlines())):
            if i >= all_sample_num * num_return_sequences:
                break
            line_dict = json.loads(json_line)
            group.append(
                line_dict["question"][0] if isinstance(line_dict["question"], list) else line_dict["question"]
            )
            if (i + 1) % num_return_sequences == 0:
                generate_question_groups.append(group)
                group = []
    print("true_questions", len(true_questions))
    print("generate_question_groups", len(generate_question_groups))
    positive = []
    negative = []
    if is_multiprocessing:
        pool = multiprocessing.Pool(processes=30)
        pool_results = []
        if is_map_async:
            map_async_inputs = []
    i = 0
    bleu_cal_time_start = time.time()
    generate_question_groups = [
        [
            generate_question if generate_question.strip() != "" else "none"
            for generate_question in generate_question_group
        ]
        for generate_question_group in generate_question_groups
    ]
    for true_question, generate_question_group in tzip(true_questions, generate_question_groups):
        if is_multiprocessing:
            if is_map_async:
                map_async_inputs.append((true_question, generate_question_group, bleu_n_size, bleu_threshold, i))
            else:
                pool_results.append(
                    pool.apply_async(
                        worker_apply_async,
                        args=(true_question, generate_question_group, bleu_n_size, bleu_threshold, i),
                    )
                )

        else:
            first_positive_pair = None
            best_pair, best_score = None, 0
            for generate_question in generate_question_group:
                try:
                    bleu_score = calc_bleu_n([generate_question], [true_question], bleu_n_size)
                except BaseException:
                    print("generate_question", generate_question)
                    print("true_question", true_question)
                if bleu_score > best_score:
                    best_pair = (generate_question, true_question)
                if bleu_score > bleu_threshold:
                    first_positive_pair = (generate_question, true_question)
            if first_positive_pair:
                positive.append((best_pair[0], best_pair[1], best_score))
            else:
                negative.append((best_pair[0], best_pair[1], best_score))
        i += 1
    if is_multiprocessing:
        if is_map_async:
            pool_results = pool.map_async(worker_map_async, map_async_inputs)
            pool.close()
            pool.join()
            for result in pool_results.get():
                is_positive, pair = result
                if is_positive:
                    positive.append(pair)
                else:
                    negative.append(pair)
        else:
            pool.close()
            pool.join()
            for result in pool_results:
                is_positive, pair = result.get()
                if is_positive:
                    positive.append(pair)
                else:
                    negative.append(pair)

    bleu_cal_time_end = time.time()
    print("bleu_cal_time_spend:", bleu_cal_time_end - bleu_cal_time_start)
    if is_log_file and log_dir:
        with open(os.path.join(log_dir, "positive_pair.txt"), "w", encoding="utf-8") as wf:
            for pair in positive:
                wf.write(
                    pair[0] + "\t" + pair[1] + "\n"
                    if len(pair) == 2
                    else pair[0] + "\t" + pair[1] + str(pair[2]) + "\n"
                )
        with open(os.path.join(log_dir, "negative_pair.txt"), "w", encoding="utf-8") as wf:
            for pair in negative:
                wf.write(
                    pair[0] + "\t" + pair[1] + "\n"
                    if len(pair) == 2
                    else pair[0] + "\t" + pair[1] + str(pair[2]) + "\n"
                )
    assert len(positive) + len(negative) == all_sample_num, (
        "the number of positive pairs "
        + str(len(positive))
        + " plus the number of negative pairs "
        + str(len(negative))
        + " should be equal to all_sample_num"
        + str(all_sample_num)
    )
    return len(positive) / (len(positive) + len(negative))


if __name__ == "__main__":
    args = parse_args()
    rate = coverage_rate(
        true_file_path=args.true_file_path,
        generate_file_path=args.generate_file_path,
        bleu_n_size=args.bleu_n_size,
        bleu_threshold=args.bleu_threshold,
        num_return_sequences=args.num_return_sequences,
        all_sample_num=args.all_sample_num,
        is_log_file=args.do_log_file,
        log_dir=args.log_dir,
        is_multiprocessing=args.do_multiprocessing,
        is_map_async=args.do_map_async,
    )
    print("coverage rate is", rate)
