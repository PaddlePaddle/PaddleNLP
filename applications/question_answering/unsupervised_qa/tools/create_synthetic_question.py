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

from tqdm import tqdm

from paddlenlp import Taskflow


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model_path', type=str, default=None, help='the model path to be loaded for question_generation taskflow')
    parser.add_argument('--max_length', type=int, default=50, help='the max decoding length')
    parser.add_argument('--num_return_sequences', type=int, default=3, help='the number of return sequences for each input sample, it should be less than num_beams')
    parser.add_argument('--source_file_path', type=str, default=None, help='the souce json file path')
    parser.add_argument('--target_file_path', type=str, default=None, help='the target json file path')
    parser.add_argument('--all_sample_num', type=int, default=None, help='the test sample number when convert_json_to_data')
    parser.add_argument('--batch_size', type=int, default=1, help='the batch size when using taskflow')
    parser.add_argument('--decode_strategy', type=str, default=None, help='the decode strategy')
    parser.add_argument('--num_beams', type=int, default=6, help='the number of beams when using beam search')
    parser.add_argument('--num_beam_groups', type=int, default=1, help='the number of beam groups when using diverse beam search')
    parser.add_argument('--diversity_rate', type=float, default=0.0, help='the diversity_rate when using diverse beam search')
    parser.add_argument('--top_k', type=float, default=0, help='the top_k when using sampling decoding strategy')
    parser.add_argument('--top_p', type=float, default=1.0, help='the top_p when using sampling decoding strategy')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when using sampling decoding strategy')
    args = parser.parse_args()
    return args
# yapf: enable


def create_fake_question(json_file, out_json, num_return_sequences, all_sample_num=None, batch_size=8):
    with open(json_file, "r", encoding="utf-8") as rf, open(out_json, "w", encoding="utf-8") as wf:
        all_lines = rf.readlines()
        num_all_lines = len(all_lines)
        context_buffer = []
        answer_buffer = []
        true_question_buffer = []
        for i, json_line in enumerate(tqdm(all_lines)):
            line_dict = json.loads(json_line)
            q = line_dict["question"]
            a = line_dict["answer"]
            c = line_dict["context"]

            context_buffer += [c]
            answer_buffer += [a]
            true_question_buffer += [q]
            if (
                (i + 1) % batch_size == 0
                or (all_sample_num and (i + 1) == all_sample_num or (i + 1))
                or (i + 1) == num_all_lines
            ):
                result_buffer = question_generation(
                    [{"context": context, "answer": answer} for context, answer in zip(context_buffer, answer_buffer)]
                )
                context_buffer_temp, answer_buffer_temp, true_question_buffer_temp = [], [], []
                for context, answer, true_question in zip(context_buffer, answer_buffer, true_question_buffer):
                    context_buffer_temp += [context] * num_return_sequences
                    answer_buffer_temp += [answer] * num_return_sequences
                    true_question_buffer_temp += [true_question] * num_return_sequences
                result_one_two_buffer = [(one, two) for one, two in zip(result_buffer[0], result_buffer[1])]
                for context, answer, true_question, result in zip(
                    context_buffer_temp, answer_buffer_temp, true_question_buffer_temp, result_one_two_buffer
                ):
                    fake_quesitons_tokens = [result[0]]
                    fake_quesitons_scores = [result[1]]
                    for fake_quesitons_token, fake_quesitons_score in zip(
                        fake_quesitons_tokens, fake_quesitons_scores
                    ):
                        out_dict = {
                            "context": context,
                            "answer": answer,
                            "question": fake_quesitons_token,
                            "true_question": true_question,
                            "score": fake_quesitons_score,
                        }
                        wf.write(json.dumps(out_dict, ensure_ascii=False) + "\n")
                context_buffer = []
                answer_buffer = []
                true_question_buffer = []

            if all_sample_num and (i + 1) >= all_sample_num:
                break


if __name__ == "__main__":
    args = parse_args()
    question_generation = Taskflow(
        "question_generation",
        task_path=args.model_path,
        output_scores=True,
        max_length=args.max_length,
        is_select_from_num_return_sequences=False,
        num_return_sequences=args.num_return_sequences,
        batch_size=args.batch_size,
        decode_strategy=args.decode_strategy,
        num_beams=args.num_beams,
        num_beam_groups=args.num_beam_groups,
        diversity_rate=args.diversity_rate,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
    )
    create_fake_question(
        args.source_file_path, args.target_file_path, args.num_return_sequences, args.all_sample_num, args.batch_size
    )
