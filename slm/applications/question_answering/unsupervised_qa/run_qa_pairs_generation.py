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
import os

from tqdm import tqdm

from paddlenlp import Taskflow


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--answer_generation_model_path', type=str, default=None, help='the model path to be loaded for answer extraction')
    parser.add_argument('--question_generation_model_path', type=str, default=None, help='the model path to be loaded for question generation')
    parser.add_argument('--filtration_model_path', type=str, default=None, help='the model path to be loaded for filtration')
    parser.add_argument('--source_file_path', type=str, default=None, help='the source file path')
    parser.add_argument('--target_file_path', type=str, default=None, help='the target json file path')
    parser.add_argument('--batch_size', type=int, default=1, help='the batch size when using taskflow')
    parser.add_argument("--do_debug", action='store_true', help="Whether to do debug")
    parser.add_argument('--a_prompt', type=str, default='答案', help='the prompt when using taskflow, separate by ,')
    parser.add_argument('--a_position_prob', type=float, default=0.01, help='confidence threshold for answer extraction')
    parser.add_argument('--a_max_answer_candidates', type=int, default=5, help='the max number of return answer candidate for each input')
    parser.add_argument('--q_num_return_sequences', type=int, default=3, help='the number of return sequences for each input sample, it should be less than num_beams')
    parser.add_argument('--q_max_question_length', type=int, default=50, help='the max decoding length')
    parser.add_argument('--q_decode_strategy', type=str, default='sampling', help='the decode strategy')
    parser.add_argument('--q_num_beams', type=int, default=6, help='the number of beams when using beam search')
    parser.add_argument('--q_num_beam_groups', type=int, default=1, help='the number of beam groups when using diverse beam search')
    parser.add_argument('--q_diversity_rate', type=float, default=0.0, help='the diversity_rate when using diverse beam search')
    parser.add_argument('--q_top_k', type=float, default=5, help='the top_k when using sampling decoding strategy')
    parser.add_argument('--q_top_p', type=float, default=1.0, help='the top_p when using sampling decoding strategy')
    parser.add_argument('--q_temperature', type=float, default=1.0, help='the temperature when using sampling decoding strategy')
    parser.add_argument("--do_filtration", action='store_true', help="Whether to do filtration")
    parser.add_argument('--f_filtration_position_prob', type=float, default=0.1, help='confidence threshold for filtration')
    args = parser.parse_args()
    return args
# yapf: enable


def answer_generation_from_paragraphs(
    paragraphs, batch_size=16, model=None, max_answer_candidates=5, schema=None, wf=None
):
    """Generate answer from given paragraphs."""
    result = []
    buffer = []
    i = 0
    len_paragraphs = len(paragraphs)
    for paragraph_tobe in tqdm(paragraphs):
        buffer.append(paragraph_tobe)
        if len(buffer) == batch_size or (i + 1) == len_paragraphs:
            predicts = model(buffer)
            paragraph_list = buffer
            buffer = []
            for predict_dict, paragraph in zip(predicts, paragraph_list):
                answers = []
                probabilitys = []
                for prompt in schema:
                    if prompt in predict_dict:
                        answer_dicts = predict_dict[prompt]
                        answers += [answer_dict["text"] for answer_dict in answer_dicts]
                        probabilitys += [answer_dict["probability"] for answer_dict in answer_dicts]
                    else:
                        answers += []
                        probabilitys += []
                candidates = sorted(list(set([(a, p) for a, p in zip(answers, probabilitys)])), key=lambda x: -x[1])
                if len(candidates) > max_answer_candidates:
                    candidates = candidates[:max_answer_candidates]
                outdict = {
                    "context": paragraph,
                    "answer_candidates": candidates,
                }
                if wf:
                    wf.write(json.dumps(outdict, ensure_ascii=False) + "\n")
                result.append(outdict)
        i += 1
    return result


def create_fake_question(
    json_file_or_pair_list, out_json=None, num_return_sequences=1, all_sample_num=None, batch_size=8
):
    if out_json:
        wf = open(out_json, "w", encoding="utf-8")
    if isinstance(json_file_or_pair_list, list):
        all_lines = json_file_or_pair_list
    else:
        rf = open(json_file_or_pair_list, "r", encoding="utf-8")
        all_lines = []
        for json_line in rf:
            line_dict = json.loads(json_line)
            all_lines.append(line_dict)
        rf.close()
    num_all_lines = len(all_lines)
    output = []
    context_buffer = []
    answer_buffer = []
    answer_probability_buffer = []
    true_question_buffer = []
    i = 0
    for index, line_dict in enumerate(tqdm(all_lines)):
        if "question" in line_dict:
            q = line_dict["question"]
        else:
            q = ""
        c = line_dict["context"]
        assert "answer_candidates" in line_dict
        answers = line_dict["answer_candidates"]
        if not answers:
            continue
        for j, pair in enumerate(answers):
            a, p = pair
            context_buffer += [c]
            answer_buffer += [a]
            answer_probability_buffer += [p]
            true_question_buffer += [q]
            if (
                (i + 1) % batch_size == 0
                or (all_sample_num and (i + 1) == all_sample_num)
                or ((index + 1) == num_all_lines and j == len(answers) - 1)
            ):
                result_buffer = question_generation(
                    [{"context": context, "answer": answer} for context, answer in zip(context_buffer, answer_buffer)]
                )
                context_buffer_temp, answer_buffer_temp, answer_probability_buffer_temp, true_question_buffer_temp = (
                    [],
                    [],
                    [],
                    [],
                )
                for context, answer, answer_probability, true_question in zip(
                    context_buffer, answer_buffer, answer_probability_buffer, true_question_buffer
                ):
                    context_buffer_temp += [context] * num_return_sequences
                    answer_buffer_temp += [answer] * num_return_sequences
                    answer_probability_buffer_temp += [answer_probability] * num_return_sequences
                    true_question_buffer_temp += [true_question] * num_return_sequences
                result_one_two_buffer = [(one, two) for one, two in zip(result_buffer[0], result_buffer[1])]
                for context, answer, answer_probability, true_question, result in zip(
                    context_buffer_temp,
                    answer_buffer_temp,
                    answer_probability_buffer_temp,
                    true_question_buffer_temp,
                    result_one_two_buffer,
                ):
                    fake_questions_tokens = [result[0]]
                    fake_questions_scores = [result[1]]
                    for fake_questions_token, fake_questions_score in zip(
                        fake_questions_tokens, fake_questions_scores
                    ):
                        out_dict = {
                            "context": context,
                            "synthetic_answer": answer,
                            "synthetic_answer_probability": answer_probability,
                            "synthetic_question": fake_questions_token,
                            "synthetic_question_probability": fake_questions_score,
                            "true_question": true_question,
                        }
                        if out_json:
                            wf.write(json.dumps(out_dict, ensure_ascii=False) + "\n")
                        output.append(out_dict)
                context_buffer = []
                answer_buffer = []
                true_question_buffer = []
            if all_sample_num and (i + 1) >= all_sample_num:
                break
            i += 1
    if out_json:
        wf.close()
    return output


def filtration(paragraphs, batch_size=16, model=None, schema=None, wf=None, wf_debug=None):
    result = []
    buffer = []
    valid_num, invalid_num = 0, 0
    i = 0
    len_paragraphs = len(paragraphs)
    for paragraph_tobe in tqdm(paragraphs):
        buffer.append(paragraph_tobe)
        if len(buffer) == batch_size or (i + 1) == len_paragraphs:
            model_inputs = []
            for d in buffer:
                context = d["context"]
                synthetic_question = d["synthetic_question"]
                prefix = "问题：" + synthetic_question + "上下文："
                content = prefix + context
                model_inputs.append(content)
            predicts = model(model_inputs)
            paragraph_list = buffer
            buffer = []
            for predict_dict, paragraph in zip(predicts, paragraph_list):
                context = paragraph["context"]
                synthetic_question = paragraph["synthetic_question"]
                synthetic_question_probability = paragraph["synthetic_question_probability"]
                synthetic_answer = paragraph["synthetic_answer"]
                synthetic_answer_probability = paragraph["synthetic_answer_probability"]

                answers = []
                probabilitys = []
                for prompt in schema:
                    if prompt in predict_dict:
                        answer_dicts = predict_dict[prompt]
                        answers += [answer_dict["text"] for answer_dict in answer_dicts]
                        probabilitys += [answer_dict["probability"] for answer_dict in answer_dicts]
                    else:
                        answers += []
                        probabilitys += []
                candidates = [
                    an for an, pro in sorted([(a, p) for a, p in zip(answers, probabilitys)], key=lambda x: -x[1])
                ]
                out_dict = {
                    "context": context,
                    "synthetic_answer": synthetic_answer,
                    "synthetic_answer_probability": synthetic_answer_probability,
                    "synthetic_question": synthetic_question,
                    "synthetic_question_probability": synthetic_question_probability,
                }
                if synthetic_answer in candidates:
                    if wf:
                        wf.write(json.dumps(out_dict, ensure_ascii=False) + "\n")
                    result.append(out_dict)
                    valid_num += 1
                else:
                    if wf_debug:
                        wf_debug.write(json.dumps(out_dict, ensure_ascii=False) + "\n")
                    invalid_num += 1
        i += 1
    print("valid synthetic question-answer pairs number:", valid_num)
    print("invalid synthetic question-answer pairs number:", invalid_num)
    return result


if __name__ == "__main__":
    args = parse_args()
    assert args.a_prompt
    schema = args.a_prompt.strip().split(",")
    answer_generator = Taskflow(
        "information_extraction",
        schema=schema,
        task_path=args.answer_generation_model_path,
        batch_size=args.batch_size,
        position_prob=args.a_position_prob,
    )
    assert args.source_file_path
    paragraphs = []
    if args.source_file_path.endswith(".json"):
        with open(args.source_file_path, "r", encoding="utf-8") as rf:
            for json_line in rf:
                line_dict = json.loads(json_line)
                assert "context" in line_dict or "content" in line_dict
                if "context" in line_dict:
                    paragraphs.append(line_dict["context"].strip())
                elif "content" in line_dict:
                    paragraphs.append(line_dict["content"].strip())
    else:
        with open(args.source_file_path, "r", encoding="utf-8") as rf:
            for line in rf:
                paragraphs.append(line.strip())

    synthetic_context_answer_pairs = answer_generation_from_paragraphs(
        paragraphs,
        batch_size=args.batch_size,
        model=answer_generator,
        max_answer_candidates=args.a_max_answer_candidates,
        schema=schema,
        wf=None,
    )
    print("create synthetic answers successfully!")

    question_generation = Taskflow(
        "question_generation",
        task_path=args.question_generation_model_path,
        output_scores=True,
        max_length=args.q_max_question_length,
        is_select_from_num_return_sequences=False,
        num_return_sequences=args.q_num_return_sequences,
        batch_size=args.batch_size,
        decode_strategy=args.q_decode_strategy,
        num_beams=args.q_num_beams,
        num_beam_groups=args.q_num_beam_groups,
        diversity_rate=args.q_diversity_rate,
        top_k=args.q_top_k,
        top_p=args.q_top_p,
        temperature=args.q_temperature,
    )
    synthetic_answer_question_pairs = create_fake_question(
        synthetic_context_answer_pairs,
        None if args.do_filtration else args.target_file_path,
        args.q_num_return_sequences,
        None,
        args.batch_size,
    )
    print("create synthetic question-answer pairs successfully!")

    wf = None
    wf_debug = None
    if args.target_file_path:
        if not os.path.exists(os.path.dirname(args.target_file_path)):
            os.makedirs(os.path.dirname(args.target_file_path))
        wf = open(args.target_file_path, "w", encoding="utf-8")
        if args.do_debug:
            wf_debug = open(args.target_file_path + ".debug.json", "w", encoding="utf-8")
    if args.do_filtration:
        filtration_model = Taskflow(
            "information_extraction",
            schema=["答案"],
            task_path=args.filtration_model_path,
            batch_size=args.batch_size,
            position_prob=args.f_filtration_position_prob,
        )
        filtration(
            synthetic_answer_question_pairs,
            batch_size=16,
            model=filtration_model,
            schema=["答案"],
            wf=wf,
            wf_debug=wf_debug,
        )
        print("filter synthetic question-answer pairs successfully!")
    rf.close()
    wf.close()
