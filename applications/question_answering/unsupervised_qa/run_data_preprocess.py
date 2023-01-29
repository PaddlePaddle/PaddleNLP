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


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source_file_path', type=str, default=None, help='the source json file path')
    parser.add_argument('--target_dir', type=str, default='data', help='the target file path')
    parser.add_argument('--do_answer_prompt', action="store_true", help="is use answer prompt")
    parser.add_argument('--do_len_prompt', action="store_true", help="is use length prompt")
    parser.add_argument('--do_domain_prompt', action="store_true", help="is use domain prompt")
    parser.add_argument('--domain', type=str, default=None, help='the domain of the dataset when using domain prompt')
    args = parser.parse_args()
    return args
# yapf: enable


def convert_from_json_to_answer_extraction_format(
    json_file, output_path, domain=None, do_answer_prompt=True, do_len_prompt=False, do_domain_prompt=False
):
    with open(json_file, "r", encoding="utf-8") as rf, open(output_path, "w", encoding="utf-8") as wf:
        for line in rf:
            json_line = json.loads(line)
            context = json_line["context"]

            answer = json_line["answer"]
            # Cut the abnormally long sample
            if len(answer) > 300:
                answer = answer[:300]

            begin_id = context.find(answer)
            assert begin_id != -1, "'" + answer + "' is not found in " + context
            end_id = begin_id + len(answer)
            result = {"text": answer, "start": begin_id, "end": end_id}
            if do_answer_prompt:
                outdict = {
                    "content": context,
                    "result_list": [result],
                    "prompt": "答案",
                }
                wf.write(json.dumps(outdict, ensure_ascii=False) + "\n")
            if do_len_prompt:
                if len(answer) < 10:
                    len_prompat = "短答案"
                elif len(answer) < 20:
                    len_prompat = "中短答案"
                elif len(answer) < 30:
                    len_prompat = "中长答案"
                else:
                    len_prompat = "长答案"

                len_outdict = {
                    "content": context,
                    "result_list": [result],
                    "prompt": len_prompat,
                }
                wf.write(json.dumps(len_outdict, ensure_ascii=False) + "\n")
            if do_domain_prompt and domain:
                domain_outdict = {
                    "content": context,
                    "result_list": [result],
                    "prompt": domain,
                }
                wf.write(json.dumps(domain_outdict, ensure_ascii=False) + "\n")


def convert_from_json_to_question_generation_format(json_file, output_path, tokenizer=None):
    with open(json_file, "r", encoding="utf-8") as rf, open(output_path, "w", encoding="utf-8") as wf:
        for line in rf:
            json_line = json.loads(line)
            context = json_line["context"]

            answer = json_line["answer"]
            # Cut the abnormally long sample
            if len(answer) > 300:
                answer = answer[:300]
            question = json_line["question"]

            outdict = {
                "question": question,
                "answer": answer,
                "context": context,
            }
            wf.write(json.dumps(outdict, ensure_ascii=False) + "\n")


def convert_from_json_to_filtration_format(json_file, output_path, tokenizer=None):
    with open(json_file, "r", encoding="utf-8") as rf, open(output_path, "w", encoding="utf-8") as wf:
        for line in rf:
            json_line = json.loads(line)
            context = json_line["context"]

            answer = json_line["answer"]
            # Cut the abnormally long sample
            if len(answer) > 300:
                answer = answer[:300]
            question = json_line["question"]

            prefix = "问题：" + question + "上下文："
            content = prefix + context

            begin_id = context.find(answer)
            assert begin_id != -1, "'" + answer + "' is not found in " + context
            end_id = begin_id + len(answer)
            begin_id += len(prefix)
            end_id += len(prefix)

            result = {"text": answer, "start": begin_id, "end": end_id}
            outdict = {
                "content": content,
                "result_list": [result],
                "prompt": "答案",
            }
            wf.write(json.dumps(outdict, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = parse_args()
    answer_extraction_target_file_path = os.path.join(
        args.target_dir, "answer_extraction", os.path.basename(args.source_file_path)
    )
    if not os.path.exists(os.path.dirname(answer_extraction_target_file_path)):
        os.makedirs(os.path.dirname(answer_extraction_target_file_path))
    convert_from_json_to_answer_extraction_format(
        json_file=args.source_file_path,
        output_path=answer_extraction_target_file_path,
        domain=args.domain,
        do_answer_prompt=args.do_answer_prompt,
        do_len_prompt=args.do_len_prompt,
        do_domain_prompt=args.do_domain_prompt,
    )

    question_generation_target_file_path = os.path.join(
        args.target_dir, "question_generation", os.path.basename(args.source_file_path)
    )
    if not os.path.exists(os.path.dirname(question_generation_target_file_path)):
        os.makedirs(os.path.dirname(question_generation_target_file_path))
    convert_from_json_to_question_generation_format(
        json_file=args.source_file_path, output_path=question_generation_target_file_path
    )

    filtration_target_file_path = os.path.join(args.target_dir, "filtration", os.path.basename(args.source_file_path))
    if not os.path.exists(os.path.dirname(filtration_target_file_path)):
        os.makedirs(os.path.dirname(filtration_target_file_path))
    convert_from_json_to_filtration_format(json_file=args.source_file_path, output_path=filtration_target_file_path)
