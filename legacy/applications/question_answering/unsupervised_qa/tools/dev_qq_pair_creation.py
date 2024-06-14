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
    parser.add_argument("--do_create_test_qq_pair", action='store_true', help="Whether to do create_test_qq_pair")
    parser.add_argument('--qq_pair_source_ori_file_path', type=str, default=None, help='the original source file path for qq-pair creating')
    parser.add_argument('--qq_pair_source_trans_file_path', type=str, default=None, help='the translated source file path for qq-pair creating')
    parser.add_argument('--qq_pair_target_file_path', type=str, default=None, help='the target file path for qq-pair creating')
    parser.add_argument('--trans_query_answer_path', type=str, default=None, help='the target query-answer file path for extract_trans_from_fake_question')
    parser.add_argument('--dev_sample_num', type=int, default=None, help='the test sample number when convert_json_to_data, if None, treat all lines as dev samples')
    args = parser.parse_args()
    return args
# yapf: enable


def extract_q_from_json_file(json_file, out_file=None, test_sample_num=None, query_answer_path=None):
    with open(json_file, "r", encoding="utf-8") as rf:
        if out_file:
            wf = open(os.path.join(out_file), "w", encoding="utf-8")
        if query_answer_path:
            qeury_answer_wf = open(query_answer_path, "w", encoding="utf-8")
        q_list = []
        for i, json_line in enumerate(rf.readlines()):
            line_dict = json.loads(json_line)
            if isinstance(line_dict["question"], list):
                question = line_dict["question"][0]
            else:
                question = line_dict["question"]
            answer = line_dict["answer"]
            if not test_sample_num or i < test_sample_num:
                if query_answer_path:
                    qeury_answer_wf.write(
                        question.replace("\n", " ").replace("\t", " ").strip()
                        + "\t"
                        + answer.replace("\n", " ").replace("\t", " ").strip()
                        + "\n"
                    )
                if out_file:
                    wf.write(question.replace("\n", " ").replace("\t", " ").strip() + "\n")
                q_list.append(question.strip())
            else:
                break
        if query_answer_path:
            qeury_answer_wf.close()
        if out_file:
            wf.colse()
        return q_list


def create_test_qq_pair(
    ori_path=None, trans_path=None, write_path=None, trans_query_answer_path=None, test_sample_num=None
):
    assert trans_path
    trans_rf = open(trans_path, "r", encoding="utf-8")
    wf = open(write_path, "w", encoding="utf-8")
    if trans_path.endswith(".json"):
        trans_q_list = extract_q_from_json_file(trans_path, None, test_sample_num, trans_query_answer_path)
    else:
        trans_q_list = [
            line.strip() for i, line in enumerate(trans_rf.readlines()) if not test_sample_num or i < test_sample_num
        ]

    if not ori_path or ori_path in ["NONE", "None", "none"]:
        origin_q_list = ["-" for _ in range(len(trans_q_list))]
    else:
        origin_rf = open(ori_path, "r", encoding="utf-8")
        if ori_path.endswith(".json"):
            origin_q_list = extract_q_from_json_file(ori_path, None, test_sample_num)
        else:
            origin_q_list = [
                line.strip()
                for i, line in enumerate(origin_rf.readlines())
                if not test_sample_num or i < test_sample_num
            ]

    for origin, trans in zip(origin_q_list, trans_q_list):
        wf.write(
            trans.replace("\n", " ").replace("\t", " ").strip()
            + "\t"
            + origin.replace("\n", " ").replace("\t", " ").strip()
            + "\n"
        )
    if not ori_path or ori_path in ["NONE", "None", "none"]:
        pass
    else:
        origin_rf.close()
    trans_rf.close()
    wf.close()


if __name__ == "__main__":
    args = parse_args()
    if args.do_create_test_qq_pair:
        create_test_qq_pair(
            ori_path=args.qq_pair_source_ori_file_path,
            trans_path=args.qq_pair_source_trans_file_path,
            write_path=args.qq_pair_target_file_path,
            trans_query_answer_path=args.trans_query_answer_path,
            test_sample_num=args.dev_sample_num,
        )
