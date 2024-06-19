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
    parser.add_argument('--target_dir_path', type=str, default=None, help='the target dir path')
    parser.add_argument('--test_sample_num', type=int, default=0, help='the test sample number when preparing qa system data')
    parser.add_argument('--train_sample_num', type=int, default=0, help='the test sample number when preparing qa system data')
    parser.add_argument('--all_sample_num', type=int, default=None, help='the all sample number when preparing qa system data')
    args = parser.parse_args()
    return args
# yapf: enable


def convert_json_to_data(json_file, out_dir, test_sample_num, train_sample_num, all_sample_num=None):
    with open(json_file, "r", encoding="utf-8") as rf, open(
        os.path.join(out_dir, "qa_pair.csv"), "w", encoding="utf-8"
    ) as qa_pair_wf, open(os.path.join(out_dir, "qac_triple.csv"), "w", encoding="utf-8") as qac_triple_wf, open(
        os.path.join(out_dir, "train.csv"), "w", encoding="utf-8"
    ) as train_wf, open(
        os.path.join(out_dir, "q_corpus.csv"), "w", encoding="utf-8"
    ) as q_corpus_wf, open(
        os.path.join(out_dir, "dev.csv"), "w", encoding="utf-8"
    ) as test_wf:
        for i, json_line in enumerate(rf.readlines()):
            line_dict = json.loads(json_line)
            context = line_dict["context"]
            if "answer" in line_dict and "question" in line_dict:
                answer = line_dict["answer"]
                question = line_dict["question"]
            elif "synthetic_answer" in line_dict and "synthetic_question" in line_dict:
                answer = line_dict["synthetic_answer"]
                question = line_dict["synthetic_question"]

            if isinstance(question, list):
                question = question[0]
            else:
                question = question

            if i < test_sample_num:
                test_wf.write(question.replace("\n", " ").replace("\t", " ").strip() + "\n")
            elif test_sample_num <= i < test_sample_num + train_sample_num:
                train_wf.write(question.replace("\n", " ").replace("\t", " ").strip() + "\n")

            if not all_sample_num or i < all_sample_num:
                qa_pair_wf.write(
                    question.replace("\n", " ").replace("\t", " ").strip()
                    + "\t"
                    + answer.replace("\n", " ").replace("\t", " ").strip()
                    + "\n"
                )
                qac_triple_wf.write(
                    question.replace("\n", " ").replace("\t", " ").strip()
                    + "\t"
                    + answer.replace("\n", " ").replace("\t", " ").strip()
                    + "\t"
                    + context
                    + "\n"
                )
                q_corpus_wf.write(question.replace("\n", " ").replace("\t", " ").strip() + "\n")


if __name__ == "__main__":
    args = parse_args()
    convert_json_to_data(
        args.source_file_path, args.target_dir_path, args.test_sample_num, args.train_sample_num, args.all_sample_num
    )
