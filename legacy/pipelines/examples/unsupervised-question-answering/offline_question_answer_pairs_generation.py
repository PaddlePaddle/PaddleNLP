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
import os

from pipelines.nodes import AnswerExtractor, QAFilter, QuestionGenerator
from pipelines.pipelines import QAGenerationPipeline

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to run dense_qa system, defaults to gpu.")
parser.add_argument("--doc_dir", default="data/my_data", type=str, help="The question-answer piars file to be loaded when building ANN index.")
parser.add_argument("--source_file", default=None, type=str, help="The source raw texts file to be loaded when creating question-answer pairs.")

args = parser.parse_args()
# yapf: enable


def offline_qa_generation():
    answer_extractor = AnswerExtractor(
        model="uie-base-answer-extractor",
        device=args.device,
        schema=["答案"],
        position_prob=0.01,
    )

    question_generator = QuestionGenerator(
        model="unimo-text-1.0-question-generator",
        device=args.device,
    )

    qa_filter = QAFilter(
        model="uie-base-qa-filter",
        device=args.device,
        schema=["答案"],
        position_prob=0.1,
    )

    pipe = QAGenerationPipeline(
        answer_extractor=answer_extractor, question_generator=question_generator, qa_filter=qa_filter
    )
    pipeline_params = {"QAFilter": {"is_filter": True}}

    if args.source_file:
        meta = []
        with open(args.source_file, "r", encoding="utf-8") as rf:
            for line in rf:
                meta.append(line.strip())
        prediction = pipe.run(meta=meta, params=pipeline_params)
        prediction = prediction["filtered_cqa_triples"]
        if not os.path.exists(args.doc_dir):
            os.makedirs(args.doc_dir)
        with open(os.path.join(args.doc_dir, "generated_qa_pairs.txt"), "w", encoding="utf-8") as wf:
            for pair in prediction:
                wf.write(pair["synthetic_question"].strip() + "\t" + pair["synthetic_answer"].strip() + "\n")


if __name__ == "__main__":
    offline_qa_generation()
