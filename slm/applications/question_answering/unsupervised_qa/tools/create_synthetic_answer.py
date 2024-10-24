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
    parser.add_argument('--source_file_path', type=str, default=None, help='the source file path')
    parser.add_argument('--target_file_path', type=str, default=None, help='the target json file path')
    parser.add_argument('--all_sample_num', type=int, default=None, help='the test sample number when convert_json_to_data')
    parser.add_argument('--num_return_sequences', type=int, default=3, help='the number of return sequences for each input sample, it should be less than num_beams')
    parser.add_argument('--batch_size', type=int, default=1, help='the batch size when using taskflow')
    parser.add_argument('--position_prob', type=float, default=0.01, help='the batch size when using taskflow')
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


def answer_generation_from_paragraphs(paragraphs, batch_size=16, model=None, wf=None):
    """Generate answer from given paragraphs."""
    result = []
    buffer = []
    for paragraph_tobe in tqdm(paragraphs):
        buffer.append(paragraph_tobe)
        if len(buffer) == batch_size:
            predicts = model(buffer)
            paragraph_list = buffer
            buffer = []
            for predict_dict, paragraph in zip(predicts, paragraph_list):
                if "答案" in predict_dict:
                    answer_dicts = predict_dict["答案"]
                    answers = [answer_dict["text"] for answer_dict in answer_dicts]
                    probabilitys = [answer_dict["probability"] for answer_dict in answer_dicts]
                else:
                    answers = []
                    probabilitys = []

                outdict = {
                    "context": paragraph,
                    "answer_candidates": sorted([(a, p) for a, p in zip(answers, probabilitys)], key=lambda x: -x[1]),
                }
                if wf:
                    wf.write(json.dumps(outdict, ensure_ascii=False) + "\n")
                result.append(outdict)
    return result


if __name__ == "__main__":
    args = parse_args()
    schema = ["答案"]
    answer_generator = Taskflow(
        "information_extraction",
        schema=schema,
        task_path=args.model_path,
        batch_size=args.batch_size,
        position_prob=args.position_prob,
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
    wf = None
    if args.target_file_path:
        wf = open(args.target_file_path, "w", encoding="utf-8")

    answer_generation_from_paragraphs(paragraphs, batch_size=args.batch_size, model=answer_generator, wf=wf)
    rf.close()
    wf.close()
