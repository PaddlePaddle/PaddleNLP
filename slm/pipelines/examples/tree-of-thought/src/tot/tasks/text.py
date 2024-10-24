# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import re

from src.tot.models import gpt
from src.tot.prompts.text import (
    compare_prompt,
    cot_prompt,
    score_prompt,
    standard_prompt,
    vote_prompt,
)
from src.tot.tasks.base import DATA_PATH, Task


class TextTask(Task):
    """
    Input (x)   : a text instruction
    Output (y)  : a text generation
    Reward (r)  : # TODO
    Input Example:
    Output Example:
    """

    def __init__(self, file="data_100_random_text.txt"):
        """
        file: a text file, each line is some sentences
        """
        super().__init__()
        path = os.path.join(DATA_PATH, "text", file)
        self.data = open(path).readlines()
        self.steps = 2
        self.stops = ["\nPassage:\n", None]

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output(self, idx: int, output: str):
        output = output.split("Passage:\n")[-1]
        prompt = score_prompt + output
        score_outputs = gpt(prompt, n=5, model="gpt-4")
        scores = []
        for score_output in score_outputs:
            # print(score_output)
            pattern = r".*coherency score is (\d+).*"
            match = re.match(pattern, score_output, re.DOTALL)
            if match:
                score = int(match.groups()[0])
                scores.append(score)
            else:
                print(f"------------------score no match: {[score_output]}")
        print(scores)
        # print('------------')
        info = {"rs": scores, "r": sum(scores) / len(scores) if scores else 0}
        return info

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = "") -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = "") -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        prompt = vote_prompt
        for i, y in enumerate(ys, 1):
            # y = y.replace('Plan:\n', '')
            # TODO: truncate the plan part?
            prompt += f"Choice {i}:\n{y}\n"
        return prompt

    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best choice is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f"vote no match: {[vote_output]}")
        return vote_results

    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:
        assert len(ys) == 2, "compare prompt only supports 2 candidates"
        ys = [y.split("Passage:\n")[-1] for y in ys]
        prompt = compare_prompt + f"Passage 1:\n{ys[0]}\n\nPassage 2:\n{ys[1]}\n"
        return prompt

    @staticmethod
    def compare_output_unwrap(compare_output: str):
        if "more coherent passage is 1" in compare_output:
            return 0
        elif "more coherent passage is 2" in compare_output:
            return 1
        elif "two passages are similarly coherent" in compare_output:
            return 0.5
        else:
            print(f"-----------------compare no match: {[compare_output]}")
            return -1
