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

# Adapted from https://github.com/ymcui/Chinese-LLaMA-Alpaca and https://github.com/SJTU-LIT/ceval
import os
import random
import re

import numpy as np
import paddle
from evaluator import Evaluator
from tqdm import tqdm

from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer


class ModelEvaluator(Evaluator):
    def __init__(self, choices, k, model_name_or_path, temperature=0.2):
        super().__init__(choices, model_name_or_path, k)
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype="float16", low_cpu_mem_usage=True)
        self.model.eval()
        self.generation_config = dict(
            temperature=temperature,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            num_beams=1,
            repetition_penalty=1.1,
            max_new_tokens=20,
        )

        self.A_id = self.tokenizer.encode("A", add_special_tokens=False)["input_ids"][0]
        self.B_id = self.tokenizer.encode("B", add_special_tokens=False)["input_ids"][0]
        self.C_id = self.tokenizer.encode("C", add_special_tokens=False)["input_ids"][0]
        self.D_id = self.tokenizer.encode("D", add_special_tokens=False)["input_ids"][0]

    def eval_subject(
        self,
        subject_name,
        test_df,
        dev_df=None,
        few_shot=False,
        cot=False,
        save_result_dir=None,
        with_prompt=False,
        constrained_decoding=False,
        do_test=False,
    ):
        all_answers = {}

        correct_num = 0
        if save_result_dir:
            result = []
            score = []
        if few_shot:
            history = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot)
        else:
            history = ""
        answers = ["NA"] * len(test_df) if do_test is True else list(test_df["answer"])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False, cot=cot, with_prompt=with_prompt)
            instruction = history + question
            inputs = self.tokenizer(instruction, return_tensors="pd")
            batch_size, length = inputs.input_ids.shape
            if constrained_decoding is True:
                # batch_size is 1, take the last logits as the logits for next token prediction
                with paddle.no_grad():
                    logits = self.model(**inputs)[0][0, -1, :]
                choices_logits = logits[[self.A_id, self.B_id, self.C_id, self.D_id]].numpy()
                assert not (np.any(np.isinf(choices_logits)) or np.any(np.isnan(choices_logits)))
                ans = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choices_logits)]
                response = self.tokenizer.decode([logits.argmax(-1).item()])
            else:
                generation_output = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **self.generation_config,
                )
                response = self.tokenizer.decode(generation_output[0][0, length:], skip_special_tokens=True)
                ans, direct_extract = self.extract_answer(row, response)
            if ans == answers[row_index]:
                correct_num += 1
                correct = 1
            else:
                correct = 0
            print(f"\n=======begin {str(row_index)}=======")
            print("question: ", question)
            print("response: ", response)
            print("ans: ", ans)
            print("ground truth: ", answers[row_index], "\n")
            if save_result_dir:
                result.append(response)
                score.append(correct)
            print(f"=======end {str(row_index)}=======")

            all_answers[str(row_index)] = ans

        correct_ratio = 100 * correct_num / len(answers)

        if save_result_dir:
            test_df["model_output"] = result
            test_df["correctness"] = score
            test_df.to_csv(os.path.join(save_result_dir, f"{subject_name}_test.csv"))

        return correct_ratio, all_answers

    def format_example(self, line, include_answer=True, cot=False, with_prompt=False):
        example = line["question"]
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        if include_answer:
            if cot:
                example += "\n答案：让我们一步一步思考，\n" + line["explanation"] + f"\n所以答案是{line['answer']}。\n\n"
            else:
                example += "\n答案：" + line["answer"] + "\n\n"
        else:
            if with_prompt is False:
                if cot:
                    example += "\n答案：让我们一步一步思考，\n1."
                else:
                    example += "\n答案："
            else:
                if cot:
                    example += "\n答案是什么？让我们一步一步思考，\n1."
                else:
                    example += "\n答案是什么？ "
        return example

    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            prompt += self.format_example(dev_df.iloc[i, :], include_answer=True, cot=cot)
        return prompt

    def extract_answer(self, line, gen_ans):
        m = re.findall(r"所以答案是(.+?)。", gen_ans, re.M)
        if len(m) > 0 and m[-1] in self.choices:
            return m[-1], True
        answer_patterns = [
            r"([ABCD])是正确的",
            r"选项([ABCD])正确",
            r"答案为([ABCD])",
            r"答案是([ABCD])",
            r"答案([ABCD])",
            r"选择([ABCD])",
            r"答案：([ABCD])",
            r"选择答案([ABCD])",
        ]
        # RE extraction
        for answer_pattern in answer_patterns:
            m = re.search(answer_pattern, gen_ans, re.M)
            if m:
                answer = m.group(1)
                return answer, False
        # only containing one choice-character
        m = re.findall(r"[ABCD]", gen_ans, re.M)
        if len(m) >= 1:
            answer = m[0]
            return answer, False
        # only containing one choice-context
        choices_dict = {}
        pattern = ""
        for c in self.choices:
            choices_dict[str(line[f"{c}"])] = c
            pattern += re.escape(str(line[f"{c}"])) + "|"
        pattern = pattern[:-1]
        m = re.findall(pattern, gen_ans, re.M)
        print("w/ escape:", repr(pattern), gen_ans, (len(m) >= 1))
        if len(m) >= 1:
            answer = choices_dict[m[0]]
            return answer, False
        return random.choice("ABCD"), False
