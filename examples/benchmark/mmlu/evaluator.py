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
# Adapted from https://github.com/hendrycks/test

import numpy as np
import paddle
from tqdm import tqdm

from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

choices = ["A", "B", "C", "D"]


class ModelEvaluator(object):
    def __init__(self, model_name_or_path, ntrain, temperature=0.2, dtype="float32", tensor_parallel_degree=1):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tensor_parallel_degree = tensor_parallel_degree
        if self.tensor_parallel_degree > 1:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=dtype,
                low_cpu_mem_usage=True,
                tensor_parallel_output=False,
                tensor_parallel_degree=self.tensor_parallel_degree,
                tensor_parallel_rank=paddle.distributed.get_rank(),
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=dtype, low_cpu_mem_usage=True)
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
        self.ntrain = ntrain

    def format_subject(self, subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def gen_prompt(self, train_df, subject, k=-1):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            self.format_subject(subject)
        )
        if k == -1:
            k = train_df.shape[0]
        for i in range(k):
            prompt += self.format_example(train_df, i)
        return prompt

    def format_example(self, df, idx, include_answer=True):
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
        return prompt

    def eval(self, subject, dev_df, test_df, do_ptq=False):
        cors = []
        all_probs = []
        for i in tqdm(range(test_df.shape[0]), total=test_df.shape[0]):
            # for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = self.ntrain
            prompt_end = self.format_example(test_df, i, include_answer=False)
            train_prompt = self.gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            inputs = self.tokenizer(prompt, return_tensors="pd")
            label = test_df.iloc[i, test_df.shape[1] - 1]

            with paddle.no_grad():
                logits = self.model(**inputs)[0][0, -1, :]
            choices_logits = logits[[self.A_id, self.B_id, self.C_id, self.D_id]].numpy()
            assert not (np.any(np.isinf(choices_logits)) or np.any(np.isnan(choices_logits)))
            ans = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choices_logits)]

            cor = ans == label
            cors.append(cor)
            all_probs.append(choices_logits)

            print(f"\n=======begin {str(i)}=======")
            print("prompt: ", prompt)
            print("ans: ", ans)
            print("ground truth: ", label, "\n")
            print(f"=======end {str(i)}=======")

        acc = np.mean(cors)
        cors = np.array(cors)

        all_probs = np.array(all_probs)
        print("Average accuracy {:.3f} - {}".format(acc, subject))

        return cors, acc, all_probs
