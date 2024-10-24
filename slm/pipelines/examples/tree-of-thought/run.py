# coding=utf8, ErnestinaQiu

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

import argparse
import json
import os
import time

from src.llm.llama import Ernie, Ernie_llm_list, llamaChatCompletion, llm_config
from src.tot.methods.bfs import naive_solve, solve
from src.tot.models import gpt_usage
from src.tot.tasks import get_task


def run(args, chatter):
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0
    if args.naive_run:
        file = f"./logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json"
        metric_fp = f"./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_select}_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}_metric.txt"
    else:
        file = f"./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json"
        metric_fp = f"./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}_metric.txt"
    os.makedirs(os.path.dirname(file), exist_ok=True)

    for i in range(args.task_start_index, args.task_end_index):
        args.log_fp = f"./logs/{args.task}/{args.backend}_{args.temperature}_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.log"
        args.query_fp = f"./logs/{args.task}/{args.backend}_{args.temperature}_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}_query.log"
        f = open(args.log_fp, "a", encoding="utf8")
        f.write(f"------ index: {i}")
        f.close()

        f = open(args.query_fp, "a", encoding="utf8")
        f.write(f"------ index: {i}")
        f.close()

        chatter.query = []
        chatter.tokenizer.init_chat_template(
            os.path.join(os.getcwd(), "pipelines", "examples", "tree-of-thought", "src", "llm", "chat_template.json")
        )

        # solve
        if args.naive_run:
            ys, info = naive_solve(args, task, i, chatter=chatter, args=args)
        else:
            ys, info = solve(args, task, i, chatter=chatter, args=args)

        # log
        infos = [task.test_output(i, y) for y in ys]
        info.update({"idx": i, "ys": ys, "infos": infos, "usage_so_far": gpt_usage(args.backend)})
        logs.append(info)
        with open(file, "w") as f:
            json.dump(logs, f, indent=4)

        # log main metric
        accs = [info["r"] for info in infos]
        cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)
        mes = f"{i}, 'sum(accs)', {sum(accs)}, 'cnt_avg', {cnt_avg}, 'cnt_any', {cnt_any}, '\n'"
        f = open(metric_fp, "a", encoding="utf8")
        f.write(mes)
        f.close()

        f = open(args.query_fp, "a", encoding="utf8")
        f.write(json.dumps(chatter.query))
        f.close()

    n = args.task_end_index - args.task_start_index
    mes2 = f"cnt_avg / n: {cnt_avg / n}, cnt_any / n: {cnt_any / n}"
    mes3 = f"'usage_so_far', {gpt_usage(args.backend)}"
    f = open(metric_fp, "a", encoding="utf8")
    f.write(mes2)
    f.write(mes3)
    f.close()


llm_backend_choices = list(llm_config.keys())


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--backend", type=str, choices=llm_backend_choices, default="llama-2-7b-chat")
    args.add_argument("--temperature", type=float, default=0.6)

    args.add_argument("--task", type=str, required=True, choices=["game24", "text", "crosswords"])
    args.add_argument("--task_start_index", type=int, default=900)
    args.add_argument("--task_end_index", type=int, default=1000)

    args.add_argument("--naive_run", action="store_true")
    args.add_argument(
        "--prompt_sample", type=str, choices=["standard", "cot"]
    )  # only used when method_generate = sample, or naive_run

    args.add_argument("--method_generate", type=str, choices=["sample", "propose"])
    args.add_argument("--method_evaluate", type=str, choices=["value", "vote"])
    args.add_argument("--method_select", type=str, choices=["sample", "greedy"], default="greedy")
    args.add_argument("--n_generate_sample", type=int, default=1)  # only thing needed if naive_run
    args.add_argument("--n_evaluate_sample", type=int, default=1)
    args.add_argument("--n_select_sample", type=int, default=1)

    args.add_argument("--query_fp", type=str, default=f"./logs/default/query_{int(time.time())}.log")

    args = args.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.backend in llm_backend_choices:
        chatter = llamaChatCompletion(args.backend)
    elif args.backend in Ernie_llm_list:
        chatter = Ernie(model=args.backend)
    run(args, chatter=chatter)
