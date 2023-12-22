#coding=utf8, ErnestinaQiu
import argparse
import json
import os

from src.tot.methods.bfs import naive_solve, solve
from src.tot.models import gpt_usage
from src.tot.tasks import get_task


def run(args):
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0
    if args.naive_run:
        file = f"./logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json"
    else:
        file = f"./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json"
    os.makedirs(os.path.dirname(file), exist_ok=True)

    for i in range(args.task_start_index, args.task_end_index):
        # solve
        if args.naive_run:
            ys, info = naive_solve(args, task, i)
        else:
            ys, info = solve(args, task, i)

        # log
        infos = [task.test_output(i, y) for y in ys]
        info.update(
            {
                "idx": i,
                "ys": ys,
                "infos": infos,
                "usage_so_far": gpt_usage(args.backend),
            }
        )
        logs.append(info)
        with open(file, "w") as f:
            json.dump(logs, f, indent=4)

        # log main metric
        accs = [info["r"] for info in infos]
        cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)
        print(i, "sum(accs)", sum(accs), "cnt_avg", cnt_avg, "cnt_any", cnt_any, "\n")

    n = args.task_end_index - args.task_start_index
    print(cnt_avg / n, cnt_any / n)
    print("usage_so_far", gpt_usage(args.backend))

llm_backend_choices = [
              "llama-2-7b", "llama-2-7b-chat",
              "llama-2-13b", "llama-2-13b-chat",
              "llama-2-70b", "llama-2-70b-chat",
              "llama-7b", "llama-13b", "llama-30b",
              "llama-65b", "ziqingyang/chinese-llama-7b", "ziqingyang/chinese-llama-13b",
              "ziqingyang/chinese-alpaca-7b", "ziqingyang/chinese-alpaca-13b",
              "idea-ccnl/ziya-llama-13b-v1", "linly-ai/chinese-llama-2-7b", "linly-ai/chinese-llama-2-13b",
              "baichuan-inc/Baichuan-7B", "baichuan-inc/Baichuan-13B-Base",
              "baichuan-inc/Baichuan-13B-Chat", "baichuan-inc/Baichuan2-7B-Base",
              "baichuan-inc/Baichuan2-7B-Chat",  "baichuan-inc/Baichuan2-13B-Base",
              "baichuan-inc/Baichuan2-13B-Chat", "FlagAlpha/Llama2-Chinese-7b-Chat",
              "FlagAlpha/Llama2-Chinese-13b-Chat"
                ]

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--backend", type=str, choices=llm_backend_choices, default="llama-2-7b-chat"
    )
    args.add_argument("--temperature", type=float, default=0.6)

    args.add_argument(
        "--task", type=str, required=True, choices=["game24", "text", "crosswords"]
    )
    args.add_argument("--task_start_index", type=int, default=900)
    args.add_argument("--task_end_index", type=int, default=1000)

    args.add_argument("--naive_run", action="store_true")
    args.add_argument(
        "--prompt_sample", type=str, choices=["standard", "cot"]
    )  # only used when method_generate = sample, or naive_run

    args.add_argument("--method_generate", type=str, choices=["sample", "propose"])
    args.add_argument("--method_evaluate", type=str, choices=["value", "vote"])
    args.add_argument(
        "--method_select", type=str, choices=["sample", "greedy"], default="greedy"
    )
    args.add_argument(
        "--n_generate_sample", type=int, default=1
    )  # only thing needed if naive_run
    args.add_argument("--n_evaluate_sample", type=int, default=1)
    args.add_argument("--n_select_sample", type=int, default=1)

    args = args.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    run(args)
