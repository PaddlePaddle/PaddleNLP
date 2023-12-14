import argparse
from src.tot.methods.bfs import solve
from src.tot.tasks.game24 import Game24Task

args = argparse.Namespace(backend='llama-2-7b-chat', temperature=0.6, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

task = Game24Task()
ys, infos = solve(args, task, 900)
print(ys[0])
print(infos)