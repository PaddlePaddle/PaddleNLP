# Tree of Thoughts (ToT)

![teaser](https://github.com/PaddlePaddle/PaddleNLP/assets/48557439/30f9e365-398a-4822-b3c2-a0768f70e310)

论文[Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) 的代码 prompts 和 model outputs 实现。


## Setup
1. 安装
```bash
git clone git@github.com:PaddlePaddle/PaddleNLP.git
cd pipelines/examples/tree-of-thought/
pip install -r requirements.txt
```

2. 请从 https://github.com/ErnestinaQiu/tree-of-thought-llm/tree/master/src/tot/data 获取测试数据，并放置在 pipelines/examples/tree-of-thought/tree/master/src/tot/data

## Quick Start
以下是脚本，该脚本尝试使用4 5 6 10解决24点游戏（由于使用 llama-7b-chat，可能会稍慢一些）


在目录 pipelines/examples/agents/tree-of-thought-llm 下运行

```
python demo.py
```

以下是文档的中文翻译：

```python
import argparse
from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task

args = argparse.Namespace(backend='llama-2-7b-chat', temperature=0.6, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

task = Game24Task()
ys, infos = solve(args, task, 900)
print(ys[0])
```

输出结果可能如下（注意它不是确定性的，有时输出可能是错误的）：
```
10 - 4 = 6 (left: 5 6 6)
5 * 6 = 30 (left: 6 30)
30 - 6 = 24 (left: 24)
Answer: (5 * (10 - 4)) - 6 = 24
```

## 论文实验

通过 ``sh scripts/{game24, text, crosswords}/{standard_sampling, cot_sampling, bfs}.sh`` 运行实验。

非常简单的 ``run.py`` 实现了 ToT + BFS 算法，以及朴素的 IO/CoT 抽样。一些关键参数：

- ``--naive_run``: 如果为 True，则运行朴素的 IO/CoT 抽样，而不是 ToT + BFS。
- ``--prompt_sample`` (choices=[``standard``, ``cot``]): 抽样提示
- ``--method_generate`` (choices=[``sample``, ``propose``]): 思维生成器，是抽样独立思维（用于创意写作）还是提出连续思维（用于24点游戏）
- ``--method_evaluate`` (choices=[``value``, ``vote``]): 状态评估器，是独立使用价值状态（用于24点游戏）还是对状态进行投票（用于创意写作）
- ``--n_generate_sample``: 提示进行思维生成的次数
- ``--n_evaluate_sample``: 提示进行状态评估的次数
- ``--n_select_sample``: 每一步保留的状态数量（即论文中的 ``b`` 在 ToT + BFS 算法中）

## 论文轨迹

``logs/`` 包含论文实验的所有轨迹，除了 ``logs/game24/gpt-4_0.7_propose1_value3_greedy5_start900_end1000.json``，该文件是在论文之后重新生成的（因为原始实验是在笔记本中进行的），由于 GPT 解码中的随机性，得分从原来的 74\% 下降到了 69\%。我们希望将来汇总多次运行以考虑抽样随机性，并更新论文，但这不应影响论文的主要结论。

## 论文实验的任务脚本
### crosswords（填字游戏）
```
python run.py \
    --task crosswords \      # 任务名：填字游戏
    --task_start_index 0 \   # 填字游戏任务数据集中开始的序号
    --task_end_index 20 \    # 填字游戏任务数据集中结束的序号
    --naive_run \
    --prompt_sample cot \    # 抽样提示的方式, cot
    --n_generate_sample 10   # 提示进行思维生成的次数, 10次
```

```
python run.py \
    --task crosswords \
    --task_start_index 0 \
    --task_end_index 20 \
    --naive_run \               # 运行朴素的 IO/CoT 抽样
    --prompt_sample standard \  # 抽样提示的方式, standard
    --n_generate_sample 10
```

### game24（24点游戏）
```
python run.py \
    --task game24 \             # 任务名：24点游戏
    --task_start_index 900 \    # 24点游戏任务数据集中开始的序号
    --task_end_index 1000 \     # 24点游戏任务数据集中结束的序号
    --method_generate propose \ # 思维生成器，是抽样独立思维（用于创意写作）还是提出连续思维（用于24点游戏）
    --method_evaluate value \   # 状态评估器，独立使用价值状态（用于24点游戏）
    --method_select greedy \    # 策略选择，"greedy"（贪婪）
    --n_evaluate_sample 3 \     # 提示进行状态评估的次数
    --n_select_sample 5 \       # 每一步保留的状态数量（即论文中的 ``b`` 在 ToT + BFS 算法中）
```

```
python run.py \
    --task game24 \
    --task_start_index 900 \
    --task_end_index 1000 \
    --naive_run \                # 运行朴素的 IO/CoT 抽样
    --prompt_sample cot \        # 抽样提示的方式, cot
    --n_generate_sample 100 \
```

```
python run.py \
    --task game24 \
    --task_start_index 900 \
    --task_end_index 1000 \
    --naive_run \
    --prompt_sample standard \
    --n_generate_sample 100 \
```

### text(创意写作)
```
python run.py \
    --task text \            # 任务名：创意写作
    --task_start_index 0 \   # 创意写作任务数据集中开始的序号
    --task_end_index 100 \   # 创意写作任务数据集中结束的序号
    --method_generate sample \  # 思维生成器，是抽样独立思维（用于创意写作）还是提出连续思维（用于24点游戏）
    --method_evaluate vote \   # 状态评估器，对状态进行投票（用于创意写作）
    --method_select greedy \   # 策略选择，"sample"（举例）
    --n_generate_sample 5 \    # 提示进行思维生成的次数
    --n_evaluate_sample 5 \    # 提示进行状态评估的次数
    --n_select_sample 1 \      # 每一步保留的状态数量（即论文中的 ``b`` 在 ToT + BFS 算法中）
    --prompt_sample cot \
    --temperature 1.0 \
```

```
python run.py \
    --task text \
    --task_start_index 0 \
    --task_end_index 100 \
    --naive_run \                # 运行朴素的 IO/CoT 抽样
    --prompt_sample cot \        # 抽样提示的方式, cot
    --n_generate_sample 10 \
    --temperature 1.0 \
```

```
python run.py \
    --task text \
    --task_start_index 0 \
    --task_end_index 100 \
    --naive_run \                # 运行朴素的 IO/CoT 抽样
    --prompt_sample standard \   # 抽样提示的方式, standard
    --n_generate_sample 10 \
    --temperature 1.0 \
```

## 测试结果
本测试采用的是 paddlenlp 中 facebook/llama-2-7b-chat 和 facebook/llama-2-13b-chat.使用的参数为 temperature=0.6, decode_strategy 为"greedy_search"，max_new_tokens=512,结果如下
|model|method|acc|
|----|----|----|
|llama-2-7b-chat|cot|0|
|llama-2-7b-chat|standard sampling| 0|
|llama-2-7b-chat|ToT| 3%|
|llama-2-13b-chat|cot|0|
|llama-2-13b-chat|standard sampling|0|
|llama-2-13b-chat|ToT|2%|


## 如何添加新任务

设置一个新任务很容易，主要包括两个步骤。
* 在 ``tot/tasks/`` 中设置一个新的任务类和任务文件在 ``tot/data/`` 中。查看 ``tot/tasks/game24.py`` 以获取示例。将任务添加到 ``tot/tasks/__init__.py`` 中。
* 在 ``tot/prompts/`` 中设置任务特定的提示。查看 ``tot/prompts/game24.py`` 以获取示例。根据任务的性质，选择 ``--method_generate`` (choices=[``sample``, ``propose``]) 和 ``--method_evaluate`` (choices=[``value``, ``vote``]) 及其相应的提示。


## 致谢

我们借鉴了 Shunyu Yao ect.出色的框架设计，在此对 Tree of Thoughts 作者及其开源社区表示感谢。

We learn form the excellent framework design of Shunyu Yao, and we would like to express our thanks to the authors of Tree of Thoughts and their open source community.

```bibtex
@misc{yao2023tree,
      title={{Tree of Thoughts}: Deliberate Problem Solving with Large Language Models},
      author={Shunyu Yao and Dian Yu and Jeffrey Zhao and Izhak Shafran and Thomas L. Griffiths and Yuan Cao and Karthik Narasimhan},
      year={2023},
      eprint={2305.10601},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
