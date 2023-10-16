# DPO: Direct Preference Optimization

DPO算法用于对齐人类偏好，包括以下两个阶段：
1. 在目标数据集上进行监督微调SFT
2. 在SFT模型上使用DPO算法进行优化

具体算法请参考：[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

| Model | Pretrain | SFT | LoRA | PrefixTuning | Generation | Quantization | DPO |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [LLaMA v1/v2](./llama) | ✅  | ✅ | ✅ | ✅ | ✅ | ✅  | ✅ |


# DPO训练流程

## 1. 环境准备

- PaddlePaddle >= 2.5.1
- PaddleNLP >= 2.6.0

## 2. 数据准备

### 2.1 SFT
    我们支持的数据格式是每行包含一个字典，每个字典包含以下字段：

    - `src` : `str, List(str)`, 模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
    - `tgt` : `str, List(str)`, 模型的输出。

    样例数据：
    ```
    {"src": "Looking for movie featuring Dragons vs. Navy (or Army, but the trailer featured battleship trying to shoot the dragons down).\n\nTrailer also showed a dogfight between helicopters and dragons, capping (on the trailer) with a dragon setting a Blackhawk's blades on fire.", "tgt": "I saw that trailer too and then could not find it again , but stumbled across it today ,it is called Crimson Skies \n\nOnly other info i could find on this is that it was originaly called Dragon seige but your guess is as good as mine as to wether its movie or game"}
    ...
    ```
### 2.2 DPO

    我们支持的数据格式是每行包含一个字典，每个字典包含以下字段：

        - `prompt` : `str, List(str)`, 模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
        - `chosen` : `str, List(str)`, 人类偏向选择的回答。
        - `rejected` : `str, List(str)`, 人类偏向拒绝的回答。

        样例数据：
        ```
        {'prompt': "Question: I am working on a web api and I am curios about the `HTTP SEARCH` verb and how you should use it.\n\nMy first approach was, well you could use it surely for a search. But asp.net WebApi doesn't support the `SEARCH` verb.\n\nMy question is basically, should I use `SEARCH` or is it not important?\n\nPS: I found the `SEARCH` verb in fiddler2 from telerik.\n\nAnswer: ", 'chosen': 'The HTTP protocol is defined by RFC documents. RFC2616 defines HTTP/1.1 the current release.\n\nSEARCH is not documented as a verb in this document.', 'rejected': 'AFAIK the `SEARCH` method is only a proposal and should not be used. Use `GET` instead.'}
        ...
        ```

## 3. 训练

### 3.1 SFT

SFT使用与PaddleNLP的SFT使用方式相同，因此可以参考PaddleNLP的[SFT](../../llm/README.md)训练脚本。SFT启动训练命令如下：

```
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" sft_main.py ./scripts/llama/sft_argument.json
```

### 3.2 DPO

DPO训练脚本如下：

```
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" dpo_main.py ./scripts/llama/dpo_argument.json
```

## 4. 模型推理

模型推理部分与PaddleNLP的SFT使用方式相同，因此可以参考PaddleNLP的[模型推理](../../llm/README.md)。DPO启动训练命令如下：

```
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" dpo_main.py ./scripts/llama/dpo_argument.json
```
