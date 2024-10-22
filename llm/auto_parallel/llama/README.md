# LLaMA 自动并行训练

## 1. 模型组网介绍

- 动静统一自动并行组网[modeling_auto.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/llama/modeling_auto.py)，支持动态图和动转静训练，建议使用。


## 2. 动静统一组网的训练方式
- 动态图训练
参考训练脚本 **run_pretrain_auto.sh**，运行8卡 dp2mp2pp2的并行策略。
- 动转静训练
参考训练脚本 **run_pretrain_auto.sh**，并开启 `to_static=1`，运行8卡 dp2mp2pp2的并行策略。

