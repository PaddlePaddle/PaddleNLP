# LLaMA 自动并行训练

## 1. 模型组网介绍

- 动静统一自动并行组网[modeling_auto.py](../../../paddlenlp/transformers/llama/modeling_auto.py)，支持动态图和动转静训练，建议使用。
- 静态图自动并行组网[modeling_auto_static.py](../../../paddlenlp/transformers/llama/modeling_auto_static.py)，仅支持静态图训练，未来可能会下线。

## 2. 动静统一组网的训练方式
- 动态图训练
参考训练脚本[run_pretrain_auto.sh](./run_pretrain_auto.sh)，运行8卡dp2mp2pp2的并行策略。
- 动转静训练
参考训练脚本[run_pretrain_auto.sh](./run_pretrain_auto.sh)，并开启 `to_static=1`，运行8卡dp2mp2pp2的并行策略。

## 3. 静态图组网的训练方式

参考训练脚本[run_pretrain_auto_static.sh](./run_pretrain_auto_static.sh)，运行8卡dp2sharding2mp2pp2vpp2的并行策略。
参考训练脚本[run_pretrain_auto_static_sp.sh](./run_pretrain_auto_static_sp.sh)，运行8卡dp2sharding2mp2pp2vpp2sp的并行策略。
