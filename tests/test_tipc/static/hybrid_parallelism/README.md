# PaddleNLP 下静态图混合并行 benchmark 模型执行说明
静态图混合并行 benchmark 测试脚本说明

# 目录说明
# Docker 运行环境
docker image: registry.baidu.com/paddlecloud/base-images:paddlecloud-ubuntu18.04-gcc8.2-cuda11.0-cudnn8

# 运行 benchmark 测试步骤
```shell
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/tests/
```

# 准备数据

```shell
bash test_tipc/static/hybrid_parallelism/gpt/benchmark_common/prepare.sh
```

# 运行模型

## 单卡

```shell
bash  test_tipc/static/hybrid_parallelism/gpt/N1C1/${shell_name}.sh
```

## 多卡

```shell
bash  test_tipc/static/hybrid_parallelism/gpt/N${node_num}C${gpu_num}/${shell_name}.sh
```
