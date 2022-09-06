# PaddleNLP 下静态图 benchmark 模型执行说明

静态图 benchmark 测试脚本说明

# 目录说明
# Docker 运行环境

docker image: registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82

# 运行 benchmark 测试步骤

```shell
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/tests/
```

# 准备数据

无需额外准备数据，`${shell_name}.sh` 脚本里面已经加上了 prepare.sh 的调用。

```shell
bash test_tipc/static/dp/${model_item}/benchmark_common/prepare.sh
```

# 运行模型

## 单卡

```shell
export CUDA_VISIBLE_DEVICES=0
bash  test_tipc/static/dp/${model_item}/N1C1/${shell_name}.sh
```

## 多卡

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash test_tipc/static/dp/${model_item}/N1C8/${shell_name}.sh
```
