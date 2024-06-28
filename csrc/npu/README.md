# PaddleNLP 自定义 OP

此文档介绍如何编译安装 PaddleNLP NPU 自定义 OP。

# 1. 安装 PaddleCustomDevice

参考 [PaddleCustomDevice NPU 安装文档](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/README_cn.md) 进行安装

# 2. 安装 paddlenlp_ops
```shell
python setup.py build bdist_wheel

pip install dist/paddlenlp_ops*.whl
```
