# PaddleNLP 自定义 OP

此文档介绍如何编译安装 PaddleNLP SDAA 自定义 OP。

# 1. 安装 PaddleCustomDevice

参考 [PaddleCustomDevice SDAA 安装文档](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/sdaa/README_cn.md) 进行安装


# 2. 安装 paddlenlp_ops
```shell
python setup_sdaa.py build bdist_wheel

pip install dist/paddlenlp_ops*.whl
```
