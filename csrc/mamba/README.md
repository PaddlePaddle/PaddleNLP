# PaddleNLP Mamba 自定义算子安装

此文档介绍如何编译`Mamba`相关的自定义算子, 当前版本代码参考[mamba 仓库 v2.2.2](https://github.com/state-spaces/mamba/tree/v2.2.2)。

## 安装依赖

```shell
pip install -r requirements.txt
```

## 编译 causal_conv1d 算子

```shell
python setup_causal_conv1d.py install
```

## 编译 selective_scan 算子

```shell
python setup_selective_scan.py install
```

## 安装 Python 算子

```shell
python setup.py install
```
