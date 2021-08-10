# DDParser

* [模型简介](#模型简介)
* [快速开始](#快速开始)
    * [环境依赖](#环境依赖)
    * [句法分析任务](#句法分析任务)
* [致谢](#致谢)
* [参考论文](#参考论文)

## 模型简介

依存句法分析任务通过分析句子中词语之间的依存关系来确定句子的句法结构，DDParser是一款依存句法分析工具，
该示例基于paddle 2.1的[baidu/ddparser](https://github.com/baidu/DDParser)实现。

以下是本项目主要代码结构及说明：

```text
ddparser/
├── model # 部署
│   ├── dropouts.py # dropout
│   ├── encoder.py # 编码器
│   ├── metric.py # 指标计算
│   ├── model.py # 模型网络
│   └── model_utils.py # 模型网络工具函数
├── README.md # 使用说明
├── data.py # 数据结构
├── env.py # 环境配置工具
├── run.py # 主入口，包含训练、评估和预测任务
└── utils.py # 任务工具函数
```

## 快速开始

## 环境依赖

- LAC

安装命令: `pip install LAC`

## 句法分析任务

### 模型训练

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run.py --mode train --device gpu
```

### 模型评估
```shell
export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch --gpus "0" run.py --mode evaluate
```

### 模型预测
```shell
export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch --gpus "0" run.py --mode predict
```

## 致谢

* 感谢[百度NLP](https://github.com/baidu/DDParser)提供ddparser的开源代码实现。

## 参考论文

[《Deep Biaffine Attention for Neural Dependency Parsing》](https://arxiv.org/abs/1611.01734)