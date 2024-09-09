# SimBERT模型

## 模型简介
[SimBERT](https://github.com/ZhuiyiTechnology/simbert)的模型权重是以Google开源的BERT模型为基础，基于微软的UniLM思想设计了融检索与生成于一体的任务，来进一步微调后得到的模型，所以它同时具备相似问生成和相似句检索能力。

## 快速开始

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
simbert/
├── data.py #训练样本的数据加载以及转换
├── predict.py # 模型预测
└── README.md # 文档说明
```

### 模型预测

启动预测：
```shell
export CUDA_VISIBLE_DEVICES=0
python predict.py --input_file ./datasets/lcqmc/dev.tsv
```

待预测数据如以下示例：


```text
世界上什么东西最小   世界上什么东西最小？
光眼睛大就好看吗  眼睛好看吗？
小蝌蚪找妈妈怎么样   小蝌蚪找妈妈是谁画的
```

按照predict.py.py进行预测得到相似度

如

```text
{'query': '世界上什么东西最小', 'title': '世界上什么东西最小？', 'similarity': 0.992725}
{'query': '光眼睛大就好看吗', 'title': '眼睛好看吗？', 'similarity': 0.74502724}
{'query': '小蝌蚪找妈妈怎么样', 'title': '小蝌蚪找妈妈是谁画的', 'similarity': 0.8192148}
```

## Reference

关于SimBERT更多信息参考[科学空间](https://spaces.ac.cn/archives/7427)

SimBERT项目地址 https://github.com/ZhuiyiTechnology/simbert
