# Ernie_doc 在iflytek数据集上的使用

## 简介
本示例将使用ERNIE-DOC模型，演示如何在长文本数据集上（e.g. iflytek）完成分类任务的训练，预测以及动转静过程。以下是本例的简要目录结构及说明：
```shell
.
├── LICENSE										
├── README.md									#文档 
├── data.py										#数据处理
├── metrics.py								#ERNIE-Doc下游任务指标
├── modeling.py								#ERNIE-Doc模型实现（针对实现静态图修改）
├── predict.py								#分类任务预测脚本（包括动态图预测和动转静）
└── train.py									#分类任务训练脚本（包括数据下载，模型导出和测试集结果导出）
```

## 快速开始