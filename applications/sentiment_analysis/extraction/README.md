# 评论维度和观点抽取模型

## 1. 方案设计

在本实践中，我们将采用序列标注的方式抽取文本数据中的评论维度及相应的观点词，为此我们基于BIO的序列标注体系进行了标签的拓展：B-Aspect, I-Aspect, B-Opinion, I-Opinion, O，其中前两者用于标注评论维度，后两者用于标注评论观点。

如图1所示，首先将文本串传入SKEP模型中，利用SKEP模型对该文本串进行语义编码后，然后基于每个位置的输出去预测相应的标签。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/574e85b9bd3f4933bf13b574856935d063091ff5bdaf4195813a5d6d39b0c583" /></center>

<br><center>图1 评价维度和观点词抽取模型</center><br/>

## 2. 项目结构说明

以下是本项目的简要目录结构及说明：

```shell
.
├── data.py        # 数据处理脚本
├── evaluate.py    # 模型评估脚本
├── model.py       # 模型组网脚本
├── train.py       # 模型训练脚本
└── utils.py       # 工具包
```

## 3. 数据说明

如上所述，本项目将采用序列标注的方式进行抽取评论维度和观点，所以本项目训练集中需要包含两列数据：文本串和相应的序列标签数据，下面给出了一条样本。

```
还不错价格不高，服务好 O B-Opinion O B-Aspect I-Aspect B-Opinion I-Opinion O B-Aspect I-Aspect B-Opinion
```

