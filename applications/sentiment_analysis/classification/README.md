# 细粒度情感分类模型



## 1. 方案设计

本项目将针对语句中的维度级别进行情感分析，对于给定的一段文本，我们在基于评论维度和观点抽取模型抽取出评论维度和观点后，便可以有针对性的对各个维度进行评论。具体来讲，本实践将抽取出的评论维度和评论观点进行拼接，然后原始语句进行拼接作为一条独立的训练语句。

如图1所示，首先将评论维度和观点词进行拼接为"味道好"，然后将"味道好"和原文进行拼接，然后传入SKEP模型，并使用"CLS"位置的向量进行细粒度情感倾向。

<center><img src="../imgs/design_cls_model.png" /></center>

<br><center>图1 细粒度情感分类模型</center><br/>

## 2. 项目结构说明

以下是本项目运行的完整目录结构及说明：

```shell
.
├── data             # 数据目录
├── checkpoints      # 模型保存目录
│   └── static       # 静态图模型保存目录
├── data.py          # 数据处理脚本
├── model.py         # 模型组网脚本
├── train.py         # 模型训练脚本
├── evaluate.py      # 模型评估脚本
├── utils.py         # 工具函数
├── run_train.sh     # 模型训练命令
├── run_evaluate.sh  # 模型评估命令
└── README.md
```

## 3. 数据说明

本模型将基于评论维度和观点进行细粒度的情感分析，因此数据集中需要包含3列数据：文本串和相应的序列标签数据，下面给出了一条样本，其中第1列是情感标签，第2列是评论维度和观点，第3列是原文。

> 1   口味清淡   口味很清淡，价格也比较公道

可点击[data_cls](https://bj.bcebos.com/v1/paddlenlp/data/data_ext.tar.gz)进行Demo数据下载，将数据解压之后放入本目录的`data`文件夹下。

## 4. 模型效果展示

在分类模型训练过程中，总共训练了10轮，并选择了评估F1得分最高的best模型， 更加详细的训练参数设置如下表所示：
|Model|训练参数配置|硬件|MD5|
| ------------ | ------------ | ------------ |-----------|
|[cls_model](https://bj.bcebos.com/paddlenlp/models/best_cls.pdparams)|<div style="width: 150pt"> learning_rate: 3e-5, batch_size: 16, max_seq_len:256, epochs：10 </div>|<div style="width: 100pt">Tesla V100-32g</div>|3de6ddf581e665d9b1d035c29b49778a|

我们基于训练过程中的best模型在验证集`dev_set`和测试集`test_set`上进行了评估测试，模型效果如下表所示:
|Model|数据集|precision|Recall|F1|
| ------------ | ------------ | ------------ |-----------|------------ |
|SKEP-Large|dev_set|0.98758|0.99251|0.99004|
|SKEP-Large|test_set|0.98497|0.99139|0.98817|

**备注**：以上数据是基于全量数据训练和测试结果，并非Demo数据集。

## 5. 模型训练
通过运行以下命令进行分类模型训练：
```shell
sh run_train.sh
```

## 6. 模型测试
通过运行以下命令进行分类模型测试：
```shell
sh run_evaluate.sh
```
