# 评论维度和观点抽取模型

## 1. 方案设计

在本实践中，我们将采用序列标注的方式抽取文本数据中的评论维度及相应的观点词，为此我们基于BIO的序列标注体系进行了标签的拓展：B-Aspect, I-Aspect, B-Opinion, I-Opinion, O，其中前两者用于标注评论维度，后两者用于标注评论观点。

如图1所示，首先将文本串传入SKEP模型中，利用SKEP模型对该文本串进行语义编码后，然后基于每个位置的输出去预测相应的标签。

<center><img src="../imgs/design_ext_model.png" /></center>
<br><center>图1 评价维度和观点词抽取模型</center><br/>

## 2. 项目结构说明

以下是本项目的简要目录结构及说明：

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

如上所述，本项目将采用序列标注的方式进行抽取评论维度和观点，所以本项目训练集中需要包含两列数据：文本串和相应的序列标签数据，下面给出了一条样本。

```
还不错价格不高，服务好 O B-Opinion O B-Aspect I-Aspect B-Opinion I-Opinion O B-Aspect I-Aspect B-Opinion
```

可点击[data_ext](https://bj.bcebos.com/v1/paddlenlp/data/data_cls.tar.gz)进行Demo数据下载，将数据解压之后放入本目录的`data`文件夹下。

## 4. 模型效果展示
在分类模型训练过程中，总共训练了10轮，并选择了评估F1得分最高的best模型， 更加详细的训练参数设置如下表所示：
|Model|训练参数配置|硬件|MD5|
| ------------ | ------------ | ------------ |-----------|
|[ext_model](https://bj.bcebos.com/paddlenlp/models/best_ext.pdparams)|<div style="width: 150pt"> learning_rate: 5e-5, batch_size: 8, max_seq_len:512, epochs：10 </div>|<div style="width: 100pt">Tesla V100-32g</div>|e3358632165aa0338225e175b57cb304|

我们基于训练过程中的best模型在验证集`dev_set`和测试集`test_set`上进行了评估测试，模型效果如下表所示:
|Model|数据集|precision|Recall|F1|
| ------------ | ------------ | ------------ |-----------|------------ |
|SKEP-Large|dev_set|0.87095|0.90056|0.88551|
|SKEP-Large|test_set|0.87125|0.89944|0.88512|

**备注**：以上数据是基于全量数据训练和测试结果，并非Demo数据集。

## 5. 模型训练
通过运行以下命令进行抽取模型训练：
```shell
sh run_train.sh
```

## 6. 模型测试
通过运行以下命令进行抽取模型测试：
```shell
sh run_evaluate.sh
```
