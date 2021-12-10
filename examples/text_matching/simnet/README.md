# 使用SimNet完成文本匹配任务

短文本语义匹配(SimilarityNet, SimNet)是一个计算短文本相似度的框架，可以根据用户输入的两个文本，计算出相似度得分。
SimNet框架在百度各产品上广泛应用，主要包括BOW、CNN、RNN、MMDNN等核心网络结构形式，提供语义相似度计算训练和预测框架，
适用于信息检索、新闻推荐、智能客服等多个应用场景，帮助企业解决语义匹配问题。
可通过[AI开放平台-短文本相似度](https://ai.baidu.com/tech/nlp_basic/simnet)线上体验。

## 模型简介


本项目通过调用[Seq2Vec](../../../paddlenlp/seq2vec/)中内置的模型进行序列建模，完成句子的向量表示。包含最简单的词袋模型和一系列经典的RNN类模型。

| 模型                                             | 模型介绍                                                     |
| ------------------------------------------------ | ------------------------------------------------------------ |
| BOW（Bag Of Words）                              | 非序列模型，将句子表示为其所包含词的向量的加和               |
| CNN                                          | 序列模型，使用卷积操作，提取局部区域地特征             |
| GRU（Gated Recurrent Unit）                      | 序列模型，能够较好地解决序列文本中长距离依赖的问题           |
| LSTM（Long Short Term Memory）                   | 序列模型，能够较好地解决序列文本中长距离依赖的问题           |


| 模型  | dev acc | test acc |
| ---- | ------- | -------- |
| BoW  | 0.7290 | 0.75232 |
| CNN  | 0.7042 | 0.73760 |
| GRU  | 0.7781 | 0.77808 |
| LSTM  | 0.73760 | 0.77320 |



## 快速开始

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
simnet/
├── model.py # 模型组网
├── predict.py # 模型预测
├── utils.py # 数据处理工具
├── train.py # 训练模型主程序入口，包括训练、评估
└── README.md # 文档说明
```

### 数据准备

#### 使用PaddleNLP内置数据集

```python
from paddlenlp.datasets import load_dataset

train_ds, dev_ds, test_ds = load_dataset("lcqmc", splits=["train", "dev", "test"])
```

部分样例数据如下：

```text
query title label
最近有什么好看的电视剧，推荐一下 近期有什么好看的电视剧，求推荐？ 1
大学生验证仅针对在读学生，已毕业学生不能申请的哦。 通过了大学生验证的用户，可以在支付宝的合作商户，享受学生优惠   0
如何在网上查户口  如何网上查户口 1
关于故事的成语 来自故事的成语 1
 湖北农村信用社手机银行客户端下载   湖北长阳农村商业银行手机银行客户端下载 0
草泥马是什么动物  草泥马是一种什么动物 1
```

### 模型训练

在模型训练之前，需要先下载词汇表文件simnet_vocab.txt，用于构造词-id映射关系。

```shell
wget https://bj.bcebos.com/paddlenlp/data/simnet_vocab.txt
```

**NOTE:** 词表的选择和实际应用数据相关，需根据实际数据选择词表。

我们以中文文本匹配数据集LCQMC为示例数据集，可以运行下面的命令，在训练集（train.tsv）上进行模型训练，并在开发集（dev.tsv）验证

CPU启动：

```shell
python train.py --vocab_path='./simnet_vocab.txt' \
   --device=cpu \
   --network=lstm \
   --lr=5e-4 \
   --batch_size=64 \
   --epochs=5 \
   --save_dir='./checkpoints'
```

GPU启动：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" train.py --vocab_path='./simnet_vocab.txt' \
   --device=gpu \
   --network=lstm \
   --lr=5e-4 \
   --batch_size=64 \
   --epochs=5 \
   --save_dir='./checkpoints'
```

以上参数表示：

* `vocab_path`: 词汇表文件路径。
* `device`: 选用什么设备进行训练，可选cpu或gpu。如使用gpu训练则参数gpus指定GPU卡号。
* `network`: 模型网络名称，默认为`lstm`， 可更换为lstm, gru, rnn，bow，cnn等。
* `lr`: 学习率， 默认为5e-4。
* `batch_size`: 运行一个batch大小，默认为64。
* `epochs`: 训练轮次，默认为5。
* `save_dir`: 训练保存模型的文件路径。
* `init_from_ckpt`: 恢复模型训练的断点路径。


程序运行时将会自动进行训练，评估，测试。同时训练过程中会自动保存模型在指定的`save_dir`中。
如：
```text
checkpoints/
├── 0.pdopt
├── 0.pdparams
├── 1.pdopt
├── 1.pdparams
├── ...
└── final.pdparams
```

**NOTE:** 如需恢复模型训练，则init_from_ckpt只需指定到文件名即可，不需要添加文件尾缀。如`--init_from_ckpt=checkpoints/0`即可，程序会自动加载模型参数`checkpoints/0.pdparams`，也会自动加载优化器状态`checkpoints/0.pdopt`。

### 模型预测

启动预测

CPU启动：

```shell
python predict.py --vocab_path='./simnet_vocab.txt' \
   --device=cpu \
   --network=lstm \
   --params_path=checkpoints/final.pdparams
```

GPU启动：

```shell
CUDA_VISIBLE_DEVICES=0 python predict.py --vocab_path='./simnet_vocab.txt' \
   --device=gpu \
   --network=lstm \
   --params_path='./checkpoints/final.pdparams'
```

将待预测数据分词完毕后，如以下示例：

```text
世界上什么东西最小   世界上什么东西最小？
光眼睛大就好看吗  眼睛好看吗？
小蝌蚪找妈妈怎么样   小蝌蚪找妈妈是谁画的
```

处理成模型所需的`Tensor`，如可以直接调用`preprocess_prediction_data`函数既可处理完毕。之后传入`predict`函数即可输出预测结果。

如

```text
Data: ['世界上什么东西最小', '世界上什么东西最小？']      Label: similar
Data: ['光眼睛大就好看吗', '眼睛好看吗？']      Label: dissimilar
Data: ['小蝌蚪找妈妈怎么样', '小蝌蚪找妈妈是谁画的']      Label: dissimilar
```
