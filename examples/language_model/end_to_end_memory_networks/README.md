# End-To-End-Memory-Networks-in-Paddle
## 一、简介

用Paddle来复现论文End-To-End Memory Networks

![模型简介](http://paddle.yulan.net.cn/model_introduction.png)

本模型是Facebook AI在Memory networks之后提出的一个更加完善的记忆网络模型，在问答系统以及语言模型中均有良好的应用。论文中使用了多个单层单元堆叠而成的多层架构。

单层架构如上图a所示，主要的参数包括A,B,C,W四个矩阵，其中A,B,C三个矩阵就是embedding矩阵，主要是将输入文本和Question编码成词向量，W是最终的输出矩阵。从上图可以看出，对于输入的句子s分别会使用A和C进行编码得到Input和Output的记忆模块，Input用来跟Question编码得到的向量相乘得到每句话跟q的相关性，Output则与该相关性进行加权求和得到输出向量。然后再加上q并传入最终的输出层。

多层网络如上图b所示，实际上是将多个单层堆叠到一起形成的网络，这里将每一层称为一个hop。
为了减少参数，模型提出了两种让各个hop之间共享Embedding参数（A与C）的方法：
* Adjacent：这种方法让相邻层之间的$A=C$。也就是说$A_{k+1}=C_{k}$，此外W等于顶层的C，B等于底层的A，这样就减少了一半的参数量。
* Layer-wise（RNN-like)：与RNN相似，采用完全共享参数的方法，即各层之间参数均相等。$A_{1}=A_{2}=...=A_{k}$,$C_{1}=C_{2}=...=C_{k}$。但这样模型的参数太少，性能会受到影响，故提出一种改进方法，在每一层之间加一个线性映射矩阵H，即令$u^{k+1}=H u^{k}+o^{k}$。

具体到语言模型，模型做出了一下调整：
1. 由于输入是单个句子，编码级别是单词级的，所以可以直接将每个单词的词向量存入memory即可，也就是说A与C现在都是单词的Embedding矩阵，mi与ci中都是单个单词的词向量。
2. 输出W矩阵的output为下一个单词的概率，即输出维度为vocab size。
3. 不同于QA任务，这里不存在Question，所以直接将q向量设置为全0.1的常量，也不需要再进行Embedding操作。
4. 采用Layer-wise的参数缩减策略。
5. 文中提出，对于每一层的u向量中一半的神经元进行ReLU操作，以帮助模型训练。

## 二、数据集

* Penn Treetank:

    * [Penn Treebank](http://paddle.yulan.net.cn/ptb.zip)

        NLP中常用的PTB语料库,语料来源为1989年华尔街日报，并做以下切分

        train：887k words

        valid：70k words

        test：78k words

        vocabulary  size：10k

    * [text8](http://paddle.yulan.net.cn/text8.zip)

        来源于enwiki8，总共100M个字符，划分为93.3M/5.7M/1M字符(train/valid/test)，将出现次数少于10次的单词替换为<UNK>

## 三、环境依赖

* 硬件：GPU
* 框架：Paddle >= 2.0.0，progress库

## 四、快速开始

下载数据集和已训练好的模型
```bash
mkdir data
mkdir models
cd data
wget http://paddle.yulan.net.cn/ptb.zip
wget http://paddle.yulan.net.cn/text8.zip
unzip -d ptb ptb.zip
unzip -d text8 text8.zip
cd ..
cd models
wget http://paddle.yulan.net.cn/model_ptb
wget http://paddle.yulan.net.cn/model_text8
cd ..
```

### 训练

训练参数可在`config.yaml`文件中调整。

Note: 由于本模型受随机因素影响较大，故每次训练的结果差异较大，即使固定随机种子，由于GPU的原因训练结果仍然无法完全一致。

#### 在ptb数据集上训练

```bash
cp config/config_ptb.yaml config.yaml
python train.py
```

#### 寻找最佳模型

由于模型受随机因素影响较大，故要进行多次训练来找到最优模型，原论文中在ptb数据集上进行了10次训练，并保留了在test集上表现最好的模型。本复现提供了一个脚本，来进行多次训练以获得能达到足够精度的模型。

```bash
cp config/config_ptb.yaml config.yaml
python train_until.py --target 111.0
```

以下是在ptb数据集上进行多次训练以达到目标精度的[log](http://paddle.yulan.net.cn/ptb_train_until.log),可以计算出20轮的平均ppl为113，方差为5.68

#### 在text8数据集上训练

```bash
cp config/config_text8.yaml config.yaml
python train.py
```

### 测试

保持`config.yaml`文件与训练时相同

```
python eval.py
```

### 使用预训练模型

#### ptb数据集上

```bash
cp config/config_ptb_test.yaml config.yaml
python eval.py
```

将得到以下结果

![](http://paddle.yulan.net.cn/test_ptb.png)

#### text8数据集上

```bash
cp config/config_text8_test.yaml config.yaml
python eval.py
```

结果如下

![](http://paddle.yulan.net.cn/test_text8.png)

## 五、复现精度

相应模型已包含在本repo中，分别位于目录`models_ptb`与`models_text8`下

| Dataset | Paper Perplexity | Our Perplexity |
| :-----: | :--------------: | :------------: |
|   ptb   |       111        |     110.75     |
|  text8  |       147        |     145.62     |

## 六、代码结构详细说明

### 6.1 代码结构

```
├── checkpoints
├── config                                        # 配置文件模板
├── config.yaml
├── README.md
├── requirements.txt
├── config.py
├── model.py
├── data.py
├── train.py                                    # 训练脚本
├── eval.py                                        # 测试脚本
├── train_until.py
└── utils.py
```

### 6.2 参数说明

可以在`config.yaml`中设置以下参数

```
# internal state dimension
edim: 150
# linear part of the state
lindim: 75
# number of hops
nhop: 7
# memory size
mem_size: 200
# initial internal state value
init_hid: 0.1
# initial learning rate
init_lr: 0.01
# weight initialization std
init_std: 0.05
# clip gradients to this norm
max_grad_norm: 50

# batch size to use during training
batch_size: 128
# number of epoch to use during training
nepoch: 100

# data directory
data_dir: "data/ptb"
# checkpoint directory
checkpoint_dir: "checkpoints"
# model name for test and recover train
model_name: "model"
# if True, load model [model_name] before train
recover_train: False
# data set name
data_name: "ptb"
# print progress, need progress module
show: True
# initial random seed
srand: 17814
# How many epochs output log once
log_epoch: 5
# Desired ppl
target_ppl: 147
```

### 七、reference
原论文地址：[Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus: “End-To-End Memory Networks”, 2015.](https://arxiv.org/pdf/1503.08895v5.pdf)

复现repo：[yulangz/End-to-End-Memory-Networks-in-Paddle](https://github.com/yulangz/End-to-End-Memory-Networks-in-Paddle)

参考repo：[https://github.com/facebookarchive/MemNN](https://github.com/facebookarchive/MemNN)

项目AiStudio地址：[https://aistudio.baidu.com/aistudio/projectdetail/2381004](https://aistudio.baidu.com/aistudio/projectdetail/2381004)
