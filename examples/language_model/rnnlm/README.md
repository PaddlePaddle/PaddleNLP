# 语言模型

# 简介

## 1. 任务说明
本文主要介绍基于lstm的语言的模型的实现，给定一个输入词序列（中文分词、英文tokenize），计算其ppl（语言模型困惑度，用户表示句子的流利程度），基于循环神经网络语言模型的介绍可以[参阅论文](https://arxiv.org/abs/1409.2329)。相对于传统的方法，基于循环神经网络的方法能够更好的解决稀疏词的问题。


## 2. 效果说明

|   |    train    |   valid    |    test      |
| :------------- | :---------: | :--------: | :----------: |
|     PaddlePaddle     |    47.234   |  86.801    |    83.159    |
|   Tensorflow   |    45.594   |  87.363    |    84.015   |



## 3. 数据集

此任务的数据集合是采用ptb dataset，下载地址为: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz


# 快速开始

### 数据准备
为了方便开发者进行测试，我们内置了数据下载脚本，默认自动下载PTB数据集。

### 训练或Fine-tune

任务训练启动命令如下：

```
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" train.py \
```

程序运行时将会自动进行训练，评估，测试。同时训练过程中会自动保存模型到checkpoint、中。
还可以在启动命令后以--的形式修改网络参数或数据位置，具体可修改的参数和参数的默认值参考`args.py`。

**NOTE:** 如需恢复模型训练，则init_from_ckpt只需指定到文件名即可，不需要添加文件尾缀。如`--init_from_ckpt=checkpoints/test`即可，程序会自动加载模型参数`checkpoints/test.pdparams`，也会自动加载优化器状态`checkpoints/test.pdopt`。

# 进阶使用

## 任务定义与建模
此任务目的是给定一个输入的词序列，预测下一个词出现的概率。

## 模型原理介绍
此任务采用了序列任务常用的rnn网络，实现了一个两层的lstm网络，然后lstm的结果去预测下一个词出现的概率。

由于数据的特殊性，每一个batch的last hidden和last cell会被作为下一个batch 的init hidden 和 init cell。


## 数据格式说明
此任务的数据格式比较简单，每一行为一个已经分好词（英文的tokenize）的词序列。

目前的句子示例如下图所示:
```
aer banknote berlitz calloway centrust cluett fromstein gitano guterman hydro-quebec ipo kia memotec mlx nahb punts rake regatta rubens sim snack-food ssangyong swapo wachter
pierre <unk> N years old will join the board as a nonexecutive director nov. N
mr. <unk> is chairman of <unk> n.v. the dutch publishing group
```

特殊说明：ptb的数据比较特殊，ptb的数据来源于一些文章，相邻的句子可能来源于一个段落或者相邻的段落，ptb 数据不能做shuffle。


## 如何组建自己的模型
+ **自定义数据：** 关于数据，如果可以把自己的数据先进行分词（或者tokenize），通过`--data_path`来指定本地数据集所在文件夹，并需要在`train.py`中修改对应的文件名称。
+ **网络结构更改：** 网络只实现了基于lstm的语言模型，用户可以自己的需求更换为gru等网络结构，这些实现都是在`model.py`中定义。
