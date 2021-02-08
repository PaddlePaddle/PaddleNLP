# 语言模型

# 简介

## 1. 任务说明
本文主要介绍基于lstm的语言的模型的实现，给定一个输入词序列（中文分词、英文tokenize），计算其ppl（语言模型困惑度，用户表示句子的流利程度），基于循环神经网络语言模型的介绍可以[参阅论文](https://arxiv.org/abs/1409.2329)。相对于传统的方法，基于循环神经网络的方法能够更好的解决稀疏词的问题。

**目前语言模型要求使用PaddlePaddle 1.7及以上版本或适当的develop版本。**

同时推荐用户参考[IPython Notebook demo](https://aistudio.baidu.com/aistudio/projectDetail/122290)

## 2. 效果说明
在small meidum large三个不同配置情况的ppl对比：

|  small config  |    train    |   valid    |    test      |
| :------------- | :---------: | :--------: | :----------: |
|     paddle     |    40.962   |  118.111   |   112.617    |
|   tensorflow   |    40.492   |  118.329   |   113.788    |

|  medium config |    train    |   valid    |    test      |
| :------------- | :---------: | :--------: | :----------: |
|     paddle     |    45.620   |  87.398    |    83.682    |
|   tensorflow   |    45.594   |  87.363    |    84.015    |

|  large config  |    train    |   valid    |    test      |
| :------------- | :---------: | :--------: | :----------: |
|     paddle     |    37.221   |  82.358    |    78.137    |
|   tensorflow   |    38.342   |  82.311    |    78.121    |

## 3. 数据集

此任务的数据集合是采用ptb dataset，下载地址为: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz


# 快速开始

## 1. 安装说明

### Paddle安装
本项目依赖于 Paddle Fluid, 关于PaddlePaddle框架的安装教程，详见[PaddlePaddle官方网站](http://paddlepaddle.org/documentation/docs/zh/1.3/beginners_guide/install/index_cn.html)。
### 安装代码
### 环境依赖

## 2. 开始第一次模型调用

### 数据准备
为了方便开发者进行测试，我们提供了数据下载脚本。用户也可以自行下载数据，并解压。

```
cd data; sh download_data.sh
```

### 训练或fine-tune
任务训练启动命令如下：
```
sh run.sh
```
需要指定数据的目录，模型的大小(默认为small，用户可以选择medium， 或者large)。

# 进阶使用
## 1. 任务定义与建模
此任务目的是给定一个输入的词序列，预测下一个词出现的概率。

## 2. 模型原理介绍
此任务采用了序列任务常用的rnn网络，实现了一个两层的lstm网络，然后lstm的结果去预测下一个词出现的概率。

由于数据的特殊性，每一个batch的last hidden和last cell会被作为下一个batch 的init hidden 和 init cell，数据的特殊性下节会介绍。


## 3. 数据格式说明
此任务的数据格式比较简单，每一行为一个已经分好词（英文的tokenize）的词序列。

目前的句子示例如下图所示:
```
aer banknote berlitz calloway centrust cluett fromstein gitano guterman hydro-quebec ipo kia memotec mlx nahb punts rake regatta rubens sim snack-food ssangyong swapo wachter
pierre <unk> N years old will join the board as a nonexecutive director nov. N
mr. <unk> is chairman of <unk> n.v. the dutch publishing group
```

特殊说明：ptb的数据比较特殊，ptb的数据来源于一些文章，相邻的句子可能来源于一个段落或者相邻的段落，ptb 数据不能做shuffle



## 4. 目录结构

```text
.
├── README.md            # 文档
├── run.sh               # 启动脚本
├── train.py             # 训练代码
├── reader.py            # 数据读取
├── args.py              # 参数读取
└── data                 # 数据下载
../
└── models
    └── language_model
        └── lm_model.py  # 模型定义文件
```

## 5. 如何组建自己的模型
+ **自定义数据：** 关于数据，如果可以把自己的数据先进行分词（或者tokenize），然后放入到data目录下，并修改reader.py中文件的名称，如果句子之间没有关联，用户可以将`train.py`中更新的代码注释掉。
    ```
    init_hidden = np.array(fetch_outs[1])
    init_cell = np.array(fetch_outs[2])
    ```

+ **网络结构更改：** 网络只实现了基于lstm的语言模型，用户可以自己的需求更换为gru或者self等网络结构，这些实现都是在lm_model.py 中定义


# 其他

## Copyright and License
Copyright 2017 Baidu.com, Inc. All Rights Reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## 如何贡献代码

如果你可以修复某个issue或者增加一个新功能，欢迎给我们提交PR。如果对应的PR被接受了，我们将根据贡献的质量和难度进行打分（0-5分，越高越好）。如果你累计获得了10分，可以联系我们获得面试机会或者为你写推荐信。
