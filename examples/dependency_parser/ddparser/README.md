# DDParser

* [模型简介](#模型简介)
* [数据格式](#数据格式)
* [快速开始](#快速开始)
    * [环境依赖](#环境依赖)
    * [文件结构](#文件结构)
    * [数据准备](#数据准备)
    * [模型训练](#模型训练)
    * [模型评估](#模型评估)
    * [模型预测](#模型预测)
    * [可配置参数说明](#可配置参数说明)
* [致谢](#致谢)
* [Reference](#Reference)

## 模型简介

依存句法分析任务通过分析句子中词语之间的依存关系来确定句子的句法结构，
该用例是基于Paddle v2.1的[baidu/ddparser](https://github.com/baidu/DDParser)实现，
模型结构为[Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)。
同时本用例引入了[ERNIE](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/transformers.rst)系列预训练模型，
用户可以基于预训练模型finetune完成依存句法分析训练（参考以下[示例](#模型训练)）。

### 模型效果
以下展示了该用例在清华大学语义依存网络语料（THU）和哈尔滨工业大学依存网络语料（HIT）开发集上的效果验证，数据集获取方式参考[数据准备](#数据准备)。

#### THU开发集

| 模型名称                   |  UAS  |   LAS |
| ------------------------- | :---: | ----: |
| `biaffine-dep`            | 82.93 | 74.07 |
| `biaffine-dep-lstm-pe`    | 80.02 | 70.17 |
| `biaffine-dep-ernie-tiny` | 89.02 | 81.39 |
| `biaffine-dep-ernie-1.0`  | 92.25 | 84.77 |
| `biaffine-dep-ernie-gram` | 92.20 | 85.10 |

#### HIT开发集

| 模型名称                   |  UAS  |   LAS |
| ------------------------- | :---: | ----: |
| `biaffine-dep`            | 83.52 | 68.25 |
| `biaffine-dep-lstm-pe`    | 77.48 | 61.34 |
| `biaffine-dep-ernie-tiny` | 83.75 | 69.01 |
| `biaffine-dep-ernie-1.0`  | 89.21 | 74.37 |
| `biaffine-dep-ernie-gram` | 89.59 | 74.75 |

其中`lstm-pe`表示lstm by positional encoding，`biaffine-dep`使用了句子的word级表示和pos词性标签，其他模型使用句子的word级表示和char级表示。在小数据集上使用`lstm`作为encoder进行训练时建议增加pos词性标签作为数据输入以增强模型效果。

指标释义：
```text
UAS = number of words assigned correct head / total words
LAS = number of words assigned correct head and relation / total words
```

## 数据格式

本用例数据格式基于[CoNLL-X](https://ilk.uvt.nl/~emarsi/download/pubs/14964.pdf)。

| 名称 | 含义 |
| --- | --- |
| ID |  单词ID，序号从1开始 |
| FORM | 当前单词 |  
| LEMMA | 当前词语的原型或词干，在中文中此列与FORM相同 |  
| CPOSTAG | 当前词语的词性（粗粒度） |
| POSTAG | 当前词语的词性（细粒度） |
| FEATS | 句法特征 | 
| HEAD | 当前单词的中心词 | 
| DEPREL | 当前单词与中心词的依存关系 |
| PHEAD | 当前单词的主观中心词 |
| PDEPREL | 当前单词与主观中心词的依存关系 |

THU数据集示例：
```
ID      FROM   LEMMA CPOSTAG POSTAG  FEATS   HEAD    DEPREL 
1       世界    世界    n       n       _       5       限定
2       第      第      m       m       _       4       限定
3       八      八      m       m       _       2       连接依存
4       大      大      a       a       _       5       限定
5       奇迹    奇迹    n       n       _       6       存现体
6       出现    出现    v       v       _       0       核心成分
```

HIT数据集示例：
```
1       城建    城建    NN      NN      _       2       relevant        _    _
2       成为    成为    VV      VV      _       0       ROOT            _    _
3       外商    外商    NN      NN      _       4       agent           _    _
4       投资    投资    VV      VV      _       7       d-restrictive   _    _
5       青海    青海    NR      NR      _       4       patient         _    _
6       新      新      JJ      JJ     _       7       d-attribute     _    _
7       热点    热点    NN      NN      _       2       isa             _    _
```

- 该用例中用户只需关注`FORM`、`POSTTAG`、`HEAD`和`DEPREL`这几列信息即可，'_'表示数值不可用。

## 快速开始

### 环境依赖
* `LAC`
* `dill`

安装命令：`pip install LAC dill`

### 文件结构

```text
ddparser/
├── model # 部署
│   ├── dropouts.py # dropout
│   ├── encoder.py # 编码器
│   └── dep.py # 模型网络
├── README.md # 使用说明
├── criterion.py # 损失函数
├── data.py # 数据结构
├── env.py # 环境配置工具
├── metric.py # 指标计算
├── parser.py # 一键预测工具
├── run.py # 主入口，包含训练、评估和预测任务
└── utils.py # 工具函数
```

### 数据准备

该用例使用的是[第二届自然语言处理与中文计算会议（NLP&CC 2013）](http://tcci.ccf.org.cn/conference/2013/pages/page04_sam.html)
提供的数据集，其中`THU`文件夹为清华大学语义依存网络语料，`HIT`文件夹为哈尔滨工业大学依存网络语料。
下载并解压[数据集](http://tcci.ccf.org.cn/conference/2013/dldoc/evsam05.zip)，
将`THU`和`HIT`文件夹分别放置在当前路径的`./data`路径下。

`./data`路径结构如下：

```text
data/
├── HIT # 清华大学语义依存网络语料
│   ├── train.conll # 训练集
│   └── dev.conll # 开发集
└── THU # 哈尔滨工业大学依存网络语料
    ├── train.conll # 训练集
    └── dev.conll # 开发集
```

### 模型训练

通过指定`--preprocess`，任务会基于训练数据自动生成词表和关系表等信息并保存`fields`文件到`--save_dir`所指定的路径下。

用户可以通过`--feat`来指定输入的特征，`--encoding_model`指定不同的encoder，

以下是以BiLSTM为encoder训练ddparser的示例：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run.py \
    --mode=train \
    --preprocess \
    --device=gpu \
    --save_dir=model_file \
    --encoding_model=lstm \
    --feat=pos \
    --train_data_path=data/THU/train.conll \
    --dev_data_path=data/THU/dev.conll
```

除了以BiLSTM作为encoder，我们还提供了`ernie-1.0`、`ernie-tiny`和`ernie-gram-zh`等预训练模型作为encoder来训练ddparser的方法。

以下是一个基于预训练模型`ernie-gram-zh`训练ddparser的示例：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run.py \
    --mode=train \
    --device=gpu \
    --encoding_model=ernie-gram-zh \
    --train_data_path=data/THU/train.conll \
    --dev_data_path=data/THU/dev.conll 
```

### 模型评估
通过`--model_file_path`指定待评估的模型文件，执行以下命令可对模型效果进行验证：
```shell
export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch --gpus "0" run.py \
    --mode=evaluate \
    --device=gpu \
    --model_file_path=model_file/best.pdparams \
    --test_data_path=data/THU/dev.conll
    --tree
```
命令执行后返回示例：
```shell
eval loss: 0.27116, UAS: 82.69%, LAS: 73.66%
```

### 模型预测
用户可以执行一下命令进行模型预测，通过`--test_data_path`指定待预测数据，`--model_file_path`指定模型文件，`--infer_result_dir`指定预测结果存放路径。
```shell
export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch --gpus "0" run.py \
    --mode=predict \
    --device=gpu \
    --test_data_path=data/THU/dev.conll \
    --model_file_path=model_file/best.pdparams \
    --infer_result_dir=infer_result \
    --tree
```
命令执行后会在`infer_result`路径下生成预测结果文件。

### 可配置参数说明

* `mode`: 任务模式，可选为train、evaluate和predict。
* `device`: 选用什么设备进行训练，可选cpu、gpu或xpu。如使用gpu训练则参数gpus指定GPU卡号。
* `encoding_model`: 选择模型编码网络，可选lstm、lstm-pe、ernie-1.0、ernie-tiny和ernie-gram-zh。
* `preprocess`: 训练模式下的使用参数，设置表示会基于训练数据进行词统计等操作，不设置则使用已统计好的信息；针对统一训练数据，多次训练可不设置该参数; 默认为True。
* `epochs`: 训练轮数。
* `save_dir`: 保存训练模型的路径；默认将当前在验证集上效果最好的模型保存在目录model_file文件夹下。
* `train_data_path`: 训练集文件路径。
* `dev_data_path`: 开发集文件路径。
* `batch_size`: 批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数，默认为1000。
* `init_from_params`: 模型参数路径，热启动模型训练；默认为None。
* `clip`: 梯度裁剪阈值，将梯度限制在阈值范围内。
* `lstm_lr`: 模型编码网络为lstm或lstm-pe时的学习率，默认为0.002。
* `ernie_lr`: 模型编码网络为ernie-1.0、ernie-tiny、ernie-gram-zh时的学习率，默认为5e-5。
* `seed`: 随机种子，默认为1000。
* `test_data_path`: 测试集文件路径。
* `model_file_path`: 评估和预测模式下的使用参数，设置后会从该路径加载已训练保存的模型文件进行模型评估或预测，默认为model_file文件夹。
* `infer_result_dir`: 预测结果保存路径，默认保存在当前目录infer_result文件夹下。
* `min_freq`: 训练模式下的使用参数，基于训练数据生成的词表的最小词频，默认为2。
* `n_buckets`: 训练模式下的使用参数，选择数据分桶数，对训练数据按照长度进行分桶。
* `tree`: 确保输出结果是正确的依存句法树，默认为True。
* `feat`: 模型编码网络为lstm时的使用参数，选择输入的特征，可选char（句子的char级表示）和pos（词性标签）；ernie类别的模型只能为None。
* `warmup_proportion`: 学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0.0。
* `weight_decay`: 控制正则项力度的参数，用于防止过拟合，默认为0.0。

## Reference

- [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)
- [baidu/ddparser](https://github.com/baidu/DDParser)