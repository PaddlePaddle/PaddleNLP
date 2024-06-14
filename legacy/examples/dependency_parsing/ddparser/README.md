# DDParser

 - [模型简介](#模型简介)
 - [快速开始](#快速开始)
    - [模型效果](#模型效果)
    - [数据格式](#数据格式)
    - [数据准备](#数据准备)
    - [文件结构](#文件结构)
    - [模型训练、预测与部署](#模型训练、预测与部署)
 - [Taskflow一键预测](#Taskflow一键预测)
 - [Reference](#Reference)

## 模型简介

依存句法分析任务通过分析句子中词语之间的依存关系来确定句子的句法结构，
该项目是基于Paddle v2.1的[baidu/ddparser](https://github.com/baidu/DDParser)实现，
模型结构为[Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)。
同时本项目引入了[ERNIE](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer) 系列预训练模型，
用户可以基于预训练模型finetune完成依存句法分析训练（参考以下[示例](#模型训练)）。

## 快速开始

本项目展示了基于NLPCC2013_EVSAM05_THU和NLPCC2013_EVSAM05_HIT数据集的进行模型训练、预测和部署的示例。

### 模型效果

以下是NLPCC2013_EVSAM05_THU和NLPCC2013_EVSAM05_HIT数据集的模型性能对比，baseline为第二届自然语言处理与中文计算会议发布的[评测报告](http://tcci.ccf.org.cn/conference/2013/dldoc/evrpt05.rar)。

#### NLPCC2013_EVSAM05_THU

| model                     | dev UAS | dev LAS | test UAS | test LAS |
| ------------------------- | :-----: | :------:| :-------:| :-------:|
| `baseline`                |  81.49  |  72.17  |  84.68   |  76.02   |
| `biaffine-dep(+char)`     |  84.11  |  75.16  |  85.31   |  76.73   |
| `biaffine-dep(+pos)`      |  83.28  |  74.20  |  84.54   |  75.33   |
| `biaffine-dep-lstm-pe`    |  81.02  |  71.20  |  82.86   |  73.86   |
| `biaffine-dep-ernie-tiny` |  89.02  |  81.39  |  89.31   |  81.51   |
| `biaffine-dep-ernie-1.0`  |  92.25  |  84.77  |  92.12   |  84.62   |
| `biaffine-dep-ernie-gram` |  92.20  |  85.10  |  91.96   |  84.10   |

#### NLPCC2013_EVSAM05_HIT

| model                     | dev UAS | dev LAS | test UAS | test LAS |
| ------------------------- | :-----: | :------:| :-------:| :-------:|
| `baseline`                |  82.96  |  65.45  |  82.65   |  65.25   |
| `biaffine-dep(+char)`     |  80.90  |  65.29  |  80.77   |  65.43   |
| `biaffine-dep(+pos)`      |  83.85  |  68.27  |  83.75   |  68.04   |
| `biaffine-dep-lstm-pe`    |  77.48  |  61.34  |  76.41   |  60.32   |
| `biaffine-dep-ernie-tiny` |  84.21  |  68.89  |  83.98   |  68.67   |
| `biaffine-dep-ernie-1.0`  |  89.24  |  74.12  |  88.64   |  74.09   |
| `biaffine-dep-ernie-gram` |  89.59  |  74.75  |  88.79   |  74.46   |

其中`lstm-pe`表示lstm by positional encoding，`biaffine-dep`的模型输入可以选择句子的word级表示加char级表示（`biaffine-dep(+char)`）或者句子的word级表示加上pos词性标签（`biaffine-dep(+pos)`），其他模型使用句子的word级表示和char级表示。

指标释义：
```text
UAS (依存准确率) = number of words assigned correct head / total words
LAS (依存标注准备率) = number of words assigned correct head and relation / total words
```

### 数据格式

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

NLPCC2013_EVSAM05_THU数据集示例：
```
ID      FROM   LEMMA CPOSTAG POSTAG  FEATS   HEAD    DEPREL
1       世界    世界    n       n       _       5       限定
2       第      第      m       m       _       4       限定
3       八      八      m       m       _       2       连接依存
4       大      大      a       a       _       5       限定
5       奇迹    奇迹    n       n       _       6       存现体
6       出现    出现    v       v       _       0       核心成分
```

NLPCC2013_EVSAM05_HIT数据集示例：
```
ID      FROM   LEMMA CPOSTAG POSTAG  FEATS   HEAD     DEPREL        PHEAD PDEPREL
1       城建    城建    NN      NN      _       2       relevant        _    _
2       成为    成为    VV      VV      _       0       ROOT            _    _
3       外商    外商    NN      NN      _       4       agent           _    _
4       投资    投资    VV      VV      _       7       d-restrictive   _    _
5       青海    青海    NR      NR      _       4       patient         _    _
6       新      新      JJ      JJ     _       7       d-attribute     _    _
7       热点    热点    NN      NN      _       2       isa             _    _
```

- 该用例中用户只需关注`FORM`、`POSTTAG`、`HEAD`和`DEPREL`这几列信息即可，'_'表示数值不可用。

### 数据准备

该用例使用的是[第二届自然语言处理与中文计算会议（NLP&CC 2013）](http://tcci.ccf.org.cn/conference/2013/pages/page04_sam.html)
提供的数据集，其中`NLPCC2013_EVSAM05_THU`为清华大学语义依存网络语料，`NLPCC2013_EVSAM05_HIT`为哈尔滨工业大学依存网络语料。

为了方便用户的快速使用，PaddleNLP Dataset API内置了数据集，一键可完成数据集加载。

加载`NLPCC2013_EVSAM05_THU`数据集：
```python
from paddlenlp.datasets import load_dataset
train_ds, dev_ds, test_ds = load_dataset("nlpcc13_evsam05_thu", splits=["train", "dev", "test"])
```

加载`NLPCC2013_EVSAM05_HIT`数据集：
```python
from paddlenlp.datasets import load_dataset
train_ds, dev_ds, test_ds = load_dataset("nlpcc13_evsam05_hit", splits=["train", "dev", "test"])
```

### 文件结构

以下是本项目主要代码结构及说明：

```text
ddparser/
├── deploy # 部署
│   └── python
│       └── predict.py # python预测部署示例
├── model
│   ├── dropouts.py # dropout
│   ├── encoder.py # 编码器
│   └── dep.py # 模型网络
├── README.md # 使用说明
├── export_model.py # 模型导出脚本
├── criterion.py # 损失函数
├── data.py # 数据结构
├── metric.py # 指标计算
├── train.py # 训练脚本
├── predict.py # 预测脚本
└── utils.py # 工具函数
```

### 模型训练、预测与部署

本项目提供了三种模型结构：LSTMEncoder+MLP+BiAffine、LSTMByWPEncoder+MLP+BiAffine和ErnieEncoder+MLP+BiAffine，用户可通过`--encoding_model`指定所使用的模型结构。

#### LSTMEncoder+MLP+BiAffine

##### 启动训练

通过如下命令，指定GPU 0卡，以`lstm`为encoder在`nlpcc13_evsam05_thu`数据集上训练与评估：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" train.py \
    --device=gpu \
    --epochs=100 \
    --task_name=nlpcc13_evsam05_thu \
    --save_dir=./model_file \
    --encoding_model=lstm \
    --feat=pos \
    --batch_size=1000 \
    --lstm_lr=0.002
```

##### 基于动态图的预测

```shell
export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch --gpus "0" predict.py \
    --device=gpu \
    --task_name=nlpcc13_evsam05_thu \
    --encoding_model=lstm \
    --feat=pos \
    --params_path=./model_file/best.pdparams \
    --infer_output_file=infer_output.conll
```

**NOTE**: 预测时的`encoding_model`和`feat`需要与训练时的参数保持一致。

#### LSTMByWPEncoder+MLP+BiAffine

##### 启动训练

通过如下命令，指定GPU 0卡，以`lstm-pe`为encoder在`nlpcc13_evsam05_hit`数据集上训练与评估：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" train.py \
    --device=gpu \
    --epochs=100 \
    --task_name=nlpcc13_evsam05_hit \
    --encoding_model=lstm-pe \
    --save_dir=./model_file \
    --lstm_lr=0.002
```

##### 基于动态图的预测

```shell
export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch --gpus "0" predict.py \
    --device=gpu \
    --task_name=nlpcc13_evsam05_hit \
    --encoding_model=lstm-pe \
    --params_path=./model_file/best.pdparams \
    --infer_output_file=infer_output.conll
```

##### 基于静态图的预测部署

使用动态图训练结束后，可以将动态图参数导出成静态图参数， 从而获得较优的预测部署性能，执行如下命令完成动态图转换静态图的功能：

```shell
python export_model.py --encoding_model=lstm-pe \
    --params_path=./model_file/best.pdparams \
    --output_path=./output
```

导出静态图模型之后，可以用于部署，`deploy/python/predict.py`脚本提供了python部署预测示例。运行方式：
```shell
python deploy/python/predict.py --model_dir=./output --task_name=nlpcc13_evsam05_hit
```

#### ErnieEncoder+MLP+BiAffine

##### 启动训练

通过如下命令，指定GPU 0卡，以预训练模型`ernie-gram-zh`为encoder在`nlpcc13_evsam05_hit`数据集上训练与评估：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" train.py \
    --device=gpu \
    --epochs=100 \
    --task_name=nlpcc13_evsam05_hit \
    --encoding_model=ernie-gram-zh \
    --save_dir=./model_file \
    --ernie_lr=5e-5
```

##### 基于动态图的预测

```shell
export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch --gpus "0" predict.py \
    --device=gpu \
    --task_name=nlpcc13_evsam05_hit \
    --encoding_model=ernie-gram-zh \
    --params_path=./model_file/best.pdparams \
    --infer_output_file=infer_output.conll
```

##### 基于静态图的预测部署

使用动态图训练结束后，可以将动态图参数导出成静态图参数， 从而获得较优的预测部署性能，执行如下命令完成动态图转换静态图的功能：

```shell
python export_model.py --encoding_model=ernie-gram-zh \
    --params_path=./model_file/best.pdparams \
    --output_path=./output
```

导出静态图模型之后，可以用于部署，`deploy/python/predict.py`脚本提供了python部署预测示例。运行方式：
```shell
python deploy/python/predict.py --model_dir=./output --task_name=nlpcc13_evsam05_hit
```

#### 参数释义

项目中的参数具体说明如下：

* `device`: 选用什么设备进行训练，可选cpu、gpu。
* `task_name`: 选择训练所用的数据集，可选nlpcc13_evsam05_thu和nlpcc13_evsam05_hit。
* `encoding_model`: 选择模型编码网络，可选lstm、lstm-pe、ernie-1.0、ernie-tiny和ernie-gram-zh。
* `epochs`: 训练轮数。
* `save_dir`: 保存训练模型的路径；默认将当前在验证集上LAS指标最高的模型`best.pdparams`和训练最近一个epoch的模型`last_epoch.pdparams`保存在目录model_file文件夹下。
* `batch_size`: 批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数，默认为1000。
* `init_from_params`: 模型参数路径，热启动模型训练；默认为None。
* `clip`: 梯度裁剪阈值，将梯度限制在阈值范围内。
* `lstm_lr`: 模型编码网络为lstm或lstm-pe时的学习率，默认为0.002。
* `ernie_lr`: 模型编码网络为ernie-1.0、ernie-tiny、ernie-gram-zh时的学习率，默认为5e-5。
* `seed`: 随机种子，默认为1000。
* `min_freq`: 训练模式下的使用参数，基于训练数据生成的词表的最小词频，默认为2。
* `n_buckets`: 选择数据分桶数，对训练数据按照长度进行分桶。
* `tree`: 确保输出结果是正确的依存句法树，默认为True。
* `feat`: 模型编码网络为lstm时的使用参数，选择输入的特征，可选char（句子的char级表示）和pos（词性标签）；ernie类别的模型只能为None。
* `warmup_proportion`: 学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0.0。
* `weight_decay`: 控制正则项力度的参数，用于防止过拟合，默认为0.0。

## Taskflow一键预测

Taskflow向用户提供了一个百度基于大规模标注数据集[DuCTB1.0](#数据来源)训练的依存句法分析工具ddparser。用户可以方便地使用该工具完成[一键预测](#一键预测)。

### 环境依赖

- LAC >= 2.1
- matplotlib >= 3.4.2

### 一键预测

```python
from paddlenlp import Taskflow

ddp = Taskflow("dependency_parsing")
ddp("百度是一家高科技公司")
'''
[{'word': ['百度', '是', '一家', '高科技', '公司'],
  'head': ['2', '0', '5', '5', '2'],
  'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']}]
'''
ddp(["百度是一家高科技公司", "他送了一本书"])
'''
[{'word': ['百度', '是', '一家', '高科技', '公司'],
  'head': ['2', '0', '5', '5', '2'],
  'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']},
 {'word': ['他', '送', '了', '一本', '书'],
  'head': ['2', '0', '2', '5', '2'],
  'deprel': ['SBV', 'HED', 'MT', 'ATT', 'VOB']}]
'''

# 输出概率和词性标签
ddp = Taskflow("dependency_parsing", prob=True, use_pos=True)
ddp("百度是一家高科技公司")
'''
[{'word': ['百度', '是', '一家', '高科技', '公司'],
  'postag': ['ORG', 'v', 'm', 'n', 'n'],
  'head': ['2', '0', '5', '5', '2'],
  'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB'],
  'prob': [1.0, 1.0, 1.0, 1.0, 1.0]}]
'''

# 使用ddparser-ernie-1.0进行预测
ddp = Taskflow("dependency_parsing", model="ddparser-ernie-1.0")
ddp("百度是一家高科技公司")
'''
[{'word': ['百度', '是', '一家', '高科技', '公司'],
  'head': ['2', '0', '5', '5', '2'],
  'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']}]
'''

# 使用ddparser-ernie-gram-zh进行预测
ddp = Taskflow("dependency_parsing", model="ddparser-ernie-gram-zh")
ddp("百度是一家高科技公司")
'''
[{'word': ['百度', '是', '一家', '高科技', '公司'],
  'head': ['2', '0', '5', '5', '2'],
  'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']}]
'''
```

### 依存关系可视化

```python
from paddlenlp import Taskflow

ddp = Taskflow("dependency_parsing", return_visual=True)
result = ddp("百度是一家高科技公司")[0]['visual']
import cv2
cv2.imwrite('test.png', result)
```

### 标注关系说明

DuCTB1.0数据集含14种标注关系，具体含义见下表：

| Label |  关系类型  | 说明                     | 示例                           |
| :---: | :--------: | :----------------------- | :----------------------------- |
|  SBV  |  主谓关系  | 主语与谓词间的关系       | 他送了一本书(他<--送)          |
|  VOB  |  动宾关系  | 宾语与谓词间的关系       | 他送了一本书(送-->书)          |
|  POB  |  介宾关系  | 介词与宾语间的关系       | 我把书卖了（把-->书）          |
|  ADV  |  状中关系  | 状语与中心词间的关系     | 我昨天买书了（昨天<--买）      |
|  CMP  |  动补关系  | 补语与中心词间的关系     | 我都吃完了（吃-->完）          |
|  ATT  |  定中关系  | 定语与中心词间的关系     | 他送了一本书(一本<--书)        |
|   F   |  方位关系  | 方位词与中心词的关系     | 在公园里玩耍(公园-->里)        |
|  COO  |  并列关系  | 同类型词语间关系        | 叔叔阿姨(叔叔-->阿姨)          |
|  DBL  |  兼语结构  | 主谓短语做宾语的结构     | 他请我吃饭(请-->我，请-->吃饭) |
|  DOB  | 双宾语结构 | 谓语后出现两个宾语       | 他送我一本书(送-->我，送-->书) |
|  VV   |  连谓结构  | 同主语的多个谓词间关系   | 他外出吃饭(外出-->吃饭)        |
|  IC   |  子句结构  | 两个结构独立或关联的单句  | 你好，书店怎么走？(你好<--走)  |
|  MT   |  虚词成分  | 虚词与中心词间的关系     | 他送了一本书(送-->了)          |
|  HED  |  核心关系  | 指整个句子的核心         |                               |

### 数据来源

**DuCTB1.0**: `Baidu Chinese Treebank1.0`是百度构建的中文句法树库，即Taskflow所提供的依存句法分析工具-DDParser的训练数据来源。

## Reference

- [baidu/ddparser](https://github.com/baidu/DDParser)
- [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)
