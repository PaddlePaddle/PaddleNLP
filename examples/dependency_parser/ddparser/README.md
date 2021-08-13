# DDParser

* [模型简介](#模型简介)
* [数据格式](#数据格式)
* [标注关系](#标注关系)
* [快速开始](#快速开始)
    * [环境依赖](#环境依赖)
    * [文件结构](#文件结构)
    * [一键预测](#一键预测)
    * [模型训练](#模型训练)
    * [模型评估](#模型评估)
    * [模型预测](#模型预测)
    * [可配置参数说明](#可配置参数说明)
* [致谢](#致谢)
* [参考论文](#参考论文)

## 模型简介

依存句法分析任务通过分析句子中词语之间的依存关系来确定句子的句法结构，DDParser是一款依存句法分析工具，
该用例是基于Paddle v2.1的[baidu/ddparser](https://github.com/baidu/DDParser)实现，
模型结构为[Biaffine Dependency Parser](https://arxiv.org/abs/1611.01734)。
同时本用例引入了[ERNIE](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/transformers.rst)系列预训练模型，
用户可以基于预训练模型finetune完成依存句法分析训练（参考以下[示例](####模型训练)）。

## 数据格式

本用例数据格式基于[CoNLL-X](https://ilk.uvt.nl/~emarsi/download/pubs/14964.pdf)数据格式。

示例：
```
ID      FROM   LEMMA CPOSTAG POSTAG  FEATS   HEAD    DEPREL   PROB   PDEPREL
1       百度    百度    -       -       -       2       SBV     1.0     -
2       是      是      -       -       -       0       HED     1.0     -
3       一家    一家    -       -       -       5       ATT     1.0     -
4       高科技  高科技  -       -       -       5       ATT     1.0     -
5       公司    公司    -       -       -       2       VOB     1.0     -
```

## 标注关系

标注关系说明：

| Label |  关系类型  | 说明                     | 示例                           |
| :---: | :--------: | :----------------------- | :----------------------------- |
|  SBV  |  主谓关系  | 主语与谓词间的关系       | 他送了一本书(他<--送)          |
|  VOB  |  动宾关系  | 宾语与谓词间的关系       | 他送了一本书(送-->书)          |
|  POB  |  介宾关系  | 介词与宾语间的关系       | 我把书卖了（把-->书）          |
|  ADV  |  状中关系  | 状语与中心词间的关系     | 我昨天买书了（昨天<--买）      |
|  CMP  |  动补关系  | 补语与中心词间的关系     | 我都吃完了（吃-->完）          |
|  ATT  |  定中关系  | 定语与中心词间的关系     | 他送了一本书(一本<--书)        |
|   F   |  方位关系  | 方位词与中心词的关系     | 在公园里玩耍(公园-->里)        |
|  COO  |  并列关系  | 同类型词语间关系         | 叔叔阿姨(叔叔-->阿姨)          |
|  DBL  |  兼语结构  | 主谓短语做宾语的结构     | 他请我吃饭(请-->我，请-->吃饭) |
|  DOB  | 双宾语结构 | 谓语后出现两个宾语       | 他送我一本书(送-->我，送-->书) |
|  VV   |  连谓结构  | 同主语的多个谓词间关系   | 他外出吃饭(外出-->吃饭)        |
|  IC   |  子句结构  | 两个结构独立或关联的单句 | 你好，书店怎么走？(你好<--走)  |
|  MT   |  虚词成分  | 虚词与中心词间的关系     | 他送了一本书(送-->了)          |
|  HED  |  核心关系  | 指整个句子的核心         |                                |

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

### 一键预测

使用默认模型进行一键预测：
```python
>>> from parser import Parser
>>> parser = Parser()
>>> parser.predict("百度是一家高科技公司")
[{'word': ['百度', '是', '一家', '高科技', '公司'], 'head': [2, 0, 5, 5, 2], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']}]
```

使用`ddparser-ernie-gram`进行一键预测：
```python
>>> from parser import Parser
>>> parser = Parser(encoding_model="ernie-gram-zh")
>>> parser.predict("百度是一家高科技公司")
[{'word': ['百度', '是', '一家', '高科技', '公司'], 'head': [2, 0, 5, 5, 2], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']}]
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
                                                --train_data_path=data/train.txt \
                                                --dev_data_path=data/dev.txt 
```

除了以BiLSTM作为encoder，我们还提供了`ernie-1.0`、`ernie-tiny`和`ernie-gram-zh`等预训练模型作为encoder来训练ddparser的方法。

以下是一个基于预训练模型`ernie-gram-zh`训练ddparser的示例：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run.py \
                                                --mode=train \
                                                --device=gpu \
                                                --encoding_model=ernie-gram-zh \
                                                --train_data_path=data/train.txt \
                                                --dev_data_path=data/dev.txt 
```

### 模型评估
通过`--model_file_path`指定待评估的模型文件，执行以下命令可对模型效果进行验证：
```shell
export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch --gpus "0" run.py \
                                                --mode=evaluate \
                                                --device=gpu \
                                                --model_file_path=model_file/best.pdparams \
                                                --tree
```
命令执行后返回示例：
```shell
eval loss: 0.27116, UAS: 95.747%, LAS: 94.034%
```
指标释义：
```text
UAS = number of words assigned correct head / total words
LAS = number of words assigned correct head and relation / total words
```

### 模型预测
用户可以执行一下命令进行模型预测，通过`--test_data_path`指定待预测数据，`--model_file_path`来指定模型文件，`--infer_result_dir`指定预测结果存放路径。
```shell
export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch --gpus "0" run.py \
                                                --mode=predict \
                                                --device=gpu \
                                                --test_data_path=data/test.txt \
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

## 致谢

* 感谢[百度NLP](https://github.com/baidu/DDParser)提供ddparser的开源代码实现。

## 参考论文

- [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)
