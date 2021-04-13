# ELMo

## 模型简介

ELMo(Embeddings from Language Models)是重要的通用语义表示模型之一，以双向LSTM为网络基本组件，以Language Model为训练目标，通过预训练得到通用的语义表示，ELMo能够学习到复杂的特征，比如语法、语义，并且能够学习在不同上下文情况下的词汇多义性。将ELMo得到的语义表示作为Feature迁移到下游NLP任务中，会显著提升下游任务的模型性能，比如问答、文本蕴含和情感分析等。ELMo模型的细节可以[参阅论文](https://arxiv.org/abs/1802.05365)。

本项目是ELMo在Paddle上的开源实现, 基于1 Billion Word Language Model Benchmark进行预训练，并接入了简单的下游任务作为示例程序。

接入的下游任务是在sentence polarity dataset v1数据集上构建的文本二分类任务，采用ELMo + BoW的简单网络结构。与base模型（Word2Vec + BoW）进行精度对比。

| 模型  | test acc |
| ---- | -------- |
| word2vec + BoW  | 0.7769   |
| ELMo + BoW  | 0.7760   |

## 环境依赖

- sklearn
- gensim

安装方式：`pip install sklearn gensim`

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
.
├── args.py # 运行参数配置文件
├── dataset.py # 数据读取
├── elmo.py # 模型组网
├── run_pretrain.py # 训练模型主程序入口
├── run_eval.py # 评估模型主程序入口
├── word2vec_base.py # 下游二分类任务base模型训练测试主程序入口
├── run_finetune.py # 下游二分类任务训练测试主程序入口
├── download_data.sh # 数据下载脚本
└── README.md # 文档说明
```

### 数据准备

运行下载数据的脚本后，会生成两个文件，1-billion-word目录下会存在训练数据目录（training-tokenized-shuffled）、测试集数据（heldout-tokenized-shuffled）以及对应的词典（vocab-15w.txt），sentence-polarity-dataset-v1目录下会存在未切分的正向样本（rt-polarity.pos）、负向样本（rt-polarity.neg）以及Google预训练好的Word2Vec向量文件GoogleNews-vectors-negative300.bin.gz。

```shell
sh download_data.sh
```

1-billion-word目录结构：

```text
.
├── training-tokenized-shuffled # 训练集
├── heldout-tokenized-shuffled # 测试集
└── vocab-15w.txt # 词典
```

sentence-polarity-dataset-v1目录结构：

```text
.
├── rt-polarity.pos # 正向样本
├── rt-polarity.neg # 负向样本
└── GoogleNews-vectors-negative300.bin.gz # 预训练好的Word2Vec向量
```

### 模型训练

基于1-billion-word数据集，可以运行下面的命令，在训练集上进行模型训练
```shell
# GPU启动, 支持单卡和多卡
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus '0' run_pretrain.py --train_data_path='./1-billion-word/training-tokenized-shuffled/*' --vocab_file='./1-billion-word/vocab-15w.txt' --save_dir='./checkpoints' --device='gpu'
```

其他可选参数和参数的默认值请参考`args.py`。

程序运行时将会自动开始训练，同时训练过程中会自动保存模型在指定的`save_dir`中。
如：
```text
checkpoints/
├── 10000.pdopt
├── 10000.pdparams
├── 20000.pdopt
├── 20000.pdparams
├── ...
├── final.pdopt
└── final.pdparams
```

**NOTE:** 如需恢复模型训练，则init_from_ckpt只需指定到文件名即可，不需要添加文件尾缀。如`--init_from_ckpt=checkpoints/10000`即可，程序会自动加载模型参数`checkpoints/10000.pdparams`，也会自动加载优化器状态`checkpoints/10000.pdopt`。

### 模型评估

基于1-billion-word数据集，可以运行下面的命令，在评测集上进行模型评估
```shell
# GPU启动，仅支持单卡
export CUDA_VISIBLE_DEVICES=0
python run_eval.py --dev_data_path='./1-billion-word/heldout-tokenized-shuffled/*' --vocab_file='./1-billion-word/vocab-15w.txt' --init_from_ckpt='./checkpoints/10000' --device='gpu'
```

### 下游任务

下游任务是基于sentence polarity dataset v1数据集的二分类任务，base模型采用Word2Vec + BoW的模型结构，其中Word2Vec采用Google预训练好的GoogleNews-vectors-negative300.bin.gz。

#### base模型

base模型可以运行下面的命令，在训练集上进行模型训练评估
```shell
# GPU启动, 支持单卡和多卡
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus '0' word2vec_base.py --data_dir='./sentence-polarity-dataset-v1/' --pretrained_word2vec_file='./sentence-polarity-dataset-v1/GoogleNews-vectors-negative300.bin' --device='gpu'
```

#### ELMo finetune

ELMo finetune可以运行下面的命令，在训练集上进行模型训练评估
```shell
# GPU启动, 支持单卡和多卡
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus '0' run_finetune.py --data_dir='./sentence-polarity-dataset-v1/' --init_from_ckpt='./checkpoints/10000' --device='gpu'
```

**NOTE:** 可以通过构建模型时的trainable参数设置ELMo参与或不参与下游任务的训练。ELMo接入下游任务的具体用法请参考`run_finetune.py`。

另外，预训练的ELMo也可以作为文本词向量编码器单独使用，即输入文本内容，输出每个词对应的词向量。用法示例如下：

```python
from elmo import ELMoEmbedder

embedder = ELMoEmbedder(params_file)
sentences = [['The', 'first', 'sentence', '.'], ['Second', 'one', '.']]

embeddings = embedder.encode(sentences)
for i, (text, emb) in enumerate(zip(sentences, embeddings)):
    print(text)
    print(emb.shape)
    print()
```

## Reference

- [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
