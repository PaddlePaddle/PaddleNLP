# ERNIE-Doc: A Retrospective Long-Document Modeling Transformer

* [模型简介](#模型简介)
* [快速开始](#快速开始)
    * [环境依赖](#环境依赖)
    * [通用参数释义](#通用参数释义)
    * [分类任务](#分类任务)
    * [阅读理解任务](#阅读理解任务)
    * [语义匹配任务](#语义匹配任务)
    * [序列标注任务](#序列标注任务)
* [致谢](#致谢)
* [参考论文](#参考论文)

## 模型简介
[ERNIE-Doc](https://arxiv.org/abs/2012.15688)是百度NLP提出的针对长文本的预训练模型。在循环Transformer机制之上，创新性地提出两阶段重复学习以及增强的循环机制，以此提高模型感受野，加强模型对长文本的理解能力。

本项目是 ERNIE-Doc 的 PaddlePaddle 动态图实现， 包含模型训练，模型验证等内容。以下是本例的简要目录结构及说明：

```text
.
├── README.md                   # 文档
├── data.py                     # 数据处理
├── metrics.py                  # ERNIE-Doc下游任务指标
├── model.py                    # 下游任务模型实现
├── optimization.py             # 优化算法
├── run_classifier.py           # 分类任务
├── run_mcq.py                  # 阅读理解任务，单项选择题
├── run_mrc.py                  # 抽取式阅读理解任务
├── run_semantic_matching.py    # 语义匹配任务
└── run_sequence_labeling.py    # 序列标注任务

```

## 快速开始

### 环境依赖

- nltk
- beautifulsoup4

安装命令：`pip install nltk==3.5 beautifulsoup4`

初次使用时，需要下载nltk的模型，可运行以下命令（下载模型可能比较慢，请耐心等待）：

```
python -c "import nltk; nltk.download('punkt')"
```

### 通用参数释义

- `model_name_or_path` 指示了Fine-tuning使用的具体预训练模型以及预训练时使用的tokenizer，目前支持的预训练模型有："ernie-doc-base-zh", "ernie-doc-base-en"。若模型相关内容保存在本地，这里也可以提供相应目录地址，例如："./checkpoint/model_xx/"。
- `dataset` 表示Fine-tuning需要加载的数据集。
- `memory_length` 表示当前的句子被截取作为下一个样本的特征的长度。
- `max_seq_length` 表示最大句子长度，超过该长度的部分将被切分成下一个样本。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔步数。
- `save_steps` 表示模型保存及评估间隔步数。
- `output_dir` 表示模型保存路径。
- `device` 表示训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。
- `seed` 表示随机数种子。
- `weight_decay` 表示AdamW的权重衰减系数。
- `warmup_proportion` 表示学习率warmup系数。
- `layerwise_decay` 表示AdamW with Layerwise decay的逐层衰减系数。

由于不同任务、不同数据集所设的超参数差别较大，可查看[ERNIE-Doc](https://arxiv.org/abs/2012.15688)论文附录中具体超参设定，此处不一一列举。

### 分类任务

分类任务支持多种数据集的评测，目前支持`imdb`, `iflytek`, `thucnews`, `hyp`四个数据集（有关数据集的描述可查看[PaddleNLP文本分类数据集](../../docs/data_prepare/dataset_list.md)）。可通过参数`dataset`指定具体的数据集，下面以`imdb`为例子运行分类任务。

#### 单卡训练

```shell
python run_classifier.py --batch_size 8 --model_name_or_path ernie-doc-base-en

```

#### 多卡训练

```shell
python -m paddle.distributed.launch --gpus "0,1" --log_dir imdb run_classifier.py --batch_size 8 --model_name_or_path ernie-doc-base-en

```

在`imdb`, `iflytek`, `thucnews`, `hyp`各数据集上Fine-tuning后，在验证集上有如下结果：

| Dataset   | Model             |      Dev ACC     |
|:---------:|:-----------------:|:----------------:|
| IMDB      | ernie-doc-base-en |      0.9506      |
| THUCNews  | ernie-doc-base-zh |      0.9854      |
| HYP       | ernie-doc-base-en |      0.7412      |
| IFLYTEK   | ernie-doc-base-zh |      0.6179      |


### 阅读理解任务

阅读理解任务支持抽取式阅读理解与单项选择题任务。

- 抽取式阅读理解

目前抽取式阅读理解支持`duredear-robust`, `drcd`,`cmrc2018`数据集。可通过参数`dataset`指定具体的数据集，下面以`dureader_robust`为例子运行抽取式阅读理解任务。

#### 单卡训练

```shell
python run_mrc.py --dataset dureader_robust --batch_size 8 --learning_rate 2.75e-4
```

#### 多卡训练

```shell
python -m paddle.distributed.launch --gpus "0,1" --log_dir dureader_robust run_mrc.py --dataset dureader_robust --batch_size 8 --learning_rate 2.75e-4
```

在`duredear-robust`, `drcd`, `cmrc2018`各数据集上Fine-tuning后，在验证集上有如下结果：

| Dataset        | Model             |      Dev EM/F1   |
|:--------------:|:-----------------:|:----------------:|
| Dureader-robust| ernie-doc-base-zh |  0.7481/0.8637   |
| DRCD           | ernie-doc-base-zh |  0.8879/0.9392   |
| CMRC2018       | ernie-doc-base-zh |  0.7061/0.9004   |


- 单项选择题

[C3](https://github.com/nlpdata/c3)是首个自由形式的多选项中文机器阅读理解数据集。该数据集每个样本提供一个上下文（文章或者对话）、问题以及至多四个答案选项，要求从答案选项中选择一个正确选项。

目前PaddleNLP提供`C3`阅读理解单项选择题数据集，可执行以下命令运行该任务。

#### 单卡训练

```shell
python run_mcq.py --batch_size 8

```

#### 多卡训练

```shell
python -m paddle.distributed.launch --gpus "0,1" --log_dir mcq run_mcq.py --batch_size 8

```

在`C3`数据集上Fine-tuning后，在验证集上有如下结果：
| Dataset        | Model             |   Dev/Test Acc   |
|:--------------:|:-----------------:|:----------------:|
| C3             | ernie-doc-base-zh |  0.7573/0.7583   |


### 语义匹配任务

[CAIL2019 SCM](https://github.com/china-ai-law-challenge/CAIL2019/tree/master/scm) 数据集是来自“中国裁判文书网”公开的法律文书,其中每份数据由三篇法律文书组成。对于每份数据，用`(A,B,C)`来代表该组数据，其中`(A,B,C)`均对应某一篇文书。该任务要求判别similarity(A, B)是否大于similarity(A, C)。

可执行以下命令运行该任务。

#### 单卡训练

```shell
python run_semantic_matching.py  --batch_size 6 --learning_rate 2e-5
```

#### 多卡训练

```shell
python -m paddle.distributed.launch --gpus "0,1" --log_dir cail run_semantic_matching.py --batch_size 6 --learning_rate 2e-5
```

在`CAIL2019-SCM`数据集上Fine-tuning后，在验证集与测试集上有如下结果：

| Dataset        | Model             |   Dev/Test Acc   |
|:--------------:|:-----------------:|:----------------:|
| CAIL2019-SCM   | ernie-doc-base-zh |  0.6420/0.6484   |


### 序列标注任务


MSRA-NER 数据集由微软亚研院发布，其目标是识别文本中具有特定意义的实体，主要包括人名、地名、机构名等。示例如下：

```
不\002久\002前\002，\002中\002国\002共\002产\002党\002召\002开\002了\002举\002世\002瞩\002目\002的\002第\002十\002五\002次\002全\002国\002代\002表\002大\002会\002。    O\002O\002O\002O\002B-ORG\002I-ORG\002I-ORG\002I-ORG\002I-ORG\002O\002O\002O\002O\002O\002O\002O\002O\002B-ORG\002I-ORG\002I-ORG\002I-ORG\002I-ORG\002I-ORG\002I-ORG\002I-ORG\002I-ORG\002I-ORG\002O
这\002次\002代\002表\002大\002会\002是\002在\002中\002国\002改\002革\002开\002放\002和\002社\002会\002主\002义\002现\002代\002化\002建\002设\002发\002展\002的\002关\002键\002时\002刻\002召\002开\002的\002历\002史\002性\002会\002议\002。    O\002O\002O\002O\002O\002O\002O\002O\002B-LOC\002I-LOC\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O
```

PaddleNLP集成的数据集MSRA-NER数据集对文件格式做了调整：每一行文本、标签以特殊字符"\t"进行分隔，每个字之间以特殊字符"\002"分隔。

可执行以下命令运行序列标注任务。

#### 单卡训练

```shell
python run_sequence_labeling.py --batch_size 8 --learning_rate 3e-5
```

#### 多卡训练

```shell
python -m paddle.distributed.launch --gpus "0,1" --log_dir msra_ner run_sequence_labeling.py --batch_size 8 --learning_rate 3e-5
```

在`MSRA-NER`数据集上Fine-tuning后，在验证集与测试集上有如下最佳结果：

| Dataset        | Model             |   Precision/Recall/F1   |
|:--------------:|:-----------------:|:-----------------------:|
| MSRA-NER       | ernie-doc-base-zh |  0.9288/0.9139/0.9213   |


## 致谢
* 感谢[百度NLP](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-doc)提供ERNIE-Doc开源代码的实现以及预训练模型。

## 参考论文

* Siyu Ding, Junyuan Shang et al. "ERNIE-Doc: A Retrospective Long-Document Modeling Transformer" ACL, 2021
