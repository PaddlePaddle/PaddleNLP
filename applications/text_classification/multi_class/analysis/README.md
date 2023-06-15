# 训练评估与模型优化指南

**目录**
   * [Analysis模块介绍](#Analysis模块介绍)
   * [环境准备](#环境准备)
   * [可解释性分析](#可解释性分析)
        * [单词级别可解释性分析](#单词级别可解释性分析)
        * [句子级别可解释性分析](#句子级别可解释性分析)
   * [数据优化](#数据优化)
        * [稀疏数据筛选方案](#稀疏数据筛选方案)
        * [脏数据清洗方案](#脏数据清洗方案)
        * [数据增强策略方案](#数据增强策略方案)

## Analysis模块介绍

Analysis模块提供了**可解释性分析、数据优化**等功能，旨在帮助开发者更好地分析文本分类模型预测结果和对模型效果进行优化。


- **可解释性分析：** 基于[TrustAI](https://github.com/PaddlePaddle/TrustAI)提供单词和句子级别的模型可解释性分析，帮助理解模型预测结果。

- **数据优化：** 结合[TrustAI](https://github.com/PaddlePaddle/TrustAI)和[数据增强API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md)提供了**稀疏数据筛选、脏数据清洗、数据增强**三种优化策略，从多角度优化训练数据提升模型效果。

<div align="center">
    <img src="https://user-images.githubusercontent.com/63761690/195241942-70068989-df17-4f53-9f71-c189d8c5c88d.png" width="600">
</div>

以下是本项目主要代码结构及说明：

```text
analysis/
├── sent_interpret.ipynb # 句子级别可解释性分析脚本
├── word_interpret.py # 单词级别可解释性分析notebook
├── sparse.py # 稀疏数据筛选脚本
├── dirty.py # 脏数据清洗脚本
├── aug.py # 数据增强脚本
└── README.md # 训练评估与模型优化指南
```

## 环境准备
需要可解释性分析和数据优化需要安装相关环境。
- trustai >= 0.1.12
- interpretdl >= 0.7.0

**安装TrustAI**（可选）如果使用可解释性分析和数据优化中稀疏数据筛选和脏数据清洗需要安装TrustAI。
```shell
pip install trustai==0.1.12
```

**安装InterpretDL**（可选）如果使用词级别可解释性分析GradShap方法，需要安装InterpretDL
```shell
pip install interpretdl==0.7.0
```

## 可解释性分析
"模型为什么会预测出这个结果?"是文本分类任务开发者时常遇到的问题，如何分析错误样本(bad case)是文本分类任务落地中重要一环，本项目基于TrustAI开源了基于词级别和句子级别的模型可解释性分析方法，帮助开发者更好地理解文本分类模型与数据，有助于后续的模型优化与数据清洗标注。

### 单词级别可解释性分析
本项目开源模型的词级别可解释性分析Notebook，提供LIME、Integrated Gradient、GradShap 三种分析方法，支持分析微调后模型的预测结果，开发者可以通过更改**数据目录**和**模型目录**在自己的任务中使用Jupyter Notebook进行数据分析。

运行 [word_interpret.ipynb](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/applications/text_classification/multi_class/analysis/README.md) 代码，即可分析影响样本预测结果的关键词以及可视化所有词对预测结果的贡献情况，颜色越深代表这个词对预测结果影响越大：
<div align="center">
    <img src="https://user-images.githubusercontent.com/63761690/192739675-63145d59-23c6-416f-bf71-998fd4995254.png" width="1000">
</div>

### 句子级别可解释性分析
本项目基于特征相似度（[FeatureSimilarity](https://arxiv.org/abs/2104.04128)）算法，计算对样本预测结果正影响的训练数据，帮助理解模型的预测结果与训练集数据的关系。


我们可以运行代码，得到支持样本模型预测结果的训练数据：
```shell
python sent_interpret.py \
    --device "gpu" \
    --dataset_dir "../data" \
    --params_path "../checkpoint/" \
    --max_seq_length 128 \
    --batch_size 16 \
    --top_k 3 \
    --train_file "train.txt" \
    --interpret_input_file "bad_case.txt" \
    --interpret_result_file "sent_interpret.txt"
```

默认在GPU环境下使用，在CPU环境下修改参数配置为`--device "cpu"`

可支持配置的参数：

* `device`: 选用什么设备进行训练，可选择cpu、gpu、xpu、npu；默认为"gpu"。
* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含dev.txt和label.txt文件;默认为None。
* `params_path`：保存训练模型的目录；默认为"../checkpoint/"。
* `max_seq_length`：分词器tokenizer使用的最大序列长度，ERNIE模型最大不能超过2048。请根据文本长度选择，通常推荐128、256或512，若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `seed`：随机种子，默认为3。
* `top_k`：筛选支持训练证据数量；默认为3。
* `train_file`：本地数据集中训练集文件名；默认为"train.txt"。
* `interpret_input_file`：本地数据集中待分析文件名；默认为"bad_case.txt"。
* `interpret_result_file`：保存句子级别可解释性结果文件名；默认为"sent_interpret.txt"。

## 数据优化

### 稀疏数据筛选方案

稀疏数据筛选适用于文本分类中**数据不平衡或训练数据覆盖不足**的场景，简单来说，就是由于模型在训练过程中没有学习到足够与待预测样本相似的数据，模型难以正确预测样本所属类别的情况。稀疏数据筛选旨在开发集中挖掘缺乏训练证据支持的数据，通常可以采用**数据增强**或**少量数据标注**的两种低成本方式，提升模型在开发集的预测效果。

本项目中稀疏数据筛选基于TrustAI，利用基于特征相似度的实例级证据分析方法，抽取开发集中样本的支持训练证据，并计算支持证据平均分（通常为得分前三的支持训练证据均分）。分数较低的样本表明其训练证据不足，在训练集中较为稀疏，实验表明模型在这些样本上表现也相对较差。更多细节详见[TrustAI](https://github.com/PaddlePaddle/TrustAI)和[实例级证据分析](https://github.com/PaddlePaddle/TrustAI/blob/main/trustai/interpretation/example_level/README.md)。


#### 稀疏数据识别—数据增强

这里我们将介绍稀疏数据识别—数据增强流程：

- **稀疏数据识别：** 挖掘开发集中的缺乏训练证据支持数据，记为稀疏数据集（Sparse Dataset）；

- **数据增强**：将稀疏数据集在训练集中的支持证据应用数据增强策略，这些数据增强后的训练数据记为支持数据集（Support Dataset）；

- **重新训练模型：** 将支持数据集加入到原有的训练集获得新的训练集，重新训练新的文本分类模型。

现在我们进行稀疏数据识别-数据增强，得到支持数据集：

```shell
python sparse.py \
    --device "gpu" \
    --dataset_dir "../data" \
    --aug_strategy "substitute" \
    --max_seq_length 128 \
    --params_path "../checkpoint/" \
    --batch_size 16 \
    --sparse_num 100 \
    --support_num 100
```

默认在GPU环境下使用，在CPU环境下修改参数配置为`--device "cpu"`

可支持配置的参数：

* `device`: 选用什么设备进行训练，可选择cpu、gpu、xpu、npu；默认为"gpu"。
* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含dev.txt和label.txt文件;默认为None。
* `aug_strategy`：数据增强类型，可选"duplicate","substitute", "insert", "delete", "swap"；默认为"substitute"。
* `params_path`：保存训练模型的目录；默认为"../checkpoint/"。
* `max_seq_length`：分词器tokenizer使用的最大序列长度，ERNIE模型最大不能超过2048。请根据文本长度选择，通常推荐128、256或512，若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `seed`：随机种子，默认为3。
* `rationale_num_sparse`：筛选稀疏数据时计算样本置信度时支持训练证据数量；认为3。
* `rationale_num_support`：筛选支持数据时计算样本置信度时支持训练证据数量，如果筛选的支持数据不够，可以适当增加；默认为6。
* `sparse_num`：筛选稀疏数据数量，建议为开发集的10%~20%，默认为100。
* `support_num`：用于数据增强的支持数据数量，建议为训练集的10%~20%，默认为100。
* `support_threshold`：支持数据的阈值，只选择支持证据分数大于阈值作为支持数据，默认为0.7。
* `train_file`：本地数据集中训练集文件名；默认为"train.txt"。
* `dev_file`：本地数据集中开发集文件名；默认为"dev.txt"。
* `label_file`：本地数据集中标签集文件名；默认为"label.txt"。
* `sparse_file`：保存在本地数据集路径中稀疏数据文件名；默认为"sparse.txt"。
* `support_file`：保存在本地数据集路径中支持训练数据文件名；默认为"support.txt"。

将得到增强支持数据`support.txt`与训练集数据`train.txt`合并得到新的训练集`train_sparse_aug.txt`重新进行训练：

```shell
cat ../data/train.txt ../data/support.txt > ../data/train_sparse_aug.txt
```

**方案效果**

我们在KUAKE-QIC数据集部分数据（训练集数据规模:500）进行实验,筛选稀疏数据数量和支持数据数量均设为100条，使用不同的数据增强方法进行评测：
|  |Accuracy(%)   |
| ---------| ------------ |
|训练集|73.50|
|训练集+支持增强集(duplicate) |73.61|
|训练集+支持增强集(substitute) |**74.32**|
|训练集+支持增强集(insert) |73.81|
|训练集+支持增强集(delete) |74.27|
|训练集+支持增强集(swap) |73.66|

#### 稀疏数据识别-数据标注

本方案能够有针对性进行数据标注，相比于随机标注数据更好提高模型预测效果。这里我们将介绍稀疏数据识别-数据标注流程：

- **稀疏数据识别：** 挖掘开发集中的缺乏训练证据支持数据，记为稀疏数据集（Sparse Dataset）；

- **数据标注**：在未标注数据集中筛选稀疏数据集的支持证据，并进行数据标注，记为支持数据集（Support Dataset）；

- **重新训练模型：** 将支持数据集加入到原有的训练集获得新的训练集，重新训练新的文本分类模型。

现在我们进行稀疏数据识别--数据标注，得到待标注数据：

```shell
python sparse.py \
    --annotate \
    --device "gpu" \
    --dataset_dir "../data" \
    --max_seq_length 128 \
    --params_path "../checkpoint/" \
    --batch_size 16 \
    --sparse_num 100 \
    --support_num 100 \
    --unlabeled_file "data.txt"
```

默认在GPU环境下使用，在CPU环境下修改参数配置为`--device "cpu"`

可支持配置的参数：

* `device`: 选用什么设备进行训练，可选择cpu、gpu、xpu、npu；默认为"gpu"。
* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含dev.txt和label.txt文件;默认为None。
* `annotate`：选择稀疏数据识别--数据标注模式；默认为False。
* `params_path`：保存训练模型的目录；默认为"../checkpoint/"。
* `max_seq_length`：分词器tokenizer使用的最大序列长度，ERNIE模型最大不能超过2048。请根据文本长度选择，通常推荐128、256或512，若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `seed`：随机种子，默认为3。
* `rationale_num_sparse`：筛选稀疏数据时计算样本置信度时支持训练证据数量；认为3。
* `rationale_num_support`：筛选支持数据时计算样本置信度时支持训练证据数量，如果筛选的支持数据不够，可以适当增加；默认为6。
* `sparse_num`：筛选稀疏数据数量，建议为开发集的10%~20%，默认为100。
* `support_num`：用于数据增强的支持数据数量，建议为训练集的10%~20%，默认为100。
* `support_threshold`：支持数据的阈值，只选择支持证据分数大于阈值作为支持数据，默认为0.7。
* `train_file`：本地数据集中训练集文件名；默认为"train.txt"。
* `dev_file`：本地数据集中开发集文件名；默认为"dev.txt"。
* `label_file`：本地数据集中标签集文件名；默认为"label.txt"。
* `unlabeled_file`：本地数据集中未标注数据文件名；默认为"data.txt"。
* `sparse_file`：保存在本地数据集路径中稀疏数据文件名；默认为"sparse.txt"。
* `support_file`：保存在本地数据集路径中支持训练数据文件名；默认为"support.txt"。

我们将筛选出的支持数据`support.txt`进行标注，可以使用标注工具帮助更快标注，详情请参考[文本分类任务doccano数据标注使用指南](../../doccano.md)进行文本分类数据标注。然后将已标注数据`support.txt`与训练集数据`train.txt`合并得到新的训练集`train_sparse_annotate.txt`重新进行训练：

```shell
cat ../data/train.txt ../data/support.txt > ../data/train_sparse_annotate.txt
```

**方案效果**

我们在KUAKE-QIC数据集部分数据（训练集数据规模:500）进行实验,筛选稀疏数据数量设为100条，筛选待标注数据数量为50和100条。我们比较了使用稀疏数据方案的策略采样和随机采样的效果，下表结果表明使用稀疏数据方案的策略采样能够支持指导训练数据扩充，在标注更少的数据情况下获得更大提升的效果：

|  |Accuracy(%)   |
| ---------| ------------ |
|训练集|73.50|
|训练集+策略采样集(50) |76.88|
|训练集+随机采样集(50) |74.32|
|训练集+策略采样集(100) |**77.64**|
|训练集+随机采样集(100) |76.37|

### 脏数据清洗方案

脏数据清洗方案是基于已训练好的文本分类模型，筛选出训练数据集中标注错误的数据，再由人工检查重新标注，获得标注正确的数据集进行重新训练。我们将介绍脏数据清洗流程：

- **脏数据筛选：** 基于TrustAI中表示点方法，计算训练数据对文本分类模型的影响分数，分数高的训练数据表明对模型影响大，这些数据有较大概率为标注错误样本，记为脏数据集（Dirty Dataset）。

- **数据清洗、训练：** 将筛选出的脏数据由人工重新检查，为数据打上正确的标签。将清洗后的训练数据重新放入文本分类模型进行训练。

现在我们进行脏数据识别，脏数据保存在`"train_dirty.txt"`,剩余训练数据保存在`"train_dirty_rest.txt"`：

```shell
python dirty.py \
    --device "gpu:3" \
    --dataset_dir "../data" \
    --max_seq_length 128 \
    --params_path "../checkpoint/" \
    --batch_size 8 \
    --dirty_num 100 \
    --dirty_file "train_dirty.txt" \
    --rest_file "train_dirty_rest.txt"
```

默认在GPU环境下使用，在CPU环境下修改参数配置为`--device "cpu"`

可支持配置的参数：

* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含train.txt和label.txt文件;默认为None。
* `max_seq_length`：分词器tokenizer使用的最大序列长度，ERNIE模型最大不能超过2048。请根据文本长度选择，通常推荐128、256或512，若出现显存不足，请适当调低这一参数；默认为128。
* `params_path`：保存训练模型的目录；默认为"../checkpoint/"。
* `device`: 选用什么设备进行训练，可选择cpu、gpu、xpu、npu；默认为"gpu"。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `seed`：随机种子，默认为3。
* `dirty_file`：保存脏数据文件名，默认为"train_dirty.txt"。
* `rest_file`：保存剩余数据（非脏数据）文件名，默认为"train_dirty_rest.txt"。
* `train_file`：本地数据集中训练集文件名；默认为"train.txt"。
* `dirty_threshold`：筛选脏数据用于重新标注的阈值，只选择影响分数大于阈值作为支持数据，默认为0。


我们将筛选出脏数据进行人工检查重新标注，可以将`train_dirty.txt`直接导入标注工具doccano帮助更快重新标注，详情请参考[文本分类任务doccano数据标注使用指南](../../doccano.md)进行文本分类数据标注。然后将已重新标注的脏数据`train_dirty.txt`与剩余训练集数据`train_dirty_rest.txt`合并得到新的训练集`train_clean.txt`重新进行训练：

```shell
cat ../data/train_dirty_rest.txt ../data/train_dirty.txt > ../data/train_clean.txt
```

**方案效果**

我们在KUAKE-QIC数据集部分数据（训练集数据规模:500）进行实验，取100条数据进行脏数据处理，也即100条训练数据为标签错误数据，选择不同`dirty_num`应用脏数据清洗策略进行评测：

|  |Accuracy(%)   |
| ---------| ------------ |
|训练集(500)|**73.50**|
|训练集(500，含100条脏数据) |65.58|
|训练集(500，含100条脏数据) + 脏数据清洗(50)|68.90|
|训练集(500，含100条脏数据) + 脏数据清洗(100)|69.36|
|训练集(500，含100条脏数据) + 脏数据清洗(150)|73.15|

### 数据增强策略方案

在数据量较少或某些类别样本量较少时，也可以通过数据增强策略的方式，生成更多的训练数据，提升模型效果。

```shell
python aug.py \
    --create_n 2 \
    --aug_percent 0.1 \
    --train_path "../data/train.txt" \
    --aug_path "../data/aug.txt"
```

可支持配置的参数：

* `train_path`：待增强训练数据集文件路径；默认为"../data/train.txt"。
* `aug_path`：增强生成的训练数据集文件路径；默认为"../data/train_aug.txt"。
* `aug_strategy`：数据增强策略，可选"mix", "substitute", "insert", "delete", "swap","mix"为多种数据策略混合使用；默认为"substitute"。
* `aug_type`：词替换/词插入增强类型，可选"synonym", "homonym", "mlm"，建议在GPU环境下使用mlm类型；默认为"synonym"。
* `create_n`：生成的句子数量，默认为2。
* `aug_percent`：生成词替换百分比，默认为0.1。
* `device`: 选用什么设备进行增强，可选择cpu、gpu、xpu、npu，仅在使用mlm类型有影响；默认为"gpu"。

生成的增强数据保存在`"aug.txt"`文件中，与训练集数据`train.txt`合并得到新的训练集`train_aug.txt`重新进行训练：

```shell
cat ../data/aug.txt ../data/train.txt > ../data/train_aug.txt
```

PaddleNLP内置多种数据增强策略，更多数据增强策略使用方法请参考[数据增强API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md)。
