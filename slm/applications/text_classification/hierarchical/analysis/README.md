# 训练评估与模型优化指南

**目录**
   * [Analysis 模块介绍](#Analysis 模块介绍)
   * [环境准备](#环境准备)
   * [模型评估](#模型评估)
   * [可解释性分析](#可解释性分析)
        * [单词级别可解释性分析](#单词级别可解释性分析)
        * [句子级别可解释性分析](#句子级别可解释性分析)
   * [数据优化](#数据优化)
        * [稀疏数据筛选方案](#稀疏数据筛选方案)
        * [脏数据清洗方案](#脏数据清洗方案)
        * [数据增强策略方案](#数据增强策略方案)

## Analysis 模块介绍

Analysis 模块提供了**模型评估、可解释性分析、数据优化**等功能，旨在帮助开发者更好地分析文本分类模型预测结果和对模型效果进行优化。

- **模型评估：** 对整体分类情况和每个类别分别进行评估，并打印预测错误样本，帮助开发者分析模型表现找到训练和预测数据中存在的问题。

- **可解释性分析：** 基于[TrustAI](https://github.com/PaddlePaddle/TrustAI)提供单词和句子级别的模型可解释性分析，帮助理解模型预测结果。

- **数据优化：** 结合[TrustAI](https://github.com/PaddlePaddle/TrustAI)和[数据增强 API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md)提供了**稀疏数据筛选、脏数据清洗、数据增强**三种优化策略，从多角度优化训练数据提升模型效果。

<div align="center">
    <img src="https://user-images.githubusercontent.com/63761690/195241942-70068989-df17-4f53-9f71-c189d8c5c88d.png" width="600">
</div>

以下是本项目主要代码结构及说明：

```text
analysis/
├── evaluate.py # 评估脚本
├── sent_interpret.py # 句子级别可解释性分析脚本
├── word_interpret.py # 单词级别可解释性分析notebook
├── sparse.py # 稀疏数据筛选脚本
├── dirty.py # 脏数据清洗脚本
├── aug.py # 数据增强脚本
└── README.md # 训练评估与模型优化指南
```

## 环境准备
需要可解释性分析和数据优化需要安装相关环境。
- trustai >= 0.1.7
- interpretdl >= 0.7.0

**安装 TrustAI**（可选）如果使用可解释性分析和数据优化中稀疏数据筛选和脏数据清洗需要安装 TrustAI。
```shell
pip install trustai==0.1.7
```

**安装 InterpretDL**（可选）如果使用词级别可解释性分析 GradShap 方法，需要安装 InterpretDL
```shell
pip install interpretdl==0.7.0
```

## 模型评估

我们使用训练好的模型计算模型的在开发集的准确率，同时打印每个类别数据量及表现：

```shell
python evaluate.py \
    --device "gpu" \
    --dataset_dir "../data" \
    --params_path "../checkpoint" \
    --max_seq_length 128 \
    --batch_size 32 \
    --bad_case_file "bad_case.txt"
```

默认在 GPU 环境下使用，在 CPU 环境下修改参数配置为`--device "cpu"`

可支持配置的参数：

* `device`: 选用什么设备进行训练，可选择 cpu、gpu、xpu、npu；默认为"gpu"。
* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含 dev.txt 和 label.txt 文件;默认为 None。
* `params_path`：保存训练模型的目录；默认为"../checkpoint/"。
* `max_seq_length`：分词器 tokenizer 使用的最大序列长度，ERNIE 模型最大不能超过2048。请根据文本长度选择，通常推荐128、256或512，若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `dev_file`：本地数据集中开发集文件名；默认为"dev.txt"。
* `label_file`：本地数据集中标签集文件名；默认为"label.txt"。
* `bad_case_path`：开发集中预测错误样本保存路径；默认为"/bad_case.txt"。


输出打印示例：

```text
[2022-08-11 03:10:14,058] [    INFO] - -----Evaluate model-------

[2022-08-11 03:10:14,059] [    INFO] - Dev dataset size: 1498
[2022-08-11 03:10:14,059] [    INFO] - Accuracy in dev dataset: 89.19%
[2022-08-11 03:10:14,059] [    INFO] - Macro avg in dev dataset: precision: 93.48 | recall: 93.26 | F1 score 93.22
[2022-08-11 03:10:14,059] [    INFO] - Micro avg in dev dataset: precision: 95.07 | recall: 95.46 | F1 score 95.26
[2022-08-11 03:10:14,095] [    INFO] - Level 1 Label Performance: Macro F1 score: 96.39 | Micro F1 score: 96.81 | Accuracy: 94.93
[2022-08-11 03:10:14,255] [    INFO] - Level 2 Label Performance: Macro F1 score: 92.79 | Micro F1 score: 93.90 | Accuracy: 89.72
[2022-08-11 03:10:14,256] [    INFO] - Class name: 交往
[2022-08-11 03:10:14,256] [    INFO] - Evaluation examples in dev dataset: 60(4.0%) | precision: 91.94 | recall: 95.00 | F1 score 93.44
[2022-08-11 03:10:14,256] [    INFO] - ----------------------------
[2022-08-11 03:10:14,256] [    INFO] - Class name: 交往##会见
[2022-08-11 03:10:14,256] [    INFO] - Evaluation examples in dev dataset: 12(0.8%) | precision: 92.31 | recall: 100.00 | F1 score 96.00
...
```

预测错误的样本保存在 bad_case.txt 文件中：

```text
Text	Label	Prediction
据猛龙随队记者JoshLewenberg报道，消息人士透露，猛龙已将前锋萨加巴-科纳特裁掉。此前他与猛龙签下了一份Exhibit10合同。在被裁掉后，科纳特下赛季大概率将前往猛龙的发展联盟球队效力。	组织关系,组织关系##加盟,组织关系##裁员	组织关系,组织关系##解雇
冠军射手被裁掉，欲加入湖人队，但湖人却无意，冠军射手何去何从	组织关系,组织关系##裁员	组织关系,组织关系##解雇
6月7日报道，IBM将裁员超过1000人。IBM周四确认，将裁减一千多人。据知情人士称，此次裁员将影响到约1700名员工，约占IBM全球逾34万员工中的0.5%。IBM股价今年累计上涨16%，但该公司4月发布的财报显示，一季度营收下降5%，低于市场预期。	组织关系,组织关系##裁员	组织关系,组织关系##裁员,财经/交易
有多名魅族员工表示，从6月份开始，魅族开始了新一轮裁员，重点裁员区域是营销和线下。裁员占比超过30%，剩余员工将不过千余人，魅族的知名工程师，爱讲真话的洪汉生已经从钉钉里退出了，外界传言说他去了OPPO。	组织关系,组织关系##退出,组织关系##裁员	组织关系,组织关系##裁员
...
```

## 可解释性分析
"模型为什么会预测出这个结果?"是文本分类任务开发者时常遇到的问题，如何分析错误样本(bad case)是文本分类任务落地中重要一环，本项目基于 TrustAI 开源了基于词级别和句子级别的模型可解释性分析方法，帮助开发者更好地理解文本分类模型与数据，有助于后续的模型优化与数据清洗标注。

### 单词级别可解释性分析
本项目开源模型的词级别可解释性分析 Notebook，提供 LIME、Integrated Gradient、GradShap 三种分析方法，支持分析微调后模型的预测结果，开发者可以通过更改**数据目录**和**模型目录**在自己的任务中使用 Jupyter Notebook 进行数据分析。

运行 `word_interpret.ipynb`代码，即可分析影响样本预测结果的关键词以及可视化所有词对预测结果的贡献情况，颜色越深代表这个词对预测结果影响越大：
<div align="center">
    <img src="https://user-images.githubusercontent.com/63761690/195334753-78cc2dc8-a5ba-4460-9fde-3b1bb704c053.png" width="1000">
</div>

### 句子级别可解释性分析
本项目基于特征相似度（[FeatureSimilarity](https://arxiv.org/abs/2104.04128)）算法，计算对样本预测结果正影响的训练数据，帮助理解模型的预测结果与训练集数据的关系。

待分析数据文件`interpret_input_file`应为以下三种格式中的一种：
**格式一：包括文本、标签、预测结果**
```text
<文本>'\t'<标签>'\t'<预测结果>
...
```

**格式二：包括文本、标签**
```text
<文本>'\t'<标签>
...
```

**格式三：只包括文本**
```text
<文本>
准予原告胡某甲与被告韩某甲离婚。
...
```

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

默认在 GPU 环境下使用，在 CPU 环境下修改参数配置为`--device "cpu"`

可支持配置的参数：

* `device`: 选用什么设备进行训练，可可选择 cpu、gpu、xpu、npu；默认为"gpu"。
* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含 dev.txt 和 label.txt 文件;默认为 None。
* `params_path`：保存训练模型的目录；默认为"../checkpoint/"。
* `max_seq_length`：分词器 tokenizer 使用的最大序列长度，ERNIE 模型最大不能超过2048。请根据文本长度选择，通常推荐128、256或512，若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `seed`：随机种子，默认为3。
* `top_k`：筛选支持训练证据数量；默认为3。
* `train_file`：本地数据集中训练集文件名；默认为"train.txt"。
* `interpret_input_file`：本地数据集中待分析文件名；默认为"bad_case.txt"。
* `interpret_result_file`：保存句子级别可解释性结果文件名；默认为"sent_interpret.txt"。

可解释性结果保存在 `interpret_result_file` 文件中：
```text
text: 据猛龙随队记者JoshLewenberg报道，消息人士透露，猛龙已将前锋萨加巴-科纳特裁掉。此前他与猛龙签下了一份Exhibit10合同。在被裁掉后，科纳特下赛季大概率将前往猛龙的发展联盟球队效力。
predict label: 组织关系,组织关系##解雇
label: 组织关系,组织关系##加盟,组织关系##裁员
examples with positive influence
support1 text: 尼克斯官方今日宣布，他们已经裁掉了前锋扎克-欧文，后者昨日才与尼克斯签约。	label: 组织关系,组织关系##加盟,组织关系##解雇	score: 0.99357
support2 text: 活塞官方今日宣布，他们已经签下了克雷格-斯沃德，并且裁掉了托德-威瑟斯。	label: 组织关系,组织关系##加盟,组织关系##解雇	score: 0.98344
support3 text: 孟菲斯灰熊今年宣布，球队已经签下后卫达斯蒂-汉纳斯（DustyHannahs，版头图）并裁掉马特-穆尼。	label: 组织关系,组织关系##加盟,组织关系##解雇	score: 0.98219
...
```

## 数据优化

### 稀疏数据筛选方案

稀疏数据筛选适用于文本分类中**数据不平衡或训练数据覆盖不足**的场景，简单来说，就是由于模型在训练过程中没有学习到足够与待预测样本相似的数据，模型难以正确预测样本所属类别的情况。稀疏数据筛选旨在开发集中挖掘缺乏训练证据支持的数据，通常可以采用**数据增强**或**少量数据标注**的两种低成本方式，提升模型在开发集的预测效果。

本项目中稀疏数据筛选基于 TrustAI，利用基于特征相似度的实例级证据分析方法，抽取开发集中样本的支持训练证据，并计算支持证据平均分（通常为得分前三的支持训练证据均分）。分数较低的样本表明其训练证据不足，在训练集中较为稀疏，实验表明模型在这些样本上表现也相对较差。更多细节详见[TrustAI](https://github.com/PaddlePaddle/TrustAI)和[实例级证据分析](https://github.com/PaddlePaddle/TrustAI/blob/main/trustai/interpretation/example_level/README.md)。


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

默认在 GPU 环境下使用，在 CPU 环境下修改参数配置为`--device "cpu"`

可支持配置的参数：

* `device`: 选用什么设备进行训练，可选择 cpu、gpu、xpu、npu；默认为"gpu"。
* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含 dev.txt 和 label.txt 文件;默认为 None。
* `aug_strategy`：数据增强类型，可选"duplicate","substitute", "insert", "delete", "swap"；默认为"substitute"。
* `params_path`：保存训练模型的目录；默认为"../checkpoint/"。
* `max_seq_length`：分词器 tokenizer 使用的最大序列长度，ERNIE 模型最大不能超过2048。请根据文本长度选择，通常推荐128、256或512，若出现显存不足，请适当调低这一参数；默认为128。
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

我们在[2020语言与智能技术竞赛：事件抽取任务](https://aistudio.baidu.com/aistudio/competition/detail/32/0/introduction)抽取部分训练数据（训练集数据规模:700）进行实验,筛选稀疏数据数量和筛选支持数据数量均设为100条，使用不同的数据增强方法进行评测：

|  |Micro F1(%)   | Macro F1(%) |
| ---------| ------------ |------------ |
|训练集|90.41|79.16|
|训练集+支持增强集(duplicate) |**90.60**|80.55|
|训练集+支持增强集(substitute) |90.21|80.11|
|训练集+支持增强集(insert) |90.53|**80.61**|
|训练集+支持增强集(delete) |90.56| 80.26|
|训练集+支持增强集(swap) |90.18|80.05|

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

默认在 GPU 环境下使用，在 CPU 环境下修改参数配置为`--device "cpu"`

可支持配置的参数：

* `device`: 选用什么设备进行训练，可选择 cpu、gpu、xpu、npu；默认为"gpu"。
* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含 dev.txt 和 label.txt 文件;默认为 None。
* `annotate`：选择稀疏数据识别--数据标注模式；默认为 False。
* `params_path`：保存训练模型的目录；默认为"../checkpoint/"。
* `max_seq_length`：分词器 tokenizer 使用的最大序列长度，ERNIE 模型最大不能超过2048。请根据文本长度选择，通常推荐128、256或512，若出现显存不足，请适当调低这一参数；默认为128。
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

我们将筛选出的支持数据`support.txt`进行标注，可以使用标注工具帮助更快标注，详情请参考[文本分类任务 doccano 数据标注使用指南](../../doccano.md)进行文本分类数据标注。然后将已标注数据`support.txt`与训练集数据`train.txt`合并得到新的训练集`train_sparse_annotate.txt`重新进行训练：

```shell
cat ../data/train.txt ../data/support.txt > ../data/train_sparse_annotate.txt
```

**方案效果**

我们在[2020语言与智能技术竞赛：事件抽取任务](https://aistudio.baidu.com/aistudio/competition/detail/32/0/introduction)抽取部分训练数据（训练集数据规模:700）进行实验,筛选稀疏数据数量设为100条，筛选待标注数据数量为50和100条。我们比较了使用稀疏数据方案的策略采样和随机采样的效果，下表结果表明使用稀疏数据方案的策略采样能够有效指导训练数据扩充，在标注更少的数据情况下获得更大提升的效果：

|  |Micro F1(%)   | Macro F1(%) |
| ---------| ------------ | ------------ |
|训练集|90.41|79.16|
|训练集+策略采样集(50) |90.79|82.37|
|训练集+随机采样集(50) |90.10|79.27|
|训练集+策略采样集(100) |91.12|**84.13**|
|训练集+随机采样集(100) |**91.24**|81.66|

### 脏数据清洗方案

脏数据清洗方案是基于已训练好的文本分类模型，筛选出训练数据集中标注错误的数据，再由人工检查重新标注，获得标注正确的数据集进行重新训练。我们将介绍脏数据清洗流程：

- **脏数据筛选：** 基于 TrustAI 中表示点方法，计算训练数据对文本分类模型的影响分数，分数高的训练数据表明对模型影响大，这些数据有较大概率为标注错误样本，记为脏数据集（Dirty Dataset）。

- **数据清洗、训练：** 将筛选出的脏数据由人工重新检查，为数据打上正确的标签。将清洗后的训练数据重新放入文本分类模型进行训练。

现在我们进行脏数据识别，脏数据保存在`"train_dirty.txt"`,剩余训练数据保存在`"train_dirty_rest.txt"`：

```shell
python dirty.py \
    --device "gpu" \
    --dataset_dir "../data" \
    --max_seq_length 128 \
    --params_path "../checkpoint/" \
    --batch_size 16 \
    --dirty_num 100 \
    --dirty_file "train_dirty.txt" \
    --rest_file "train_dirty_rest.txt"
```

默认在 GPU 环境下使用，在 CPU 环境下修改参数配置为`--device "cpu"`

可支持配置的参数：

* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含 train.txt 和 label.txt 文件;默认为 None。
* `max_seq_length`：分词器 tokenizer 使用的最大序列长度，ERNIE 模型最大不能超过2048。请根据文本长度选择，通常推荐128、256或512，若出现显存不足，请适当调低这一参数；默认为128。
* `params_path`：保存训练模型的目录；默认为"../checkpoint/"。
* `device`: 选用什么设备进行训练，可选择 cpu、gpu、xpu、npu；默认为"gpu"。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `seed`：随机种子，默认为3。
* `dirty_file`：保存脏数据文件名，默认为"train_dirty.txt"。
* `rest_file`：保存剩余数据（非脏数据）文件名，默认为"train_dirty_rest.txt"。
* `train_file`：本地数据集中训练集文件名；默认为"train.txt"。
* `dirty_threshold`：筛选脏数据用于重新标注的阈值，只选择影响分数大于阈值作为支持数据，默认为0。


我们将筛选出脏数据进行人工检查重新标注，可以将`train_dirty.txt`直接导入标注工具 doccano 帮助更快重新标注，详情请参考[文本分类任务 doccano 数据标注使用指南](../../doccano.md)进行文本分类数据标注。然后将已重新标注的脏数据`train_dirty.txt`与剩余训练集数据`train_dirty_rest.txt`合并得到新的训练集`train_clean.txt`重新进行训练：

```shell
cat ../data/train_dirty_rest.txt ../data/train_dirty.txt > ../data/train_clean.txt
```

**方案效果**

我们在[2020语言与智能技术竞赛：事件抽取任务](https://aistudio.baidu.com/aistudio/competition/detail/32/0/introduction)抽取部分训练数据（训练集数据规模:2000）进行实验，取200条数据进行脏数据处理，也即200条训练数据为标签错误数据，选择不同`dirty_num`应用脏数据清洗策略进行评测：

|  |Micro F1(%)   | Macro F1(%) |
| ---------| ------------ |------------ |
|训练集(2000)|92.54|86.04|
|训练集(2000，含200条脏数据) |89.11|73.33|
|训练集(2000，含200条脏数据) + 脏数据清洗(50)|90.00|77.67|
|训练集(2000，含200条脏数据) + 脏数据清洗(100)|92.48|**87.83**|
|训练集(2000，含200条脏数据) + 脏数据清洗(150)|**92.55**|83.73|

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
* `aug_type`：词替换/词插入增强类型，可选"synonym", "homonym", "mlm"，建议在 GPU 环境下使用 mlm 类型；默认为"synonym"。
* `create_n`：生成的句子数量，默认为2。
* `aug_percent`：生成词替换百分比，默认为0.1。
* `device`: 选用什么设备进行增强，可选择 cpu、gpu、xpu、npu，仅在使用 mlm 类型有影响；默认为"gpu"。

生成的增强数据保存在`"aug.txt"`文件中，与训练集数据`train.txt`合并得到新的训练集`train_aug.txt`重新进行训练：

```shell
cat ../data/aug.txt ../data/train.txt > ../data/train_aug.txt
```

PaddleNLP 内置多种数据增强策略，更多数据增强策略使用方法请参考[数据增强 API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md)。

**方案效果**

我们在[2020语言与智能技术竞赛：事件抽取任务](https://aistudio.baidu.com/aistudio/competition/detail/32/0/introduction)抽取部分训练数据（训练集数据规模:2000）进行实验，采用不同数据增强策略进行两倍数据增强（每条样本生成两条增强样本）：

|  |Micro F1(%)   | Macro F1(%) |
| ---------| ------------ |------------ |
|训练集(2000)|92.54|86.04|
|训练集(2000)+数据增强(×2, mix) |93.23|89.69|
|训练集(2000)+支持增强集(×2, substitute) |93.07|89.49|
|训练集(2000)+支持增强集(×2, insert) |**93.63**|**89.69**|
|训练集(2000)+支持增强集(×2, delete) |91.53| 84.47|
|训练集(2000)+支持增强集(×2, swap) |93.24|89.02|
