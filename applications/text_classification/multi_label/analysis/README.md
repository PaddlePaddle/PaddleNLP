# 多标签训练评估与模型优化指南

**目录**
   * [analysis模块介绍](#analysis模块介绍)
   * [模型评估](#模型评估)
   * [稀疏数据筛选方案](#稀疏数据筛选方案)
   * [脏数据清洗方案](#脏数据清洗方案)
   * [数据增强策略方案](#数据增强策略方案)

## analysis模块介绍

analysis模块提供了**模型评估**脚本对整体分类情况和每个类别分别进行评估，并打印预测错误样本，帮助开发者分析模型表现找到训练和预测数据中存在的问题问题。同时基于[可信AI工具集](https://github.com/PaddlePaddle/TrustAI)和[数据增强API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md)提供了**稀疏数据筛选、脏数据清洗、数据增强**三种优化方案从多角度帮助开发者提升模型效果。

<div align="center">
    <img src="https://user-images.githubusercontent.com/63761690/184790352-7f91da02-aadd-48fe-9130-8323a89c6a43.jpg" width="600">
</div>

以下是本项目主要代码结构及说明：

```text
analysis/
├── evaluate.py # 评估脚本
├── sparse.py # 稀疏数据筛选脚本
├── dirty.py # 脏数据清洗脚本
├── aug.py # 数据增强脚本
└── README.md # 多标签训练评估与模型优化指南
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
    --bad_case_path "./bad_case.txt"
```

默认在GPU环境下使用，在CPU环境下修改参数配置为`--device "cpu"`

可支持配置的参数：

* `device`: 选用什么设备进行训练，选择cpu、gpu、xpu、npu；默认为"gpu"。
* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含train.txt、dev.txt和label.txt文件;默认为None。
* `params_path`：保存训练模型的目录；默认为"../checkpoint/"。
* `max_seq_length`：分词器tokenizer使用的最大序列长度，ERNIE模型最大不能超过2048。请根据文本长度选择，通常推荐128、256或512，若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `train_file`：本地数据集中开发集文件名；默认为"train.txt"。
* `dev_file`：本地数据集中开发集文件名；默认为"dev.txt"。
* `label_file`：本地数据集中标签集文件名；默认为"label.txt"。
* `bad_case_path`：开发集中预测错误样本保存路径；默认为"/bad_case.txt"。

输出打印示例：

```text
[2022-08-12 02:24:48,193] [    INFO] - -----Evaluate model-------
[2022-08-12 02:24:48,194] [    INFO] - Train dataset size: 14377
[2022-08-12 02:24:48,194] [    INFO] - Dev dataset size: 1611
[2022-08-12 02:24:48,194] [    INFO] - Accuracy in dev dataset: 74.24%
[2022-08-12 02:24:48,194] [    INFO] - Macro avg in dev dataset: precision: 82.96 | recall: 77.59 | F1 score 79.36
[2022-08-12 02:24:48,194] [    INFO] - Micro avg in dev dataset: precision: 91.50 | recall: 89.66 | F1 score 90.57
[2022-08-12 02:24:48,195] [    INFO] - Class name: 婚后有子女
[2022-08-12 02:24:48,195] [    INFO] - Evaluation examples in train dataset: 6759(47.0%) | precision: 99.78 | recall: 99.59 | F1 score 99.68
[2022-08-12 02:24:48,195] [    INFO] - Evaluation examples in dev dataset: 784(48.7%) | precision: 97.07 | recall: 97.32 | F1 score 97.20
[2022-08-12 02:24:48,195] [    INFO] - ----------------------------
[2022-08-12 02:24:48,195] [    INFO] - Class name: 限制行为能力子女抚养
[2022-08-12 02:24:48,195] [    INFO] - Evaluation examples in train dataset: 4358(30.3%) | precision: 99.36 | recall: 99.56 | F1 score 99.46
[2022-08-12 02:24:48,195] [    INFO] - Evaluation examples in dev dataset: 492(30.5%) | precision: 88.57 | recall: 88.21 | F1 score 88.39
...
```

预测错误的样本保存在bad_case.txt文件中：

```text
Prediction    Label    Text
不动产分割    不动产分割,有夫妻共同财产    2014年，王X以其与肖X协议离婚时未分割该套楼房的首付款为由，起诉至法院，要求分得楼房的首付款15万元。
婚后分居,准予离婚    二次起诉离婚,准予离婚,婚后分居,法定离婚    但原、被告对已建立起的夫妻感情不够珍惜，因琐事即发生吵闹并最终分居，对夫妻感情造成了严重的影响，现原、被告已分居六年有余，且经人民法院判决不准离婚后仍未和好，夫妻感情确已破裂，依法应准予原、被告离婚。
婚后有子女,限制行为能力子女抚养    婚后有子女    婚后生有一女，取名彭某乙，已11岁，现已由被告从铁炉白族乡中心小学转入走马镇李桥小学读书。
婚后分居    不履行家庭义务,婚后分居    2015年2月23日，被告将原告赶出家门，原告居住于娘家待产，双方分居至今。
...
```

## 稀疏数据筛选方案

稀疏数据指缺乏足够训练数据支持导致低置信度的待预测数据，简单来说，由于模型在训练过程中没有学习到足够与待预测样本相似的数据，模型难以正确预测样本所属类别。本项目中稀疏数据筛选基于TrustAI（可信AI）工具集，利用基于特征相似度的实例级证据分析方法，抽取开发集中样本的支持训练证据，并计算支持证据平均分（通常为得分前三的支持训练证据均分）。分数较低的样本表明其训练证据不足，在训练集中较为稀疏，实验表明模型在这些样本上表现也相对较差。更多细节详见[TrustAI](https://github.com/PaddlePaddle/TrustAI)和[实例级证据分析](https://github.com/PaddlePaddle/TrustAI/blob/main/trustai/interpretation/example_level/README.md)。

稀疏数据筛选旨在开发集中挖掘缺乏训练证据支持的稀疏数据，通常可以采用**数据增强**或**少量数据标注**的两种低成本方式，提升模型预测效果。

**安装TrustAI**
```shell
pip install trustai==0.1.4
```

### 稀疏数据识别--数据增强

这里我们将介绍稀疏数据识别--数据增强流程，首先使用数据增强脚本挖掘开发集中的稀疏数据，然后筛选训练集中对稀疏数据的支持数据进行数据增强，然后将得到的数据增强后的支持数据加入到训练集中进行训练。

现在我们进行稀疏数据识别--数据增强，得到新增训练数据：

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

* `device`: 选用什么设备进行训练，选择cpu、gpu、xpu、npu；默认为"gpu"。
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

我们在CAIL2019—婚姻家庭要素提取数据集抽取部分训练数据（训练集数据规模:500）进行实验,筛选稀疏数据数量和筛选支持数据数量均设为100条，使用不同的数据增强方法进行评测：

|  |Micro F1(%)   | Macro F1(%) |
| ---------| ------------ |------------ |
|训练集|84.43|50.01|
|训练集+支持增强集(duplicate) |84.80|**51.78**|
|训练集+支持增强集(substitute) |84.66|50.61|
|训练集+支持增强集(insert) |84.48|49.95|
|训练集+支持增强集(delete) |84.83| 51.04|
|训练集+支持增强集(swap) |**84.84**|51.06|

### 稀疏数据识别--数据标注

这里我们将介绍稀疏数据识别--数据标注流程，首先使用数据增强脚本挖掘开发集中的稀疏数据，然后筛选对稀疏数据支持的未标注数据，然后将得到支持数据进行标注后加入到训练集中进行训练。

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

* `device`: 选用什么设备进行训练，选择cpu、gpu、xpu、npu；默认为"gpu"。
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

我们在CAIL2019—婚姻家庭要素提取数据集抽取部分训练数据（训练集数据规模:500）进行实验,筛选稀疏数据数量设为100条，筛选待标注数据数量为50和100条。我们比较了使用稀疏数据方案的策略采样和随机采样的效果，下表结果表明使用稀疏数据方案的策略采样能够有效指导训练数据扩充，在标注更少的数据情况下获得更大提升的效果：

|  |Micro F1(%)   | Macro F1(%) |
| ---------| ------------ | ------------ |
|训练集|84.43|50.01|
|训练集+策略采样集(50) |85.77|**57.13**|
|训练集+随机采样集(50) |84.91|54.40|
|训练集+策略采样集(100) |**86.14**|56.93|
|训练集+随机采样集(100) |84.69|50.76|

## 脏数据清洗方案

训练数据标注质量对模型效果有较大影响，但受限于标注人员水平、标注任务难易程度等影响，训练数据中都存在一定比例的标注较差的数据（脏数据）。当标注数据规模较大时，数据标注检查就成为一个难题。本项目中脏数据清洗基于TrustAI（可信AI）工具集，利用基于表示点方法的实例级证据分析方法，计算训练数据对模型的影响分数，分数高的训练数据表明对模型影响大，这些数据有较大概率为脏数据（标注错误样本）。更多细节详见[TrustAI](https://github.com/PaddlePaddle/TrustAI)和[实例级证据分析](https://github.com/PaddlePaddle/TrustAI/blob/main/trustai/interpretation/example_level/README.md)。

**安装TrustAI**
```shell
pip install trustai==0.1.4
```

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

默认在GPU环境下使用，在CPU环境下修改参数配置为`--device "cpu"`

可支持配置的参数：

* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含train.txt和label.txt文件;默认为None。
* `max_seq_length`：分词器tokenizer使用的最大序列长度，ERNIE模型最大不能超过2048。请根据文本长度选择，通常推荐128、256或512，若出现显存不足，请适当调低这一参数；默认为128。
* `params_path`：保存训练模型的目录；默认为"../checkpoint/"。
* `device`: 选用什么设备进行训练，选择cpu、gpu、xpu、npu；默认为"gpu"。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `seed`：随机种子，默认为3。
* `dirty_file`：保存脏数据文件名，默认为"train_dirty.txt"。
* `rest_file`：保存剩余数据（非脏数据）文件名，默认为"train_dirty_rest.txt"。
* `train_file`：本地数据集中训练集文件名；默认为"train.txt"。
* `dirty_num`：筛选脏数据用于重新标注的数量；默认为100。
* `dirty_threshold`：筛选脏数据用于重新标注的阈值，只选择影响分数大于阈值作为有效数据，默认为0。


我们将筛选出脏数据进行重新标注，可以将`train_dirty.txt`直接导入标注工具doccano帮助更快重新标注，详情请参考[文本分类任务doccano数据标注使用指南](../../doccano.md)进行文本分类数据标注。然后将已重新标注的脏数据`train_dirty.txt`与剩余训练集数据`train_dirty_rest.txt`合并得到新的训练集`train_clean.txt`重新进行训练：

```shell
cat ../data/train_dirty_rest.txt ../data/train_dirty.txt > ../data/train_clean.txt
```
**方案效果**

我们在CAIL2019—婚姻家庭要素提取数据集抽取部分训练数据（训练集数据规模:500）进行实验,取50条数据进行脏数据处理，也即50条训练数据为标签错误数据。选择不同`dirty_num`应用脏数据清洗策略进行评测：

|  |Micro F1(%)   | Macro F1(%) |
| ---------| ------------ |------------ |
|训练集(500，含50条脏数据) |82.89|47.83|
|训练集(500，含50条脏数据) + 脏数据清洗(25)|82.42|49.57|
|训练集(500，含50条脏数据) + 脏数据清洗(50)|83.38|50.40|
|训练集(500，含50条脏数据) + 脏数据清洗(100)|84.50|51.28|


## 数据增强策略方案

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
* `aug_strategy`：数据增强策略，可选"mix", "substitute", "insert", "delete", "swap"为多种数据策略混合使用；默认为"substitute"。
* `aug_type`：词替换/词插入增强类型，可选"synonym", "homonym", "mlm"，建议在GPU环境下使用mlm类型；默认为"synonym"。
* `create_n`：生成的句子数量，默认为2。
* `aug_percent`：生成词替换百分比，默认为0.1。
* `device`: 选用什么设备进行增强，选择cpu、gpu、xpu、npu，仅在使用mlm类型有影响；默认为"gpu"。

生成的增强数据保存在`"aug.txt"`文件中，与训练集数据`train.txt`合并得到新的训练集`train_aug.txt`重新进行训练：

```shell
cat ../data/aug.txt ../data/train.txt > ../data/train_aug.txt
```

PaddleNLP内置多种数据增强策略，更多数据增强策略使用方法请参考[数据增强API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md)。

我们在CAIL2019—婚姻家庭要素提取数据集抽取部分训练数据（训练集数据规模:500）进行实验，采用不同数据增强策略进行两倍数据增强（每条样本生成两条增强样本）：

|  |Micro F1(%)   | Macro F1(%) |
| ---------| ------------ |------------ |
|训练集(500)|84.43|50.01|
|训练集(500)+数据增强(×2, mix) |84.72|51.86|
|训练集(500)+支持增强集(×2, substitute) |84.50|53.23|
|训练集(500)+支持增强集(×2, insert) |**85.03**|53.54|
|训练集(500)+支持增强集(×2, delete) |84.74| **55.89**|
|训练集(500)+支持增强集(×2, swap) |84.44|52.50|
