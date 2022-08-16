# 文本多标签分类任务指南

**目录**
   * [多标签任务介绍](#多标签任务介绍)
   * [代码结构说明](#代码结构说明)
   * [环境准备](#环境准备)
   * [数据集准备](#数据集准备)
   * [模型训练](#模型训练)
       * [训练评估与模型优化](#训练评估与模型优化)
       * [训练效果](#训练效果)
   * [模型预测](#模型预测)
   * [静态图导出](#静态图导出)
   * [模型裁剪](#模型裁剪)
       * [环境准备](#环境准备)
       * [裁剪API使用](#裁剪API使用)
       * [裁剪效果](#裁剪效果)
   * [模型部署](#模型部署)

## 多标签任务介绍

文本多标签分类是自然语言处理（NLP）中常见的文本分类任务，多标签数据集中样本包含一个或多个标签类别，多标签任务的目标是预测**样本属于哪些标签类别，这些类别具有不相互排斥的属性**。文本多标签分类在各种现实场景中具有广泛的适用性，例如商品分类、网页标签、新闻标注、蛋白质功能分类、电影分类、语义场景分类等。

近年来，大量包含了案件事实及其适用法律条文信息的裁判文书逐渐在互联网上公开，海量的数据使自然语言处理技术的应用成为可能。现实中的案情错综复杂，案情描述通常涉及多个重要事实，以CAIL2019数据集中婚姻家庭领域的案情要素抽取为例：

```text
"2013年11月28日原、被告离婚时自愿达成协议，婚生子张某乙由被告李某某抚养，本院以（2013）宝渭法民初字第01848号民事调解书对该协议内容予以了确认，该协议具有法律效力，对原、被告双方均有约束力。"
```
该案件中涉及`婚后有子女`、`限制行为能力子女抚养`两项要素。接下来我们将讲解如何利用多标签模型，对输入文本中进行案情重要要素抽取。

## 代码结构说明

以下是本项目主要代码结构及说明：

```text
multi_label/
├── deploy # 部署
│   └── predictor # 离线部署
│   │   ├── infer.py # 测试脚本
│   │   ├── predictor.py 离线部署脚本
│   │   └── README.md # 离线部署使用说明
│   ├── paddle_serving # PaddleServing在线服务化部署
│   │   ├──config.yml # 服务端的配置文件
│   │   ├──rpc_client.py # 客户端预测脚本
│   │   ├──service.py # 服务端的脚本
│   │   └── README.md # 在线服务化部署使用说明
│   └── triton_serving # Triton在线服务化部署
│       ├── README.md # Triton部署使用说明
│       ├── seqcls_grpc_client.py # 客户端预测脚本
│       └── models
│           ├── seqcls
│           │   └── config.pbtxt
│           ├── seqcls_model
│           │   └──config.pbtxt
│           ├── seqcls_postprocess
│           │   ├── 1
│           │   │   └── model.py
│           │   └── config.pbtxt
│           └── tokenizer
│               ├── 1
│               │   └── model.py
│               └── config.pbtxt
├── train.py # 训练评估脚本
├── predict.py # 预测脚本
├── export_model.py # 静态图模型导出脚本
├── utils.py # 工具函数脚本
├── metric.py # metric脚本
├── prune.py # 裁剪脚本
├── prune_trainer.py # 裁剪trainer脚本
└── README.md # 使用说明
```

## 环境准备
**文本分类所需用到的环境配置：**

- python >= 3.6
- paddlepaddle >= 2.3
- paddlenlp >= 2.3.4
- scikit-learn >= 1.0.2

**安装PaddlePaddle**

环境中paddlepaddle-gpu或paddlepaddle版本应大于或等于2.3, 请参见[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)根据自己需求选择合适的PaddlePaddle下载命令。


**安装PaddleNLP**
```shell
python3 -m pip install paddlenlp==2.3.4 -i https://mirror.baidu.com/pypi/simple
```
安装PaddleNLP默认开启百度镜像源来加速下载，如果您使用 HTTP 代理可以关闭(删去 -i https://mirror.baidu.com/pypi/simple)，更多关于PaddleNLP安装的详细教程请查见[PaddleNLP快速安装](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/get_started/installation.rst)。

**安装sklearn**
```shell
pip install scikit-learn==1.0.2
```

## 数据集准备
训练需要准备指定格式的本地数据集,如果没有已标注的数据集，可以参考[文本分类任务doccano数据标注使用指南](../doccano.md)进行文本分类数据标注。

**指定格式本地数据集目录结构**

```text
data/
├── train.txt # 训练数据集文件
├── dev.txt # 开发数据集文件
├── test.txt # 可选，测试数据集文件
├── label.txt # 分类标签文件
└── data.txt # 待预测数据文件
```

**训练、开发、测试数据集**

train.txt(训练数据集文件)， dev.txt(开发数据集文件)，test.txt(可选，测试数据集文件)，文件中文本与标签类别名用tab符`'\t'`分隔开，标签中多个标签之间用`','`逗号分隔开。训练集指用于训练模型的数据；开发集指用于评测模型表现的数据，可以根据模型在开发集上的精度调整训练参数和模型；测试集用于测试模型表现，没有测试集时可以使用开发集代替；通常建议训练集：开发集比例为8:2或7:3; 建议训练集、开发集、测试集的比例为8:1:1或6:2:2。**注意文本中不能包含tab符`'\t'`**。

- train.txt/dev.txt/test.txt 文件格式：
```text
<文本>'\t'<标签>','<标签>','<标签>
<文本>'\t'<标签>','<标签>
...
```

- train.txt/dev.txt/test.txt 文件样例：

```text
现在原告已是第二次申请与被告离婚了。    二次起诉离婚
双方均认可价值6万元。    不动产分割,有夫妻共同财产
2004年4月，原、被告发生纠纷后，被告离家外出未归，直到现在，双方长期分居生活，十几年间互无联系，夫妻感情已经完全破裂。    婚后分居
婚生子杨某甲由原告抚养，高中阶段之前的相关费用由原告承担，高中阶段之后的相关费用由双方协商，被告可以随时探望孩子；    婚后有子女,支付抚养费,限制行为能力子女抚养
...
```
**分类标签**

label.txt(分类标签文件)记录数据集中所有标签集合，每一行为一个标签名。
- label.txt 文件格式：

```text
<标签>
<标签>
...
```
- label.txt 文件样例：
```text
婚后有子女
限制行为能力子女抚养
有夫妻共同财产
支付抚养费
...
```

**待预测数据**

data.txt(待预测数据文件)，需要预测标签的文本数据。

- data.txt 文件格式：

```text
<文本>
<文本>
...
```
- data.txt 文件样例：
```text
原、被告另购置橱柜、碗架、电磁炉、电饭锅各一个归原告王某某所有。
于是原告到儿子就读的幼儿园进行探望，被告碰见后对原告破口大骂，还不让儿子叫原告妈妈，而叫被告现在的妻子做妈妈。
6、被告父亲给的房屋装修款2.3万元在原告处，要求依法分割；
由我全额出资购买的联想台式电脑，我均依次放弃。
...
```

## 模型训练

我们以公开数据集CAIL2019—婚姻家庭要素提取任务为示例，在训练集上进行模型微调，并在开发集上验证。

下载CAIL2019—婚姻家庭要素提取任务数据集：
```shell
wget https://paddlenlp.bj.bcebos.com/datasets/divorce.tar.gz
tar -zxvf divorce.tar.gz
mv divorce data
```

使用CPU/GPU训练：
```shell
python train.py \
    --device "gpu" \
    --dataset_dir "data" \
    --save_dir "./checkpoint" \
    --max_seq_length 128 \
    --model_name "ernie-3.0-medium-zh" \
    --batch_size 32 \
    --early_stop \
    --learning_rate 3e-5 \
    --epochs 100 \
    --logging_steps 5 \
    --train_file "train.txt"
```
默认为GPU训练，使用CPU训练只需将设备参数配置改为`--device "cpu"`

如果在CPU环境下训练，可以指定`nproc_per_node`参数进行多核训练：
```shell
python -m paddle.distributed.launch --nproc_per_node=8 --backend='gloo' train.py \
    --device "cpu" \
    --dataset_dir "data" \
    --save_dir "./checkpoint" \
    --max_seq_length 128 \
    --model_name "ernie-3.0-medium-zh" \
    --batch_size 32 \
    --early_stop \
    --learning_rate 3e-5 \
    --epochs 100 \
    --logging_steps 5 \
    --train_file "train.txt"
```

如果在GPU环境中使用，可以指定`gpus`参数进行单卡/多卡训练：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" train.py \
    --device "gpu" \
    --dataset_dir "data" \
    --save_dir "./checkpoint" \
    --max_seq_length 128 \
    --model_name "ernie-3.0-medium-zh" \
    --batch_size 32 \
    --early_stop \
    --learning_rate 3e-5 \
    --epochs 100 \
    --logging_steps 5 \
    --train_file "train.txt"
```

使用多卡训练可以指定多个GPU卡号，例如 --gpus "0,1"。如果设备只有一个GPU卡号默认为0，可使用`nvidia-smi`命令查看GPU使用情况。

可支持配置的参数：

* `device`: 选用什么设备进行训练，选择cpu、gpu、xpu、npu。如使用gpu训练，可使用参数--gpus指定GPU卡号；默认为"gpu"。
* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含train.txt，dev.txt和label.txt文件;默认为None。
* `save_dir`：保存训练模型的目录；默认保存在当前目录checkpoint文件夹下。
* `max_seq_length`：分词器tokenizer使用的最大序列长度，ERNIE模型最大不能超过2048。请根据文本长度选择，通常推荐128、256或512，若出现显存不足，请适当调低这一参数；默认为128。
* `model_name`：选择预训练模型,可选"ernie-3.0-xbase-zh", "ernie-3.0-base-zh", "ernie-3.0-medium-zh", "ernie-3.0-micro-zh", "ernie-3.0-mini-zh", "ernie-3.0-nano-zh", "ernie-2.0-base-en", "ernie-2.0-large-en"；默认为"ernie-3.0-medium-zh"。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：训练最大学习率；默认为3e-5。
* `epochs`: 训练轮次，使用早停法时可以选择100；默认为10。
* `early_stop`：选择是否使用早停法(EarlyStopping)，模型在开发集经过一定epoch后精度表现不再上升，训练终止；默认为False。
* `early_stop_nums`：在设定的早停训练轮次内，模型在开发集上表现不再上升，训练终止；默认为4。
* `logging_steps`: 训练过程中日志打印的间隔steps数，默认5。
* `weight_decay`：控制正则项力度的参数，用于防止过拟合，默认为0.0。
* `warmup`：是否使用学习率warmup策略，使用时应设置适当的训练轮次（epochs）；默认为False。
* `warmup_steps`：学习率warmup策略的比例数，如果设为1000，则学习率会在1000steps数从0慢慢增长到learning_rate, 而后再缓慢衰减；默认为0。
* `init_from_ckpt`: 模型初始checkpoint参数地址，默认None。
* `seed`：随机种子，默认为3。
* `train_file`：本地数据集中训练集文件名；默认为"train.txt"。
* `dev_file`：本地数据集中开发集文件名；默认为"dev.txt"。
* `label_file`：本地数据集中标签集文件名；默认为"label.txt"。

程序运行时将会自动进行训练，评估。同时训练过程中会自动保存开发集上最佳模型在指定的 `save_dir` 中，保存模型文件结构如下所示：

```text
checkpoint/
├── model_config.json
├── model_state.pdparams
├── tokenizer_config.json
└── vocab.txt
```

**NOTE:**
* 如需恢复模型训练，则可以设置 `init_from_ckpt` ， 如 `init_from_ckpt=checkpoint/model_state.pdparams` 。
* 如需训练英文文本分类任务，只需更换预训练模型参数 `model_name` 。英文训练任务推荐使用"ernie-2.0-base-en"，更多可选模型可参考[Transformer预训练模型](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer)。

### 训练评估与模型优化

训练后的模型我们可以使用[评估脚本](analysis/evaluate.py)对每个类别分别进行评估，并输出预测错误样本（bad case）：

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

模型表现常常受限于数据质量，在analysis模块中我们提供了基于[TrustAI](https://github.com/PaddlePaddle/TrustAI)的稀疏数据筛选、脏数据清洗、数据增强三种优化方案助力开发者提升模型效果，更多模型评估和优化方案细节详见[训练评估与模型优化指南](analysis/README.md)。


### 训练效果

PaddleNLP提供ERNIE 3.0 全系列轻量化模型，对于中文训练任务可以根据需求选择不同的预训练模型参数进行训练，我们评测了不同预训练模型在CAIL2019—婚姻家庭要素提取任务的表现，测试配置如下：

1. 数据集：CAIL2019—婚姻家庭要素提取任务开发集

2. 物理机环境

    系统: CentOS Linux release 7.7.1908 (Core)

    GPU: Tesla V100-SXM2-32GB

    CPU: Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz

    CUDA: 11.2

    cuDNN: 8.1.0

    Driver Version: 460.27.04

    内存: 630 GB

3. PaddlePaddle 版本：2.3.0

4. PaddleNLP 版本：2.3.1

5. 性能数据指标：latency。latency 测试方法：固定 batch size 为 32，GPU部署运行时间 total_time，计算 latency = total_time / total_samples

6. 精度评价指标：Micro F1分数、Macro F1分数

|  model_name  | 模型结构  |Micro F1(%)   | Macro F1(%) | latency(ms) |
| -------------------------- | ------------ | ------------ | ------------ |------------ |
|"ernie-3.0-base-zh" |12-layer, 768-hidden, 12-heads|90.38|80.14| 2.70 |
|"ernie-3.0-medium-zh"| 6-layer, 768-hidden, 12-heads|90.57|79.36| 1.46|
|"ernie-3.0-mini-zh" |6-layer, 384-hidden, 12-heads|89.27|76.78| 0.56|
|"ernie-3.0-micro-zh" | 4-layer, 384-hidden, 12-heads|89.43|77.20| 0.34|
|"ernie-3.0-nano-zh" |4-layer, 312-hidden, 12-heads|85.39|75.07|0.32|

## 模型预测

训练结束后，输入待预测数据(data.txt)和类别标签对照列表(label.txt)，使用训练好的模型进行。

在CPU环境下进行预测，预测结果将保存在`output_file`：
```shell
python predict.py \
    --device "cpu" \
    --dataset_dir "data" \
    --output_file "output.txt" \
    --params_path "./checkpoint" \
    --max_seq_length 128 \
    --batch_size 32
```
在GPU环境下进行预测，预测结果将保存在`output_file`：
```shell
python predict.py \
    --device "gpu" \
    --dataset_dir "data" \
    --output_file "output.txt" \
    --params_path "./checkpoint" \
    --max_seq_length 128 \
    --batch_size 32
```

可支持配置的参数：

* `device`: 选用什么设备进行预测，可选cpu、gpu、xpu、npu；默认为gpu。
* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含data.txt和label.txt文件;默认为None。
* `params_path`：待预测模型的目录；默认为"./checkpoint/"。
* `max_seq_length`：模型使用的最大序列长度,建议与训练时最大序列长度一致, 若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `data_file`：本地数据集中未标注待预测数据文件名；默认为"data.txt"。
* `label_file`：本地数据集中标签集文件名；默认为"label.txt"。

## 静态图导出

使用动态图训练结束之后，还可以将动态图参数导出成静态图参数，静态图模型将用于**后续的推理部署工作**。具体代码见[静态图导出脚本](export_model.py)，静态图参数保存在`output_path`指定路径中。运行方式：

```shell
python export_model.py \
    --params_path ./checkpoint/ \
    --output_path ./export
```

可支持配置的参数：

* `params_path`：动态图训练保存的参数路径；默认为"./checkpoint/"。
* `output_path`：静态图图保存的参数路径；默认为"./export"。

程序运行时将会自动导出模型到指定的 `output_path` 中，保存模型文件结构如下所示：

```text
export/
├── float32.pdiparams
├── float32.pdiparams.info
└── float32.pdmodel
```
 导出模型之后用于部署，项目提供了基于ONNXRuntime的 [离线部署方案](./deploy/predictor/README.md) 和基于Paddle Serving的 [在线服务化部署方案](./deploy/predictor/README.md)。

## 模型裁剪

**如果有模型部署上线的需求，需要进一步压缩模型体积**，可以使用本项目基于 PaddleNLP 的 Trainer API 发布提供了模型裁剪 API。裁剪 API 支持用户对 ERNIE 等Transformers 类下游任务微调模型进行裁剪，用户只需要简单地调用脚本`prune.py` 即可一键启动裁剪和并自动保存裁剪后的模型参数。
### 环境准备

使用裁剪功能需要安装 paddleslim 包

```shell
pip install paddleslim==2.2.2
```

### 裁剪 API 使用

使用CPU进行裁剪训练
```shell
python prune.py \
    --device "cpu" \
    --output_dir "./prune" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --logging_steps 5 \
    --save_steps 50 \
    --seed 3 \
    --dataset_dir "data" \
    --max_seq_length 128 \
    --params_dir "./checkpoint" \
    --width_mult '2/3'
```

使用GPU单卡/多卡训练
```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" prune.py \
    --device "gpu" \
    --output_dir "./prune" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --logging_steps 5 \
    --save_steps 50 \
    --seed 3 \
    --dataset_dir "data" \
    --max_seq_length 128 \
    --params_dir "./checkpoint" \
    --width_mult '2/3'

```
使用多卡训练可以指定多个GPU卡号，例如 --gpus "0,1"。如果设备只有一个GPU卡号默认为0，可使用`nvidia-smi`命令查看GPU使用情况。

可支持配置的参数：
* `TrainingArguments`
  * `output_dir`：必须，保存模型输出和和中间checkpoint的输出目录;默认为 `None` 。
  * `device`: 选用什么设备进行裁剪，选择cpu、gpu。如使用gpu训练，可使用参数--gpus指定GPU卡号。
  * `per_device_train_batch_size`：训练集裁剪训练过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
  * `per_device_eval_batch_size`：开发集评测过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
  * `learning_rate`：训练最大学习率；默认为3e-5。
  * `num_train_epochs`: 训练轮次，使用早停法时可以选择100；默认为10。
  * `logging_steps`: 训练过程中日志打印的间隔steps数，默认5。
  * `save_steps`: 训练过程中保存模型checkpoint的间隔steps数，默认100。
  * `seed`：随机种子，默认为3。
  * `TrainingArguments` 包含了用户需要的大部分训练参数，所有可配置的参数详见[TrainingArguments 参数介绍](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/trainer.md#trainingarguments-%E5%8F%82%E6%95%B0%E4%BB%8B%E7%BB%8D)。

* `DataArguments`
  * `dataset_dir`：本地数据集路径，需包含train.txt,dev.txt,label.txt;默认为None。
  * `max_seq_length`：模型使用的最大序列长度，建议与训练过程保持一致, 若出现显存不足，请适当调低这一参数；默认为128。

* `ModelArguments`
  * `params_dir`：待预测模型参数文件；默认为"./checkpoint/"。
  * `width_mult`：裁剪宽度保留的比例，表示对self_attention中的 `q`、`k`、`v` 以及 `ffn` 权重宽度的保留比例，默认是 '2/3'。

以上参数都可通过 `python prune.py --dataset_dir xx --params_dir xx` 的方式传入）

程序运行时将会自动进行训练，评估，测试。同时训练过程中会自动保存开发集上最佳模型在指定的 `output_dir` 中，保存模型文件结构如下所示：

```text
prune/
├── 0.6666666666666666
│   ├── float32.pdiparams
│   ├── float32.pdiparams.info
│   ├── float32.pdmodel
│   ├── model_state.pdparams
│   └── model_config.json
└── ...
```

**NOTE:**

1. 目前支持的裁剪策略需要训练，训练时间视下游任务数据量而定，且和微调的训练时间是一个量级。

2. 裁剪类似蒸馏过程，方便起见，可以直接使用微调时的超参。为了进一步提升精度，可以对 `per_device_train_batch_size`、`learning_rate`、`num_train_epochs`、`max_seq_length` 等超参进行网格搜索（grid search）。

3. 模型裁剪主要用于推理部署，因此裁剪后的模型都是静态图模型，只可用于推理部署，不能再通过 `from_pretrained` 导入继续训练。

4. 导出模型之后用于部署，项目提供了基于ONNXRuntime的 [离线部署方案](./deploy/predictor/README.md) 和基于Paddle Serving的 [在线服务化部署方案](./deploy/predictor/README.md)。


### 裁剪效果
本案例我们对ERNIE 3.0模型微调后的模型使用裁剪 API 进行裁剪，我们评测了不同裁剪保留比例在CAIL2019—婚姻家庭要素提取任务的表现，测试配置如下：

1. 数据集：CAIL2019—婚姻家庭要素提取任务开发集

2. 物理机环境

    系统: CentOS Linux release 7.7.1908 (Core)

    GPU: Tesla V100-SXM2-32GB

    CPU: Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz

    CUDA: 11.2

    cuDNN: 8.1.0

    Driver Version: 460.27.04

    内存: 630 GB

3. PaddlePaddle 版本：2.3.0

4. PaddleNLP 版本：2.3.1

5. 性能数据指标：latency。latency 测试方法：固定 batch size 为 32，GPU部署运行时间 total_time，计算 latency = total_time / total_samples

6. 精度评价指标：Micro F1分数、Macro F1分数

|                            | Micro F1(%)   | Macro F1(%) | latency(ms) |
| -------------------------- | ------------ | ------------- |------------- |
| ERNIE 3.0 Medium             | 90.57|79.36| 1.46|
| ERNIE 3.0 Medium +裁剪(保留比例3/4)    | 89.94|79.35| 0.81   |
| ERNIE 3.0 Medium +裁剪(保留比例2/3)    | 89.99|79.37 | 0.75  |
| ERNIE 3.0 Medium +裁剪(保留比例1/2)    | 89.19 | 76.35| 0.61 |

## 模型部署

- 离线部署搭建请参考[离线部署](deploy/predictor/README.md)。

- 在线服务化部署搭建请参考 [Paddle Serving部署指南](deploy/paddle_serving/README.md) (Paddle Serving支持X86、Arm CPU、NVIDIA GPU、昆仑/昇腾等多种硬件)或[Triton部署指南](deploy/triton_serving/README.md)。
