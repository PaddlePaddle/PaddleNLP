# 文本二分类/多分类任务指南

**目录**
   * [二分类/多分类任务介绍](#二分类/多分类任务介绍)
   * [代码结构说明](#代码结构说明)
   * [环境准备](#环境准备)
   * [数据集准备](#数据集准备)
   * [模型训练](#模型训练)
       * [训练效果](#训练效果)
   * [模型预测](#模型预测)
   * [静态图导出](#静态图导出)
   * [模型裁剪](#模型裁剪)
      * [环境准备](#环境准备)
       * [裁剪API使用](#裁剪API使用)
       * [裁剪效果](#裁剪效果)
   * [模型部署](#模型部署)

## 二分类/多分类任务介绍

文本分类是自然语言处理（NLP）基本任务之一，文本二分类/多分类任务的目标是预测**样本最可能来自所有标签中的哪一个类别**。在本项目中二分类任务被视为多分类任务中标签集包含两个类别的情况，以下统一称为多分类任务。多分类任务在商品分类、网页标签、新闻分类、医疗文本分类等各种现实场景中具有广泛的适用性。在医学搜索中，对搜索问题的意图分类可以极大提升搜索结果的相关性，CBLUE数据集中医疗搜索检索词意图分类(KUAKE-QIC)任务共有10880条医学问题检索文本涵盖11种意图分类类型，接下来我们将介绍如何使用多分类模型,根据输入的检索文本进行多分类任务。

## 代码结构说明

以下是本项目主要代码结构及说明：

```text
multi_class/
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
├── prune.py # 裁剪脚本
├── prune_trainer.py # 裁剪API脚本
└── README.md # 多分类使用说明
```

## 准备环境
**文本分类所需用到的环境配置：**

- python >= 3.6
- paddlepaddle >= 2.3
- paddlenlp >= 2.3.4

**安装PaddlePaddle**

环境中paddlepaddle-gpu或paddlepaddle版本应大于或等于2.3, 请参见[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)根据自己需求选择合适的PaddlePaddle下载命令。


**安装PaddleNLP**
```shell
python3 -m pip install paddlenlp==2.3.4 -i https://mirror.baidu.com/pypi/simple
```
安装PaddleNLP默认开启百度镜像源来加速下载，如果您使用 HTTP 代理可以关闭(删去 -i https://mirror.baidu.com/pypi/simple)，更多关于PaddleNLP安装的详细教程请查见[PaddleNLP快速安装](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/get_started/installation.rst)。


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

train.txt(训练数据集文件)， dev.txt(开发数据集文件)，test.txt(测试数据集文件)，文件中文本与标签类别名用tab符`'\t'`分隔开。训练集指用于训练模型的数据；开发集指用于评测模型表现的数据，可以根据模型在开发集上的精度调整训练参数和模型；测试集用于测试模型表现，没有测试集时可以使用开发集代替；通常建议训练集、开发集、测试集的比例为8:1:1或6:2:2；只有训练集和开发集的情况时建议训练集：开发集比例为8:2或7:3。**注意文本中不能包含tab符**
- train.txt/dev.txt/test.txt 文件格式：
```text
<文本>'\t'<标签>
<文本>'\t'<标签>
...
```
- train.txt/dev.txt/test.txt 文件样例：
```text
25岁已经感觉脸部松弛了怎么办	治疗方案
小孩的眉毛剪了会长吗？	其他
172的身高还能长高吗？	其他
冻疮用三金冻疮酊有效果么？	功效作用
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
病情诊断
治疗方案
病因分析
指标解读
就医建议
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
黑苦荞茶的功效与作用及食用方法
交界痣会凸起吗
检查是否能怀孕挂什么科
鱼油怎么吃咬破吃还是直接咽下去
...
```
## 模型训练

接下来我们将以公开数据集KUAKE-QIC任务为示例，介绍如何在训练集上进行模型训练，并在开发集上使用准确率评估模型表现。

下载KUAKE-QIC数据集：
```shell
wget https://paddlenlp.bj.bcebos.com/datasets/KUAKE_QIC.tar.gz
tar -zxvf KUAKE_QIC.tar.gz
mv KUAKE_QIC data
```

使用CPU训练
```shell
python train.py \
    --device "cpu" \
    --dataset_dir "data" \
    --save_dir "./checkpoint" \
    --max_seq_length 128 \
    --model_name "ernie-3.0-medium-zh" \
    --batch_size 32 \
    --early_stop \
    --learning_rate 3e-5 \
    --epochs 100 \
    --logging_steps 5
```

使用GPU单卡/多卡训练
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
    --logging_steps 5
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


### 训练效果
PaddleNLP提供ERNIE 3.0 全系列轻量化模型，对于中文训练任务可以根据需求选择不同的预训练模型参数进行训练，我们评测了不同预训练模型在KUAKE-QIC任务的表现，测试配置如下：

1. 数据集：CBLUE数据集中医疗搜索检索词意图分类(KUAKE-QIC)任务开发集

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

6. 精度评价指标：Accuracy

|  model_name  | 模型结构  |Accuracy(%)   | latency(ms) |
| -------------------------- | ------------ | ------------ | ------------ |
|"ernie-3.0-base-zh" |12-layer, 768-hidden, 12-heads|82.25| 2.07 |
|"ernie-3.0-medium-zh"| 6-layer, 768-hidden, 12-heads|81.79| 1.07|
|"ernie-3.0-mini-zh" |6-layer, 384-hidden, 12-heads|79.80| 0.38|
|"ernie-3.0-micro-zh" | 4-layer, 384-hidden, 12-heads|79.80| 0.26|
|"ernie-3.0-nano-zh" |4-layer, 312-hidden, 12-heads|78.57|0.22|
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

如果有模型部署上线的需求，需要进一步压缩模型体积，可以使用本项目基于 PaddleNLP 的 Trainer API 发布提供了模型裁剪 API。裁剪 API 支持用户对 ERNIE 等Transformers 类下游任务微调模型进行裁剪，用户只需要简单地调用脚本`prune.py` 即可一键启动裁剪和并自动保存裁剪后的模型。
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
本案例我们对ERNIE 3.0模型微调后的模型使用裁剪 API 进行裁剪，我们评测了不同裁剪保留比例在KUAKE-QIC任务的表现，测试配置如下：

1. 数据集：CBLUE数据集中医疗搜索检索词意图分类(KUAKE-QIC)任务开发集

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

6. 精度评价指标：Accuracy

|                            | Accuracy(%)   | latency(ms) |
| -------------------------- | ------------ | ------------- |
| ERNIE 3.0 Medium             | 81.79 | 1.07  |
| ERNIE 3.0 Medium +裁剪(保留比例3/4)    | 81.79| 0.83   |
| ERNIE 3.0 Medium +裁剪(保留比例2/3)    | 81.07  | 0.79  |
| ERNIE 3.0 Medium +裁剪(保留比例1/2)    | 81.07 | 0.64  |


## 模型部署

- 离线部署搭建请参考[离线部署](deploy/predictor/README.md)。

- 在线服务化部署搭建请参考 [Paddle Serving部署指南](deploy/paddle_serving/README.md) (Paddle Serving支持X86、Arm CPU、NVIDIA GPU、昆仑/昇腾等多种硬件)或[Triton部署指南](deploy/triton_serving/README.md)。
