# 文本层次分类任务指南

**目录**
   * [层次分类任务介绍](#层次分类任务介绍)
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


## 层次分类任务介绍

多标签层次分类任务指自然语言处理任务中，**每个样本具有多个标签标记，并且标签集合中标签之间存在预定义的层次结构**，多标签层次分类需要充分考虑标签集之间的层次结构关系来预测层次化预测结果。层次分类任务中标签层次结构分为两类，一类为树状结构，另一类为有向无环图(DAG)结构。有向无环图结构与树状结构区别在于，有向无环图中的节点可能存在不止一个父节点。在现实场景中，大量的数据如新闻分类、专利分类、学术论文分类等标签集合存在层次化结构，需要利用算法为文本自动标注更细粒度和更准确的标签。

层次分类问题是一个天然多标签问题，以下图一个树状标签结构(宠物为根节点)为例，如果一个样本属于美短虎斑，样本也天然地同时属于类别美国短毛猫和类别猫两个样本标签。本项目采用通用多标签层次分类算法，**将每个结点的标签路径视为一个多分类标签，使用单个多标签分类器进行决策**，以上面美短虎斑的例子为例，该样本包含三个标签：猫、猫##美国短毛猫、猫##美国短毛猫##美短虎斑(不同层的标签之间使用`##`作为分割符)。下图的**标签结构标签集合**为猫、猫##波斯猫、猫##缅因猫、猫##美国短毛猫、猫##美国短毛猫##美短加白、猫##美国短毛猫##美短虎斑、猫##美国短毛猫##美短起司、兔、兔##侏儒兔、兔##垂耳兔总共10个标签。

<div align="center">
    <img src="https://user-images.githubusercontent.com/63761690/175248039-ce1673f1-9b03-4804-b1cb-29e4b4193f86.png" width="600">
</div>

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
- scikit-learn >= 0.24.2

**安装PaddlePaddle**

环境中paddlepaddle-gpu或paddlepaddle版本应大于或等于2.3, 请参见[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)根据自己需求选择合适的PaddlePaddle下载命令。


**安装PaddleNLP**
```shell
python3 -m pip install paddlenlp==2.3.4 -i https://mirror.baidu.com/pypi/simple
```
安装PaddleNLP默认开启百度镜像源来加速下载，如果您使用 HTTP 代理可以关闭(删去 -i https://mirror.baidu.com/pypi/simple)，更多关于PaddleNLP安装的详细教程请查见[PaddleNLP快速安装](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/get_started/installation.rst)。

**安装sklearn**
```shell
pip install scikit-learn==0.24.2
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

train.txt(训练数据集文件)， dev.txt(开发数据集文件)，test.txt(可选，测试数据集文件)，文件中文本与标签类别名用tab符`'\t'`分隔开，标签中多个标签之间用`','`英文逗号分隔开。训练集指用于训练模型的数据；开发集指用于评测模型表现的数据，可以根据模型在开发集上的精度调整训练参数和模型；测试集用于测试模型表现，没有测试集时可以使用开发集代替；通常建议训练集：开发集比例为8:2或7:3; 建议训练集、开发集、测试集的比例为8:1:1或6:2:2。

**注意文本中不能包含tab符`'\t'`**。本项目选择为标签层次结构中的每一个节点生成对应的标签路径，详见[层次分类任务介绍](#层次分类任务介绍)

- train.txt/dev.txt/test.txt 文件格式：
```text
<文本>'\t'<标签>','<标签>','<标签>
<文本>'\t'<标签>','<标签>
...
```

- train.txt/dev.txt/test.txt 文件样例：
```text
嗓子紧，有点咳咳，吃什么消炎药好，我现在吃阿莫西林胶囊    药物,药物##适用症
支气管扩张发病率大概是多少    其他,其他##无法确定
支气管扩张是癌症吗    病症,病症##定义
支气管扩张引发的疾病有哪些    病症,病症##相关病症
支气管扩张和哮喘哪个严重    其他,其他##对比
...
```
**分类标签**

label.txt(层次分类标签文件)记录数据集中所有标签路径集合，在标签路径中，高层的标签指向底层标签，标签之间用`'##'`连接，本项目选择为标签层次结构中的每一个节点生成对应的标签路径，详见[层次分类任务介绍](#层次分类任务介绍)。

- label.txt 文件格式：

```text
<一级标签>
<一级标签>'##'<二级标签>
<一级标签>'##'<二级标签>'##'<三级标签>
...
```
- label.txt  文件样例：
```text
药物
药物##适用症
病症
病症##定义
其他
其他##无法确定
其他##对比
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
请问木竭胶囊能同高血压药、氨糖同时服吗？
低压100*高压140*头涨，想吃点降压药。谢谢！
脑穿通畸形易发人群有哪些
...
```
## 模型训练
我们以公开数据集CMID为示例，在训练集上进行模型微调，并在开发集上验证。CMID是一个人工标注的医疗意图分类信息数据集，包含4个一级标签和36个二级标签，数据集按8：2划分成训练集和开发集。

下载CMID数据集：
```shell
wget https://paddlenlp.bj.bcebos.com/datasets/cmid.tar.gz
tar -zxvf cmid.tar.gz
mv cmid data
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

PaddleNLP提供ERNIE 3.0 全系列轻量化模型，对于中文训练任务可以根据需求选择不同的预训练模型参数进行训练，我们评测了不同预训练模型在医疗意图分类CMID任务的表现，测试配置如下：

1. 数据集：医疗意图分类CMID开发集

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
|"ernie-3.0-base-zh" |12-layer, 768-hidden, 12-heads|63.13|45.92| 4.16 |
|"ernie-3.0-medium-zh"| 6-layer, 768-hidden, 12-heads|63.51|46.98| 2.12|
|"ernie-3.0-mini-zh" |6-layer, 384-hidden, 12-heads|61.70|42.30| 0.83|
|"ernie-3.0-micro-zh" | 4-layer, 384-hidden, 12-heads|61.14|42.97| 0.60|
|"ernie-3.0-nano-zh" |4-layer, 312-hidden, 12-heads|60.15|32.62|0.25|

## 模型预测
训练结束后，输入待预测数据(data.txt)和类别标签对照列表(label.txt)，使用训练好的模型进行。

在CPU环境下进行预测：
```shell
python predict.py \
    --device "cpu" \
    --dataset_dir "data" \
    --params_path "./checkpoint" \
    --max_seq_length 128 \
    --batch_size 32
```

在GPU环境下进行预测：

```shell
python predict.py \
    --device "gpu" \
    --dataset_dir "data" \
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
本案例我们对ERNIE 3.0模型微调后的模型使用裁剪 API 进行裁剪，我们评测了不同裁剪保留比例在医疗意图分类CMID的表现，测试配置如下：

1. 数据集：医疗意图分类CMID开发集

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
| ERNIE 3.0 Medium             | 63.13|45.92| 4.16 |
| ERNIE 3.0 Medium +裁剪(保留比例3/4)    | 62.11|45.04| 0.81   |
| ERNIE 3.0 Medium +裁剪(保留比例2/3)    | 62.41|45.06 | 0.74  |
| ERNIE 3.0 Medium +裁剪(保留比例1/2)    | 59.86 | 41.42| 0.61 |

## 模型部署

- 离线部署搭建请参考[离线部署](deploy/predictor/README.md)。

- 在线服务化部署搭建请参考 [Paddle Serving部署指南](deploy/paddle_serving/README.md) (Paddle Serving支持X86、Arm CPU、NVIDIA GPU、昆仑/昇腾等多种硬件)或[Triton部署指南](deploy/triton_serving/README.md)。
