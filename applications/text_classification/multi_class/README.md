# 文本多分类任务指南

**目录**
   * [多分类任务介绍](#多分类任务介绍)
   * [代码结构说明](#代码结构说明)
   * [模型微调](#模型微调)
       * [从本地文件创建数据集](#从本地文件创建数据集)
   * [模型预测](#模型预测)
   * [模型静态图导出](#模型预测)
   * [模型裁剪](#模型裁剪)
       * [环境准备](#环境准备)
       * [裁剪API使用](#裁剪API使用)
       * [裁剪效果](#裁剪效果)
   * [模型部署](#模型部署)

## 多分类任务介绍

文本分类是自然语言处理（NLP）基本任务之一，文本多分类任务的目标是预测输入的文本样本最可能来自`n_classes` 个标签类别中的某一类别，多分类任务在商品分类、网页标签、新闻分类、医疗文本分类等各种现实场景中具有广泛的适用性。在医学搜索中，对搜索问题的意图分类可以极大提升搜索结果的相关性，CBLUE数据集中医疗搜索检索词意图分类(KUAKE-QIC)任务共有10880条医学问题检索文本涵盖11种意图分类类型，接下来我们将介绍如何使用多分类模型,根据输入的检索文本进行多分类任务。

## 代码结构说明

以下是本项目主要代码结构及说明：

```text
multi_class/
├── deploy # 部署
│   └── predictor # 导出ONNX模型并基于ONNXRuntime部署
│   │   ├── infer.py # ONNXRuntime推理部署示例
│   │   ├── predictor.py
│   │   └── README.md # 使用说明
│   ├── paddle_serving # 基于Paddle Serving 部署
│   │   ├──config.yml # 分类任务启动服务端的配置文件
│   │   ├──rpc_client.py # 分类任务发送pipeline预测请求的脚本
│   │   ├──service.py # 分类任务启动服务端的脚本
│   │   └── README.md # 使用说明
│   └── triton_serving # 基于Triton server部署
│       ├── README.md # 使用说明
│       ├── seqcls_grpc_client.py # 客户端测试代码
│       └── models # 部署模型
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
├── export_model.py # 动态图参数导出静态图参数脚本
├── utils.py # 工具函数脚本
├── metric.py # metric脚本
├── prune.py # 裁剪脚本
├── prune_trainer.py # 裁剪trainer脚本
├── prune_config.py # 裁剪训练参数配置
├── requirements.txt # 环境依赖
└── README.md # 使用说明
```
## 模型微调

我们以公开数据集CBLUE数据集中医疗搜索检索词意图分类(KUAKE-QIC)任务为示例，在训练集上进行模型微调，并在开发集上使用准确率Accuracy评估模型表现。

单卡训练
```shell
python train.py --warmup --early_stop --epochs 100
```

指定GPU卡号/多卡训练
```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" train.py --warmup --early_stop
```
使用多卡训练可以指定多个GPU卡号，例如 --gpus "0,1"

可支持配置的参数：

* `save_dir`：保存训练模型的目录；默认保存在当前目录checkpoint文件夹下。
* `dataset`：训练数据集;默认为"cblue"。
* `dataset_dir`：本地数据集路径，数据集路径中应包含train.txt，dev.txt和label.txt文件;默认为None。
* `task_name`：训练数据集;默认为"KUAKE-QIC"。
* `max_seq_length`：ERNIE模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `model_name`：选择预训练模型；默认为"ernie-3.0-base-zh"。
* `device`: 选用什么设备进行训练，可选cpu、gpu、xpu、npu。如使用gpu训练，可使用参数gpus指定GPU卡号。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：Fine-tune的最大学习率；默认为6e-5。
* `weight_decay`：控制正则项力度的参数，用于防止过拟合，默认为0.01。
* `early_stop`：选择是否使用早停法(EarlyStopping)；默认为False。
* `early_stop_nums`：在设定的早停训练轮次内，模型在开发集上表现不再上升，训练终止；默认为4。
* `epochs`: 训练轮次，默认为100。
* `warmup`：是否使用学习率warmup策略；默认为False。
* `warmup_proportion`：学习率warmup策略的比例数，如果设为0.1，则学习率会在前10%steps数从0慢慢增长到learning_rate, 而后再缓慢衰减；默认为0.1。
* `logging_steps`: 日志打印的间隔steps数，默认5。
* `init_from_ckpt`: 模型初始checkpoint参数地址，默认None。
* `seed`：随机种子，默认为3。


程序运行时将会自动进行训练，评估，测试。同时训练过程中会自动保存开发集上最佳模型在指定的 `save_dir` 中，保存模型文件结构如下所示：

```text
checkpoint/
├── model_config.json
├── model_state.pdparams
├── tokenizer_config.json
└── vocab.txt
```

**NOTE:**
* 如需恢复模型训练，则可以设置 `init_from_ckpt` ， 如 `init_from_ckpt=checkpoint/model_state.pdparams` 。
* 如需训练中文文本分类任务，只需更换预训练模型参数 `model_name` 。中文训练任务推荐使用"ernie-3.0-base-zh"，更多可选模型可参考[Transformer预训练模型](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer)。

### 从本地文件创建数据集
在许多情况，我们需要使用本地数据集来训练我们的文本分类模型，本项目支持使用固定格式本地数据集文件进行训练。如果需要对本地数据集进行数据标注，可以参考[文本分类任务doccano数据标注使用指南](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/applications/text_classification/doccano.md)进行文本分类数据标注。本项目将以CBLUE数据集中医疗搜索检索词意图分类(KUAKE-QIC)任务为例进行介绍如何加载本地固定格式数据集进行训练：


```shell
wget https://paddlenlp.bj.bcebos.com/datasets/KUAKE_QIC.tar.gz
tar -zxvf KUAKE_QIC.tar.gz
mv KUAKE_QIC data
```

本地数据集目录结构如下：

```text
data/
├── train.txt # 训练数据集文件
├── dev.txt # 开发数据集文件
├── label.txt # 分类标签文件
└── data.txt # 可选，待预测数据文件
```

train.txt(训练数据集文件), dev.txt(开发数据集文件),输入文本序列与标签类别名用`'\t'`分隔开。
- train.txt/dev.txt 文件格式：
```text
<输入序列1>'\t'<标签1>'\n'
<输入序列2>'\t'<标签2>'\n'
...
```
- train.txt/dev.txt 文件样例：
```text
25岁已经感觉脸部松弛了怎么办	治疗方案
小孩的眉毛剪了会长吗？	其他
172的身高还能长高吗？	其他
冻疮用三金冻疮酊有效果么？	功效作用
...
```


label.txt(分类标签文件)记录数据集中所有标签集合，每一行为一个标签名。
- label.txt 文件格式：
```text
<标签名1>'\n'
<标签名2>'\n'
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

data.txt(可选，待预测数据文件)。
- data.txt 文件格式：
```text
<输入序列1>'\n'
<输入序列2>'\n'
...
```
- data.txt 文件样例：
```text
黑苦荞茶的功效与作用及食用方法
交界痣会凸起吗
...
```
在训练过程中通过指定数据集路径参数`dataset_dir`进行：
单卡训练
```shell
python train.py --warmup --dataset_dir data
```

指定GPU卡号/多卡训练
```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" train.py --warmup --dataset_dir data
```
使用多卡训练可以指定多个GPU卡号，例如 --gpus "0,1"

更多数据集读取格式详见[数据集加载](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_load.html#)和[自定义数据集](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html)。


## 模型预测

输入待预测数据和数据标签对照列表，模型预测数据对应的标签

使用默认数据进行预测：
```shell
python predict.py --params_path ./checkpoint/
```
也可以选择使用本地数据文件data/data.txt进行预测：
```shell
python predict.py --params_path ./checkpoint/ --dataset_dir data
```
可支持配置的参数：

* `params_path`：待预测模型参数文件夹；默认为"./checkpoint/"。
* `dataset_dir`：本地数据集路径，数据集路径中应包含data.txt和label.txt文件;默认为None。
* `max_seq_length`：ERNIE模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为512。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `device`: 选用什么设备进行训练，可选cpu、gpu、xpu、npu；默认为gpu。

## 模型静态图导出

使用动态图训练结束之后，还可以将动态图参数导出成静态图参数，具体代码见[静态图导出脚本](export_model.py)。静态图参数保存在`output_path`指定路径中。运行方式：

```shell
python export_model.py --params_path ./checkpoint/ --output_path ./export
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


导出模型之后，可以用于部署，项目提供了[onnxruntime部署脚本](./deploy/predictor/infer.py),用法详见[ONNX Runtime推理部署指南](./deploy/predictor/README.md)。运行方式：

使用内置数据集进行部署：
```shell
python deploy/predictor/infer.py --model_path_prefix ./export/float32
```
也可以选择使用本地数据文件data/data.txt进行部署：
```shell
python deploy/predictor/infer.py --model_path_prefix ./export/float32 --dataset_dir data
```

此外，本项目还提供了基于[Paddle Serving](./deploy/paddle_serving)的服务化部署，用法详见[基于Paddle Serving的服务化部署](./deploy/predictor/README.md)。

## 模型裁剪
### 环境准备

使用裁剪功能需要安装 paddleslim 包

```shell
pip install paddleslim==2.2.2
```

### 裁剪 API 使用
本项目基于 PaddleNLP 的 Trainer API 发布提供了模型裁剪 API。裁剪 API 支持用户对 ERNIE 等Transformers 类下游任务微调模型进行裁剪，用户只需要简单地调用 `prune()` 即可一键启动裁剪和并自动保存裁剪后的模型。

可以这样使用裁剪 API (示例代码只提供了核心调用，如需跑通完整的例子可参考[完整样例脚本](prune.py)):

```python
trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        criterion=criterion)

output_dir = os.path.join(training_args.output_dir, data_args.dataset)

trainer.prune(output_dir, prune_config=DynabertConfig(width_mult=2/3))
```
由于裁剪 API 基于 Trainer，所以首先需要初始化一个 Trainer 实例，对于模型裁剪来说必要传入的参数如下：

* `model`：ERNIE 等模型在下游任务中微调后的模型，通过`AutoModelForSequenceClassification.from_pretrained(model_args.params_dir)` 来获取
* `data_collator`：使用 PaddleNLP 预定义好的[DataCollator 类](../../../paddlenlp/data/data_collator.py)，`data_collator` 可对数据进行 `Pad` 等操作,使用方法参考本项目中代码即可
* `train_dataset`：裁剪训练需要使用的训练集
* `eval_dataset`：裁剪训练使用的评估集(开发集)
* `tokenizer`：模型`model`对应的 `tokenizer`，可使用 `AutoTokenizer.from_pretrained(model_args.params_dir)` 来获取
* `criterion`： 定义criterion计算损失，分类中使用损失函数 paddle.nn.BCEWithLogitsLoss()

然后可以直接调用 `prune` 启动裁剪，其中 `prune` 的参数释义如下：
* `output_dir`：裁剪后模型保存目录
* `prune_config`：裁剪配置，目前裁剪配置仅支持`DynabertConfig`类。

当默认参数不满足需求时，可通过传入参数对裁剪过程进行特殊配置，`DynabertConfig`中可以传的参数有：
* `width_mult_list`：裁剪宽度保留的比例，表示对 `q`、`k`、`v` 以及 `ffn` 权重宽度的保留比例，默认是 `2/3`
* `output_filename_prefix`：裁剪导出模型的文件名前缀，默认是`"float32"`


选择使用默认数据集启动裁剪：
```shell
python prune.py --output_dir ./prune --params_dir ./checkpoint/
```
也可以选择使用本地数据文件启动裁剪：
```shell
python prune.py --output_dir ./prune --params_dir ./checkpoint/ --dataset_dir data
```

可支持配置的参数：
* `TrainingArguments`
  * `output_dir`：必须，保存模型输出和和中间checkpoint的输出目录;默认为 `None` 。
  * `TrainingArguments` 包含了用户需要的大部分训练参数，所有可配置的参数详见[TrainingArguments 参数介绍](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/trainer.md#trainingarguments-%E5%8F%82%E6%95%B0%E4%BB%8B%E7%BB%8D)，示例通过`prune_config.json`对TrainingArguments 参数进行配置。

* `DataArguments`
  * `dataset`：训练数据集;默认为 cblue 数据集。
  * `task_name`：训练数据集任务名;默认为"KUAKE-QIC"。
  * `dataset_dir`：本地数据集路径，需包含train.txt,dev.txt,label.txt;默认为None。
  * `max_seq_length`：ERNIE模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。

* `ModelArguments`
  * `params_dir`：待预测模型参数文件；默认为"./checkpoint/"。

以上参数都可通过 `python prune.py --dataset xx --params_dir xx` 的方式传入）

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

1. 目前支持的裁剪策略需要训练，训练时间视下游任务数据量而定，且和微调的训练时间是一个量级；

2. 裁剪类似蒸馏过程，方便起见，可以直接使用微调时的超参。为了进一步提升精度，可以对 `per_device_train_batch_size`、`learning_rate`、`num_train_epochs`、`max_seq_length` 等超参进行 grid search；

3. 模型裁剪主要用于推理部署，因此裁剪后的模型都是静态图模型，只可用于推理部署，不能再通过 `from_pretrained` 导入继续训练。

4. 导出模型之后用于部署，项目提供了[onnxruntime部署脚本](./deploy/predictor/infer.py)，用法详见[ONNX Runtime推理部署指南](./deploy/predictor/README.md)。运行方式：

使用内置数据集进行部署：
```shell
python deploy/preditor/infer.py --model_path_prefix ./prune/0.6666666666666666/float32
```
也可以选择使用本地数据文件data/data.txt进行部署：
```shell
python deploy/preditor/infer.py --model_path_prefix ./prune/0.6666666666666666/float32 --dataset_dir data
```
5. 本项目提供了基于[Paddle Serving](./deploy/paddle_serving)的服务化部署，用法详见[基于Paddle Serving的服务化部署](./deploy/predictor/README.md)。

### 裁剪效果
本案例我们对ERNIE 3.0模型微调后的模型使用裁剪 API 进行裁剪,将模型转为ONNX模型，并基于ONNXRuntime引擎GPU部署，测试配置如下：

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

5. 性能数据指标：latency。latency 测试方法：固定 batch size 为 200，GPU部署运行时间 total_time，计算 latency = total_time / total_samples

6. 精度评价指标：Accuracy

|                            | Accuracy(%)   | latency(ms) |
| -------------------------- | ------------ | ------------- |
| ERNIE 3.0             | 81.43 | 1.95  |
| ERNIE 3.0+裁剪(保留比例3/4)    | 82.53(+1.1) | 1.55   |
| ERNIE 3.0+裁剪(保留比例2/3)    | 82.20(+0.77)  | 1.34  |
| ERNIE 3.0+裁剪(保留比例1/2)    | 81.13(-0.3) | 1.06   |

## 模型部署


- 服务化部署请参考：[基于Paddle Serving的服务化部署指南](deploy/paddle_serving/README.md)，Paddle Serving支持X86、Arm CPU、NVIDIA GPU、昆仑/昇腾等多种硬件的服务化部署

- ONNXRuntime 部署请参考：[ONNX导出及ONNXRuntime部署指南](deploy/predictor/README.md)

- 基于ONNXRuntime的服务化部署请参考：[基于Triton Inference Server的服务化部署指南](deploy/triton_serving/README.md)
