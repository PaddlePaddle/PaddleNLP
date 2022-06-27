# 多标签层次分类任务

**目录**
   * [多标签层次分类任务介绍](#层次分类任务介绍)
   * [代码结构说明](#代码结构说明)
   * [模型微调](#模型微调)
       * [以内置数据集格式读取本地数据集](#以内置数据集格式读取本地数据集)
   * [模型预测](#模型预测)
   * [模型静态图导出](#模型静态图导出)
   * [模型裁剪](#模型裁剪)
       * [环境准备](#环境准备)
       * [裁剪API使用](#裁剪API使用)
       * [裁剪效果](#裁剪效果)
   * [模型部署](#模型部署)


## 层次分类任务介绍

多标签层次分类任务指自然语言处理任务中，每个样本具有多个标签标记，并且标签集合中标签之间存在预定义的层次结构，多标签层次分类需要充分考虑标签集之间的层次结构关系来预测层次化预测结果。层次分类任务中标签层次结构分为两类，一类为树状结构，另一类为有向无环图(DAG)结构。有向无环图结构与树状结构区别在于，有向无环图中的节点可能存在不止一个父节点。在现实场景中，大量的数据如新闻分类、专利分类、学术论文分类等标签集合存在层次化结构，需要利用算法为文本自动标注更细粒度和更准确的标签。

层次分类问题可以被视为一个多标签问题，以下图一个树状标签结构(宠物为根节点)为例，如果一个样本属于美短虎斑，样本也天然地同时属于类别美国短毛猫和类别猫两个样本标签。本项目采用通用多标签层次分类算法，将每个结点的标签路径视为一个多分类标签，使用单个多标签分类器进行决策。以上面的例子为例，该样本包含三个标签：猫、猫--美国短毛猫、猫--美国短毛猫--美短虎斑。

<div align="center">
    <img src="https://user-images.githubusercontent.com/63761690/175248039-ce1673f1-9b03-4804-b1cb-29e4b4193f86.png" width="600">
</div>

## 代码结构说明

以下是本项目主要代码结构及说明：

```text
hierarchical_classification/
├── deploy # 部署
│   └── predictor # 导出ONNX模型并基于ONNXRuntime部署
│   │   ├── infer.py # ONNXRuntime推理部署示例
│   │   ├── predictor.py
│   │   └── README.md # 使用说明
│   ├── paddle_serving # 基于Paddle Serving 部署
│   │   ├──config.yml # 层次分类任务启动服务端的配置文件
│   │   ├──rpc_client.py # 层次分类任务发送pipeline预测请求的脚本
│   │   └──service.py # 层次分类任务启动服务端的脚本
│   └── triton_serving # 基于Triton server 部署
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

请使用以下命令安装所需依赖

```shell
pip install -r requirements.txt
```

我们以层次分类公开数据集WOS(Web of Science)为示例，在训练集上进行模型微调，并在开发集上验证。WOS数据集是一个两层的层次文本分类数据集，包含7个父类和134子类，每个样本对应一个父类标签和子类标签，父类标签和子类标签间具有树状层次结构关系。


单卡训练
```shell
python train.py --early_stop
```

指定GPU卡号/多卡训练
```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" train.py --early_stop
```
使用多卡训练可以指定多个GPU卡号，例如 --gpus "0,1"

可支持配置的参数：

* `save_dir`：保存训练模型的目录；默认保存在当前目录checkpoints文件夹下。
* `dataset_dir`：本地训练数据集;默认为None。
* `dataset`：训练数据集;默认为wos数据集。
* `max_seq_length`：ERNIE/BERT模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为512。
* `model_name`：选择预训练模型；默认为"ernie-2.0-base-en"，中文数据集推荐使用"ernie-3.0-base-zh"。
* `device`: 选用什么设备进行训练，可选cpu、gpu、xpu、npu。如使用gpu训练则参数gpus指定GPU卡号。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为12。
* `learning_rate`：Fine-tune的最大学习率；默认为3e-5。
* `weight_decay`：控制正则项力度的参数，用于防止过拟合，默认为0.00。
* `early_stop`：选择是否使用早停法(EarlyStopping)；默认为False。
* `early_stop_nums`：在设定的早停训练轮次内，模型在开发集上表现不再上升，训练终止；默认为6。
* `epochs`: 训练轮次，默认为1000。
* `warmup`：是否使用学习率warmup策略；默认为False。
* `warmup_steps`：学习率warmup策略的steps数，如果设为2000，则学习率会在前2000 steps数从0慢慢增长到learning_rate, 而后再缓慢衰减；默认为2000。
* `logging_steps`: 日志打印的间隔steps数，默认100。
* `init_from_ckpt`: 模型初始checkpoint参数地址，默认None。
* `seed`：随机种子，默认为3。
* `depth`：层次结构最大深度，默认为2。


程序运行时将会自动进行训练，评估，测试。同时训练过程中会自动保存开发集上最佳模型在指定的 `save_dir` 中，保存模型文件结构如下所示：

```text
checkpoint/
├── model_config.json
├── model_state.pdparams
├── tokenizer_config.json
└── vocab.txt

```

**NOTE:**
* 如需恢复模型训练，则可以设置 `init_from_ckpt` ， 如 `init_from_ckpt=checkpoints/macro/model_state.pdparams` 。
* 如需训练中文层次分类任务，只需更换预训练模型参数 `model_name` 。中文训练任务推荐使用"ernie-3.0-base-zh"，更多可选模型可参考[Transformer预训练模型](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer)。

### 以内置数据集格式读取本地数据集

在许多情况下，我们希望使用本地的数据集来训练层次分类模型。本项目将以wos数据集为例，介绍如何使用本地指定格式数据集文件以内置数据集方式读取层次分类训练数据，本项目默认不同层的标签之间存在层次关系。

下载wos数据集到本地(如果有本地数据集，跳过此步骤)

```shell
wget https://paddlenlp.bj.bcebos.com/datasets/wos_data.tar.gz

tar -zxvf wos_data.tar.gz
```

本地数据集目录结构如下：

```text
wos_data/
├── train.tsv # 训练数据集文件
├── dev.tsv # 开发数据集文件
├── test.tsv # 可选，测试训练集文件
├── taxonomy.tsv # 层次分类标签文件
└── data.tsv # 可选，待预测数据文件

```

- train.tsv(训练数据集文件), dev.tsv(开发数据集文件), test.tsv(可选，测试训练集文件)中数据格式为：
```text

<输入序列1>'\t'<level 1 标签>'\t'<level 2 标签>'\t'...'\t'<level n 标签>'\n'

<输入序列2>'\t'<level 1 标签>'\t'<level 2 标签>'\t'...'\t'<level n 标签>'\n'

...

```
其中n表示标签层次结构中最大层数，<level i 标签> 代表数据的第i层标签。输入文本序列及不同层的标签数据用`'\t'`分隔开，每一层标签中多个标签之间用`','`逗号分隔开。注意，对于第i层数据没有标签的，使用空字符`''`来表示<level i 标签>。

- taxonomy.tsv(层次分类标签文件)记录数据集中所有标签路径集合，标签格式为：

```text

<level 1: 标签1>'\n'

<level 1: 标签1>'--'<level 2: 标签1>'\n'

...

<level 1: 标签 i1>'--'...'--'<level n: 标签in>'\n'

```

标签路径中，高层的标签指向底层标签，标签之间用`'--'`连接，本项目选择为标签层次结构中的每一个节点生成对应的标签路径。

- data.tsv(可选，待预测数据文件)，数据格式为：

```text

<输入序列1>'\n'

<输入序列2>'\n'

...

在训练过程中通过指定数据集路径参数`dataset_dir`进行训练：

```

单卡训练
```shell

python train.py --early_stop --dataset_dir 'wos_data'

```

指定GPU卡号/多卡训练
```shell

unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus "0" train.py --early_stop --dataset_dir 'wos_data'

```
使用多卡训练可以指定多个GPU卡号，例如 --gpus "0,1"

更多数据集读取格式详见[数据集加载](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_load.html#)和[自定义数据集](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html)。

## 模型预测

输入待预测数据和数据标签对照列表，模型预测数据对应的标签

启动预测：
```shell
python predict.py --params_path ./checkpoints/model_state.pdparams
```

也可以通过指定本地数据文件路径进行预测：
```shell
python predict.py --params_path ./checkpoints/model_state.pdparams --dataset_dir "wos_data"
```

可支持配置的参数：

* `params_path`：待预测模型参数文件；默认为"./checkpoint/model_state.pdparams"。
* `max_seq_length`：ERNIE/BERT模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为512。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为12。
* `device`: 选用什么设备进行训练，可选cpu、gpu、xpu、npu；默认为gpu。
* `model_name`：选择预训练模型；默认为"ernie-2.0-base-en"，中文数据集推荐使用"ernie-3.0-base-zh"。
* `depth`：层次结构最大深度，默认为2。
* `dataset_dir`：本地训练数据集;默认为None。


## 模型静态图导出

使用动态图训练结束之后，还可以将动态图参数导出成静态图参数，具体代码见[静态图导出脚本](export_model.py)。静态图参数保存在`output_path`指定路径中。运行方式：

```shell
python export_model.py --params_path=./checkpoint/model_state.pdparams --output_path=./export
```
可支持配置的参数：

* `params_path`：动态图训练保存的参数路径；默认为"./checkpoint/model_state.pdparams"。
* `output_path`：静态图图保存的参数路径；默认为"./export"。
* `num_classes`：任务标签类别数;默认为wos数据集类别数141。
* `model_name`：选择预训练模型；默认为"ernie-2.0-base-en"，中文数据集推荐使用"ernie-3.0-base-zh"。

程序运行时将会自动导出模型到指定的 `output_path` 中，保存模型文件结构如下所示：

```text
export/
├── float32.pdiparams
├── float32.pdiparams.info
└── float32.pdmodel
```


导出模型之后，可以用于部署，项目提供了[onnxruntime部署预测示例](./deploy/predictor/infer.py),用法详见[ONNX Runtime推理部署](./deploy/predictor/README.md)。运行方式：

```shell
# 使用默认数据集
python deploy/predictor/infer.py --model_path_prefix ./export/float32

# 使用本地数据集
python deploy/predictor/infer.py --model_path_prefix ./export/float32 --data_dir "wos_data"
```

## 模型裁剪
### 环境准备

使用裁剪功能需要安装 paddleslim 包

```shell
pip install paddleslim
```

### 裁剪 API 使用
本项目基于 PaddleNLP 的 Trainer API 发布提供了模型裁剪 API。裁剪 API 支持用户对 ERNIE、BERT 等Transformers 类下游任务微调模型进行裁剪，用户只需要简单地调用 `prune()` 即可一键启动裁剪和并自动保存裁剪后的模型。

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

* `model`：ERNIE、BERT 等模型在下游任务中微调后的模型，通过`AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path)` 来获取
* `data_collator`：使用 PaddleNLP 预定义好的[DataCollator 类](../../../paddlenlp/data/data_collator.py)，`data_collator` 可对数据进行 `Pad` 等操作,使用方法参考本项目中代码即可
* `train_dataset`：裁剪训练需要使用的训练集
* `eval_dataset`：裁剪训练使用的评估集
* `tokenizer`：模型`model`对应的 `tokenizer`，可使用 `AutoTokenizer.from_pretrained(model_args.model_name_or_path)` 来获取
* `criterion`： 定义criterion计算损失，层次分类中使用损失函数 paddle.nn.BCEWithLogitsLoss()

然后可以直接调用 `prune` 启动裁剪，其中 `prune` 的参数释义如下：
* `output_dir`：裁剪后模型保存目录
* `prune_config`：裁剪配置，目前裁剪配置仅支持`DynabertConfig`类。

当默认参数不满足需求时，可通过传入参数对裁剪过程进行特殊配置，`DynabertConfig`中可以传的参数有：
* `width_mult_list`：裁剪宽度保留的比例，表示对 `q`、`k`、`v` 以及 `ffn` 权重宽度的保留比例，默认是 `2/3`
* `output_filename_prefix`：裁剪导出模型的文件名前缀，默认是`"float32"`


启动裁剪：
```shell
# 使用默认数据集
python prune.py --output_dir ./prune --params_dir ./checkpoint/model_state.pdparams

# 使用本地数据集
python prune.py --output_dir ./prune --params_dir ./checkpoint/model_state.pdparams --data_dir "wos_data"
```


可支持配置的参数：
* `TrainingArguments`
  * `output_dir`：必须，保存模型输出和和中间checkpoints的输出目录;默认为 `None` 。
  * `TrainingArguments` 包含了用户需要的大部分训练参数，所有可配置的参数详见[TrainingArguments 参数介绍](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/trainer.md#trainingarguments-%E5%8F%82%E6%95%B0%E4%BB%8B%E7%BB%8D)，示例默认通过`prune_config.json`对TrainingArguments 参数进行配置

* `DataArguments`
  * `dataset`：训练数据集;默认为wos数据集。
  * `dataset`：本地数据集路径，路径内需要包含train.tsv, dev.tsv, taxonomy.tsv文件;默认为None。
  * `depth`：层次分类数据标签最大深度;默认为2。
  * `max_seq_length`：ERNIE/BERT模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为512。

* `ModelArguments`
  * `params_dir`：待预测模型参数文件；默认为"./checkpoint/model_state.pdparams"。
  * `model_name_or_path`：选择预训练模型；默认为"ernie-2.0-base-en"，中文数据集推荐使用"ernie-3.0-base-zh"。

以上参数都可通过 `python prune.py --dataset xx --params_dir xx` 的方式传入）

程序运行时将会自动进行训练，评估，测试。同时训练过程中会自动保存开发集上最佳模型在指定的 `output_dir` 中，保存模型文件结构如下所示：

```text
prune/
├── 0.25
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

导出模型之后用于部署，项目提供了[onnxruntime部署预测示例](./deploy/predictor/infer.py)，用法详见[ONNX Runtime推理部署](./deploy/predictor/README.md)。运行方式：

```shell
# 使用内置数据集
python deploy/predictor/infer.py --model_path_prefix ./prune/0.25/float32

#使用本地数据集
python deploy/predictor/infer.py --model_path_prefix ./prune/0.25/float32 --data_dir "wos_data"
```

### 裁剪效果
本案例我们对ERNIE 2.0模型微调后的模型使用裁剪 API 进行裁剪,将模型转为ONNX模型，并基于ONNXRuntime引擎GPU部署，测试配置如下：

1. 数据集：WOS（英文层次分类测试集）

2. 物理机环境

    系统: CentOS Linux release 7.7.1908 (Core)

    GPU: Tesla V100-SXM2-32GB * 8

    CPU: Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz

    CUDA: 11.2

    cuDNN: 8.1.0

    Driver Version: 460.27.04

    内存: 630 GB

3. PaddlePaddle 版本：2.3.0

4. PaddleNLP 版本：2.3.1

5. 性能数据指标：latency。latency 测试方法：固定 batch size 为 200，GPU部署运行时间 total_time，计算 latency = total_time / total_samples

6. 精度评价指标：Micro F1 和 Macro F1


|                            | Micro F1   | Macro F1   | latency(ms) |
| -------------------------- | ------------ | ------------ | ------------- |
| ERNIE 2.0             | 85.71 | 80.82 | 8.80  |
| ERNIE 2.0+裁剪(保留比例3/4)    | 86.83(+1.12) | 81.78(+0.96) | 6.85   |
| ERNIE 2.0+裁剪(保留比例2/3)    | 86.74(+1.03) | 81.64(+0.82) | 5.98  |
| ERNIE 2.0+裁剪(保留比例1/4)    | 85.79(+0.08) | 79.53(-1.29) | 2.51   |

## 模型部署


- 服务化部署请参考：[基于Paddle Serving的服务化部署指南](deploy/paddle_serving/README.md)，Paddle Serving支持X86、Arm CPU、NVIDIA GPU、昆仑/昇腾等多种硬件的服务化部署

- ONNXRuntime 部署请参考：[ONNX导出及ONNXRuntime部署指南](deploy/predictor/README.md)

- 基于ONNXRuntime的服务化部署请参考：[基于Triton Inference Server的服务化部署指南](deploy/triton_serving/README.md)
