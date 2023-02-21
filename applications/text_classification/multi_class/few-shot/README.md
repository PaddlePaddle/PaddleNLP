# 小样本场景下的二/多分类任务指南

**零样本/小样本文本分类推荐使用 UTC 模型，详情见[目录](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/zero_shot_text_classification)，本项目将会在2.5.2版本下线。**

## 目录

- [1. 项目说明](#项目说明)
- [2. 效果展示](#效果展示)
- [3. 定制训练](#定制训练)
  - [3.1 运行环境](#运行环境)
  - [3.2 代码结构](#代码结构)
  - [3.3 数据标注](#数据标注)
  - [3.4 模型训练](#模型训练)
  - [3.5 模型评估](#模型评估)
  - [3.6 模型部署](#模型部署)
- [4. References](#References)

<a name="项目说明"></a>
## 1. 项目说明

本项目提供了小样本场景下文本二/多分类的解决方案，在 ERNIE3.0 的基础上利用提示学习取得比微调更好的分类效果，充分利用标注信息。

### 模型介绍

**文本二/多分类** 用于预测样本属于标签候选集中的哪个类别，在商品分类、网页分类、新闻分类、医疗文本分类等现实场景中有着广泛应用。
现有的主流解决方案是在预训练语言模型上进行微调，因为二/多分类任务与预训练阶段的掩码预测任务有着天然的差异，想要取得较好的分类效果往往需要大量数据标注。

**提示学习(Prompt Learning)** 的主要思想是将二/多分类任务转换为掩码预测任务，充分利用预训练语言模型学习到的特征，从而降低样本需求。以情感分类任务为例，标签分为`1-正向`，`0-负向`两类，如下图所示，通过提示`我[MASK]喜欢。`，原有`1-正向`，`0-负向`的标签被转化为了预测空格是`很`还是`不`。

<div align="center">
    <img src=https://user-images.githubusercontent.com/25607475/183909263-6ead8871-699c-4c2d-951f-e33eddcfdd9c.png width=800 height=300 />
</div>

微调方法和提示方法的区别如图所示：

【微调学习】需要学习的参数是以 `[CLS]` 向量为输入，以负向/正向为输出的随机初始化的分类器。

【提示学习】通过构造提示，将原有的分类任务转化为掩码预测，即掩盖原句中的某个字，用模型预测该字。此时的分类器不再是随机初始化，而是利用了待预测字的预训练向量来初始化，充分利用了预训练模型学习到的参数。

【方案选择】对于标注样本充足的场景可以直接使用[微调学习](../README.md)实现文本多分类，对于尚无标注或者标注样本较少的任务场景我们推荐使用提示学习，以取得更好的效果。

### 方案特点

- **标注成本低**：以往的微调方式需要大量的数据标注才能保证模型分类效果。提示学习可以降低数据标注依赖，在少样本（few-shot）的场景下取得比微调更好的分类效果。
- **全流程打通**：提供了从训练到部署的完整解决方案，可以低成本迁移至实际应用场景。

<a name="效果展示"></a>
## 2.效果展示

本项目中使用了 ERNIE3.0 模型，对于中文训练任务可以根据需求选择不同的预训练模型参数进行训练，我们测评了 Base 模型在新闻分类任务上的表现。测试配置如下：

1. 数据集：[FewCLUE](https://github.com/CLUEbenchmark/FewCLUE)中的新闻分类（tnews）任务测试集。

2. 物理机环境

   系统: CentOS Linux release 7.7.1908 (Core)

   GPU: Tesla V100-SXM2-32GB

   CPU: Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz

   CUDA: 11.2

   cuDNN: 8.1.0

   Driver Version: 460.27.04

   内存: 630 GB

3. PaddlePaddle 版本：2.4rc

4. PaddleNLP 版本：2.4.3

5. 评估设置

   每个 epoch 评估一次，按照验证集上的评价指标，取验证集上分数最高的模型参数用于测试集的测试。为了避免过拟合，这里使用了早停机制 (Early-stopping)。表格中的最终结果为重复 10 次的均值。

- 微调

```
cd ../
python train.py --dataset_dir "./data/" --save_dir "./checkpoints" --max_seq_length 128 --model_name "ernie-3.0-base-zh" --batch_size 8 --learning_rate 3e-5 --epochs 100 --logging_steps 5 --early_stop
```

- 提示学习

```
python train.py --data_dir ./data/ --output_dir ./checkpoints/ --prompt "这条新闻写的是" --model_name_or_path ernie-3.0-base-zh --max_seq_length 128  --learning_rate 3e-5 --ppt_learning_rate 3e-4 --do_train --do_eval --num_train_epochs 100 --logging_steps 5 --per_device_eval_batch_size 32 --per_device_train_batch_size 8 --do_predict --metric_for_best_model accuracy --load_best_model_at_end --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1
```

6. 精度评价指标：Accuracy


| model_name | 训练方式 | Accuracy |
| ---------- | ------- | ---- |
| ernie-3.0-base-zh | 微调学习 | 0.5046 |
| ernie-3.0-base-zh | 提示学习 | 0.5521 |


<a name="定制训练"></a>
## 3.定制训练

下边通过**新闻分类**的例子展示如何使用小样本学习来进行文本分类。

<a name="运行环境"></a>
### 3.1 运行环境

- python >= 3.7
- paddlepaddle >= 2.4rc
- paddlenlp >= 2.4.3
- paddle2onnx >= 1.0.3

<a name="代码结构"></a>
### 3.2 代码结构

```text
.
├── train.py    # 模型组网训练脚本
├── utils.py    # 数据处理工具
├── infer.py    # 模型部署脚本
└── README.md
```

<a name="数据标注"></a>
### 3.3 数据标注

我们推荐使用数据标注平台[doccano](https://github.com/doccano/doccano)进行自定义数据标注，本项目也打通了从标注到训练的通道，即doccano导出数据后可通过[doccano.py](../../doccano.py)脚本轻松将数据转换为输入模型时需要的形式，实现无缝衔接。标注方法的详细介绍请参考[doccano数据标注指南](../../doccano.md)。

**示例数据**

这里我们使用[FewCLUE](https://github.com/CLUEbenchmark/FewCLUE)中的新闻分类tnews数据集后缀为0的子集作为示例数据集，可点击[这里](https://paddlenlp.bj.bcebos.com/datasets/few-shot/tnews.tar.gz)下载解压并放入`./data/`文件夹，或者运行以下脚本

```
wget https://paddlenlp.bj.bcebos.com/datasets/few-shot/tnews.tar.gz
tar zxvf tnews.tar.gz
mv tnews data
```

**数据格式**

下边主要介绍二/多分类任务自定义数据集的格式要求，整体目录如下

```text
data/
├── train.txt  # 训练数据集
├── dev.txt    # 验证数据集
├── test.txt   # 测试数据集（可选）
├── data.txt   # 待预测数据（可选）
└── label.txt  # 分类标签集
```

**训练/验证/测试数据**

对于训练/验证/测试数据集文件，每行数据表示一条样本，包括文本和标签两部分，由tab符`\t`分隔。格式如下
```text
<文本>'\t'<标签>
<文本>'\t'<标签>
...
```
例如，在新闻分类数据集中
```
文登区这些公路及危桥将进入封闭施工，请注意绕行！ news_car
普洱茶要如何醒茶？ news_culture
...
```

**预测数据**

对于待预测数据文件，每行包含一条待预测样本，无标签。格式如下
```text
<文本>
<文本>
...
```
例如，在新闻分类数据集中
```
互联网时代如何保护个人信息
清秋暮雨读柳词：忍把浮名，换了浅斟低唱丨周末读诗
...
```

**标签数据**

对于分类标签集文件，存储了数据集中所有的标签集合，每行为一个标签名。如果需要自定义标签映射用于分类器初始化，则每行需要包括标签名和相应的映射词，由`==`分隔。格式如下
```text
<标签>'=='<映射词>
<标签>'=='<映射词>
...
```
例如，对于新闻分类数据集，原标签`news_car`可被映射为中文`汽车`等等。
```
news_car==汽车
news_culture==文化
...
```
**Note**: 这里的标签映射词定义遵循的规则是，不同映射词尽可能长度一致，映射词和提示需要尽可能构成通顺的语句。越接近自然语句，小样本下模型训练效果越好。如果原标签名已经可以构成通顺语句，也可以不构造映射词，每行一个标签即可。

<a name="模型训练"></a>
### 3.4 模型训练

**单卡训练**

```
python train.py \
--device gpu \
--data_dir ./data \
--output_dir ./checkpoints/ \
--prompt "这条新闻标题的主题是" \
--max_seq_length 128  \
--learning_rate 3e-6 \
--ppt_learning_rate 3e-5 \
--do_train \
--do_eval \
--use_rdrop \
--max_steps 1000 \
--eval_steps 10 \
--logging_steps 5 \
--save_total_limit 1 \
--load_best_model_at_end True \
--per_device_eval_batch_size 32 \
--per_device_train_batch_size 8 \
--do_predict \
--do_export
```
**多卡训练**

```
unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus 0,1,2,3 train.py \
--data_dir ./data \
--output_dir ./checkpoints/ \
--prompt "这条新闻标题的主题是" \
--max_seq_length 128  \
--learning_rate 3e-6 \
--ppt_learning_rate 3e-5 \
--do_train \
--do_eval \
--use_rdrop \
--do_eval \
--max_steps 1000 \
--eval_steps 10 \
--logging_steps 5 \
--save_total_limit 1 \
--load_best_model_at_end True \
--per_device_eval_batch_size 32 \
--per_device_train_batch_size 8 \
--do_predict \
--do_export
```

可配置参数说明：
- `model_name_or_path`: 内置模型名，或者模型参数配置目录路径。默认为`ernie-3.0-base-zh`。
- `data_dir`: 训练数据集路径，数据格式要求详见[数据标注](#数据标注)。
- `output_dir`: 模型参数、训练日志和静态图导出的保存目录。
- `prompt`: 提示模板。定义了如何将文本和提示拼接结合。
- `soft_encoder`: 提示向量的编码器，`lstm`表示双向LSTM, `mlp`表示双层线性层, None表示直接使用提示向量。默认为`lstm`。
- `use_rdrop`: 使用 [R-Drop](https://arxiv.org/abs/2106.14448) 策略。
- `use_rgl`: 使用 [RGL](https://aclanthology.org/2022.findings-naacl.81/) 策略。
- `encoder_hidden_size`: 提示向量的维度。若为None，则使用预训练模型字向量维度。默认为200。
- `max_seq_length`: 最大句子长度，超过该长度的文本将被截断，不足的以Pad补全。提示文本不会被截断。
- `learning_rate`: 预训练语言模型参数基础学习率大小，将与learning rate scheduler产生的值相乘作为当前学习率。
- `ppt_learning_rate`: 提示相关参数的基础学习率大小，当预训练参数不固定时，与其共用learning rate scheduler。一般设为`learning_rate`的十倍。
- `do_train`: 是否进行训练。
- `do_eval`: 是否进行评估。
- `do_predict`: 是否进行预测。
- `do_export`: 是否在运行结束时将模型导出为静态图，保存路径为`output_dir/export`。
- `max_steps`: 训练的最大步数。此设置将会覆盖`num_train_epochs`。
- `save_total_limit`: 模型检查点保存数量。
- `eval_steps`: 评估模型的间隔步数。
- `device`: 使用的设备，默认为`gpu`。
- `logging_steps`: 打印日志的间隔步数。
- `per_device_train_batch_size`: 每次训练每张卡上的样本数量。可根据实际GPU显存适当调小/调大此配置。
- `per_device_eval_batch_size`: 每次评估每张卡上的样本数量。可根据实际GPU显存适当调小/调大此配置。

更多参数介绍可参考[配置文件](https://paddlenlp.readthedocs.io/zh/latest/trainer.html)。

<a name="模型评估"></a>
### 3.5 模型评估

在模型训练时开启`--do_predict`，训练结束后直接在测试集上`test.txt`进行评估，也可以在训练结束后，通过运行以下命令加载模型参数进行评估：
```
python train.py --do_predict --data_dir ./data --output_dir ./predict_checkpoint --resume_from_checkpoint ./checkpoints/ --max_seq_length 128
```

可配置参数说明：

- `data_dir`: 测试数据路径。测试数据应存放在该目录下`test.txt`文件中，每行一条待预测文本。
- `output_dir`: 日志的保存目录。
- `resume_from_checkpoint`: 训练时模型参数的保存目录，用于加载模型参数。
- `do_predict`: 是否进行预测。
- `max_seq_length`: 最大句子长度，超过该长度的文本将被截断，不足的以Pad补全。提示文本不会被截断。

<a name="模型部署"></a>
### 3.6 模型部署

#### 模型导出

在训练结束后，需要将动态图模型导出为静态图参数用于部署推理。可以在模型训练时开启`--do_export`在训练结束后直接导出，也可以运行以下命令加载并导出训练后的模型参数，默认导出到在`output_dir`指定的目录下。
```
python train.py --do_export --data_dir ./data --output_dir ./export_checkpoint --resume_from_checkpoint ./checkpoints/
```

可配置参数说明：

- `data_dir`: 标签数据路径。
- `output_dir`: 静态图模型参数和日志的保存目录。
- `resume_from_checkpoint`: 训练时模型参数的保存目录，用于加载模型参数。
- `do_export`: 是否将模型导出为静态图，保存路径为`output_dir/export`。
- `export_type`: 模型导出的格式，默认为`paddle`，即导出静态图。

#### ONNXRuntime部署

**运行环境**

模型转换与ONNXRuntime预测部署依赖Paddle2ONNX和ONNXRuntime，Paddle2ONNX支持将Paddle静态图模型转化为ONNX模型格式，算子目前稳定支持导出ONNX Opset 7~15，更多细节可参考：[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)。

- 如果基于GPU部署，请先确保机器已正确安装NVIDIA相关驱动和基础软件，确保CUDA >= 11.2，CuDNN >= 8.2，并使用以下命令安装所需依赖:
```shell
pip install psutil
python -m pip install onnxruntime-gpu onnx onnxconverter-common
```

- 如果基于CPU部署，请使用如下命令安装所需依赖:
```shell
pip install psutil
python -m pip install onnxruntime
```

**CPU端推理样例**

```
python infer.py --model_path_prefix checkpoints/export/model --data_dir ./data --batch_size 32 --device cpu
```

**GPU端推理样例**

```
python infer.py --model_path_prefix checkpoints/export/model --data_dir ./data --batch_size 32 --device gpu --device_id 0
```

可配置参数说明：

- `model_path_prefix`: 导出的静态图模型路径及文件前缀。
- `model_name`: 内置预训练模型名，用于加载tokenizer。默认为`ernie-3.0-base-zh`。
- `data_dir`: 待推理数据所在路径，数据应存放在该目录下的`data.txt`文件。
- `max_length`: 最大句子长度，超过该长度的文本将被截断，不足的以Pad补全。提示文本不会被截断。
- `batch_size`: 每次预测的样本数量。
- `device`: 选择推理设备，包括`cpu`和`gpu`。默认为`gpu`。
- `device_id`: 指定GPU设备ID。
- `use_fp16`: 是否使用半精度加速推理。仅在GPU设备上有效。
- `num_threads`: 设置CPU使用的线程数。默认为机器上的物理内核数。

**Note**: 在GPU设备的CUDA计算能力 (CUDA Compute Capability) 大于7.0，在包括V100、T4、A10、A100、GTX 20系列和30系列显卡等设备上可以开启FP16进行加速，在CPU或者CUDA计算能力 (CUDA Compute Capability) 小于7.0时开启不会带来加速效果。

<a name="References"></a>
## 4. References

- Liu, Xiao, et al. "GPT understands, too." arXiv preprint arXiv:2103.10385 (2021). [[PDF]](https://arxiv.org/abs/2103.10385)
- Hambardzumyan, Karen, Hrant Khachatrian, and Jonathan May. "Warp: Word-level adversarial reprogramming." arXiv preprint arXiv:2101.00121 (2021). [[PDF]](https://arxiv.org/abs/2101.00121)
- Ding, Ning, et al. "Openprompt: An open-source framework for prompt-learning." arXiv preprint arXiv:2111.01998 (2021). [[PDF]](https://arxiv.org/abs/2111.01998)
