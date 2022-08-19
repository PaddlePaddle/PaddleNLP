# 小样本多标签分类任务指南

## 目录

  * [小样本学习简介](小样本学习简介)
  * [环境要求](环境要求)
  * [训练示例](训练示例)
    * [代码结构](代码结构)
    * [数据准备](数据准备)
    * [模型训练](模型训练)
    * [模型预测](模型预测)
  * [部署示例](模型部署)
    * [模型导出](模型导出)
    * [模型部署](模型部署)

## 小样本学习简介

[多标签分类任务](../README.md/#多标签任务介绍)在商品分类、网页分类、新闻分类、医疗文本分类等现实场景中有着广泛应用。现有的主流解决方案是在大规模预训练语言模型进行微调，因为下游任务和预训练任务训练目标不同，想要取得较好的分类效果往往需要大量标注数据，因此学界和业界开始研究如何在小样本学习（Few-shot Learning）场景下取得更好的学习效果。

**提示学习(Prompt Learning)**
的主要思想是通过任务转换使得下游任务和预训练任务尽可能相似，充分利用预训练语言模型学习到的特征，从而降低样本需求量。除此之外，我们往往还需要在原有的输入文本上拼接一段“提示”，来引导预训练模型输出期望的结果。

我们以Ernie为例，回顾一下这类预训练语言模型的训练任务。
与考试中的完形填空相似，给定一句文本，遮盖掉其中的部分字词，要求语言模型预测出这些遮盖位置原本的字词。

因此，我们也将多标签分类任务转换为与完形填空相似的形式。例如影评情感分类任务，标签分为`1-正向`，`0-负向`两类。

- 在经典的微调方式中，需要学习的参数是以`[CLS]`向量为输入，以负向/正向为输出的随机初始化的分类器。
- 在提示学习中，我们通过构造提示，将原有的分类任务转化为完形填空。如下图所示，通过提示`我[MASK]喜欢。`，原有`1-正向`，`0-负向`的标签被转化为了预测空格是`很`还是`不`。此时的分类器也不再是随机初始化，而是利用了这两个字的预训练向量来初始化，充分利用了预训练模型学习到的参数。

<div align="center">
    <img src=https://user-images.githubusercontent.com/25607475/183909263-6ead8871-699c-4c2d-951f-e33eddcfdd9c.png />
</div>

## 环境要求

- python >= 3.6
- paddlepaddle >= 2.3
- paddlenlp >= 2.4.0

## 训练示例

对于标注样本充足的场景可以直接使用[预训练模型微调](../README.md)实现多标签文本分类，对于尚无标注或者标注样本较少的任务场景我们推荐使用小样本学习，以取得比微调方法更好的效果。下边通过**婚姻家庭领域的案情要素分类**的例子展示如何使用小样本学习来进行文本分类，每个标签有16条标注样本。

### 代码结构

```text
.
├── train.py    # 模型组网训练脚本
├── utils.py    # 数据处理工具
├── infer.py    # 模型部署脚本
└── README.md
```

### 数据准备

我们推荐使用数据标注平台[doccano](https://github.com/doccano/doccano)进行自定义数据标注，然后使用[doccano脚本](../../doccano.py)进行格式转换，具体流程可参考[doccano数据标注指南](../../doccano.md)。对于已有的数据集，需要将数据转换为下述文本分类任务的统一格式。这里我们使用婚姻家庭领域的案情要素分类数据集的采样子集作为示例数据集，可点击[这里](https://paddlenlp.bj.bcebos.com/datasets/few-shot/elements.tar.gz)下载解压并放入`./data/`文件夹，或者运行以下脚本

```
wget https://paddlenlp.bj.bcebos.com/datasets/few-shot/elements.tar.gz
tar zxvf elements.tar.gz
mv elements data
```

#### 目录结构

```text
data/
├── train.txt  # 训练数据集
├── dev.txt    # 验证数据集
├── test.txt   # 测试数据集（可选）
├── data.txt   # 待预测数据（可选）
└── label.txt  # 分类标签集
```

#### 数据格式

对于训练/验证/测试数据集文件，每行数据表示一条样本，包括文本和标签两部分。文本和标签由tab符`\t`分隔，多个标签以英文逗号`,`分隔。格式如下

```text
<文本>'\t'<标签>','<标签>','<标签>
<文本>'\t'<标签>','<标签>
...
```

婚姻家庭领域的案情要素分类数据集示例如下
```text
二、婚生子朱x1由被告刘×抚养，原告朱×自二〇一三年十一月起每月支付子女抚养费一千八百元，至朱x1十八周岁止；    婚后有子女,按月给付抚养费,支付抚养费,限制行为能力子女抚养
被告另要求原告赔偿其精神损害抚慰金、支付一次性生活困难扶助金各5万元。   损害赔偿
...
```

对于待预测数据文件，每行包含一条待预测样本，无标签。格式如下
```text
<文本>
<文本>
...
```
例如，
```
五松新村房屋是被告婚前购买的；
2、判令被告返还借婚姻索取的现金33万元，婚前个人存款10万元；
...
```

对于分类标签集文件，存储了数据集中所有的标签集合，每行为一个标签名。如果需要自定义标签映射，则每行需要包括标签名和相应的映射词，由`==`分隔。格式如下
```text
<标签>'=='<映射词>
<标签>'=='<映射词>
...
```
例如，在婚姻家庭要素标签中，原标签字数较多，因此同一个标签依赖的输出也多。为了降低训练难度，我们可以将其映射为较短的短语
```
有夫妻共同债务==共同债务
存在非婚生子==非婚生子
...
```
**Note**: 这里的标签映射词定义遵循的规则是，不同映射词尽可能长度一致，映射词和提示需要尽可能构成通顺的语句。越接近自然语句，小样本下模型训练效果越好。如果原标签名已经可以构成通顺语句，也可以不构造映射词，每行一个标签即可，即
```
有夫妻共同债务
存在非婚生子
...
```

### 模型训练

**单卡训练**

```
export CUDA_VISIBLE_DEVICES=0
python train.py \
--data_dir ./data \
--output_dir ./checkpoints/ \
--prompt "这句话包括的案件要素有" \
--max_seq_length 128  \
--learning_rate 3e-5 \
--ppt_learning_rate 3e-4 \
--do_train \
--do_eval \
--max_steps 1000 \
--eval_steps 100 \
--logging_steps 10 \
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
--prompt "这句话包括的案件要素有" \
--max_seq_length 128  \
--learning_rate 3e-5 \
--ppt_learning_rate 3e-4 \
--do_train \
--do_eval \
--max_steps 1000 \
--eval_steps 100 \
--logging_steps 10 \
--per_device_eval_batch_size 32 \
--per_device_train_batch_size 8 \
--do_predict \
--do_export
```

可配置参数说明：
- `model_name_or_path`: 内置模型名，或者模型参数配置目录路径。默认为`ernie-3.0-base-zh`。
- `data_dir`: 训练数据集路径，数据格式要求详见[数据准备](数据准备)。
- `output_dir`: 模型参数、训练日志和静态图导出的保存目录。
- `prompt`: 提示模板。定义了如何将文本和提示拼接结合。
- `soft_encoder`: 提示向量的编码器，`lstm`表示双向LSTM, `mlp`表示双层线性层, None表示直接使用提示向量。默认为`lstm`。
- `encoder_hidden_size`: 提示向量的维度。若为None，则表示预训练模型字向量维度。默认为200。
- `max_seq_length`: 最大句子长度，超过该长度的文本将被截断，不足的以Pad补全。提示文本不会被截断。
- `learning_rate`: 预训练语言模型参数基础学习率大小，将与learning rate scheduler产生的值相乘作为当前学习率。
- `ppt_learning_rate`: 提示相关参数的基础学习率大小，当预训练参数不固定时，与其共用learning rate scheduler。一般设为`learning_rate`的十倍。
- `do_train`: 是否进行训练。
- `do_eval`: 是否进行评估。
- `do_predict`: 是否进行预测。
- `do_export`: 是否在运行结束时将模型导出为静态图，保存路径为`output_dir/export`。
- `max_steps`: 训练的最大步数。此设置将会覆盖`num_train_epochs`。
- `eval_steps`: 评估模型的间隔步数。
- `logging_steps`: 打印日志的间隔步数。
- `per_device_train_batch_size`: 每次训练每张卡上的样本数量。可根据实际GPU显存适当调小/调大此配置。
- `per_device_eval_batch_size`: 每次评估每张卡上的样本数量。可根据实际GPU显存适当调小/调大此配置。

更多参数介绍可参考[配置文件](../../../../paddlenlp/trainer/trainer_args.py)。


### 模型预测

在模型训练时开启`--do_predict`，训练结束后直接进行预测，也可以在训练结束后，通过运行以下命令加载模型参数进行预测：
```
python train.py --do_predict --data_dir ./data --output_dir ./predict_ckpt --resume_from_checkpoint ./checkpoints/ --max_seq_length 128
```

可配置参数说明：

- `data_dir`: 测试数据路径。数据格式要求详见[数据准备](数据准备)，数据应存放在该目录下`test.txt`文件中，每行一条待预测文本。
- `output_dir`: 日志的保存目录。
- `resume_from_checkpoint`: 训练时模型参数的保存目录，用于加载模型参数。
- `do_predict`: 是否进行预测。
- `per_device_eval_batch_size`: 每次评估每张卡上的样本数量。默认为8，可根据实际GPU显存适当调小/调大此配置。
- `max_seq_length`: 最大句子长度，超过该长度的文本将被截断，不足的以Pad补全。提示文本不会被截断。

## 部署示例

### 模型导出

在训练结束后，需要将动态图模型导出为静态图参数用于部署推理。可以在模型训练时开启`--do_export`在训练结束后直接导出，也可以运行以下命令加载并导出训练后的模型参数，默认导出到在`output_dir`指定的目录下。
```
python train.py --do_export --data_dir ./data --output_dir ./export_ckpt --resume_from_checkpoint ./checkpoints/
```

可配置参数说明：

- `data_dir`: 标签数据路径。数据格式要求详见[数据准备](数据准备)。
- `output_dir`: 静态图模型参数和日志的保存目录。
- `resume_from_checkpoint`: 训练时模型参数的保存目录，用于加载模型参数。
- `do_export`: 是否将模型导出为静态图，保存路径为`output_dir/export`。

### 模型部署

模型转换与ONNXRuntime预测部署依赖Paddle2ONNX和ONNXRuntime，Paddle2ONNX支持将Paddle静态图模型转化为ONNX模型格式，算子目前稳定支持导出ONNX Opset 7~15，更多细节可参考：[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)。

ONNXRuntime推理部署需要安装以下依赖：
```shell
pip install psutil
pip install paddle2onnx==1.0.0rc3
```

如果基于GPU部署，请先确保机器已正确安装NVIDIA相关驱动和基础软件，确保CUDA >= 11.2，CuDNN >= 8.2，并使用以下命令安装所需依赖:
```shell
python -m pip install onnxruntime-gpu onnx onnxconverter-common
```

如果基于CPU部署，请使用如下命令安装所需依赖:
```shell
python -m pip install onnxruntime
```

#### CPU端推理样例

```
python infer.py --model_path_prefix checkpoints/export/model --data_dir ./data --batch_size 32 --device cpu
```

#### GPU端推理样例

```
python infer.py --model_path_prefix checkpoints/export/model --data_dir ./data --batch_size 32 --device gpu --device_id 0
```

可配置参数说明：

- `model_path_prefix`: 导出的静态图模型路径及文件前缀。
- `model_name_or_path`: 内置预训练模型名，或者模型参数配置目录路径，用于加载tokenizer。默认为`ernie-3.0-base-zh`。
- `data_dir`: 待推理数据所在路径，数据应存放在该目录下的`data.txt`文件。
- `max_seq_length`: 最大句子长度，超过该长度的文本将被截断，不足的以Pad补全。提示文本不会被截断。
- `batch_size`: 每次预测的样本数量。
- `device`: 选择推理设备，包括`cpu`和`gpu`。默认为`gpu`。
- `device_id`: 指定GPU设备ID。
- `use_fp16`: 是否使用半精度加速推理。仅在GPU设备上有效。
- `num_threads`: 设置CPU使用的线程数。默认为机器上的物理内核数。

**Note**: 在GPU设备的CUDA计算能力 (CUDA Compute Capability) 大于7.0，在包括V100、T4、A10、A100、GTX 20系列和30系列显卡等设备上可以开启FP16进行加速，在CPU或者CUDA计算能力 (CUDA Compute Capability) 小于7.0时开启不会带来加速效果。
