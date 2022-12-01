# 二分类/多分类任务指南

**目录**

- [1. 二分类/多分类简介](#二分类/多分类简介)
- [2. 快速开始](#快速开始)
    - [2.1 运行环境](#运行环境)
    - [2.2 代码结构](#代码结构)
    - [2.3 数据准备](#数据准备)
    - [2.4 模型训练](#模型训练)
    - [2.5 模型部署](#模型部署)
    - [2.6 模型效果](#模型效果)


<a name="二分类/多分类简介"></a>

## 1. 二分类/多分类简介

本项目提供通用场景下**基于预训练模型微调的二分类/多分类端到端应用方案**，打通数据标注-模型训练-模型调优-模型压缩-预测部署全流程，有效缩短开发周期，降低AI开发落地门槛。

二分类/多分类数据集的标签集含有两个或两个以上的类别，所有输入句子/文本有且只有一个标签。在文本多分类场景中，我们需要预测**输入句子/文本最可能来自 `n` 个标签类别中的哪一个类别**。在本项目中二分类任务被视为多分类任务中标签集包含两个类别的情况，以下统一称为多分类任务。以下图为例，该新闻文本的最可能的标签为 `娱乐`。多分类任务在商品分类、网页标签、新闻分类、医疗文本分类等各种现实场景中具有广泛的适用性。

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/187588832-daa4294e-248e-4c69-9a4b-1cf8f25a2fcc.png width="550"/>
</div>
<br>

**方案亮点：**

- **效果领先🏃：** 使用在中文领域内模型效果和模型计算效率有突出效果的ERNIE 3.0 轻量级系列模型作为训练基座，ERNIE 3.0 轻量级系列提供多种尺寸的预训练模型满足不同需求，具有广泛成熟的实践应用性。
- **高效调优✊：** 文本分类应用依托[TrustAI](https://github.com/PaddlePaddle/TrustAI)可信增强能力和[数据增强API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md)，提供模型分析模块助力开发者实现模型分析，并提供稀疏数据筛选、脏数据清洗、数据增强等多种解决方案。
- **简单易用👶：** 开发者**无需机器学习背景知识**，仅需提供指定格式的标注分类数据，一行命令即可开启文本分类训练，轻松完成上线部署，不再让技术成为文本分类的门槛。

**更多选择：**

对于大多数多分类任务，我们推荐使用预训练模型微调作为首选的文本分类方案，多分类项目中还提供 提示学习(小样本)和语义索引的两种全流程文本分类方案满足不同开发者需求，更多技术细节请参见[文本分类技术特色介绍](../README.md)。

- 【标注成本高、标注样本较少的小样本场景】 👉 [提示学习多分类方案](./few-shot#readme)

- 【标签类别不固定场景、标签类别众多】 👉 [语义索引多分类方案](./retrieval_based#readme)


<a name="快速开始"></a>

## 2. 快速开始

接下来我们将以CBLUE公开数据集KUAKE-QIC任务为示例，演示多分类全流程方案使用。下载数据集：

```shell
wget https://paddlenlp.bj.bcebos.com/datasets/KUAKE_QIC.tar.gz
tar -zxvf KUAKE_QIC.tar.gz
mv KUAKE_QIC data
rm KUAKE_QIC.tar.gz
```

<div align="center">
    <img width="900" alt="image" src="https://user-images.githubusercontent.com/63761690/187828356-e2f4f627-f5fe-4c83-8879-ed6951f7511e.png">
</div>
<div align="center">
    <font size ="2">
    多分类数据标注-模型训练-模型分析-模型压缩-预测部署流程图
     </font>
</div>

<a name="运行环境"></a>

### 2.1 运行环境

- python >= 3.6
- paddlepaddle >= 2.3
- paddlenlp >= 2.4
- scikit-learn >= 1.0.2

**安装PaddlePaddle：**

 环境中paddlepaddle-gpu或paddlepaddle版本应大于或等于2.3, 请参见[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)根据自己需求选择合适的PaddlePaddle下载命令。


**安装PaddleNLP：**

安装PaddleNLP默认开启百度镜像源来加速下载，如果您使用 HTTP 代理可以关闭(删去 -i https://mirror.baidu.com/pypi/simple)，更多关于PaddleNLP安装的详细教程请查见[PaddleNLP快速安装](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/get_started/installation.rst)。
```shell
python3 -m pip install --upgrade paddlenlp -i https://mirror.baidu.com/pypi/simple
```


**安装sklearn：**
```shell
python3 -m  pip install scikit-learn==1.0.2
```

<a name="代码结构"></a>

### 2.2 代码结构

```text
multi_class/
├── few-shot # 小样本学习方案
├── retrieval_based # 语义索引方案
├── analysis # 分析模块
├── deploy # 部署
│   └── predictor # 离线部署
│   ├── paddle_serving # PaddleServing在线服务化部署
│   └── triton_serving # Triton在线服务化部署
├── train.py # 训练评估脚本
├── predict.py # 预测脚本
├── export_model.py # 静态图模型导出脚本
├── utils.py # 工具函数脚本
├── prune.py # 裁剪脚本
└── README.md # 多分类使用说明
```

<a name="数据准备"></a>

### 2.3 数据准备

训练需要准备指定格式的本地数据集,如果没有已标注的数据集，可以参考[文本分类任务doccano数据标注使用指南](../doccano.md)进行文本分类数据标注。指定格式本地数据集目录结构：

```text
data/
├── train.txt # 训练数据集文件
├── dev.txt # 开发数据集文件
├── test.txt # 可选，测试数据集文件
├── label.txt # 分类标签文件
└── data.txt # 待预测数据文件
```

**训练、开发、测试数据集** 文件中文本与标签类别名用tab符`'\t'`分隔开，文本中避免出现tab符`'\t'`。

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

<a name="模型训练"></a>

### 2.4 模型训练

#### 2.4.1 预训练模型微调


使用CPU/GPU训练，默认为GPU训练，使用CPU训练只需将设备参数配置改为`--device cpu`：
```shell
python train.py \
    --model_name_or_path ernie-3.0-medium-zh \
    --data_dir ./data/ \
    --output_dir checkpoint \
    --device gpu \
    --learning_rate 3e-5 \
    --num_train_epochs 100 \
    --early_stopping_patience 4 \
    --max_seq_length 128 \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 100 \
    --do_train \
    --do_eval \
    --metric_for_best_model accuracy \
    --load_best_model_at_end \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1
```

如果在CPU环境下训练，可以指定`nproc_per_node`参数进行多核训练：
```shell
python -m paddle.distributed.launch --nproc_per_node 8 --backend gloo train.py \
    --model_name_or_path ernie-3.0-medium-zh \
    --data_dir ./data/ \
    --output_dir checkpoint \
    --device cpu \
    --learning_rate 3e-5 \
    --num_train_epochs 100 \
    --max_seq_length 128 \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 100 \
    --early_stopping_patience 4 \
    --do_train \
    --do_eval \
    --metric_for_best_model accuracy \
    --load_best_model_at_end \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1
```

如果在GPU环境中使用，可以指定`gpus`参数进行单卡/多卡训练。使用多卡训练可以指定多个GPU卡号，例如 --gpus 0,1。如果设备只有一个GPU卡号默认为0，可使用`nvidia-smi`命令查看GPU使用情况:

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus 0,1 train.py \
    --data_dir ./data/ \
    --output_dir checkpoint \
    --device gpu \
    --learning_rate 3e-5 \
    --num_train_epochs 100 \
    --max_seq_length 128 \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 100 \
    --early_stopping_patience 4 \
    --do_train \
    --do_eval \
    --metric_for_best_model accuracy \
    --load_best_model_at_end \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1
```

主要的配置的参数为：
- `model_name_or_path`: 内置模型名，或者模型参数配置目录路径。默认为`ernie-3.0-base-zh`。
- `data_dir`: 训练数据集路径，数据格式要求详见[数据标注](#数据标注)。
- `output_dir`: 模型参数、训练日志和静态图导出的保存目录。
- `max_seq_length`: 最大句子长度，超过该长度的文本将被截断，不足的以Pad补全。提示文本不会被截断。
- `num_train_epochs`: 训练轮次，使用早停法时可以选择100
- `early_stopping_patience`: 在设定的早停训练轮次内，模型在开发集上表现不再上升，训练终止；默认为4。
- `learning_rate`: 预训练语言模型参数基础学习率大小，将与learning rate scheduler产生的值相乘作为当前学习率。
- `do_train`: 是否进行训练。
- `do_eval`: 是否进行评估。
- `device`: 使用的设备，默认为`gpu`。
- `per_device_train_batch_size`: 每次训练每张卡上的样本数量。可根据实际GPU显存适当调小/调大此配置。
- `per_device_eval_batch_size`: 每次评估每张卡上的样本数量。可根据实际GPU显存适当调小/调大此配置。

训练脚本支持所有`TraingArguments`的参数，更多参数介绍可参考[TrainingArguments 参数介绍](https://paddlenlp.readthedocs.io/zh/latest/trainer.html#trainingarguments)。

程序运行时将会自动进行训练，评估。同时训练过程中会自动保存开发集上最佳模型在指定的 `output_dir` 中，保存模型文件结构如下所示：

```text
checkpoint/
├── model_config.json
├── model_state.pdparams
├── tokenizer_config.json
└── vocab.txt
```

**NOTE:**

* 如需恢复模型训练，则可以设置 `resume_from_checkpoint` ， 如 `resume_from_checkpoint=./checkpoints/checkpoint-217` 。
* 如需训练英文文本分类任务，只需更换预训练模型参数 `model_name_or_path` 。英文训练任务推荐使用"ernie-2.0-base-en"、"ernie-2.0-large-en"。
* 英文和中文以外语言的文本分类任务，推荐使用基于96种语言（涵盖法语、日语、韩语、德语、西班牙语等几乎所有常见语言）进行预训练的多语言预训练模型"ernie-m-base"、"ernie-m-large"，详情请参见[ERNIE-M论文](https://arxiv.org/pdf/2012.15674.pdf)。

#### 2.4.2 训练评估与模型优化

文本分类预测过程中常会遇到诸如"模型为什么会预测出错误的结果"，"如何提升模型的表现"等问题。[Analysis模块](./analysis) 提供了**模型评估、可解释性分析、数据优化**等功能，旨在帮助开发者更好地分析文本分类模型预测结果和对模型效果进行优化。

<div align="center">
    <img src="https://user-images.githubusercontent.com/63761690/195241942-70068989-df17-4f53-9f71-c189d8c5c88d.png" width="600">
</div>

**模型评估：** 训练后的模型我们可以使用 [Analysis模块](./analysis) 对每个类别分别进行评估，并输出预测错误样本（bad case），默认在GPU环境下使用，在CPU环境下修改参数配置为`--device "cpu"`:

```shell
python analysis/evaluate.py --device "gpu" --max_seq_length 128 --batch_size 32 --bad_case_file "bad_case.txt" --dataset_dir "data" --params_path "./checkpoint"
```

输出打印示例：

```text
[2022-08-10 06:28:37,219] [    INFO] - -----Evaluate model-------
[2022-08-10 06:28:37,219] [    INFO] - Train dataset size: 6931
[2022-08-10 06:28:37,220] [    INFO] - Dev dataset size: 1955
[2022-08-10 06:28:37,220] [    INFO] - Accuracy in dev dataset: 81.79%
[2022-08-10 06:28:37,221] [    INFO] - Top-2 accuracy in dev dataset: 92.48%
[2022-08-10 06:28:37,222] [    INFO] - Top-3 accuracy in dev dataset: 97.24%
[2022-08-10 06:28:37,222] [    INFO] - Class name: 病情诊断
[2022-08-10 06:28:37,222] [    INFO] - Evaluation examples in train dataset: 877(12.7%) | precision: 97.14 | recall: 96.92 | F1 score 97.03
[2022-08-10 06:28:37,222] [    INFO] - Evaluation examples in dev dataset: 288(14.7%) | precision: 80.32 | recall: 86.46 | F1 score 83.28
[2022-08-10 06:28:37,223] [    INFO] - ----------------------------
[2022-08-10 06:28:37,223] [    INFO] - Class name: 治疗方案
[2022-08-10 06:28:37,223] [    INFO] - Evaluation examples in train dataset: 1750(25.2%) | precision: 96.84 | recall: 99.89 | F1 score 98.34
[2022-08-10 06:28:37,223] [    INFO] - Evaluation examples in dev dataset: 676(34.6%) | precision: 88.46 | recall: 94.08 | F1 score 91.18
...
```

预测错误的样本保存在bad_case.txt文件中：

```text
Text	Label	Prediction
您好，请问一岁三个月的孩子可以服用复方锌布颗粒吗？	其他	注意事项
输卵管粘连的基本检查	其他	就医建议
会是胎动么？	其他	病情诊断
经常干呕恶心，这是生病了吗	其他	病情诊断
菏泽哪个医院治疗白癜风比较好?怎么治好	就医建议	治疗方案
...
```

**可解释性分析：** 基于[TrustAI](https://github.com/PaddlePaddle/TrustAI)提供单词和句子级别的模型可解释性分析，帮助理解模型预测结果，用于错误样本（bad case）分析，细节详见[训练评估与模型优化指南](analysis/README.md)。

- 单词级别可解释性分析，也即分析待预测样本中哪一些单词对模型预测结果起重要作用。以下图为例，用颜色深浅表示单词对预测结果的重要性。
<div align="center">
    <img src="https://user-images.githubusercontent.com/63761690/195086276-6ee16e96-4ec3-4a0f-821f-37546d21746b.png" width="1000">
</div>

- 句子级别可解释性分析 ，也即分析对待预测样本的模型预测结果与训练集中中哪些样本有重要关系。下面的例子表明句子级别可解释性分析可以帮助理解待预测样本的预测结果与训练集中样本之间的关联。
```text
text: 您好，请问一岁三个月的孩子可以服用复方锌布颗粒吗？
predict label: 注意事项
label: 其他
examples with positive influence
support1 text: 感冒期间钙产品要继续服用吗? 钙尔奇就可以，也可以吃婴儿吃的乳钙	label: 注意事项	score: 0.96602
support2 text: 打喷嚏可以吃布洛芬缓释胶囊么	label: 注意事项	score: 0.95687
support3 text: 孕后期可以足疗吗	label: 注意事项	score: 0.94021
...
```

**数据优化：** 结合[TrustAI](https://github.com/PaddlePaddle/TrustAI)和[数据增强API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md)提供了**稀疏数据筛选、脏数据清洗、数据增强**三种优化策略，从多角度优化训练数据提升模型效果，策略细节详见[训练评估与模型优化指南](analysis/README.md)。

- 稀疏数据筛选主要是解决数据不均衡、训练数据覆盖不足的问题，通过数据增强和数据标注两种方式解决这一问题。
- 脏数据清洗可以帮助开发者筛选训练集中错误标注的数据，对这些数据重新进行人工标注，得到标注正确的数据再重新进行训练。
- 数据增强策略提供多种数据增强方案，可以快速扩充数据，提高模型泛化性和鲁棒性。

#### 2.4.3 模型预测
训练结束后，输入待预测数据(data.txt)和类别标签对照列表(label.txt)，使用训练好的模型进行，默认在GPU环境下使用，在CPU环境下修改参数配置为`--device "cpu"`：

```shell
python predict.py --device "gpu" --max_seq_length 128 --batch_size 32 --dataset_dir "data"
```

可支持配置的参数：

* `device`: 选用什么设备进行预测，可选cpu、gpu、xpu、npu；默认为gpu。
* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含data.txt和label.txt文件;默认为None。
* `params_path`：待预测模型的目录；默认为"./checkpoint/"。
* `max_seq_length`：模型使用的最大序列长度,建议与训练时最大序列长度一致, 若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `data_file`：本地数据集中未标注待预测数据文件名；默认为"data.txt"。
* `label_file`：本地数据集中标签集文件名；默认为"label.txt"。

<a name="模型部署"></a>

### 2.5 模型部署
#### 2.5.1 静态图导出

使用动态图训练结束之后，还可以将动态图参数导出成静态图参数，静态图模型将用于**后续的推理部署工作**。具体代码见[静态图导出脚本](export_model.py)，静态图参数保存在`output_path`指定路径中。运行方式：

```shell
python export_model.py --params_path ./checkpoint/ --output_path ./export
```

如果使用多语言模型 ERNIE M作为预训练模型，运行方式：
```shell
python export_model.py --params_path ./checkpoint/ --output_path ./export --multilingual
```

可支持配置的参数：
* `multilingual`：是否为多语言任务（是否使用ERNIE M作为预训练模型）；默认为False。
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

#### 2.5.2 模型裁剪

如果有模型部署上线的需求，需要进一步压缩模型体积，可以使用 PaddleNLP 的 [压缩API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/compression.md), 一行命令即可启动模型裁剪。

使用裁剪功能需要安装 paddleslim：

```shell
pip install paddleslim==2.2.2
```

开始模型裁剪训练，默认为GPU训练，使用CPU训练只需将设备参数配置改为`--device "cpu"`：
```shell
python prune.py \
    --device "gpu" \
    --dataset_dir "data" \
    --output_dir "prune" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 10 \
    --max_seq_length 128 \
    --logging_steps 5 \
    --save_steps 100 \
    --width_mult_list '3/4' '2/3' '1/2'
```


可支持配置的参数：
* `output_dir`：必须，保存模型输出和和中间checkpoint的输出目录;默认为 `None` 。
* `device`: 选用什么设备进行裁剪，选择cpu、gpu。如使用gpu训练，可使用参数--gpus指定GPU卡号。
* `per_device_train_batch_size`：训练集裁剪训练过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `per_device_eval_batch_size`：开发集评测过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：训练最大学习率；默认为3e-5。
* `num_train_epochs`: 训练轮次，使用早停法时可以选择100；默认为10。
* `logging_steps`: 训练过程中日志打印的间隔steps数，默认100。
* `save_steps`: 训练过程中保存模型checkpoint的间隔steps数，默认100。
* `seed`：随机种子，默认为3。
* `width_mult_list`：裁剪宽度（multi head）保留的比例列表，表示对self_attention中的 `q`、`k`、`v` 以及 `ffn` 权重宽度的保留比例，保留比例乘以宽度（multi haed数量）应为整数；默认是None。
* `dataset_dir`：本地数据集路径，需包含train.txt,dev.txt,label.txt;默认为None。
* `max_seq_length`：模型使用的最大序列长度，建议与训练过程保持一致, 若出现显存不足，请适当调低这一参数；默认为128。
* `params_dir`：待预测模型参数文件；默认为"./checkpoint/"。

程序运行时将会自动进行训练，评估，测试。同时训练过程中会自动保存开发集上最佳模型在指定的 `output_dir` 中，保存模型文件结构如下所示：

```text
prune/
├── width_mult_0.75
│   ├── pruned_model.pdiparams
│   ├── pruned_model.pdiparams.info
│   ├── pruned_model.pdmodel
│   ├── model_state.pdparams
│   └── model_config.json
└── ...
```

**NOTE:**

1. 目前支持的裁剪策略需要训练，训练时间视下游任务数据量而定，且和微调的训练时间是一个量级。 裁剪类似蒸馏过程，方便起见，可以直接使用微调时的超参。为了进一步提升精度，可以对 `per_device_train_batch_size`、`learning_rate`、`num_train_epochs`、`max_seq_length` 等超参进行网格搜索（grid search）。

2. 模型裁剪主要用于推理部署，因此裁剪后的模型都是静态图模型，只可用于推理部署，不能再通过 `from_pretrained` 导入继续训练。导出模型之后用于部署，项目提供了基于ONNXRuntime的 [离线部署方案](./deploy/predictor/README.md) 和基于Paddle Serving的 [在线服务化部署方案](./deploy/predictor/README.md)。

3. ERNIE Base、Medium、Mini、Micro、Nano的模型宽度（multi head数量）为12，ERNIE Xbase、Large 模型宽度（multi head数量）为16，保留比例`width_mult`乘以宽度（multi haed数量）应为整数。


#### 2.5.3 部署方案

- 离线部署搭建请参考[离线部署](deploy/predictor/README.md)。

- 在线服务化部署搭建请参考 [Paddle Serving部署指南](deploy/paddle_serving/README.md) (Paddle Serving支持X86、Arm CPU、NVIDIA GPU、昆仑/昇腾等多种硬件)或[Triton部署指南](deploy/triton_serving/README.md)。


<a name="模型效果"></a>

### 2.6 模型效果

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
|ERNIE 1.0 Large Cw |24-layer, 1024-hidden, 20-heads|82.30| 5.62 |
|ERNIE 3.0 Base  |12-layer, 768-hidden, 12-heads|82.25| 2.07 |
|ERNIE 3.0 Medium| 6-layer, 768-hidden, 12-heads|81.79| 1.07|
|ERNIE 3.0 Mini |6-layer, 384-hidden, 12-heads|79.80| 0.38|
|ERNIE 3.0 Micro | 4-layer, 384-hidden, 12-heads|79.80| 0.26|
|ERNIE 3.0 Nano |4-layer, 312-hidden, 12-heads|78.57|0.22|
| ERNIE 3.0 Medium + 裁剪(保留比例3/4)|6-layer, 768-hidden, 9-heads| 81.79| 0.83   |
| ERNIE 3.0 Medium + 裁剪(保留比例2/3)|6-layer, 768-hidden, 8-heads| 81.07  | 0.79  |
| ERNIE 3.0 Medium + 裁剪(保留比例1/2)|6-layer, 768-hidden, 6-heads| 81.07 | 0.64  |
