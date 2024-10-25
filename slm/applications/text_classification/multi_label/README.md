# 多标签分类指南

**目录**
- [1. 多标签分类简介](#多标签分类简介)
- [2. 快速开始](#快速开始)
    - [2.1 运行环境](#运行环境)
    - [2.2 代码结构](#代码结构)
    - [2.3 数据准备](#数据准备)
    - [2.4 模型训练](#模型训练)
    - [2.5 模型部署](#模型部署)
    - [2.6 模型效果](#模型效果)

<a name="多标签分类简介"></a>

## 1. 多标签分类简介

本项目提供通用场景下**基于预训练模型微调的多标签分类端到端应用方案**，打通数据标注-模型训练-模型调优-模型压缩-预测部署全流程，有效缩短开发周期，降低 AI 开发落地门槛。

多标签数据集的标签集含有两个或两个以上的类别，输入句子/文本具有一个或多个标签，多标签任务的目标是预测**样本属于哪些标签类别，这些类别具有不相互排斥的属性**。文本多标签分类在各种现实场景中具有广泛的适用性，例如商品分类、网页标签、新闻标注、蛋白质功能分类、电影分类、语义场景分类等。以下图为例，该新闻文本具有 `相机` 和 `芯片` 两个标签。


<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/187823132-3590bfff-8248-4e92-900d-bac350328743.png width="550"/>
</div>
<br>

**方案亮点：**

- **效果领先🏃：** 使用在中文领域内模型效果和模型计算效率有突出效果的 ERNIE 3.0 轻量级系列模型作为训练基座，ERNIE 3.0 轻量级系列提供多种尺寸的预训练模型满足不同需求，具有广泛成熟的实践应用性。
- **高效调优✊：** 文本分类应用依托[TrustAI](https://github.com/PaddlePaddle/TrustAI)可信增强能力和[数据增强 API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md)，提供模型分析模块助力开发者实现模型分析，并提供稀疏数据筛选、脏数据清洗、数据增强等多种解决方案。
- **简单易用👶：** 开发者**无需机器学习背景知识**，仅需提供指定格式的标注分类数据，一行命令即可开启文本分类训练，轻松完成上线部署，不再让技术成为文本分类的门槛。

**更多选择：**

对于大多数多标签分类任务，我们推荐使用预训练模型微调作为首选的文本分类方案，多标签分类项目中还提供 提示学习(小样本)和语义索引的两种全流程文本分类方案满足不同开发者需求，更多技术细节请参见[文本分类技术特色介绍](../README.md)。

- 【标注成本高、标注样本较少的小样本场景】 👉 [提示学习多标签分类方案](./few-shot#readme)
- 【标签类别不固定场景、标签类别众多】 👉 [语义索引多分类方案](./retrieval_based#readme)
<a name="快速开始"></a>

## 2. 快速开始

我们以公开数据集 CAIL2019—婚姻家庭要素提取任务为示例，演示多标签分类全流程方案使用。下载数据集：
```shell
wget https://paddlenlp.bj.bcebos.com/datasets/divorce.tar.gz
tar -zxvf divorce.tar.gz
mv divorce data
rm divorce.tar.gz
```

<div align="center">
    <img width="900" alt="image" src="https://user-images.githubusercontent.com/63761690/187828356-e2f4f627-f5fe-4c83-8879-ed6951f7511e.png">
</div>
<div align="center">
    <font size ="2">
    多标签分类数据标注-模型训练-模型分析-模型压缩-预测部署流程图
     </font>
</div>

<a name="运行环境"></a>

### 2.1 运行环境

- python >= 3.6
- paddlepaddle >= 2.3
- paddlenlp >= 2.4.8
- scikit-learn >= 1.0.2

**安装 PaddlePaddle：**

 环境中 paddlepaddle-gpu 或 paddlepaddle 版本应大于或等于2.3, 请参见[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)根据自己需求选择合适的 PaddlePaddle 下载命令。


**安装 PaddleNLP：**

安装 PaddleNLP 默认开启百度镜像源来加速下载，如果您使用 HTTP 代理可以关闭(删去 -i https://mirror.baidu.com/pypi/simple)，更多关于 PaddleNLP 安装的详细教程请查见[PaddleNLP 快速安装](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/get_started/installation.rst)。
```shell
python3 -m pip install --upgrade paddlenlp -i https://mirror.baidu.com/pypi/simple
```


**安装 sklearn：**
```shell
python3 -m  pip install scikit-learn==1.0.2
```

<a name="代码结构"></a>

### 2.2 代码结构

```text
multi_label/
├── few-shot # 小样本学习方案
├── analysis # 分析模块
├── deploy # 部署
│   └── predictor # 离线部署
│   ├── paddle_serving # PaddleServing在线服务化部署
│   └── triton_serving # Triton在线服务化部署
├── train.py # 训练评估脚本
├── predict.py # 预测脚本
├── export_model.py # 静态图模型导出脚本
├── utils.py # 工具函数脚本
├── metric.py # metric脚本
├── prune.py # 裁剪脚本
└── README.md # 使用说明
```
<a name="数据准备"></a>

### 2.3 数据准备

训练需要准备指定格式的标注数据集,如果没有已标注的数据集，可以参考 [数据标注指南](../doccano.md) 进行文本分类数据标注。指定格式本地数据集目录结构：

```text
data/
├── train.txt # 训练数据集文件
├── dev.txt # 开发数据集文件
├── test.txt # 测试数据集文件（可选）
├── label.txt # 分类标签文件
└── data.txt # 待预测数据文件（可选）
```
**训练、开发、测试数据集**文件中文本与标签类别名用 tab 符`'\t'`分隔开，标签中多个标签之间用`','`逗号分隔开，文本中避免出现 tab 符`'\t'`。

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

**分类标签**包含数据集中所有标签集合，每一行为一个标签名。

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

**待预测数据文件：** 包含需要预测标签的文本数据，每条数据一行。

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
<a name="模型训练"></a>

### 2.4 模型训练

#### 2.4.1 预训练模型微调

使用 CPU/GPU 训练，默认为 GPU 训练。使用 CPU 训练只需将设备参数配置改为`--device cpu`，可以使用`--device gpu:0`指定 GPU 卡号：
```shell
python train.py \
    --dataset_dir "data" \
    --device "gpu" \
    --max_seq_length 128 \
    --model_name "ernie-3.0-medium-zh" \
    --batch_size 32 \
    --early_stop \
    --epochs 100
```

如果在 GPU 环境中使用，可以指定`gpus`参数进行单卡/多卡训练。使用多卡训练可以指定多个 GPU 卡号，例如 --gpus "0,1"。如果设备只有一个 GPU 卡号默认为0，可使用`nvidia-smi`命令查看 GPU 使用情况。

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" train.py \
    --dataset_dir "data" \
    --device "gpu" \
    --max_seq_length 128 \
    --model_name "ernie-3.0-medium-zh" \
    --batch_size 32 \
    --early_stop \
    --epochs 100
```
可支持配置的参数：

* `device`: 选用什么设备进行训练，选择 cpu、gpu、xpu、npu。如使用 gpu 训练，可使用参数--gpus 指定 GPU 卡号；默认为"gpu"。
* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含 train.txt，dev.txt 和 label.txt 文件;默认为 None。
* `save_dir`：保存训练模型的目录；默认保存在当前目录 checkpoint 文件夹下。
* `max_seq_length`：分词器 tokenizer 使用的最大序列长度，ERNIE 模型最大不能超过2048。请根据文本长度选择，通常推荐128、256或512，若出现显存不足，请适当调低这一参数；默认为128。
* `model_name`：选择预训练模型,可选"ernie-1.0-large-zh-cw","ernie-3.0-xbase-zh", "ernie-3.0-base-zh", "ernie-3.0-medium-zh", "ernie-3.0-micro-zh", "ernie-3.0-mini-zh", "ernie-3.0-nano-zh", "ernie-2.0-base-en", "ernie-2.0-large-en","ernie-m-base","ernie-m-large"；默认为"ernie-3.0-medium-zh"。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：训练最大学习率；默认为3e-5。
* `epochs`: 训练轮次，使用早停法时可以选择100；默认为10。
* `early_stop`：选择是否使用早停法(EarlyStopping)，模型在开发集经过一定 epoch 后精度表现不再上升，训练终止；默认为 False。
* `early_stop_nums`：在设定的早停训练轮次内，模型在开发集上表现不再上升，训练终止；默认为4。
* `logging_steps`: 训练过程中日志打印的间隔 steps 数，默认5。
* `weight_decay`：控制正则项力度的参数，用于防止过拟合，默认为0.0。
* `warmup`：是否使用学习率 warmup 策略，使用时应设置适当的训练轮次（epochs）；默认为 False。
* `warmup_steps`：学习率 warmup 策略的比例数，如果设为1000，则学习率会在1000steps 数从0慢慢增长到 learning_rate, 而后再缓慢衰减；默认为0。
* `init_from_ckpt`: 模型初始 checkpoint 参数地址，默认 None。
* `seed`：随机种子，默认为3。
* `train_file`：本地数据集中训练集文件名；默认为"train.txt"。
* `dev_file`：本地数据集中开发集文件名；默认为"dev.txt"。
* `label_file`：本地数据集中标签集文件名；默认为"label.txt"。

程序运行时将会自动进行训练，评估。同时训练过程中会自动保存开发集上最佳模型在指定的 `save_dir` 中，保存模型文件结构如下所示：

```text
checkpoint/
├── config.json # 模型配置文件，paddlenlp 2.4.5以前为model_config.json
├── model_state.pdparams # 模型参数文件
├── tokenizer_config.json # 分词器配置文件
├── vocab.txt
└── ...
```

**NOTE:**
* 如需恢复模型训练，则可以设置 `init_from_ckpt` ， 如 `init_from_ckpt=checkpoint/model_state.pdparams` 。
* 如需训练英文文本分类任务，只需更换预训练模型参数 `model_name` 。英文训练任务推荐使用"ernie-2.0-base-en"、"ernie-2.0-large-en"。
* 英文和中文以外语言的文本分类任务，推荐使用基于96种语言（涵盖法语、日语、韩语、德语、西班牙语等几乎所有常见语言）进行预训练的多语言预训练模型"ernie-m-base"、"ernie-m-large"，详情请参见[ERNIE-M 论文](https://arxiv.org/pdf/2012.15674.pdf)。

#### 2.4.2 训练评估与模型优化

文本分类预测过程中常会遇到诸如"模型为什么会预测出错误的结果"，"如何提升模型的表现"等问题。[Analysis 模块](./analysis) 提供了**模型评估、可解释性分析、数据优化**等功能，旨在帮助开发者更好地分析文本分类模型预测结果和对模型效果进行优化。

<div align="center">
    <img src="https://user-images.githubusercontent.com/63761690/195241942-70068989-df17-4f53-9f71-c189d8c5c88d.png" width="600">
</div>

**模型评估：** 训练后的模型我们可以使用 [Analysis 模块](./analysis) 对每个类别分别进行评估，并输出预测错误样本（bad case），默认在 GPU 环境下使用，在 CPU 环境下修改参数配置为`--device "cpu"`:

```shell
python analysis/evaluate.py --device "gpu" --max_seq_length 128 --batch_size 32 --bad_case_file "bad_case.txt" --dataset_dir "data" --params_path "./checkpoint"
```

输出打印示例：

```text
[2022-08-12 02:24:48,193] [    INFO] - -----Evaluate model-------
[2022-08-12 02:24:48,194] [    INFO] - Dev dataset size: 1611
[2022-08-12 02:24:48,194] [    INFO] - Accuracy in dev dataset: 74.24%
[2022-08-12 02:24:48,194] [    INFO] - Macro avg in dev dataset: precision: 82.96 | recall: 77.59 | F1 score 79.36
[2022-08-12 02:24:48,194] [    INFO] - Micro avg in dev dataset: precision: 91.50 | recall: 89.66 | F1 score 90.57
[2022-08-12 02:24:48,195] [    INFO] - Class name: 婚后有子女
[2022-08-12 02:24:48,195] [    INFO] - Evaluation examples in dev dataset: 784(48.7%) | precision: 97.07 | recall: 97.32 | F1 score 97.20
[2022-08-12 02:24:48,195] [    INFO] - ----------------------------
[2022-08-12 02:24:48,195] [    INFO] - Class name: 限制行为能力子女抚养
[2022-08-12 02:24:48,195] [    INFO] - Evaluation examples in dev dataset: 492(30.5%) | precision: 88.57 | recall: 88.21 | F1 score 88.39
...
```

预测错误的样本保存在 bad_case.txt 文件中：

```text
Text    Label    Prediction
2014年，王X以其与肖X协议离婚时未分割该套楼房的首付款为由，起诉至法院，要求分得楼房的首付款15万元。    不动产分割,有夫妻共同财产    不动产分割
但原、被告对已建立起的夫妻感情不够珍惜，因琐事即发生吵闹并最终分居，对夫妻感情造成了严重的影响，现原、被告已分居六年有余，且经人民法院判决不准离婚后仍未和好，夫妻感情确已破裂，依法应准予原、被告离婚。    二次起诉离婚,准予离婚,婚后分居,法定离婚    婚后分居,准予离婚
婚后生有一女，取名彭某乙，已11岁，现已由被告从铁炉白族乡中心小学转入走马镇李桥小学读书。    婚后有子女    婚后有子女,限制行为能力子女抚养
...
```
**可解释性分析：** 基于[TrustAI](https://github.com/PaddlePaddle/TrustAI)提供单词和句子级别的模型可解释性分析，帮助理解模型预测结果，用于错误样本（bad case）分析，细节详见[训练评估与模型优化指南](analysis/README.md)。

- 单词级别可解释性分析，也即分析待预测样本中哪一些单词对模型预测结果起重要作用。以下图为例，用颜色深浅表示单词对预测结果的重要性。
<div align="center">
    <img src="https://user-images.githubusercontent.com/63761690/192739675-63145d59-23c6-416f-bf71-998fd4995254.png" width="1000">
</div>

- 句子级别可解释性分析 ，也即分析对待预测样本的模型预测结果与训练集中中哪些样本有重要关系。下面的例子表明句子级别可解释性分析可以帮助理解待预测样本的预测结果与训练集中样本之间的关联。
```text
text: 2015年2月23日，被告将原告赶出家门，原告居住于娘家待产，双方分居至今。
predict label: 婚后分居
label: 不履行家庭义务,婚后分居
examples with positive influence
support1 text: 2014年中秋节原告回了娘家，原、被告分居至今。    label: 婚后分居    score: 0.99942
support2 text: 原告于2013年8月13日离开被告家，分居至今。    label: 婚后分居    score: 0.99916
support3 text: 2014年4月，被告外出务工，双方分居至今。    label: 婚后分居    score: 0.99902
...
```

**数据优化：** 结合[TrustAI](https://github.com/PaddlePaddle/TrustAI)和[数据增强 API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md)提供了**稀疏数据筛选、脏数据清洗、数据增强**三种优化策略，从多角度优化训练数据提升模型效果，策略细节详见[训练评估与模型优化指南](analysis/README.md)。

- 稀疏数据筛选主要是解决数据不均衡、训练数据覆盖不足的问题，通过数据增强和数据标注两种方式解决这一问题。
- 脏数据清洗可以帮助开发者筛选训练集中错误标注的数据，对这些数据重新进行人工标注，得到标注正确的数据再重新进行训练。
- 数据增强策略提供多种数据增强方案，可以快速扩充数据，提高模型泛化性和鲁棒性。


#### 2.4.3 模型预测
训练结束后，输入待预测数据(data.txt)和类别标签对照列表(label.txt)，使用训练好的模型进行，默认在 GPU 环境下使用，在 CPU 环境下修改参数配置为`--device "cpu"`：

```shell
python predict.py --device "gpu" --max_seq_length 128 --batch_size 32 --dataset_dir "data"
```

可支持配置的参数：

* `device`: 选用什么设备进行预测，可选 cpu、gpu、xpu、npu；默认为 gpu。
* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含 data.txt 和 label.txt 文件;默认为 None。
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

如果使用多语言模型 ERNIE M 作为预训练模型，运行方式：

```shell
python export_model.py --params_path ./checkpoint/ --output_path ./export --multilingual
```

可支持配置的参数：

* `multilingual`：是否为多语言任务（是否使用 ERNIE M 作为预训练模型）；默认为 False。
* `params_path`：动态图训练保存的参数路径；默认为"./checkpoint/"。
* `output_path`：静态图图保存的参数路径；默认为"./export"。

程序运行时将会自动导出模型到指定的 `output_path` 中，保存模型文件结构如下所示：

```text
export/
├── float32.pdiparams
├── float32.pdiparams.info
└── float32.pdmodel
```
 导出模型之后用于部署，项目提供了基于 ONNXRuntime 的 [离线部署方案](./deploy/predictor/README.md) 和基于 Paddle Serving 的 [在线服务化部署方案](./deploy/predictor/README.md)。

#### 2.5.2 模型裁剪

如果有模型部署上线的需求，需要进一步压缩模型体积，可以使用 PaddleNLP 的 [压缩 API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/compression.md), 一行命令即可启动模型裁剪。

使用裁剪功能需要安装 paddleslim：

```shell
pip install paddleslim==2.4.1
```

开始模型裁剪训练，默认为 GPU 训练，使用 CPU 训练只需将设备参数配置改为`--device "cpu"`：
```shell
python prune.py \
    --device "gpu" \
    --dataset_dir "data" \
    --output_dir "prune" \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 10 \
    --max_seq_length 128 \
    --logging_steps 5 \
    --save_steps 100 \
    --width_mult_list '3/4' '2/3' '1/2'
```


可支持配置的参数：
* `output_dir`：必须，保存模型输出和中间 checkpoint 的输出目录;默认为 `None` 。
* `device`: 选用什么设备进行裁剪，选择 cpu、gpu。如使用 gpu 训练，可使用参数--gpus 指定 GPU 卡号。
* `per_device_train_batch_size`：训练集裁剪训练过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `per_device_eval_batch_size`：开发集评测过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：训练最大学习率；默认为5e-5。
* `num_train_epochs`: 训练轮次，使用早停法时可以选择100；默认为10。
* `logging_steps`: 训练过程中日志打印的间隔 steps 数，默认100。
* `save_steps`: 训练过程中保存模型 checkpoint 的间隔 steps 数，默认100。
* `seed`：随机种子，默认为3。
* `width_mult_list`：裁剪宽度（multi head）保留的比例列表，表示对 self_attention 中的 `q`、`k`、`v` 以及 `ffn` 权重宽度的保留比例，保留比例乘以宽度（multi haed 数量）应为整数；默认是 None。
* `dataset_dir`：本地数据集路径，需包含 train.txt,dev.txt,label.txt;默认为 None。
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

2. 模型裁剪主要用于推理部署，因此裁剪后的模型都是静态图模型，只可用于推理部署，不能再通过 `from_pretrained` 导入继续训练。导出模型之后用于部署，项目提供了基于 ONNXRuntime 的 [离线部署方案](./deploy/predictor/README.md) 和基于 Paddle Serving 的 [在线服务化部署方案](./deploy/predictor/README.md)。

3. ERNIE Base、Medium、Mini、Micro、Nano 的模型宽度（multi head 数量）为12，ERNIE Xbase、Large 模型宽度（multi head 数量）为16，保留比例`width_mult`乘以宽度（multi haed 数量）应为整数。


#### 2.5.3 部署方案

- 离线部署搭建请参考[离线部署](deploy/predictor/README.md)。

- 在线服务化部署搭建请参考 [PaddleNLP SimpleServing 部署指南](deploy/simple_serving/README.md) 或 [Triton 部署指南](deploy/triton_serving/README.md)。

<a name="模型效果"></a>

### 2.6 模型效果

我们在 CAIL2019—婚姻家庭要素提取任务数据集评测模型表现，测试配置如下：

1. 数据集：CAIL2019—婚姻家庭要素提取任务数据集

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

5. 性能数据指标：latency。latency 测试方法：固定 batch size 为 32，GPU 部署运行时间 total_time，计算 latency = total_time / total_samples

6. 精度评价指标：Micro F1分数、Macro F1分数

|  model_name  | 模型结构  |Micro F1(%)   | Macro F1(%) | latency(ms) |
| -------------------------- | ------------ | ------------ | ------------ |------------ |
|ERNIE 1.0 Large Cw |24-layer, 1024-hidden, 20-heads|91.14|81.68 |5.66 |
|ERNIE 3.0 Base  |12-layer, 768-hidden, 12-heads|90.38|80.14| 2.70 |
|ERNIE 3.0 Medium| 6-layer, 768-hidden, 12-heads|90.57|79.36| 1.46|
|ERNIE 3.0 Mini |6-layer, 384-hidden, 12-heads|89.27|76.78| 0.56|
|ERNIE 3.0 Micro | 4-layer, 384-hidden, 12-heads|89.43|77.20| 0.34|
|ERNIE 3.0 Nano |4-layer, 312-hidden, 12-heads|85.39|75.07|0.32|
| ERNIE 3.0 Medium + 裁剪(保留比例3/4)|6-layer, 768-hidden, 9-heads| 89.94|79.35| 0.81   |
| ERNIE 3.0 Medium + 裁剪(保留比例2/3)|6-layer, 768-hidden, 8-heads| 89.99|79.37 | 0.75  |
| ERNIE 3.0 Medium + 裁剪(保留比例1/2)|6-layer, 768-hidden, 6-heads| 89.19 | 76.35| 0.61 |
