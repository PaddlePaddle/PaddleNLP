# 层次分类指南

**目录**
- [1. 层次分类简介](#层次分类简介)
- [2. 快速开始](#快速开始)
    - [2.1 运行环境](#运行环境)
    - [2.2 代码结构](#代码结构)
    - [2.3 数据准备](#数据准备)
    - [2.4 模型训练](#模型训练)
    - [2.5 模型部署](#模型部署)
    - [2.6 模型效果](#模型效果)

<a name="层次分类简介"></a>

## 1. 层次分类简介

本项目提供通用场景下**基于预训练模型微调的层次分类端到端应用方案**，打通数据标注-模型训练-模型调优-模型压缩-预测部署全流程，有效缩短开发周期，降低 AI 开发落地门槛。

层次文本分类任务的中数据样本具有多个标签且标签之间存在特定的层级结构，目标是**预测输入句子/文本可能来自于不同级标签类别中的某一个或几个类别**。以下图新闻文本分类为例，该新闻的一级标签为体育，二级标签为足球，体育与足球之间存在层级关系。在现实场景中，大量的数据如新闻分类、专利分类、学术论文分类等标签集合存在层次化结构，需要利用算法为文本自动标注更细粒度和更准确的标签。

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/186654723-6a287f18-56cc-4727-9347-09cfaf11b1dc.png width="550"/>
</div>
<br>

**方案亮点：**

- **效果领先🏃：** 使用在中文领域内模型效果和模型计算效率有突出效果的 ERNIE 3.0 轻量级系列模型作为训练基座，ERNIE 3.0 轻量级系列提供多种尺寸的预训练模型满足不同需求，具有广泛成熟的实践应用性。
- **高效调优✊：** 文本分类应用依托[TrustAI](https://github.com/PaddlePaddle/TrustAI)可信增强能力和[数据增强 API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md)，提供模型分析模块助力开发者实现模型分析，并提供稀疏数据筛选、脏数据清洗、数据增强等多种解决方案。
- **简单易用👶：** 开发者**无需机器学习背景知识**，仅需提供指定格式的标注分类数据，一行命令即可开启文本分类训练，轻松完成上线部署，不再让技术成为文本分类的门槛。

**更多选择：**

对于大多数层次分类任务，我们推荐使用预训练模型微调作为首选的文本分类方案，层次分类项目中还提供 提示学习(小样本)和语义索引的两种全流程文本分类方案满足不同开发者需求，更多技术细节请参见[文本分类技术特色介绍](../README.md)。

- 【标注成本高、标注样本较少的小样本场景】 👉 [提示学习层次分类方案](./few-shot#readme)

- 【标签类别不固定场景、标签数量众多】 👉 [语义索引层次分类方案](./retrieval_based#readme)

<a name="快速开始"></a>

## 2. 快速开始

我们以[2020语言与智能技术竞赛：事件抽取任务](https://aistudio.baidu.com/aistudio/competition/detail/32/0/introduction)抽取的多标签层次数据集为例，演示层次分类全流程方案使用。下载数据集：
```shell
wget https://paddlenlp.bj.bcebos.com/datasets/baidu_extract_2020.tar.gz
tar -zxvf baidu_extract_2020.tar.gz
mv baidu_extract_2020 data
rm baidu_extract_2020.tar.gz
```

<div align="center">
    <img width="900" alt="image" src="https://user-images.githubusercontent.com/63761690/187828356-e2f4f627-f5fe-4c83-8879-ed6951f7511e.png">
</div>
<div align="center">
    <font size ="2">
    层次分类数据标注-模型训练-模型分析-模型压缩-预测部署流程图
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
hierarchical/
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

**训练、开发、测试数据集文件：** 文本与标签类别名用 tab 符`'\t'`分隔开，标签中多个标签之间用英文逗号`','`分隔开，文本中避免出现 tab 符`'\t'`。

- train.txt/dev.txt/test.txt 文件格式：
```text
<文本>'\t'<标签>','<标签>','<标签>
<文本>'\t'<标签>','<标签>
...
```

- train.txt/dev.txt/test.txt 文件样例：
```text
又要停产裁员6000！通用汽车罢工危机再升级股价大跌市值蒸发近300亿！    组织行为,组织行为##罢工,组织关系,组织关系##裁员
上海一改建厂房坍塌已救出19人其中5人死亡    人生,人生##死亡,灾害/意外,灾害/意外##坍/垮塌
车闻：广本召回9万余辆；领动上市，10.98万起；艾力绅混动    产品行为,产品行为##召回
86岁老翁过马路遭挖掘机碾压身亡警方：正在侦办中    灾害/意外,灾害/意外##车祸,人生,人生##死亡
...
```

**分类标签文件：** 包含数据集中所有标签，每个标签一行。

- label.txt 文件格式：

```text
<一级标签>
<一级标签>'##'<二级标签>
<一级标签>'##'<二级标签>'##'<三级标签>
...
```
- label.txt  文件样例：
```text
人生
人生##死亡
灾害/意外
灾害/意外##坍/垮塌
灾害/意外##车祸
产品行为
产品行为##召回
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
金属卡扣安装不到位，上海乐扣乐扣贸易有限公司将召回捣碎器1162件
卡车超载致使跨桥侧翻，没那么简单
消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了
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
* `model_name`：选择预训练模型,可选"ernie-1.0-large-zh-cw","ernie-3.0-xbase-zh", "ernie-3.0-base-zh", "ernie-3.0-medium-zh", "ernie-3.0-micro-zh", "ernie-3.0-mini-zh", "ernie-3.0-nano-zh", "ernie-2.0-base-en", "ernie-2.0-large-en","ernie-m-base","ernie-m-large"；默认为"ernie-3.0-medium-zh",根据任务复杂度和硬件条件进行选择。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：训练最大学习率；默认为3e-5。
* `epochs`: 训练轮次，使用早停法时可以选择100；默认为10。
* `early_stop`：选择是否使用早停法(EarlyStopping)，模型在开发集经过一定 epoch 后精度表现不再上升，训练终止；默认为 False。
* `early_stop_nums`：在设定的早停训练轮次内，模型在开发集上表现不再上升，训练终止；默认为10。
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
* 如需恢复模型训练，则可以设置 `--init_from_ckpt checkpoint/model_state.pdparams` 。
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
[2022-08-11 03:10:14,058] [    INFO] - -----Evaluate model-------
[2022-08-11 03:10:14,059] [    INFO] - Dev dataset size: 1498
[2022-08-11 03:10:14,059] [    INFO] - Accuracy in dev dataset: 89.19%
[2022-08-11 03:10:14,059] [    INFO] - Macro avg in dev dataset: precision: 93.48 | recall: 93.26 | F1 score 93.22
[2022-08-11 03:10:14,059] [    INFO] - Micro avg in dev dataset: precision: 95.07 | recall: 95.46 | F1 score 95.26
[2022-08-11 03:10:14,095] [    INFO] - Level 1 Label Performance: Macro F1 score: 96.39 | Micro F1 score: 96.81 | Accuracy: 94.93
[2022-08-11 03:10:14,255] [    INFO] - Level 2 Label Performance: Macro F1 score: 92.79 | Micro F1 score: 93.90 | Accuracy: 89.72
[2022-08-11 03:10:14,256] [    INFO] - Class name: 交往
[2022-08-11 03:10:14,256] [    INFO] - Evaluation examples in dev dataset: 60(4.0%) | precision: 91.94 | recall: 95.00 | F1 score 93.44
[2022-08-11 03:10:14,256] [    INFO] - ----------------------------
[2022-08-11 03:10:14,256] [    INFO] - Class name: 交往##会见
[2022-08-11 03:10:14,256] [    INFO] - Evaluation examples in dev dataset: 12(0.8%) | precision: 92.31 | recall: 100.00 | F1 score 96.00
...
```

预测错误的样本保存在 bad_case.txt 文件中：

```text
Text    Label    Prediction
据猛龙随队记者JoshLewenberg报道，消息人士透露，猛龙已将前锋萨加巴-科纳特裁掉。此前他与猛龙签下了一份Exhibit10合同。在被裁掉后，科纳特下赛季大概率将前往猛龙的发展联盟球队效力。    组织关系,组织关系##加盟,组织关系##裁员    组织关系,组织关系##解雇
冠军射手被裁掉，欲加入湖人队，但湖人却无意，冠军射手何去何从    组织关系,组织关系##裁员    组织关系,组织关系##解雇
6月7日报道，IBM将裁员超过1000人。IBM周四确认，将裁减一千多人。据知情人士称，此次裁员将影响到约1700名员工，约占IBM全球逾34万员工中的0.5%。IBM股价今年累计上涨16%，但该公司4月发布的财报显示，一季度营收下降5%，低于市场预期。    组织关系,组织关系##裁员    组织关系,组织关系##裁员,财经/交易
有多名魅族员工表示，从6月份开始，魅族开始了新一轮裁员，重点裁员区域是营销和线下。裁员占比超过30%，剩余员工将不过千余人，魅族的知名工程师，爱讲真话的洪汉生已经从钉钉里退出了，外界传言说他去了OPPO。    组织关系,组织关系##退出,组织关系##裁员    组织关系,组织关系##裁员
...
```

**可解释性分析：** 基于[TrustAI](https://github.com/PaddlePaddle/TrustAI)提供单词和句子级别的模型可解释性分析，帮助理解模型预测结果，用于错误样本（bad case）分析，细节详见[训练评估与模型优化指南](analysis/README.md)。

- 单词级别可解释性分析，也即分析待预测样本中哪一些单词对模型预测结果起重要作用。以下图为例，用颜色深浅表示单词对预测结果的重要性。
<div align="center">
    <img src="https://user-images.githubusercontent.com/63761690/195334753-78cc2dc8-a5ba-4460-9fde-3b1bb704c053.png" width="1000">
</div>

- 句子级别可解释性分析 ，也即分析对待预测样本的模型预测结果与训练集中中哪些样本有重要关系。下面的例子表明句子级别可解释性分析可以帮助理解待预测样本的预测结果与训练集中样本之间的关联。
```text
text: 据猛龙随队记者JoshLewenberg报道，消息人士透露，猛龙已将前锋萨加巴-科纳特裁掉。此前他与猛龙签下了一份Exhibit10合同。在被裁掉后，科纳特下赛季大概率将前往猛龙的发展联盟球队效力。
predict label: 组织关系,组织关系##解雇
label: 组织关系,组织关系##加盟,组织关系##裁员
examples with positive influence
support1 text: 尼克斯官方今日宣布，他们已经裁掉了前锋扎克-欧文，后者昨日才与尼克斯签约。    label: 组织关系,组织关系##加盟,组织关系##解雇    score: 0.99357
support2 text: 活塞官方今日宣布，他们已经签下了克雷格-斯沃德，并且裁掉了托德-威瑟斯。    label: 组织关系,组织关系##加盟,组织关系##解雇    score: 0.98344
support3 text: 孟菲斯灰熊今年宣布，球队已经签下后卫达斯蒂-汉纳斯（DustyHannahs，版头图）并裁掉马特-穆尼。    label: 组织关系,组织关系##加盟,组织关系##解雇    score: 0.98219
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

我们在[2020语言与智能技术竞赛：事件抽取任务](https://aistudio.baidu.com/aistudio/competition/detail/32/0/introduction)的多标签层次数据集评测模型表现，测试配置如下：

1. 数据集：2020语言与智能技术竞赛抽取的多标签层次数据集

2. 物理机环境

    系统: CentOS Linux release 7.7.1908 (Core)

    GPU: Tesla V100-SXM2-32GB

    CPU: Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz

    CUDA: 11.2

    cuDNN: 8.1.0

    Driver Version: 460.27.04

    内存: 630 GB

3. PaddlePaddle 版本：2.3.0

4. PaddleNLP 版本：2.4

5. 性能数据指标：latency。latency 测试方法：固定 batch size 为 32，GPU 部署运行时间 total_time，计算 latency = total_time / total_samples

6. 精度评价指标：Micro F1分数、Macro F1分数

|   | 模型结构  |Micro F1(%)   | Macro F1(%) | latency(ms) |
| -------------------------- | ------------ | ------------ | ------------ |------------ |
|ERNIE 1.0 Large Cw  |24-layer, 1024-hidden, 20-heads|96.24|94.24 |5.59 |
|ERNIE 3.0 Xbase |20-layer, 1024-hidden, 16-heads|96.21|94.13| 5.51 |
|ERNIE 3.0 Base |12-layer, 768-hidden, 12-heads|95.68|93.39| 2.01 |
|ERNIE 3.0 Medium| 6-layer, 768-hidden, 12-heads|95.26|93.22| 1.01|
|ERNIE 3.0 Mini|6-layer, 384-hidden, 12-heads|94.72|93.03| 0.36|
|ERNIE 3.0 Micro | 4-layer, 384-hidden, 12-heads|94.24|93.08| 0.24|
|ERNIE 3.0 Nano |4-layer, 312-hidden, 12-heads|93.98|91.25|0.19|
| ERNIE 3.0 Medium + 裁剪(保留比例3/4)|6-layer, 768-hidden, 9-heads| 95.45|93.40| 0.81   |
| ERNIE 3.0 Medium + 裁剪(保留比例2/3)|6-layer, 768-hidden, 8-heads| 95.23|93.27 | 0.74  |
| ERNIE 3.0 Medium + 裁剪(保留比例1/2)|6-layer, 768-hidden, 6-heads| 94.92 | 92.70| 0.61 |
