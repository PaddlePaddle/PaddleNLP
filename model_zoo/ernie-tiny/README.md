
# ERNIE Tiny

 **目录**
   * [模型介绍](#模型介绍)
   * [模型效果](#模型效果)
   * [代码结构](#代码结构)
   * [开始运行](#开始运行)
       * [环境要求](#环境要求)
       * [数据准备](#数据准备)
   * [模型训练](#模型训练)
   * [模型压缩](#模型压缩)
       * [环境依赖](#环境依赖)
       * [压缩效果](#压缩效果)
   * [部署](#部署)
   * [参考文献](#参考文献)


<a name="模型介绍"></a>

## 模型介绍
由于预训练语言模型越来愈大，过大的参数量导致了模型难以使用。[ERNIE-Tiny v1](../ernie-3.0/) 通过 task-agnostic 知识蒸馏的方式将大模型压缩成开箱即用的小模型，在下游任务上直接微调就能取得不错的效果。然而我们发现蒸馏出来的小模型和教师模型仍然存在效果差距，对此我们提出并开源了 **ERNIE-Tiny v2** 。 ERNIE-Tiny v2 通过在教师模型融入**多任务训练**，大大提高了小模型在下游任务上的效果。

### 注入下游知识
ERNIE-Tiny v1 通过 task-agnostic 蒸馏技术将预训练大模型压缩成预训练小模型，然而由于小模型在微调之前没有接触到下游任务的相关知识，导致效果和大模型仍然存在差距。因此我们提出 ERNIE-Tiny v2，通过微调教师模型，从而让教师模型学习到下游任务的相关知识，进而能够在蒸馏的过程中传导给学生模型。尽管学生模型完全没有见过下游数据，通过预先注入下游知识到教师模型，蒸馏得到的学生模型也能够获取到下游任务的相关知识，进而提升下游任务上的效果。

### 多任务学习提升泛化性

多任务学习已经被证明对增强模型泛化性有显著的效果，例如 MT-DNN、MUPPET、FLAN 等。通过对教师模型加入多下游任务微调，不但能够对教师模型注入下游知识、提高教师模型的泛化性，并且能够通过蒸馏传给学生模型，大幅度提升小模型的泛化性。具体地，我们对教师模型进行了28个任务的多任务微调。ERNIE Tiny v2 在 in-domain、out-domain、low-resourced 数据上获得显著的提升。

[](https://user-images.githubusercontent.com/26483581/210303124-c9df89a9-e291-4322-a6a5-37d2c4c1c008.png)

<p align="center">
        <img width="644" alt="image" src="https://user-images.githubusercontent.com/26483581/210303124-c9df89a9-e291-4322-a6a5-37d2c4c1c008.png" title="ERNIE Tiny v2">
</p>

<a name="模型效果"></a>

## 模型效果

本项目开源 **ERNIE 3.0 Tiny _Base_ v2** 、**ERNIE 3.0 Tiny _Medium_ v2** 、 **ERNIE 3.0 Tiny _Mini_ v2** 、 **ERNIE 3.0 Tiny _Micro_ v2** 、 **ERNIE 3.0 Tiny _Nano_ v2**、**ERNIE 3.0 Tiny _Pico_ v2** 六个中文模型：

- [**ERNIE 3.0-Tiny_Base_v2_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_tiny_base_v2.pdparams) (_12-layer, 768-hidden, 12-heads_)
- [**ERNIE 3.0-Tiny_Medium_v2_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_tiny_medium_v2.pdparams) (_6-layer, 768-hidden, 12-heads_)
- [**ERNIE 3.0-Tiny_Mini_v2_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_tiny_mini_v2.pdparams) (_6-layer, 384-hidden, 12-heads_)
- [**ERNIE 3.0-Tiny_Micro_v2_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_tiny_micro_v2.pdparams) (_4-layer, 384-hidden, 12-heads_)
- [**ERNIE 3.0-Tiny_Nano_v2_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_tiny_nano_v2.pdparams) (_4-layer, 312-hidden, 12-heads_)
- [**ERNIE 3.0-Tiny_Pico_v2_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_tiny_pico_v2.pdparams) (_4-layer, 312-hidden, 2-heads_)

模型效果表格TBD

<a name="代码结构"></a>

## 代码结构

以下是本项目代码结构

```text
.
├── train.py                     # 微调和压缩脚本
├── utils.py                     # 训练工具脚本
├── model.py                     # 模型结构脚本
├── test.py                      # 评估脚本
├── deploy                       # 部署目录
└── README.md                    # 文档

```

<a name="开始运行"></a>

## 开始运行

### 任务介绍

ERNIE Tiny 模型可以用于文本分类、文本推理、实体抽取、问答等多类 NLU 任务中。本案例是车载语音场景下的口语理解（Spoken Language Understanding，SLU）任务，SLU 任务主要将用户的自然语言表达解析为结构化信息。结构化信息的解析主要包括意图识别和槽位填充两个步骤。

此次微调数据使用了 [NLPCC2018 Shared Task 4](http://tcci.ccf.org.cn/conference/2018/taskdata.php) 数据集，数据集来源于中文真实商用车载语音任务型对话系统的对话日志。需要说明的一点是，在本案例中，只考虑了意图识别和槽位填充任务，纠错数据在此被忽略掉了。

数据样例：

```text
- 输入：来一首周华健的花心
- 输出
	- 意图识别任务：music.play
	- 槽位填充任务：来一首<singer>周华健</singer>的<song>花心</song>
```
在本案例中，意图识别和槽位填充任务分别被建模为文本分类和序列标注任务，二者共用一个 ERNIE Tiny 模型，只有最后的任务层是独立的。

### 环境要求
- python >= 3.7
- paddlepaddle >= 2.4.1
- paddlenlp >= 2.5
- paddleslim >= 2.4

### 数据准备

训练集的下载地址为[链接](http://tcci.ccf.org.cn/conference/2018/dldoc/trainingdata04.zip)。

需要经过以下处理：
TBD

<a name="模型训练"></a>

## 模型训练

使用 PaddleNLP 只需要一行代码可以下载和调用 ERNIE Tiny 模型，之后可以在自己的下游数据下进行微调，从而获得具体任务上效果更好的模型。

```python

from paddlenlp.transformers import *

tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-tiny-medium-v2-zh")

# 用于分类任务
seq_cls_model = AutoModelForSequenceClassification.from_pretrained("ernie-3.0-tiny-medium-v2-zh")

# 用于序列标注任务
token_cls_model = AutoModelForTokenClassification.from_pretrained("ernie-3.0-tiny-medium-v2-zh")

```

在本案例中，使用的预训练模型是 `ernie-3.0-tiny-nano-v2-zh`，也可以根据使用需求选用 ERNIE-Tiny 系列的其他模型。模型训练使用 Trainer API，非常简洁实用。

运行下面的脚本，即可使用 ERNIE-Tiny 模型启动训练：

```shell
BS=64
LR=5e-5
WD=0.01
WR=0.1
EPOCHS=20

export finetuned_model=./output/BS${BS}_LR${LR}_${EPOCHS}EPOCHS_WD${WD}_WR${WR}
mkdir $finetuned_model

python train.py \
    --device gpu \
    --logging_steps 100 \
    --save_steps 100 \
    --eval_steps 100 \
    --seed 42 \
    --model_name_or_path ernie-3.0-tiny-nano-v2-zh \
    --output_dir $finetuned_model \
    --train_path data/train.txt \
    --dev_path data/dev.txt \
    --intent_label_path data/intent_label.txt \
    --slot_label_path data/slots_label.txt \
    --label_names  'intent_label' 'slot_label' \
    --max_seq_length 16  \
    --per_device_eval_batch_size ${BS} \
    --per_device_train_batch_size  ${BS} \
    --learning_rate ${LR} \
    --weight_decay ${WD} \
    --warmup_ratio ${WR} \
    --do_train \
    --do_eval \
    --do_export \
    --disable_tqdm True \
    --overwrite_output_dir \
    --num_train_epochs $EPOCHS \
    --load_best_model_at_end  True \
    --save_total_limit 1 \
    --metric_for_best_model eval_accuracy \

```

可配置参数说明：

* `model_name_or_path`：必须，进行微调使用的预训练模型。可选择的有 "ernie-3.0-tiny-base-v2-zh"、"ernie-3.0-tiny-medium-v2-zh"、"ernie-3.0-tiny-mini-v2-zh"、"ernie-3.0-tiny-micro-v2-zh"、"ernie-3.0-tiny-nano-v2-zh"、"ernie-3.0-tiny-poco-v2-zh"。
* `output_dir`：必须，模型训练或压缩后保存的模型目录；默认为 `None` 。
* `device`: 训练设备，可选择 'cpu'、'gpu' 其中的一种；默认为 GPU 训练。
* `per_device_train_batch_size`：训练集训练过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为 32。
* `per_device_eval_batch_size`：开发集评测过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为 32。
* `learning_rate`：训练最大学习率。
* `num_train_epochs`: 训练轮次，使用早停法时可以选择 100；默认为10。
* `logging_steps`: 训练过程中日志打印的间隔 steps 数，默认100。
* `save_steps`: 训练过程中保存模型 checkpoint 的间隔 steps 数，默认100。
* `seed`：全局随机种子，默认为 42。
* `weight_decay`：除了所有 bias 和 LayerNorm 权重之外，应用于所有层的权重衰减数值。可选；默认为 0.0；
* `do_train`:是否进行微调训练，设置该参数表示进行微调训练。
* `do_eval`:是否进行评估，设置该参数表示进行评估。
* `--do_export`：训练完是否导出模型。


## 模型评估

```shell
python test.py  \
    --device gpu \
    --model_name_or_path output/BS64_LR5e-5_20EPOCHS_WD0.01_WR0.1/checkpoint-6460/ \
    --infer_prefix output/BS64_LR5e-5_20EPOCHS_WD0.01_WR0.1/infer_model \
    --output_dir ./ \
    --test_path data/dev.txt \
    --intent_label_path data/intent_label.txt \
    --slot_label_path data/slots_label.txt \
    --max_seq_length 16  \
    --per_device_eval_batch_size 512 \
    --do_eval
```

如果设置`--do_eval`，脚本会开启评估模式，最终会输出精度指标。如果不设置`--do_eval`，则会输出模型后处理后的结果。

<a name="模型压缩"></a>

## 模型压缩

尽管 ERNIE Tiny 已提供了效果不错的轻量级模型可以微调后直接使用，但如果有模型部署上线的需求，则需要进一步压缩模型体积，可以使用这里提供的一套模型压缩方案对上一步微调后的模型进行压缩。


在本案例中，运行下面的脚本，即可使用上面微调后的模型进行压缩：

```shell
BS=64
LR=5e-5
WD=0.01
WR=0.1
EPOCHS=20

export finetuned_model=./output/BS${BS}_LR${LR}_${EPOCHS}EPOCHS_WD${WD}_WR${WR}
mkdir $finetuned_model

EPOCHS=1

python train.py \
    --device gpu \
    --logging_steps 100 \
    --save_steps 100 \
    --eval_steps 100 \
    --seed 42 \
    --model_name_or_path $finetuned_model/checkpoint-6460 \
    --output_dir $finetuned_model \
    --train_path data/train.txt \
    --dev_path data/dev.txt \
    --intent_label_path data/intent_label.txt \
    --slot_label_path data/slots_label.txt \
    --label_names  'intent_label' 'slot_label' \
    --max_seq_length 16  \
    --per_device_eval_batch_size ${BS} \
    --per_device_train_batch_size  ${BS} \
    --learning_rate ${LR} \
    --weight_decay ${WD} \
    --onnx_format False \
    --warmup_ratio ${WR} \
    --do_compress \
    --strategy 'dynabert+qat+embeddings' \
    --disable_tqdm True \
    --num_train_epochs $EPOCHS \
    --save_total_limit 1 \
    --metric_for_best_model eval_accuracy
```

可配置参数说明：

* `strategy`：压缩策略，本案例中推荐使用`"dynabert+qat+embeddings"`。这是一个自行拼接的策略组合。其中`"dynabert"` 是裁剪策略，能直接对模型结构进行裁剪，从而减少参数量；`"qat"` 是一种量化方法，用于将模型中矩阵乘(底层是 matmul_v2 算子)的权重及激活值数据类型由 FP32 转成 INT8，并使精度尽量保持无损，`"embeddings"` 则代 Embedding 量化策略，将 Embedding API（底层是 lookup_table_v2 算子）的权重由 FP32 转成 INT8 存储，由于词表参数量占比非常大，Embedding 量化能够大幅度减少模型的内存占用。
* `model_name_or_path`：必须，进行压缩所使用的微调模型，需要是上面微调后的`$finetuned_model`模型。
* `output_dir`：必须，模型训练或者压缩后保存的模型目录；默认为 `None` 。


其他参数同训练参数的部分，是指压缩过程中的训练所使用的参数。

### 压缩效果

模型经过压缩后，在TBD芯片上使用 Paddle Lite 上进行了测试（max_seq_length=16，batch_size=1），精度、时延、内存占用的数据如下：

| 模型                    | 策略                      | 精度 | 时延(ms) | 内存占用 Pss (KB) | 磁盘占用（KB） |
|-----------------------|-------------------------|----|--------|---------------|----------|
| 原模型                   | -                       |    |        |               |          |
| 原模型+裁剪                | dynabert                |    |        |               |          |
| 原模型+裁剪+量化             | dynabert+qat            |    |        |               |          |
| 原模型+裁剪+量化+Embedding量化 | dynabert+qat+embeddings |    |        |               |          |


由此可见，经过压缩后，TBD

<a name="部署"></a>

## 部署

部署请参考：[部署指南](./deploy/)


<a name="参考文献"></a>

## 参考文献
TBD
