 **目录**

* [PP-MiniLM中文小模型](#PP-MiniLM中文小模型)
    * [导入PP-MiniLM](#导入PP-MiniLM)
    * [在下游任务上使用PP-MiniLM](#在下游任务上使用PP-MiniLM)
        * [数据介绍](#数据介绍)
        * [环境依赖](#环境依赖)
        * [在下游任务上微调](#在下游任务上微调)
            * [运行方式](#运行方式)
            * [微调后模型精度](#微调后模型精度)
            * [导出微调后模型](#导出微调后模型)
        * [对任务上的模型进行裁剪](#对任务上的模型进行裁剪)
            * [原理简介](#原理简介)
            * [运行方式](#运行方式)
            * [裁剪后模型精度](#裁剪后模型精度)
            * [导出裁剪后的模型](#导出裁剪后的模型)
        * [对任务上的模型进行量化](#对任务上的模型进行量化)
            * [原理简介](#原理简介)
            * [运行方式](#运行方式)
            * [量化后模型精度](#量化后模型精度)
        * [预测](#预测)
            * [环境要求](#环境要求)
            * [运行方式](#运行方式)
            * [性能测试](#性能测试)
    * [参考文献](#参考文献)

<a name="PP-MiniLM中文小模型"></a>

# PP-MiniLM中文小模型

PP-MiniLM 中文特小模型案例旨在提供训推一体的高精度、高性能小模型及解决方案。

当前解决方案依托业界领先的 Task Agnostic 模型蒸馏技术、裁剪技术、量化技术，使得小模型兼具推理速度快、模型效果好、参数规模小的三大特点。

- 推理速度快: 依托 PaddleSlim 的裁剪、量化技术进一步对小模型进行压缩, 使得 PP-MiniLM 量化模型 GPU 推理速度相比 BERT base 加速比高达 4.15；

- 精度高: 我们以 MiniLMv2 提出的 Multi-Head Self-Attention Relation Distillation 技术为基础，通过引入样本间关系知识蒸馏做了进一步算法优化，6 层 PP-MiniLM 模型在 CLUE 数据集上比 12 层 `bert-base-chinese` 高 0.23%，比同等规模的 TinyBERT、UER-py RoBERTa 分别高 2.66%、1.51%；

- 参数规模小：依托 PaddleSlim 裁剪技术，在精度几乎无损(-0.15%)条件下将模型隐层宽度压缩 1/4，模型参数量减少 28%。

**整体效果**

| Model                   | #Params | #FLOPs | Speedup | AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | WSC   | CSL   | CLUE平均值 |
| ----------------------- | ------- | ------ | ------- | ----- | ----- | ------- | ----- | ----- | ----- | ----- | ---------- |
| BERT<sub>base</sub>     | 102.3M  | 10.87B | 1.00x   | 74.17 | 57.17 | 61.14   | 81.14 | 75.08 | 80.26 | 81.47 | 72.92      |
| TinyBERT<sub>6</sub>    | 59.7M   | 5.44B  | 1.66x   | 72.22 | 55.82 | 58.10   | 79.53 | 74.00 | 75.99 | 80.57 | 70.89      |
| UER-py RoBERTa L6-H768 | 59.7M   | 5.44B  | 1.66x   | 69.74 | 66.36 | 59.95   | 77.00 | 71.39 | 71.05 | 82.83 | 71.19      |
| RBT6, Chinese           | 59.7M   | 5.44B  | 1.66x   | 73.96 | 56.37 | 59.72   | 79.37 | 73.05 | 76.97 | 80.80 | 71.46      |
| ERNIE-Tiny              | 90.7M   | 4.83B  | 1.76x   | 70.78 | 55.70 | 59.95   | 75.40 | 70.98 | 67.43 | 76.60 | 68.12      |
| PP-MiniLM 6L-768H       | 59.7M   | 5.44B  | 1.66x   | 74.28 | 57.33 | 61.72   | 81.06 | 76.2  | 86.51 | 78.77 | 73.70      |
| PP-MiniLM裁剪后         | 49.1M   | 4.08B  | 2.00x   | 73.82 | 57.33 | 61.60   | 81.38 | 76.20 | 85.52 | 79.00 | 73.55      |
| PP-MiniLM量化后         | 49.2M   |   -   | 4.15x   | 73.61 | 57.18 | 61.49   | 81.26 | 76.31 | 84.54 | 77.67 | 73.15      |


**NOTE：** 量化后的模型比量化前参数量多了0.1M是因为保存了scale值。

**方案流程**

<p align="center">
<img src="./pp-minilm.png" width="950"/><br />
方案流程图
</p>

如上流程图所示，完整的中文小模型方案分为：导入PP-MiniLM中文预训练小模型、下游任务微调、裁剪、离线量化、预测部署五大步。下面会对这里的每一个步骤进行介绍。除了下游任务微调步骤，其余步骤均可以省略，但我们建议保留下面的每一个步骤。

以下是本范例模型的简要目录结构及说明：

```shell
.
├── general_distill              # 任务无关知识蒸馏目录
│ └── general_distill.py         # 任务无关知识蒸馏脚本
│ └── run.sh                     # 任务无关知识蒸馏启动脚本
│ └── README.md                  # 任务无关知识蒸馏文档
├── finetuning                   # 下游任务训练目录
│ └── run_clue.py                # clue上的微调脚本
│ └── run_clue.sh                # clue上的微调启动脚本
│ └── run_one_search.sh          # 单数据集下精调脚本
│ └── run_all_search.sh          # clue数据集下精调脚本
│ └── export_model.sh            # 导出fine-tuned部署模型脚本
├── ofa                          # ofa裁剪、蒸馏目录
│ └── run_ofa.py                 # ofa裁剪、蒸馏脚本
│ └── run_ofa.sh                 # ofa裁剪、蒸馏启动脚本
│ └── export_model.py            # 导出ofa训练得到的子模型（动、静态图模型）
├── quantization                 # 离线量化目录
│ └── quant_post.py              # 离线量化脚本
│ └── quant.sh                   # 离线量化启动脚本
├── inference                    # 预测目录
│ └── infer.py                   # 预测脚本
│ └── infer_all.sh               # 批量预测量化模型启动脚本
│ └── infer_perf.py              # 量化模型性能测试脚本
│ └── infer_perf.sh              # 量化模型性能测试启动脚本
├── data.py                      # 数据处理脚本
├── pp-minilm.png                # PP-MiniLM方案流程图
└── README.md                    # 文档，本文件

```

<a name="导入PP-MiniLM"></a>

## 导入PP-MiniLM

PP-MiniLM是使用任务无关蒸馏方法，以`roberta-wwm-ext-large`做教师模型蒸馏产出的包含 6 层 Transformer Encoder Layer、Hidden Size为768的中文预训练小模型，在[中文任务测评基准CLUE benchmark](https://github.com/CLUEbenchmark/CLUE)上七个分类任务上的模型精度超过BERT<sub>base</sub>、TinyBERT<sub>6</sub>、UER-py RoBERTa L6-H768、RBT6。

可以这样导入PP-MiniLM：

```python

from paddlenlp.transformers import ErnieModel, ErnieForSequenceClassification

model = ErnieModel.from_pretrained('ppminilm-6l-768h')
model = ErnieForSequenceClassification.from_pretrained('ppminilm-6l-768h') # 用于分类任务
```

PP-MiniLM是一个6层的预训练模型，使用`from_pretrained`导入PP-MiniLM之后，就可以在自己的数据集上进行Fine-tuning。接下来会介绍如何用下游任务数据在导入的 PP-MiniLM 上进行微调、进一步压缩及推理部署。

**NOTE：** 如果对PP-MiniLM的训练过程感兴趣，可以查看[任务无关蒸馏文档](../general_distill/README.md)了解相关细节。

<a name="在下游任务上使用PP-MiniLM"></a>

## 在下游任务上使用PP-MiniLM

PP-MiniLM预训练小模型在[CLUE benchmark](https://github.com/CLUEbenchmark/CLUE)中的 7 个分类数据集的平均精度上比 12 层 `bert-base-chinese` 高 0.23%，比同等规模的 TinyBERT、UER-py RoBERTa 分别高 2.66%、1.51%，因此我们推荐将 PP-MiniLM 运用在中文下游任务上。当然，如果想对已有模型想要进一步压缩，也可以参考这里的压缩方案，因为压缩方案是通用的。

本案例中会以 CLUE 中 7 个分类数据集为例介绍如何在下游任务上使用 PP-MiniLM。首先用 CLUE 中的数据集对预训练小模型 PP-MiniLM 进行微调，然后提供了一套压缩方案，即借助[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)进行裁剪和量化，进一步对模型规模进行压缩，最终使用基于TensorRT的[Paddle Inference](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/inference_cn.html)预测库对量化后的模型进行预测部署。裁剪、量化前，6 层 PP-MiniLM 的推理速度达`bert-base-chinese`的1.66倍，在下游任务上压缩完成后，模型推理速度高达`bert-base-chinese`的4.15倍。

<a name="数据介绍"></a>

### 数据介绍

本案例中下游任务使用的数据是CLUE benchmark中的7个分类数据集，包括AFQMC、TNEWS、IFKYTEK、OCNLI、CNMLI、CSL、CLUEWSC2020。在Linux环境下，运行`run_clue.py`这个fine-tuning脚本会将该数据集自动下载到`~/.paddlenlp/datasets/Clue/`目录下。

<a name="环境依赖"></a>

### 环境依赖

压缩方案依赖[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)提供的裁剪、量化功能，因此需要安装paddleslim。PaddleSlim是个专注于深度学习模型压缩的工具库，提供剪裁、量化、蒸馏、和模型结构搜索等模型压缩策略，帮助用户快速实现模型的小型化。

```shell
pip install -U paddleslim -i https://pypi.org/simple
```

<a name="在下游任务上微调"></a>

### 在下游任务上微调

基于如下超参范围对 PP-MiniLM 在各个任务上进行Grid Search超参寻优：

- batch sizes: 16, 32, 64
- learning rates: 3e-5, 5e-5, 1e-4

<a name="运行方式"></a>

#### 运行方式

```shell
cd finetuning
mkdir ppminilm-6l-768h
sh run_all_search.sh ppminilm-6l-768h
```

如果只是在单个数据集上用特定`batch_size`、`learning_rate`微调，可以使用如下命令：

```
sh run_clue.sh CLUEWSC2020 1e-4 32 3 128 0 ppminilm-6l-768h
```

其中每个参数依次表示：CLUE中的任务名称、学习率、batch size、epoch数、最大序列长度、gpu id、模型名称（模型保存目录）。

<a name="微调后模型精度"></a>

#### 微调后模型精度

经过超参寻优后，我们可以得到在CLUE每个任务上验证集上有最高准确率的模型，CLUE上各个任务上的最高准确率如下表：

| AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | WSC   | CSL   | CLUE平均值 |
| ----- | ----- | ------- | ----- | ----- | ----- | ----- | ---------- |
| 74.28 | 57.33 | 61.72   | 81.06 | 76.20 | 86.51 | 78.77 | 73.70      |

超参寻优完成后，保存下每个数据集下有最高准确率的模型，以及其对应的超参数，因裁剪、量化等后续步骤需要用到最好的模型和超参数。

<a name="导出微调后模型"></a>

#### 导出微调后模型

如果模型经过了超参寻优，在这一步我们可以在每个任务上选择表现最好的模型进行导出。

假设待导出的模型的地址为`ppminilm-6l-768h/models/CLUEWSC2020/1e-4_32`，可以运行下方命令将动态图模型导出为可用于部署的静态图模型：

```shell
python export_model.py --model_type ernie --model_path ppminilm-6l-768h/models/CLUEWSC2020/1e-4_32  --output_path fine_tuned_infer_model/float
cd ..
```

<a name="对任务上的模型进行裁剪"></a>

### 对任务上的模型进行裁剪

这一步主要使用PaddleSlim中的OFA功能对下游任务上的模型宽度进行裁剪，进一步压缩模型的大小。

该过程会以上一步的模型（即finetuning后得到的最好模型）当作教师模型，蒸馏宽度为3/4的学生模型。经过我们的实验，在 6L768H 条件下，模型宽度压缩为原来的 3/4，精度几乎无损（-0.15)。

<a name="原理简介"></a>

#### 原理简介

基于OFA的裁剪方法参考了[DynaBERT-Dynamic BERT with Adaptive Width and Depth](https://arxiv.org/pdf/2004.04037)中的策略。首先对预训练模型和Head进行重要性排序，保证重要的Head不容易被裁掉，然后用原模型作为蒸馏过程中的教师模型，宽度更小的（本案例是 3/4 宽度）模型作为学生模型，蒸馏得到的学生模型就是我们裁剪得到的模型。

<a name="运行方式"></a>

#### 运行方式

假设需要对上一步fine-tuned模型`../finetuning/ppminilm-6l-768h/models/CLUEWSC2020/1e-4_32`进行裁剪，其中`learning_rate`、`batch_size`可以继续使用fine-tuning时的参数，这里执行的是宽度`0.75`的裁剪，可以使用如下命令运行：

```shell
cd ofa
export FT_MODELS=../finetuning/ppminilm-6l-768h/models/CLUEWSC2020/1e-4_32

sh run_ofa.sh CLUEWSC2020 5e-5 16 50 128 4 ${FT_MODELS} 0.75
```
其中每个参数依次表示：CLUE中的任务名称、学习率、batch size、epoch数、最大序列长度、gpu id、学生模型的地址、裁剪后宽度比例列表。执行完成后，模型保存的路径位于`ofa_models/CLUEWSC2020/0.75/best_model/`。

<a name="裁剪后模型精度"></a>

#### 裁剪后模型精度

经过裁剪后，CLUE上各个任务上的精度如下表所示。相比起裁剪前，CLUE数据集上平均值下降0.15。模型的参数量由59.7M下降到49.1M。

| AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | WSC   | CSL   | CLUE平均值 |
| ----- | ----- | ------- | ----- | ----- | ----- | ----- | ---------- |
| 73.82 | 57.33 | 61.60   | 81.38 | 76.20 | 85.52 | 79.00 | 73.55      |

<a name="导出裁剪后的模型"></a>

#### 导出裁剪后的模型

这一步可以同时导出经过OFA裁剪后特定宽度下模型的动、静态图的模型和参数等文件。

以CLUEWSC2020数据集为例，导出模型：

```shell

export MODEL_PATH=ofa_models
export TASK_NAME=CLUEWSC2020
sh export.sh ${MODEL_PATH} ${TASK_NAME}
```

或者可以批量导出CLUE各个任务上的模型：

```shell

sh export_all.sh
cd ..
```

导出后的模型位于`${MODEL_PATH}/${TASK_NAME}/0.75/sub_static/float`。

<a name="对任务上的模型进行量化"></a>

### 对任务上的模型进行量化

```shell
cd quantization
```

<a name="原理简介"></a>

#### 原理简介

这里的量化采用的是静态离线量化方法，即不需要训练，只使用少量校准数据计算量化因子，就可快速得到量化模型。这一步需要有训练好的预测（静态图）模型。因此，需要对前序步骤产出的模型进行导出（参考上方导出模型的运行方式）。

量化我们可以借助PaddleSlim提供的离线量化API `paddleslim.quant.quant_post_static`实现，我们这一步使用了`mse`、`avg`、`abs_max`、`hist`四种策略，并使用4、8两种校准集数量，对`matmul/matmul_v2`算子进行`'channel_wise_abs_max'`类型的量化。

<a name="运行方式"></a>

#### 运行方式

运行如下脚本可以得到静态离线量化后的模型：

```shell
python quant_post.py --task_name $TASK_NAME --input_dir ${MODEL_DIR}/${TASK_NAME}/0.75/sub_static
```

可以批量对所有数据集下的FP32模型进行量化：

```shell
sh quant_all.sh
cd ..
```

<a name="量化后模型精度"></a>

#### 量化后模型精度

经过量化后，CLUE上各个任务上的精度如下表，比上一步（裁剪后）平均精度下降了0.4：

| AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | WSC   | CSL   | CLUE平均值 |
| ----- | ----- | ------- | ----- | ----- | ----- | ----- | --------- |
| 73.61 | 57.18 | 61.49   | 81.26 | 76.31 | 84.54 | 77.67 | 73.15     |

最后，值得注意的是，PP-MiniLM是基于`roberta-wwm-ext-large`做教师模型蒸馏得到的学生模型，如果你有更好的24层中文预训练模型，可以基于[任务无关蒸馏文档](../general_distill/README.md)中介绍的蒸馏过程，训练出一个比 PP-MiniLM 精度更高，在下游任务上表现更好的 6 层小模型。

<a name="预测"></a>

### 预测

预测部署借助Paddle安装包中自带的[Paddle Inference](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/inference_cn.html)进行预测。

<a name="环境要求"></a>

#### 环境要求

这一步依赖安装有预测库的PaddlePaddle 2.2.1。可以在[PaddlePaddle官网](https://paddleinference.paddlepaddle.org.cn/quick_start/workflow.html)根据机器环境选择合适的Python预测库进行安装。

想要得到更明显的加速效果，推荐在NVIDA Tensor Core GPU（如T4、A10、A100)上进行测试，本案例基于T4测试。若在V系列GPU卡上测试，由于其不支持Int8 Tensor Core，加速效果将达不到本文档表格中的效果。

本案例是在NVIDIA Tesla T4 单卡上，使用cuda11.1、cudnn8.1、TensorRT7.2进行预测。

<a name="运行方式"></a>

#### 运行方式

这里使用了动态shape功能，因此需要设置获取shape的范围。Paddle Inference提供了相应的接口，即首先通过离线输入数据来统计出所有临时tensor的shape范围，TRT子图的tensor输入shape范围可直接根据上一步tune出来的结果来设置，即可完成自动shape范围设置。统计完成后，只需设置统计结果路径，即可启用`tuned_dynamic_shape`功能。在本案例中，只需要先设置`--collect_shape`参数，运行`infer.py`，然后再取消传入这个参数，再次运行`infer.py`。例如：

INT8预测运行脚本：

```shell
cd inference

python infer.py --task_name ${task}  --model_path  ../quantization/${task}_quant_models/${algo}${bs}/int8  --int8 --use_trt --collect_shape # 生成shape range info文件
python infer.py --task_name ${task}  --model_path  ../quantization/${task}_quant_models/${algo}${bs}/int8  --int8 --use_trt # load shape range info文件进行预测
```
如果想要批量对Int8模型进行预测并输出不同量化策略产出模型的精度，可以使用如下的脚本批量预测：

```shell
sh infer_all.sh
```

FP32预测运行脚本：

```shell
python infer.py --task_name ${task}  --model_path  $MODEL_PATH  --use_trt --collect_shape
python infer.py --task_name ${task}  --model_path  $MODEL_PATH --use_trt
```

<a name="性能测试"></a>

#### 性能测试

测试性能环境同上。本案例测试采用的是CLUE TNEWS数据集下量化方法为`mse`、校准集数量为4得到的量化模型，在TNEWS的验证集上统计5次端到端预测的总耗时（前20个steps作为warmup steps跳过）并求平均。下表后三行分别是微调后的模型、OFA裁剪蒸馏后的模型、量化后模型的总耗时情况，加速倍数列是较`bert-base-chinese`的推理加速倍数。

运行性能测试脚本可以得到FP32、裁剪后、量化后模型的耗时：

```shell

sh infer_perf.sh
cd ..
```

取5个非`--collect_shape`阶段打印出的时长取平均，可以发现借助PaddleSlim裁剪、量化后的模型是原BERT<sub>base</sub>模型推理速度的4.15倍，其中裁剪后的模型是BERT<sub>base</sub>推理速度的2.00倍。

|                     | 平均耗时(s) | 加速比  |
| ------------------- | ----------- | -------- |
| BERT<sub>base</sub> | 21.04       | 1.00x    |
| FP32                | 12.64       | 1.66x    |
| FP32+裁剪           | 10.54       | 2.00x    |
| FP32+裁剪+INT8量化  | 5.07        | 4.15x    |

<a name="参考文献"></a>

## 参考文献

1.Wang W, Bao H, Huang S, Dong L, Wei F. MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers[J]. arXiv preprint arXiv:2012.15828v2, 2021.

2.Hou L, Huang Z, Shang L, Jiang X, Chen X and Liu Q. DynaBERT: Dynamic BERT with Adaptive Width and Depth[J]. arXiv preprint arXiv:2004.04037, 2020.

3.Cai H, Gan C, Wang T, Zhang Z, and Han S. Once for all: Train one network and specialize it for efficient deployment[J]. arXiv preprint arXiv:1908.09791, 2020.

4.Wu H, Judd P, Zhang X, Isaev M and Micikevicius P. Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation[J]. arXiv preprint arXiv:2004.09602v1, 2020.
