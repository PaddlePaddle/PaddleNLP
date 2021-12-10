 **目录**

* [PP-MiniLM中文小模型](#PP-MiniLM中文小模型)
    * [快速开始](#快速开始)
        * [一、任务无关知识蒸馏（可选）](#一、任务无关知识蒸馏（可选）)
            * [原理介绍](#原理介绍)
            * [数据介绍](#数据介绍)
            * [运行方式](#运行方式)
        * [二、在下游任务上微调](#二、在下游任务上微调)
            * [数据介绍](#数据介绍)
            * [运行方式](#运行方式)
            * [微调后模型精度](#微调后模型精度)
            * [导出微调后模型](#导出微调后模型)
        * [三、对任务上的模型进行裁剪](#三、对任务上的模型进行裁剪)
            * [原理介绍](#原理简介)
            * [环境依赖](#环境依赖)
            * [数据介绍](#数据介绍)
            * [运行方式](#运行方式)
            * [导出裁剪后的模型](#导出裁剪后的模型)
            * [裁剪后模型精度](#裁剪后模型精度)
        * [四、对任务上的模型进行量化](#四、对任务上的模型进行量化)
            * [原理简介](#原理简介)
            * [环境依赖](#环境依赖)
            * [运行方式](#运行方式)
            * [量化后模型精度](#量化后模型精度)
        * [五、预测](#五、预测)
            * [环境要求](#环境要求)
            * [运行方式](#运行方式)
            * [性能测试](#性能测试)
    * [参考文献](#参考文献)

<a name="PP-MiniLM中文小模型"></a>

# PP-MiniLM中文小模型

PP-MiniLM 中文特小模型案例旨在提供训推一体的高精度、高性能小模型及解决方案。

当前解决方案依托业界领先的 Task Agnostic 模型蒸馏技术、裁剪技术、量化技术，使得小模型兼具推理速度快、模型效果好、参数规模小的 3 大特点。

- 推理速度快: 依托 PaddleSlim 的裁剪、量化技术进一步小模型进行压缩, 使得 PP-MiniLM 量化模型 GPU 推理速度相比 Bert base 加速比高达 4.15；

- 精度高: 我们以 MiniLMv2 提出的 Multi-Head Self-Attention Relation Distillation 技术为基础，通过引入样本间关系知识蒸馏做了进一步算法优化, 6 层 PP-MiniLM 模型在 CLUE 数据集上比 12 层 Bert-base-chinese 高 0.23%，比同等规模的 TinyBERT、UER-py RoBERTa 分别高 2.66%、1.51%；

- 参数规模小：依托 PaddleSlim 裁剪技术，在精度几乎无损(-0.15%)条件下将模型隐层宽度压缩 1/4，模型参数量减少 28%。

**整体效果**

| Model                   | #Params | #FLOPs | Speedup | AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | WSC   | CSL   | CLUE平均值 |
| ----------------------- | ------- | ------ | ------- | ----- | ----- | ------- | ----- | ----- | ----- | ----- | ---------- |
| Bert<sub>base</sub>     | 102.3M  | 10.87B | 1.00x   | 74.17 | 57.17 | 61.14   | 81.14 | 75.08 | 80.26 | 81.47 | 72.92      |
| TinyBERT<sub>6</sub>    | 59.7M   | 5.44B  | 1.66x   | 72.22 | 55.82 | 58.10   | 79.53 | 74.00 | 75.99 | 80.57 | 70.89      |
| UER-py RoBERTa L6- H768 | 59.7M   | 5.44B  | 1.66x   | 69.74 | 66.36 | 59.95   | 77.00 | 71.39 | 71.05 | 82.83 | 71.19      |
| RBT6, Chinese           | 59.7M   | 5.44B  | 1.66x   | 73.96 | 56.37 | 59.72   | 79.37 | 73.05 | 76.97 | 80.80 | 71.46      |
| ERNIE-Tiny              | 90.7M   | 4.83B  | 1.76x   | 70.78 | 55.70 | 59.95   | 75.40 | 70.98 | 67.43 | 76.60 | 68.12      |
| PP-MiniLM 6L-768H       | 59.7M   | 5.44B  | 1.66x   | 74.28 | 57.33 | 61.72   | 81.06 | 76.2  | 86.51 | 78.77 | 73.70      |
| PP-MiniLM裁剪后         | 49.1M   | 4.08B  | 2.00x   | 73.82 | 57.33 | 61.60   | 81.38 | 76.20 | 85.52 | 79.00 | 73.55      |
| PP-MiniLM量化后         | 49.2M   | 4.08B  | 4.15x   | 73.61 | 57.18 | 61.49   | 81.26 | 76.31 | 84.54 | 77.67 | 73.15      |


**NOTE** 量化后的模型比量化前参数量多了0.1M是因为保存了scale值。

**方案流程**

<p align="center">
<img src="./pp-minilm.png" width="950"/><br />
整体流程图
</p>

<a name="快速开始"></a>

## 快速开始
以下是本范例模型的简要目录结构及说明：

```shell
.
├── general_distill              # 任务无关知识蒸馏目录
│ └── general_distill.py         # 任务无关知识蒸馏脚本
│ └── run.sh                     # 任务无关知识蒸馏启动脚本
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
完整的中文小模型方案分为：任务无关知识蒸馏（可选）、下游任务微调、OFA裁剪、离线量化、预测。下面会对这里的每一个步骤进行介绍。我们建议保留下面的每一个步骤，但如果受限于机器资源等原因希望简化步骤，除了下游任务微调步骤，其余步骤均可以省略。

<a name="一、任务无关知识蒸馏"></a>

### 一、任务无关知识蒸馏（可选）

<a name=""></a>
#### 环境说明

本实验基于NVIDIA Tesla V100 32G 8卡进行，训练周期约为2-3天。若资源有限，可以直接[下载PP-MiniLM(6L768H)](https://bj.bcebos.com/paddlenlp/models/transformers/ppminilm/6l-768h)用于下游任务的微调。

<a name="原理介绍"></a>

#### 原理介绍

任务无关知识蒸馏是用较大（层数更多、宽度更宽的）的基于 Transformer Layer的预训练模型对较小（层数更少、宽度更窄的）的基于Transformer Layer 的预训练模型进行蒸馏，从而得到更小、效果与较大模型更接近的预训练模型。

PP-MiniLM参考了 MiniLMv2 提出的 Multi-Head Self-Attention Relation Distillation 蒸馏策略。MiniLMv2 算法是用24层large-size的教师模型倒数几层的 Q-Q、K-K、V-V 之间的relation对6层学生模型最后一层 Q-Q、K-K、V-V 之间的relation进行蒸馏。具体的做法是，首先将学生、教师用于蒸馏的层上的Q、K、V的head数进行统一，然后计算各自Q—Q、K-K、V-V的点积，最后对教师和学生的点积计算KL散度损失。由于relation的shape是`[batch_size, head_num, seq_len, seq_len]`，因此可以认为这里的relation是一种Token与Token之间的关系。

本方案在MiniLMv2策略的基础上，做了进一步优化: 通过引入多视角的注意力关系知识来进一步提升模型效果。MiniLMv2 的自注意力关系知识仅建模了 Token 与 Token 之间的关系，PP-MiniLM 在此基础上额外引入了样本与样本间的自注意力关系知识，也就是挖掘出更多教师模型所蕴含的知识，从而进一步优化模型效果。

具体来说，PP-MiniLM 利用了`roberta-wwm-ext-large`第 20 层的 Q-Q、K-K、V-V 之间的Sample与Sample之间关系对 6 层学生模型 PP-MiniLM 第 6 层的 Q-Q、K-K、V-V 之间的Sample与Sample之间的关系进行蒸馏。与MiniLMv2不同的是，PP-MiniLM的策略需要在统一Q、K、V的Head数之后，对Q、K、V转置为`[seq_len, head_num, batch_size, head_dim]`，这样Q—Q、K- K、V-V 的点积则可以表达样本间的关系。经过我们的实验，这种方法比使用原始 MiniLMv2 算法在 CLUE 上平均准确率高0.36。

<a name="数据介绍"></a>

#### 数据介绍

任务无关知识蒸馏的训练数据一般是预训练语料，可以使用公开的预训练语料[CLUECorpus2020](https://github.com/CLUEbenchmark/CLUECorpus2020/)。需要将数据处理成一行一个句子的格式，再将数据文件分割成多个子文件（例如64个），放在同一个目录下。

<a name="运行方式"></a>

#### 运行方式

```shell
cd general_distill
sh run.sh # 包含general_distill.py的运行配置
cd ..
```

其中general_distill.py参数释义如下：

- `model_type` 指示了学生模型类型，当前仅支持'ernie'、'roberta'。
- `num_relation_heads` relation heads的个数，一般对于large size的教师模型是64，对于base size的教师模型是48。
- `teacher_model_type`指示了教师模型类型，当前仅支持'ernie'、'roberta'。
- `teacher_layer_index`蒸馏时使用的教师模型的层
- `student_layer_index` 蒸馏时使用的学生模型的层
- `teacher_model_name_or_path`教师模型的名称，例如`'roberta-wwm-ext-large'`
- `max_seq_length` 最大的样本长度
- `num_layers` 学生模型的层数，目前仅支持2，4，6
- `logging_steps` 日志间隔
- `max_steps` 最大迭代次数
- `warmup_steps` 学习率增长得到`learning_rate`所需要的步数
- `save_steps`保存模型的间隔步数
- `weight_decay` 表示AdamW优化器中使用的weight_decay的系数。
- `output_dir`训练相关文件以及模型保存的输出路径
- `device`设备选择，推荐使用gpu
- `input_dir` 训练数据目录
- `use_amp` 是否使用混合精度训练，默认False
- `alpha`head间关系的权重，默认0.0
- `beta`样本间关系的权重，默认0.0

将最终得到的模型绝对路径保存至`$GENERAL_MODEL_DIR`，例如：

```shell
GENERAL_MODEL_DIR=PaddleNLP/examples/model_compression/PP-MiniLM/general_distill/pretrain/model_400000
```

<a name="二、在下游任务上微调"></a>

### 二、在下游任务上微调

<a name="数据介绍"></a>

#### 数据介绍

本案例中下游任务使用的数据是[CLUE benchmark](https://github.com/CLUEbenchmark/CLUE)，这是一个中文任务测评基准。在Linux环境下，运行run_clue.py这个fine-tuning脚本会将该数据集自动下载到`~/.paddlenlp/datasets/Clue/`目录下。

基于如下超参范围对第一步蒸馏产出的小模型`GENERAL_MODEL_DIR`进行Grid Search超参寻优：

- batch sizes: 16, 32, 64
- learning rates: 3e-5, 5e-5, 1e-4

<a name="运行方式"></a>

#### 运行方式

```shell
cd finetuning
sh run_all_search.sh $GENERAL_MODEL_DIR
```

如果只是单个数据集上特定`batch_size`、`learning_rate`的微调，可以使用如下命令：

```
sh run_clue.sh CLUEWSC2020 1e-4 32 3 128 0 $GENERAL_MODEL_DIR
```

其中每个参数依次表示：CLUE中的任务名称、学习率、batch size、epoch数、最大序列长度、gpu id。

<a name="微调后模型精度"></a>

#### 微调后模型精度

经过超参寻优后，我们可以得到在CLUE每个任务上的有最高准确率的模型，CLUE上各个任务上的最高准确率如下表：

| AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | WSC   | CSL   | CLUE平均值 |
| ----- | ----- | ------- | ----- | ----- | ----- | ----- | ---------- |
| 74.28 | 57.33 | 61.72   | 81.06 | 76.20 | 86.51 | 78.77 | 73.70      |

<a name="导出微调后模型"></a>

#### 导出微调后模型

如果模型经过了超参寻优，在这一步我们可以在每个任务上选择表现最好的模型进行导出。
假设待导出的模型的地址为`$GENERAL_MODEL_DIR/models/CLUEWSC2020/1e-4_32`，可以运行下方命令对动态图模型导出为可用于部署的静态图模型：

```shell
python export_model.py --model_type ernie --model_path $GENERAL_MODEL_DIR/models/CLUEWSC2020/1e-4_32  --output_path fine_tuned_infer_model/float
cd ..
```

<a name="三、对任务上的模型进行裁剪"></a>

### 三、对任务上的模型进行裁剪

这一步主要使用[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)中的OFA功能对下游任务上的模型宽度进行裁剪，进一步压缩模型的大小。

该过程会以上一步的模型（即finetuning后得到的最好模型）当作教师模型，蒸馏宽度为3/4的学生模型。经过我们的实验，在6L768H 条件下，模型宽度压缩为原来的 3/4, 精度几乎无损（-0.15)。

<a name="原理简介"></a>

#### 原理简介

基于OFA的裁剪、蒸馏方法参考了[DynaBERT-Dynamic BERT with Adaptive Width and Depth](https://arxiv.org/pdf/2004.04037)中的蒸馏策略。首先对预训练模型和head进行重要性排序，保证重要的head不容易被裁掉，然后用原模型作为蒸馏过程中的教师模型，宽度更小的（本案例是3/4宽度）模型作为学生模型，蒸馏得到的学生模型就是我们裁剪得到的模型。

<a name="环境依赖"></a>

#### 环境依赖

执行这部分内容需要安装paddleslim的最新包：

```shell
pip install -U paddleslim -i https://pypi.org/simple
cd ofa
```

<a name="数据介绍"></a>

#### 数据介绍

同上步fine-tuning，同样基于CLUE数据集。

<a name="运行方式"></a>

#### 运行方式

假设需要对上一步fine-tuned模型`$GENERAL_MODEL_DIR/models/CLUEWSC2020/1e-4_32`进行进一步的裁剪，其中`learning_rate`、`batch_size`可以继续使用fine-tuning时的参数，这里执行的是宽度`0.75`的裁剪，可以使用如下命令运行：

```shell

export FT_MODELS=$GENERAL_MODEL_DIR/models/CLUEWSC2020/1e-4_32

sh run_ofa.sh CLUEWSC2020 5e-5 16 50 128 4 ${FT_MODELS} 0.75
```
其中每个参数依次表示：CLUE中的任务名称、学习率、batch size、epoch数、最大序列长度、gpu id、学生模型的地址、裁剪后宽度列表。执行完成后，模型保存的路径位于`ofa_models/CLUEWSC2020/0.75/best_model/`。

<a name="导出裁剪后的模型"></a>

#### 导出裁剪后的模型

这一步可以同时导出经过OFA裁剪后特定宽度下模型的动态图、静态图的模型和参数等文件。

以CLUEWSC2020数据集为例，导出模型：

```shell

export MODEL_PATH=ofa_models
export TASK_NAME=CLUEWSC2020
sh export.sh ${MODEL_PATH} ${TASK_NAME}
```

或者可以批量导出各个任务上的模型：

```shell

sh export_all.sh
```

最终模型保存的位置位于` ${MODEL_PATH}/${TASK_NAME}/0.75/sub_static/float`。

```shell
cd ..
```

<a name="裁剪后模型精度"></a>

#### 裁剪后模型精度

经过裁剪、蒸馏后，CLUE上各个任务上的精度如下表所示。相比起裁剪前，CLUE数据集上平均值下降0.15。模型的参数量由59.7M到49.1M。

| AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | WSC   | CSL   | CLUE平均值 |
| ----- | ----- | ------- | ----- | ----- | ----- | ----- | ---------- |
| 73.82 | 57.33 | 61.60   | 81.38 | 76.20 | 85.52 | 79.00 | 73.55      |

<a name="四、对任务上的模型进行量化"></a>

### 四、对任务上的模型进行量化

```shell
cd quantization
```

<a name="原理简介"></a>

#### 原理简介

这里的量化采用的是静态离线量化方法，即不需要训练，只使用少量校准数据计算量化因子，就可快速得到量化模型。这一步需要有训练好的预测（静态图）模型。因此，需要对前序步骤产出的模型进行导出（参考上方运行方式）。

量化我们可以借助PaddleSlim提供的离线量化API `paddleslim.quant.quant_post_static`实现，我们这一步使用了`mse`、`avg`、`abs_max`、`hist`多种策略，并使用4、8两种量化时的校准集数量，对`matmul/matmul_v2` 算子进行`'channel_wise_abs_max'`类型的量化。

<a name="环境依赖"></a>

#### 环境依赖

执行离线量化的内容需要安装paddleslim的最新包：

```shell
pip install -U paddleslim -i https://pypi.org/simple
cd ofa
```

<a name="运行方式"></a>

#### 运行方式

运行如下的脚本可以得到静态离线量化后的模型：

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

再经过量化后，CLUE上各个任务上的精度如下表，比上一步（裁剪后）下降了0.4：

| AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | WSC   | CSL   | CLUE平均值 |
| ----- | ----- | ------- | ----- | ----- | ----- | ----- | ---------- |
| 73.61 | 57.18 | 61.49   | 81.26 | 76.31 | 84.54 | 77.67 | 73.15      |


<a name="五、预测"></a>

### 五、预测

预测部署借助Paddle安装包中自带的[Paddle Inference](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/inference_cn.html)进行预测。

<a name=环境要求"></a>

#### 环境要求

这一步需要依赖安装有预测库的paddle2.2.1。可以在[PaddlePaddle官网](https://paddleinference.paddlepaddle.org.cn/quick_start/workflow.html)根据机器环境选择合适的Python预测库进行安装。

想要得到更明显的加速效果，推荐在NVIDA Tensor Core GPU（如T4、A10、A100)上进行测试，本案例基于T4测试。若在V系列卡上测试，由于其不支持Int8 Tensor Core，加速效果将达不到本文档表格中的效果。

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

测试性能环境同上。本案例测试采用的是CLUE TNEWS数据集下得到的量化模型，在TNEWS的验证集上计算端到端预测的总耗时（前20个steps作为warmup steps跳过）。下表后三行分别是微调后的模型、OFA裁剪蒸馏后的模型、量化方法为`mse`、校准集数量为4的量化模型的总耗时情况，后面一列是较`bert-base-chinese`的推理加速倍数。

运行性能测试脚本可以得到FP32、裁剪后、量化后模型的耗时

```shell

sh infer_perf.sh
cd ..
```

取5个非--collect_shap阶段打印出的时长取平均，可以发现借助PaddleSlim裁剪、量化后的模型比原BERT<sub>base</sub>模型推理速度4.15倍，其中裁剪后比起Bert<sub>base</sub>可以加速2.00倍。

|                     | 平均耗时(s) | 加速倍数 |
| ------------------- | ----------- | -------- |
| Bert<sub>base</sub> | 21.04       | -        |
| FP32                | 12.64       | 1.66x    |
| FP32+裁剪           | 10.54       | 2.00x    |
| FP32+裁剪+INT8量化  | 5.07        | 4.15x    |

<a name="参考文献"></a>

## 参考文献

1.Wang W, Bao H, Huang S, Dong L, Wei F. MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers[J]. arXiv preprint arXiv:2012.15828v2, 2021.

2.Hou L, Huang Z, Shang L, Jiang X, Chen X and Liu Q. DynaBERT: Dynamic BERT with Adaptive Width and Depth[J]. arXiv preprint arXiv:2004.04037, 2020.

3.Cai H, Gan C, Wang T, Zhang Z, and Han S. Once for all: Train one network and specialize it for efficient deployment[J]. arXiv preprint arXiv:1908.09791, 2020.

4.Wu H, Judd P, Zhang X, Isaev M and Micikevicius P. Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation[J]. arXiv preprint arXiv:2004.09602v1, 2020.
