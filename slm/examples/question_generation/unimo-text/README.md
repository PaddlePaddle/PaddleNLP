# 问题生成


**目录**
- [问题生成](#问题生成)
  - [简介](#简介)
    <!-- - [基于预训练语言模型的问题生成](#基于预训练语言模型的问题生成) -->
  <!-- - [效果展示](#效果展示) -->
  - [开箱即用](#开箱即用)
  - [训练定制](#训练定制)
    - [环境依赖](#环境依赖)
    - [代码结构说明](#代码结构说明)
    - [问题生成应用定制训练全流程介绍](#问题生成定制训练全流程介绍)
    - [数据准备](#数据准备)
      - [数据加载](#数据加载)
      - [数据处理](#数据处理)
      - [从本地文件创建数据集-可选](#从本地文件创建数据集-可选)
    - [模型训练](#模型训练)
    - [模型预测](#模型预测)
    - [模型转换部署](#模型转换部署)
      - [FasterTransformer 加速及模型静态图导出](#fastertransformer 加速及模型静态图导出)
      - [模型部署](#模型部署)
  - [References](#references)

## 简介
Question Generation（QG），即问题生成，指的是给定一段上下文，自动生成一个流畅且符合上下文主题的问句。问题生成通常可以分为，无答案问题生成和有答案问题生成，这里只关注应用更广的有答案问题生成。

问题生成技术在教育、咨询、搜索、推荐等多个领域均有着巨大的应用价值。具体来说，问题生成可广泛应用于问答系统语料库构建，事实性问题生成，教育行业题库生成，对话提问，聊天机器人意图理解，对话式搜索意图提问，闲聊机器人主动提问等等场景。

本项目是基于预训练语言模型 UNIMO-Text 的问题生成，具有以下优势：

- 效果领先。基于百度自研中文预训练语言模型 UNIMO-Text，并提供基于模版策略和大规模多领域问题生成数据集训练的通用问题生成预训练模型`unimo-text-1.0-question-generation`。
- 开箱即用。本项目提供 TaskFlow 接口，无需训练，仅需几行代码便可预测。
- 高性能推理。本项目基于 FasterTransformer 进行推理加速，能够提供更高性能的推理体验，优化后的推理模型在 dureader_qg 开发集的推理耗时缩短为优化前的1/5。
- 训练推理部署全流程打通。本项目提供了全面的定制训练流程，从数据准备、模型训练预测，到模型推理部署，一应俱全。

<!-- ### 基于预训练语言模型的问题生成

基于预训练语言模型（Pretrained Language Models, PLMs）范式的问题生成是目前最常用、效果最好(SOTA)的方式。
预训练模型是在超大规模的语料采用无监督或者弱监督的方式进行预训练，能够学习如何准确地理解自然语言并以自然语言的形式流畅表达，这两项都是完成文本生成任务的重要能力。

PaddleNLP 提供了方便易用的接口，可指定模型名或模型参数文件路径通过 from_pretrained()方法加载不同网络结构的预训练模型，且相应预训练模型权重下载速度快速、稳定。
Transformer 预训练模型汇总包含了如 ERNIE、BERT、T5、UNIMO 等主流预训练模型。下面以中文 unimo-text-1.0模型为例，演示如何加载预训练模型和分词器：
```
from paddlenlp.transformers import  ErnieForGeneration, ErnieTokenizer
model_name = "ernie-1.0"
model = UNIMOLMHeadModel.from_pretrained(model_name)
tokenizer = UNIMOTokenizer.from_pretrained(model_name)
``` -->

## 开箱即用
PaddleNLP 提供开箱即用的产业级 NLP 预置任务能力，无需训练，一键预测。
#### 支持单条、批量预测
```python
>>> from paddlenlp import Taskflow
# 默认模型为 unimo-text-1.0-dureader_qg
>>> question_generator = Taskflow("question_generation")
# 单条输入
>>> question_generator([
  {"context": "奇峰黄山千米以上的山峰有77座，整座黄山就是一座花岗岩的峰林，自古有36大峰，36小峰，最高峰莲花峰、最险峰天都峰和观日出的最佳点光明顶构成黄山的三大主峰。", "answer": "莲花峰"}
  ])
'''
  ['黄山最高峰是什么']
'''
# 多条输入
>>> question_generator([
  {"context": "奇峰黄山千米以上的山峰有77座，整座黄山就是一座花岗岩的峰林，自古有36大峰，36小峰，最高峰莲花峰、最险峰天都峰和观日出的最佳点光明顶构成黄山的三大主峰。", "answer": "莲花峰"},
  {"context": "弗朗索瓦·韦达外文名：franciscusvieta国籍：法国出生地：普瓦图出生日期：1540年逝世日期：1603年12月13日职业：数学家主要成就：为近代数学的发展奠定了基础。", "answer": "法国"}
  ])
'''
  ['黄山最高峰是什么',  '弗朗索瓦是哪里人']
'''
```
关键配置参数说明：
* `model`：可选模型，默认为 unimo-text-1.0-dureader_qg，支持的模型有["unimo-text-1.0", "unimo-text-1.0-dureader_qg", "unimo-text-1.0-question-generation", "unimo-text-1.0-question-generation-dureader_qg"]。

具体参数配置可参考[Taskflow 文档](../../../../docs/model_zoo/taskflow.md)。

## 训练定制

### 环境依赖
- nltk
- evaluate
- tqdm

安装方式：`pip install -r requirements.txt`

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
├── deploy # 部署
│   ├── paddle_inference # PaddleInference高性能推理部署
│   │   ├── inference_unimo_text.py # 推理部署脚本
│   │   └── README.md # 说明文档
│   └── paddle_serving
│       ├── config.yml # 配置文件
│       ├── pipeline_client.py # 客户端程序
│       ├── pipeline_service.py # 服务器程序
│       └── README.md # 说明文档
├── export_model.py # 动态图参数导出静态图参数脚本
├── train.py # 训练脚本
├── predict.py # 预测评估脚本
├── utils.py # 工具函数脚本
└── README.md # 说明文档
```

### 问题生成定制训练全流程介绍
接下来，我们将按数据准备、训练、预测、推理部署等四个阶段对问题生成应用的全流程进行介绍。
1. **数据准备**
- 默认使用中文问题生成数据集 DuReader_QG 进行实验，该数据集已集成到 PaddleNLP。
- 如果已有标注好的本地数据集，我们需要根据将数据集整理为文档要求的格式，请参考[从本地文件创建数据集（可选）](#从本地文件创建数据集（可选）)。

2. **模型训练**

- 数据准备完成后，可以开始使用我们的数据集对预训练模型进行微调训练。我们可以根据任务需求，调整可配置参数，选择使用 GPU 或 CPU 进行模型训练，脚本默认保存在开发集最佳表现模型。中文任务默认使用`unimo-text-1.0`模型，unimo-text-1.0还支持 large 模型。此外本项目还提供基于大规模多领域问题生成数据集训练的通用问题生成预训练模型`unimo-text-1.0-question-generation`，详见[UNIMO 模型汇总](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers/UNIMO/contents.html)，用户可以根据任务和设备需求进行选择。


3. **模型预测**

- 训练结束后，我们可以加载保存的最佳模型进行模型测试，打印模型预测结果。

4. **模型转换部署**
- 在现实部署场景中，我们通常不仅对模型的精度表现有要求，也需要考虑模型性能上的表现。我们可以使用模型裁剪进一步压缩模型体积，问题生成应用已提供裁剪 API 对上一步微调后的模型进行裁剪，模型裁剪之后会默认导出静态图模型。

- 模型部署需要将保存的最佳模型参数（动态图）导出成静态图参数，用于后续的推理部署。

- 问题生成应用提供了基于 Paddle Serving 的本地部署 predictor，并且支持在 GPU 设备使用 Faster Generation 进行加速。

- 问题生成应用提供了基于 Paddle Serving 的服务端部署方案。

### 数据准备
#### 数据加载
[**DuReader_QG**数据集](https://www.luge.ai/#/luge/dataDetail?id=8)是一个中文问题生成数据集，我们使用该数据集作为应用案例进行实验。**DuReader_QG**中的数据主要由由上下文、问题、答案3个主要部分组成，其任务描述为给定上下文 p 和答案 a，生成自然语言表述的问题 q，且该问题符合段落和上下文的限制。

为了方便用户快速测试，PaddleNLP Dataset API 内置了 DuReader_QG 数据集，一键即可完成数据集加载，示例代码如下：

```python
from paddlenlp.datasets import load_dataset
train_ds, dev_ds = load_dataset('dureader_qg', splits=('train', 'dev'))
```

#### 数据处理
针对**DuReader_QG**数据集，我们需要将 QA 任务格式的数据进行转换从而得到 text2text 形式的数据，我们默认使用模版的方式构造输入数据，默认模版如下，其他形式输入数据用户可以在 convert_example 函数中自行定义。
```text
答案: <answer_text> 上下文: <context_text>
问题: <question_text>
```

#### 从本地文件创建数据集-可选
在许多情况下，我们需要使用本地数据集来训练我们的问题生成模型，本项目支持使用固定格式本地数据集文件进行训练。
使用本地文件，只需要在模型训练时指定`train_file` 为本地训练数据地址，`predict_file` 为本地测试数据地址即可。

本地数据集目录结构如下：

```text
data/
├── train.json # 训练数据集文件
├── dev.json # 开发数据集文件
└── test.json # 可选，待预测数据文件
```
本地数据集文件格式如下：
- train.json/dev.json/test.json 文件格式：
```text
{
  "context": <context_text>,
  "answer": <answer_text>,
  "question": <question_text>,
}
...
```
- train.json/dev.json/test.json 文件样例：
```text
{
  "context": "欠条是永久有效的,未约定还款期限的借款合同纠纷,诉讼时效自债权人主张债权之日起计算,时效为2年。 根据《中华人民共和国民法通则》第一百三十五条:向人民法院请求保护民事权利的诉讼时效期间为二年,法律另有规定的除外。 第一百三十七条:诉讼时效期间从知道或者应当知道权利被侵害时起计算。但是,从权利被侵害之日起超过二十年的,人民法院不予保护。有特殊情况的,人民法院可以延长诉讼时效期间。 第六十二条第(四)项:履行期限不明确的,债务人可以随时履行,债权人也可以随时要求履行,但应当给对方必要的准备时间。",
  "answer": "永久有效",
  "question": "欠条的有效期是多久"
}
...
```

更多数据集读取格式详见[数据集加载](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_load.html#)和[自定义数据集](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html)。

### 模型训练
运行如下命令即可在样例训练集上进行 finetune，并在样例验证集上进行验证。
```shell
# GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
# 例如使用1号和2号卡，则：`--gpu 1,2`
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "1,2" --log_dir ./unimo/finetune/log train.py \
    --dataset_name=dureader_qg \
    --model_name_or_path="unimo-text-1.0" \
    --save_dir=./unimo/finetune/checkpoints \
    --output_path ./unimo/finetune/predict.txt \
    --logging_steps=100 \
    --save_steps=500 \
    --epochs=20 \
    --batch_size=16 \
    --learning_rate=1e-5 \
    --warmup_proportion=0.02 \
    --weight_decay=0.01 \
    --max_seq_len=512 \
    --max_target_len=30 \
    --do_train \
    --do_predict \
    --max_dec_len=20 \
    --min_dec_len=3 \
    --num_return_sequences=1 \
    --template=1 \
    --device=gpu
```


关键参数释义如下：
- `gpus` 指示了训练所用的 GPU，使用多卡训练可以指定多个 GPU 卡号，例如 --gpus "0,1"。
- `dataset_name` 数据集名称，当`train_file`和`predict_file`为 None 时将加载`dataset_name`的训练集和开发集，默认为`dureader_qg`。
- `train_file` 本地训练数据地址，数据格式必须与`dataset_name`所指数据集格式相同，默认为 None。
- `predict_file` 本地测试数据地址，数据格式必须与`dataset_name`所指数据集格式相同，默认为 None。
- `model_name_or_path` 指示了 finetune 使用的具体预训练模型，可以是 PaddleNLP 提供的预训练模型，或者是本地的预训练模型。如果使用本地的预训练模型，可以配置本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含 paddle 预训练模型 model_state.pdparams。如果使用 PaddleNLP 提供的预训练模型，可以选择下面其中之一。
   | 可选预训练模型        |
   |---------------------------------|
   | unimo-text-1.0      |
   | unimo-text-1.0-large |
   | unimo-text-1.0-question-generation |

   <!-- | T5-PEGASUS |
   | ernie-1.0 |
   | ernie-gen-base-en |
   | ernie-gen-large-en |
   | ernie-gen-large-en-430g | -->

- `save_dir` 表示模型的保存路径。
- `output_path` 表示预测结果的保存路径。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `seed` 表示随机数生成器的种子。
- `epochs` 表示训练轮数。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于 learning rate scheduler 产生的值相乘作为当前学习率。
- `weight_decay` 表示 AdamW 优化器中使用的 weight_decay 的系数。
- `warmup_proportion` 表示学习率逐渐升高到基础学习率（即上面配置的 learning_rate）所需要的迭代数占总步数的比例。
- `max_seq_len` 模型输入序列的最大长度。
- `max_target_len` 模型训练时标签的最大长度。
- `min_dec_len` 模型生成序列的最小长度。
- `max_dec_len` 模型生成序列的最大长度。
- `do_train` 是否进行训练。
- `do_predict` 是否进行预测，在验证集上会自动评估。
- `device` 表示使用的设备，从 gpu 和 cpu 中选择。
- `template` 表示使用的模版，从[0, 1, 2, 3, 4]中选择，0表示不选择模版，1表示使用默认模版。

程序运行时将会自动进行训练和验证，训练过程中会自动保存模型在指定的`save_dir`中。如：

```text
./unimo/finetune/checkpoints
├── model_1000
│   ├── model_config.json
│   ├── model_state.pdparams
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
└── ...
```

**NOTE:** 如需恢复模型训练，`model_name_or_path`配置本地模型的目录地址即可。

微调的模型在 dureader_qg 验证集上有如下结果(指标为 BLEU-4)，其中`unimo-text-1.0-dureader_qg-w/o-template`表示不使用模版策略微调的结果，`unimo-text-1.0-large-dureader_qg`表示使用 large 模型微调的结果，`unimo-text-1.0-question-generation-dureader_qg`表示在通用问题生成预训练模型`unimo-text-1.0-question-generation`上微调的结果：

|       model_name        | DuReaderQG |
| :-----------------------------: | :-----------: |
|    unimo-text-1.0-dureader_qg-w/o-template    | 39.61 |
|    unimo-text-1.0-dureader_qg    | 41.08 |
|    unimo-text-1.0-large-dureader_qg    | 41.51 |
|    unimo-text-1.0-question-generation-dureader_qg    | 44.02 |

### 模型预测

运行下方脚本可以使用训练好的模型进行预测。

```shell
export CUDA_VISIBLE_DEVICES=0
python -u predict.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=your_model_path \
    --output_path=./predict.txt \
    --logging_steps=100 \
    --batch_size=16 \
    --max_seq_len=512 \
    --max_target_len=30 \
    --do_predict \
    --max_dec_len=20 \
    --min_dec_len=3 \
    --template=1 \
    --device=gpu
```
关键参数释义如下：
- `output_path` 表示预测输出结果保存的文件路径，默认为./predict.txt。
- `model_name_or_path` 指示了 finetune 使用的具体预训练模型，可以是 PaddleNLP 提供的预训练模型，或者是本地的微调好的预训练模型。如果使用本地的预训练模型，可以配置本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含 paddle 预训练模型 model_state.pdparams。

### 模型转换部署

#### FasterTransformer 加速及模型静态图导出

使用动态图训练结束之后，可以通过[静态图导出脚本](export_model.py)实现基于 FasterTransformer 的高性能预测加速，并将动态图参数导出成静态图参数，静态图参数保存在`output_path`指定路径中。运行方式：

```shell
python export_model.py \
    --model_name_or_path ./checkpoint \
    --inference_model_dir ./export_checkpoint \
    --max_dec_len 50 \
    --use_fp16_decoding
```
关键参数释义如下：

* `model_name_or_path`：动态图训练保存的参数路径；默认为"./checkpoint"。
* `inference_model_dir`：静态图图保存的参数路径；默认为"./export_checkpoint"。
* `max_dec_len`：最大输出长度。
* `use_fp16_decoding`:是否使用 fp16解码进行预测。

执行命令后将会自动导出模型到指定的 `inference_model_dir` 中，保存模型文件结构如下所示：

```text
├── unimo_text.pdiparams
├── unimo_text.pdiparams.info
└── unimo_text.pdmodel
```

#### 模型部署
本项目提供多种不同场景的部署方案，请根据实际情况进行选择：
|部署方案|特色|场景|硬件|
|-|-|-|-|
|Paddle Inference<br>服务端／云端|通用性|模型算法复杂<br>硬件高性能|X86 CPU<br>NVIDIA 全系列 GPU<br>龙芯／飞腾等国产 CPU<br>昆仑／昇腾／海光 DCU 等 AI 加速芯片
|Paddle Serving<br>服务化|高并发|大流量、高并发、低延时、高吞吐<br>资源弹性调控应对服务流量变化<br>支持模型组合、加密、热更新等|X86/Arm CPU<br>NVIDIA GPU<br>昆仑／昇腾等


问题生成应用已打通多种场景部署方案，点击链接获取具体的使用教程。
- [Paddle Inference 推理 (Python)](./deploy/paddle_inference/README.md)
- [Paddle Serving 服务化部署（Python）](./deploy/paddle_serving/README.md)


## References
Zheng, Chujie, and Minlie Huang. "Exploring prompt-based few-shot learning for grounded dialog generation." arXiv preprint arXiv:2109.06513 (2021).
Li, Wei, et al. "Unimo: Towards unified-modal understanding and generation via cross-modal contrastive learning." arXiv preprint arXiv:2012.15409 (2020).
