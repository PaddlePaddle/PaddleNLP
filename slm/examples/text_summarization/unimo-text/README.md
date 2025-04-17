# 生成式文本摘要应用

**目录**
- [生成式文本摘要应用](#生成式文本摘要应用)
  - [简介](#简介)
    - [基于预训练语言模型的文本摘要](#基于预训练语言模型的文本摘要)
  - [效果展示](#效果展示)
  - [开箱即用](#开箱即用)
    - [支持单条、批量预测](#支持单条批量预测)
    - [可配置参数说明](#可配置参数说明)
  - [训练定制](#训练定制)
    - [文本摘要应用定制训练全流程介绍](#文本摘要应用定制训练全流程介绍)
    - [环境依赖](#环境依赖)
    - [代码结构说明](#代码结构说明)
    - [数据准备](#数据准备)
      - [数据加载](#数据加载)
      - [从本地文件创建数据集](#从本地文件创建数据集)
    - [模型训练](#模型训练)
    - [模型预测](#模型预测)
    - [模型推理部署](#模型推理部署)
      - [FastGeneration 加速及模型静态图导出](#fastgeneration 加速及模型静态图导出)
      - [模型部署](#模型部署)
  - [References](#references)


## 简介
文本摘要的目标是自动地将输入文本转换成简短摘要,为用户提供简明扼要的内容描述，是缓解文本信息过载的一个重要手段。
文本摘要也是自然语言生成领域中的一个重要任务，有很多应用场景，如新闻摘要、论文摘要、财报摘要、传记摘要、专利摘要、对话摘要、评论摘要、观点摘要、电影摘要、文章标题生成、商品名生成、自动报告生成、搜索结果预览等。

本项目是基于预训练语言模型 UNIMO-Text 的文本摘要，具有以下优势：
- 效果领先。
- 开箱即用。本项目提供 TaskFlow 接口，无需训练，仅需几行代码便可预测。
- 训练推理全流程打通。本项目提供了全面的定制训练流程，从数据准备、模型训练预测，到模型推理部署，一应俱全。

### 基于预训练语言模型的文本摘要

基于预训练语言模型（Pretrained Language Models, PLMs）范式的自动文本摘要是目前最常用、效果最好(SOTA)的方式。
预训练模型是在超大规模的语料采用无监督（unsupervised）或者弱监督（weak-supervised）的方式进行预训练，能够学习如何准确地理解自然语言并以自然语言的形式流畅表达，这两项都是完成文本摘要任务的重要能力。

PaddleNLP 提供了方便易用的接口，可指定模型名或模型参数文件路径通过 from_pretrained()方法加载不同网络结构的预训练模型，且相应预训练模型权重下载速度快速、稳定。下面以中文 unimo-text-1.0-summary 模型为例，演示如何加载预训练模型和分词器：
```
from paddlenlp.transformers import  UNIMOLMHeadModel, UNIMOTokenizer
model_name = "unimo-text-1.0-summary"
model = UNIMOLMHeadModel.from_pretrained(model_name)
tokenizer = UNIMOTokenizer.from_pretrained(model_name)
```

## 效果展示

## 开箱即用
PaddleNLP 提供开箱即用的产业级 NLP 预置任务能力，无需训练，一键预测。
### 支持单条、批量预测

```python
>>> from paddlenlp import Taskflow
>>> summarizer = Taskflow("text_summarization")
# 单条输入
>>> summarizer("雪后的景色可真美丽呀！不管是大树上，屋顶上，还是菜地上，都穿上了一件精美的、洁白的羽绒服。放眼望去，整个世界变成了银装素裹似的，世界就像是粉妆玉砌的一样。")
# 输出：'雪后的景色可真美丽呀！'

# 多条输入
>>> summarizer([
  "雪后的景色可真美丽呀！不管是大树上，屋顶上，还是菜地上，都穿上了一件精美的、洁白的羽绒服。放眼望去，整个世界变成了银装素裹似的，世界就像是粉妆玉砌的一样。",
  "根据“十个工作日”原则，下轮调价窗口为8月23日24时。卓创资讯分析，原油价格或延续震荡偏弱走势，且新周期的原油变化率仍将负值开局，消息面对国内成品油市场并无提振。受此影响，预计国内成品油批发价格或整体呈现稳中下滑走势，但“金九银十”即将到来，卖方看好后期市场，预计跌幅较为有限。"
  ])
#输出：['雪后的景色可真美丽呀！', '成品油调价窗口8月23日24时开启']
```

### 可配置参数说明
* `model`：可选模型，默认为`unimo-text-1.0-summary`。
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。


## 训练定制
### 文本摘要应用定制训练全流程介绍
接下来，我们将按数据准备、训练、预测、推理部署对文本摘要应用的全流程进行介绍。
1. **数据准备**
- 如果没有已标注的数据集，我们推荐[doccano](https://github.com/doccano/doccano)数据标注工具。
如果已有标注好的本地数据集，我们需要根据将数据集整理为文档要求的格式，请参考[从本地文件创建数据集](#从本地文件创建数据集)。

2. **模型训练**

- 数据准备完成后，可以开始使用我们的数据集对预训练模型进行微调训练。我们可以根据任务需求，调整可配置参数，选择使用 GPU 或 CPU 进行模型训练，脚本默认保存在开发集最佳表现模型。中文任务默认使用"unimo-text-1.0-summary"模型，unimo-text-1.0-summary 还支持 large 模型，详见[UNIMO 模型汇总](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers/UNIMO/contents.html)，可以根据任务和设备需求进行选择。


3. **模型预测**

- 训练结束后，我们可以加载保存的最佳模型进行模型测试，打印模型预测结果。

4. **模型推理部署**

- 模型部署需要将保存的最佳模型参数（动态图）导出成静态图参数，用于后续的推理部署。

- 文本摘要应用提供了基于 Paddle Inference 的本地部署 predictor，并且支持在 GPU 设备使用 FastGeneration 进行加速。

- 文本摘要应用提供了基于 Paddle Serving 的服务端部署方案。

### 环境依赖

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
text_summarization/
├── deploy # 部署
│   ├── paddle_inference # PaddleInference高性能推理部署
│   │   ├── inference_unimo_text.py # 推理部署脚本
│   │   └── README.md # 说明文档
│   └── paddle_serving
│       ├── config.yml # 配置文件
│       ├── pipeline_client.py # 客户端程序
│       ├── pipeline_service.py # 服务器程序
│       ├── export_serving.sh # serving模型导出脚本
│       └── README.md # 说明文档
├── export_model.py # 动态图参数导出静态图参数脚本
├── export_model.sh # 动态图参数导出静态图参数shell脚本
├── train.py # 训练评估脚本
├── train.sh # 训练评估shell脚本
├── utils.py # 工具函数脚本
└── README.md # 说明文档
```

### 数据准备

#### 数据加载
#### 从本地文件创建数据集

在许多情况，我们需要使用本地数据集来训练我们的文本摘要模型，本项目支持使用固定格式本地数据集文件进行训练。

本地数据集目录结构如下：

```text
data/
├── train.json # 训练数据集文件
└── test.json # 可选，待预测数据文件
```
本地数据集文件格式如下：
- train.json/test.json 文件每行格式：
```text
{
"title": "任志强抨击政府把土地作为投机品地产业被人为破坏",
"content": "“北京的保障房市场就像一个巨大的赌场，每个人都在期待中奖。”面对中国目前现行的保障性住房政策，华远地产董事长任志强再次语出惊人。（分享自@第一财经-中国房地产金融）"
}
```

更多数据集读取格式详见[数据集加载](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_load.html#)和[自定义数据集](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html)。


### 模型训练
运行如下命令即可在样例训练集上进行 finetune，并在样例验证集上进行验证。

```shell
# GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
unset CUDA_VISIBLE_DEVICES

log_dir=output
rm -rf ${log_dir}
mkdir -p ${log_dir}

python -m paddle.distributed.launch --gpus "0,1,2,3" --log_dir ${log_dir} train.py \
    --model_name_or_path=unimo-text-1.0-summary \
    --train_file train.json \
    --eval_file test.json \
    --save_dir=${log_dir}/checkpoints \
    --logging_steps=100 \
    --save_steps=10000 \
    --epochs=10 \
    --batch_size=32 \
    --learning_rate=5e-5 \
    --warmup_proportion=0.02 \
    --weight_decay=0.01 \
    --max_seq_len=60 \
    --max_target_len=30 \
    --max_dec_len=20 \
    --min_dec_len=3 \
    --do_train \
    --do_eval \
    --device=gpu \
```
也可以直接使用`train.sh`.

关键参数释义如下：
- `gpus` 指示了训练所用的 GPU 卡号。
- `dataset_name` 数据集名称。
- `train_file` 本地训练数据地址。
- `eval_file` 本地测试数据地址。
- `model_name_or_path` 指示了 finetune 使用的具体预训练模型，可以是 PaddleNLP 提供的预训练模型（详见[UNIMO 模型汇总](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers/UNIMO/contents.html)），或者是本地的预训练模型。如果使用本地的预训练模型，可以配置本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含 paddle 预训练模型 model_state.pdparams。如果使用 PaddleNLP 提供的预训练模型，可以选择下面其中之一。

   | PaddleNLP 提供的预训练模型        |
   |---------------------------------|
   | unimo-text-1.0-summary      |
   | unimo-text-1.0      |
   | unimo-text-1.0-large |

- `save_dir` 表示模型的保存路径。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `seed` 表示随机数生成器的种子。
- `epochs` 表示训练轮数。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于 learning rate scheduler 产生的值相乘作为当前学习率。
- `weight_decay` 表示 AdamW 优化器中使用的 weight_decay 的系数。
- `warmup_proportion` 表示学习率逐渐升高到基础学习率（即上面配置的 learning_rate）所需要的迭代数占总步数的比例，最早的使用可以参考[这篇论文](https://arxiv.org/pdf/1706.02677.pdf)。
- `max_seq_len` 模型输入序列的最大长度。
- `max_target_len` 模型训练时标签的最大长度。
- `min_dec_len` 模型生成序列的最小长度。
- `max_dec_len` 模型生成序列的最大长度。
- `do_train` 是否进行训练。
- `do_eval` 是否进行预测，在验证集上会自动评估。
- `device` 表示使用的设备，从 gpu 和 cpu 中选择。

更多参数详情和参数的默认值请参考`train.py`。

程序运行时将会自动进行训练和验证，训练过程中会自动保存模型在指定的`save_dir`中。
如：
```text
./checkpoints/
├── model_8000
│   ├── model_config.json
│   ├── model_state.pdparams
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
└── ...
```

**NOTE:** 如需恢复模型训练，`model_name_or_path`配置本地模型的目录地址即可。


### 模型预测

运行下方脚本可以使用训练好的模型进行预测。

```shell
export CUDA_VISIBLE_DEVICES=0
python train.py \
    --do_eval \
    --eval_file test.json \
    --model_name_or_path=your_model_path \
    --logging_steps=100 \
    --batch_size=16 \
    --max_seq_len=60 \
    --max_target_len=30 \
    --max_dec_len=20 \
    --min_dec_len=3 \
    --device=gpu
```

程序运行结束后会将预测结果保存在`output_path`中。


Finetuned baseline 的模型在[LCSTS](https://aclanthology.org/D15-1229/)测试集上有如下结果：
|       model_name        | Rouge-1 | Rouge-2 |    Rouge-L    | BLEU-4 |
| :-----------------------------: | :---: | :-----------: | :-------------------: |:-------------------: |
|   finetuned unimo-text-1.0-summary    | 39.56 | 26.24 |     36.35     |     21.48     |


### 模型推理部署

#### FastGeneration 加速及模型静态图导出

使用动态图训练结束之后，可以通过[静态图导出脚本](export_model.py)实现基于 FastGeneration 的高性能预测加速，并将动态图参数导出成静态图参数，静态图参数保存在`output_path`指定路径中。运行方式：

```shell
python export_model.py \
    --model_name_or_path unimo-text-1.0-summary \
    --decoding_strategy beam_search \
    --inference_model_dir ./inference_model \
    --max_out_len 30 \
```
关键参数释义如下：

* `model_name_or_path`：动态图训练保存的参数路径；默认为"unimo-text-1.0-summary"。
* `inference_model_dir`：静态图图保存的参数路径；默认为"./inference_model"。
* `max_out_len`：最大输出长度。

执行命令后将会自动导出模型到指定的 `inference_model` 中，保存模型文件结构如下所示：

```text
inference_model/
├── unimo_text.pdiparams
├── unimo_text.pdiparams.info
└── unimo_text.pdmodel
```

#### 模型部署
文本摘要应用已打通多种场景部署方案，点击链接获取具体的使用教程。
- [Paddle Inference 推理 (Python)](./deploy/paddle_inference/README.md)
- [Paddle Serving 服务化部署（Python）](./deploy/paddle_serving/README.md)

## References
Li, Wei, et al. "Unimo: Towards unified-modal understanding and generation via cross-modal contrastive learning." arXiv preprint arXiv:2012.15409 (2020).
