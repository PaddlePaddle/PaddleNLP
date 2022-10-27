# 生成式文本摘要应用

**目录**
- [生成式文本摘要应用](#生成式文本摘要应用)
  - [简介](#简介)
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
      - [FasterGeneration加速及模型静态图导出](#fastergeneration加速及模型静态图导出)
      - [模型部署](#模型部署)
  - [References](#references)


## 简介
文本摘要的目标是自动地将输入文本转换成简短摘要,为用户提供简明扼要的内容描述，是缓解文本信息过载的一个重要手段。
文本摘要也是自然语言生成领域中的一个重要任务，有很多应用场景，如新闻摘要、论文摘要、财报摘要、传记摘要、专利摘要、对话摘要、评论摘要、观点摘要、电影摘要、文章标题生成、商品名生成、自动报告生成、搜索结果预览等。

本项目是基于预训练语言模型PEGASUS的中文文本摘要产业实践，具有以下优势：
- 效果领先。在LCSTS上效果达到SOTA。
- 开箱即用。本项目提供TaskFlow接口，无需训练，仅需几行代码便可预测。
- 高性能推理。本项目基于[FasterGeneration](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/faster_generation)进行推理加速，能够提供更高性能的推理体验。
- 训练推理全流程打通。本项目提供了全面的定制训练流程，从数据准备、模型训练预测，到模型推理部署，一应俱全。

## 效果展示

## 开箱即用
PaddleNLP提供开箱即用的产业级NLP预置任务能力，无需训练，一键预测。
### 支持单条、批量预测

```python
>>> from paddlenlp import Taskflow
>>> summarizer = Taskflow("text_summarization")
# 单条输入
>>> summarizer('2022年，中国房地产进入转型阵痛期，传统“高杠杆、快周转”的模式难以为继，万科甚至直接喊话，中国房地产进入“黑铁时代”')
# 输出：['万科喊话中国房地产进入“黑铁时代”']

# 多条输入
>>> summarizer([
  '据悉，2022年教育部将围绕“巩固提高、深化落实、创新突破”三个关键词展开工作。要进一步强化学校教育主阵地作用，继续把落实“双减”作为学校工作的重中之重，重点从提高作业设计水平、提高课后服务水平、提高课堂教学水平、提高均衡发展水平四个方面持续巩固提高学校“双减”工作水平。',
  '党参有降血脂，降血压的作用，可以彻底消除血液中的垃圾，从而对冠心病以及心血管疾病的患者都有一定的稳定预防工作作用，因此平时口服党参能远离三高的危害。另外党参除了益气养血，降低中枢神经作用，调整消化系统功能，健脾补肺的功能。'
  ])
#输出：['教育部：将从四个方面持续巩固提高学校“双减”工作水平', '党参能降低三高的危害']
```

### 可配置参数说明
* `model`：可选模型，默认为`IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese`。
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。


## 训练定制
### 文本摘要应用定制训练全流程介绍
接下来，我们将按数据准备、训练、预测、推理部署对文本摘要应用的全流程进行介绍。
1. **数据准备**
- 如果没有已标注的数据集，我们推荐[doccano](https://github.com/doccano/doccano)数据标注工具。
如果已有标注好的本地数据集，我们需要根据将数据集整理为文档要求的格式，请参考[从本地文件创建数据集](#从本地文件创建数据集)。

2. **模型训练**

- 数据准备完成后，可以开始使用我们的数据集对预训练模型进行微调训练。我们可以根据任务需求，调整可配置参数，选择使用GPU或CPU进行模型训练，脚本默认保存在开发集最佳表现模型。中文任务默认使用"IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese"模型，还支持large模型: "IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese"。


3. **模型预测**

- 训练结束后，我们可以加载保存的最佳模型进行模型测试，打印模型预测结果。

4. **模型推理部署**

- 模型部署需要将保存的最佳模型参数（动态图）导出成静态图参数，用于后续的推理部署。

- 文本摘要应用提供了基于Paddle Inference的本地部署predictor，并且支持在GPU设备使用FasterGeneration进行加速。

- 文本摘要应用提供了基于Paddle Serving的服务端部署方案。

### 环境依赖

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
text_summarization/
├── deploy # 部署
│   ├── paddle_inference # PaddleInference高性能推理部署
│   │   ├── inference_pegasus.py # 推理部署脚本
│   │   └── README.md # 说明文档
│   └── paddle_serving
│       ├── config.yml # 配置文件
│       ├── pipeline_client.py # 客户端程序
│       ├── pipeline_service.py # 服务器程序
│       ├── export_serving.sh # serving模型导出脚本
│       └── README.md # 说明文档
├── export_model.py # 动态图参数导出静态图参数脚本
├── export_model.sh # 动态图参数导出静态图参数shell脚本
├── run_summarization.py # 训练评估脚本
├── run_train.sh # 训练评估shell脚本
├── run_generate.py # 预测脚本
├── run_generate.sh # 预测shell脚本
├── utils.py # 工具函数脚本
├── requirements.txt # 依赖包
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
运行如下命令即可在样例训练集上进行finetune，并在样例验证集上进行验证。

```shell
# GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus "2,3,4,5,6,7" run_summarization.py \
    --model_name_or_path=IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese \
    --train_file train.json \
    --eval_file test.json \
    --output_dir pegesus_out \
    --max_source_length 128 \
    --max_target_length 64 \
    --num_train_epochs 20 \
    --logging_steps 1 \
    --save_steps 10000 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --learning_rate 5e-5 \
    --warmup_proportion 0.02 \
    --weight_decay=0.01 \
    --device=gpu \
```
也可以直接使用`run_train.sh`.

关键参数释义如下：
- `gpus` 指示了训练所用的GPU卡号。
- `train_file` 本地训练数据地址。
- `eval_file` 本地测试数据地址。
- `model_name_or_path` 指示了finetune使用的具体预训练模型，可以是PaddleNLP提供的预训练模型，或者是本地的预训练模型。如果使用本地的预训练模型，可以配置本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含paddle预训练模型model_state.pdparams。如果使用PaddleNLP提供的预训练模型，可以选择下面其中之一。

   | PaddleNLP提供的预训练模型        |
   |---------------------------------|
   | IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese      |
   | IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese      |

- `output_dir` 表示模型的保存路径。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `seed` 表示随机数生成器的种子。
- `num_train_epochs` 表示训练轮数。
- `train_batch_size` 表示每次巡礼哪**每张卡**上的样本数目。
- `eval_batch_size` 表示每次验证**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `weight_decay` 表示AdamW优化器中使用的weight_decay的系数。
- `warmup_propotion` 表示学习率逐渐升高到基础学习率（即上面配置的learning_rate）所需要的迭代数占总步数的比例，最早的使用可以参考[这篇论文](https://arxiv.org/pdf/1706.02677.pdf)。
- `max_source_length` 模型输入序列的最大长度。
- `max_target_length` 模型训练时标签的最大长度。
- `device` 表示使用的设备，从gpu和cpu中选择。

更多参数详情和参数的默认值请参考`run_summarization.py`。

程序运行时将会自动进行训练和验证，训练过程中会自动保存模型在指定的`output_dir`中。
如：
```text
./pegeaus_model/
├── pegeaus_model_10000
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
unset CUDA_VISIBLE_DEVICES

python run_generate.py \
    --model_name_or_path=pegesus_out/pegeaus_model_10000 \
    --prefict_file valid.json \
    --max_source_length 128 \
    --max_target_length 64 \
    --batch_size 128 \
    --output_path generate.txt \
    --device=gpu \
```

程序运行结束后会将预测结果保存在`output_path`中。


Finetuned baseline的模型在[LCSTS](https://aclanthology.org/D15-1229/)测试集上有如下结果：
|       model_name        | Rouge-1 | Rouge-2 |    Rouge-L    | BLEU-4 |
| :-----------------------------: | :---: | :-----------: | :-------------------: |:-------------------: |
|   finetuned IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese    | 43.30 | 30.08 |     40.12     |     24.50     |
|   finetuned IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese    | 48.13 | 36.41 |     45.39     |     31.99     |


### 模型推理部署

#### FasterGeneration加速及模型静态图导出

使用动态图训练结束之后，可以通过[静态图导出脚本](export_model.py)实现基于FasterGeneration的高性能预测加速，并将动态图参数导出成静态图参数，静态图参数保存在`output_path`指定路径中。运行方式：

```shell
python export_model.py \
    --model_name_or_path IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese \
    --decoding_strategy beam_search \
    --inference_model_dir ./inference_model \
    --max_out_len 30 \
```
关键参数释义如下：

* `model_name_or_path`：动态图训练保存的参数路径；默认为"IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese"。
* `inference_model_dir`：静态图图保存的参数路径；默认为"./inference_model"。
* `max_out_len`：最大输出长度。

执行命令后将会自动导出模型到指定的 `inference_model` 中，保存模型文件结构如下所示：

```text
inference_model/
├── pegasus.pdiparams
├── pegasus.pdiparams.info
└── pegasus.pdmodel
```

#### 模型部署
文本摘要应用已打通多种场景部署方案，点击链接获取具体的使用教程。
- [Paddle Inference 推理 (Python)](./deploy/paddle_inference/README.md)
- [Paddle Serving 服务化部署（Python）](./deploy/paddle_serving/README.md)

## References
- Zhang J, Zhao Y, Saleh M, et al. Pegasus: Pre-training with extracted gap-sentences for abstractive summarization[C]//International Conference on Machine Learning. PMLR, 2020: 11328-11339.
- Wang J, Zhang Y, Zhang L, et al. Fengshenbang 1.0: Being the Foundation of Chinese Cognitive Intelligence[J]. arXiv preprint arXiv:2209.02970, 2022.
