# 通用信息抽取大模型 PP-UIE

 **目录**

- [1. 模型简介](#模型简介)
- [2. 开箱即用](#开箱即用)
  - [2.1 实体抽取](#实体抽取)
  - [2.2 关系抽取](#关系抽取)
  - [2.3 模型选择](#模型选择)
  - [2.4 更多配置](#更多配置)
- [3. 训练定制](#训练定制)
  - [3.1 代码结构](#代码结构)
  - [3.2 数据标注](#数据标注)
  - [3.3 模型微调](#模型微调)
  - [3.4 定制模型一键预测](#定制模型一键预测)
  - [3.5 实验指标](#实验指标)

<a name="模型简介"></a>

## 1. 模型简介

通用信息抽取大模型（PP-UIE）是 PaddleNLP 团队基于开源模型和高质量数据集构建的通用信息抽取大模型， PaddleNLP 基于百度 UIE 的建模思路，通过大模型的能力来训练并开源了一款面向中、英文通用信息抽取的大模型。 支持统一训练信息抽取任务包括命名实体识别（NER），关系抽取（RE）和事件抽取（EE）。模型共包含0.5B、1.5B、7B 和14B 共4个版本，以适配不同场景下信息抽取任务使用。在多个数据集（包含 Boson、CLUENER、CCIR2021等常见数据）相比其他通用信息抽取大模型在 ACC 和 F1 指标上有大幅度提升。



<a name="开箱即用"></a>

## 2. 开箱即用

```paddlenlp.Taskflow```提供通用信息抽取等能力，可抽取多种类型的信息，包括但不限于命名实体识别（如人名、地名、机构名等）、关系（如电影的导演、歌曲的发行时间等）、事件（如某路口发生车祸、某地发生地震等）等信息。用户可以使用自然语言自定义抽取目标，无需训练即可统一抽取输入文本中的对应信息。**实现开箱即用，并满足各类信息抽取需求**

<a name="实体抽取"></a>

#### 2.1 实体抽取

  命名实体识别（Named Entity Recognition，简称 NER），是指识别文本中具有特定意义的实体。在开放域信息抽取中，抽取的类别没有限制，用户可以自己定义。

  - 例如抽取的目标实体类型是"时间"、"选手"和"赛事名称", schema 构造如下：

    ```text
    ['时间', '选手', '赛事名称']
    ```

    调用示例：

    ```python
    from pprint import pprint
    from paddlenlp import Taskflow

    schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
    ie = Taskflow('information_extraction',
                  schema= ['时间', '选手', '赛事名称'],
                  schema_lang="zh",
                  batch_size=1,
                  model='paddlenlp/PP-UIE-0.5B',
                  precision='float16')
    pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")) # Better print results using pprint
    # 输出
    [{'时间': [{'text': '2月8日上午'}],
      '赛事名称': [{'text': '北京冬奥会自由式滑雪女子大跳台决赛'}],
      '选手': [{'text': '谷爱凌'}]}]
    ```


<a name="关系抽取"></a>

#### 2.2 关系抽取

  关系抽取（Relation Extraction，简称 RE），是指从文本中识别实体并抽取实体之间的语义关系，进而获取三元组信息，即<主体，谓语，客体>。

  - 例如以"竞赛名称"作为抽取主体，抽取关系类型为"主办方"、"承办方"和"时间", schema 构造如下：

    ```text
    {
      '竞赛名称': [
        '主办方',
        '承办方',
        '时间'
      ]
    }
    ```

    调用示例：

    ```python
    schema = {'竞赛名称': ['主办方', '承办方', '时间']} # Define the schema for relation extraction
    ie.set_schema(schema) # Reset schema
    pprint(ie('2022年语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。'))
    # 输出
    [{'竞赛名称': [{'relations': {'主办方': [{'text': '中国中文信息学会,中国计算机学会'}],
                          '时间': [{'text': '2022年'}],
                          '承办方': [{'text': '百度公司,中国中文信息学会评测工作委员会,中国计算机学会自然语言处理专委会'}]},
            'text': '语言与智能技术竞赛'}]}]
    ```

<a name="模型选择"></a>

#### 2.3 模型选择

- 多模型选择，满足精度、速度要求

  | 模型 |  结构  | 语言 |
  | :---: | :--------: | :--------: |
  | `paddlenlp/PP-UIE-0.5B` | 24-layers, 896-hidden, 14-heads | 中、英文 |
  | `paddlenlp/PP-UIE-1.5B` | 28-layers, 1536-hidden, 12-heads | 中、英文 |
  | `paddlenlp/PP-UIE-7B` | 28-layers, 3584-hidden, 28-heads | 中、英文 |
  | `paddlenlp/PP-UIE-14B` | 48-layers, 5120-hidden, 40-heads | 中、英文 |

<a name="更多配置"></a>

#### 2.4 更多配置

```python
>>> from paddlenlp import Taskflow

>>> ie = Taskflow('information_extraction',
                  schema = {'竞赛名称': ['主办方', '承办方', '时间']},
                  schema_lang="zh",
                  batch_size=1,
                  model='paddlenlp/PP-UIE-0.5B',
                  precision='float16')
```

* `schema`：定义任务抽取目标，可参考开箱即用中不同任务的调用示例进行配置。
* `schema_lang`：设置 schema 的语言，默认为`zh`, 可选有`zh`和`en`。因为中英 schema 的构造有所不同，因此需要指定 schema 的语言。
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `model`：选择任务使用的模型，可选有`paddlenlp/PP-UIE-0.5B`, `paddlenlp/PP-UIE-1.5B`, `paddlenlp/PP-UIE-7B`, `paddlenlp/PP-UIE-14B`。
* `precision`：选择模型精度，默认为`float16`，可选有`float16`、`bfloat16`和`float32`和。如果选择`float16`，在 GPU 硬件环境下，请先确保机器正确安装 NVIDIA 相关驱动和基础软件，**确保 CUDA>=11.2，cuDNN>=8.1.1**，初次使用需按照提示安装相关依赖。其次，需要确保 GPU 设备的 CUDA 计算能力（CUDA Compute Capability）大于7.0，典型的设备包括 V100、T4、A10、A100、GTX 20系列和30系列显卡等。如果选择`bfloat16`，能有效加速处理大模型和批量数据，尤其与混合精度结合使用时性能表现更优。但需确保硬件和软件环境支持该精度。支持 `bfloat16`的硬件包括 NVIDIA A100 和 H800 GPU，同时需要确保使用 CUDA>=11.2、cuDNN>=8.1.1 等软件环境。更多关于 CUDA Compute Capability 和精度支持情况请参考 NVIDIA 文档：[GPU 硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)。


除此之外，也可通过以下代码快速调用模型并进行推理

```python
from paddlenlp.transformers import AutoModelForCausalLM
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.generation import GenerationConfig
from paddlenlp.trl import llm_utils

model_id = "paddlenlp/PP-UIE-0.5B"

model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention=False)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
generation_config = GenerationConfig.from_pretrained(model_id)


template = """
你是一个阅读理解专家，请提取所给句子与问题，提取实体。请注意，如果存在实体，则一定在原句中逐字出现，请输出对应实体的原文，不要进行额外修改；如果无法提取，请输出“无相应实体”。
 **句子开始**
 {sentence}
 **句子结束**
 **问题开始**
 {prompt}
 **问题结束**
 **回答开始**
 """

sentences = [
    "2月12日，哈尔滨亚冬会花样滑冰女子个人滑短节目比赛中，中国选手朱易第一个登场且表现出色，拿到62.90分，创职业生涯短节目最高分。",
    "2月12日，在哈尔滨亚冬会越野滑雪男子4×7.5公里接力决赛中，由李明林、次仁占堆、宝林、王强组成的中国队夺得金牌。",
    "2月13日，在哈尔滨亚冬会冬季两项女子4×6公里接力比赛中，由唐佳琳、文颖、褚源蒙和孟繁棋组成的中国队夺得金牌。",
    "中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。",
    "《告别了》是孙耀威在专辑爱的故事里面的歌曲。",
]

prompts = [
    "时间, 选手, 赛事名称",
    "时间, 选手, 赛事名称",
    "时间, 选手, 赛事名称",
    "地震强度, 时间, 震中位置, 震源深度",
    "歌曲名称, 歌手, 所属专辑",
]

inputs = [template.format(sentence=sentence, prompt=prompt) for sentence, prompt in zip(sentences, prompts)]
inputs = [tokenizer.apply_chat_template(sentence, tokenize=False) for sentence in inputs]
input_features = tokenizer(
    inputs,
    max_length=512,
    return_position_ids=False,
    truncation=True,
    truncation_side="left",
    padding=True,
    return_tensors="pd",
    add_special_tokens=False,
)

outputs = model.generate(
    **input_features,
    max_new_tokens=200,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=llm_utils.get_eos_token_id(tokenizer, generation_config),
    pad_token_id=tokenizer.pad_token_id,
    decode_strategy="greedy_search",
    temperature=1.0,
    top_k=1,
    top_p=1.0,
    repetition_penalty=1.0,
)


def get_clean_entity(text):
    ind1 = text.find("\n **回答结束**\n\n")
    if ind1 != -1:
        pred = text[:ind1]
    else:
        pred = text
    return pred


results = tokenizer.batch_decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
results = [get_clean_entity(result) for result in results]

for sentence, prompt, result in zip(sentences, prompts, results):
    print("-" * 50)
    print(f"Sentence: {sentence}")
    print(f"Prompt: {prompt}")
    print(f"Result: {result}")
```

<a name="训练定制"></a>

## 3. 训练定制

对于简单的抽取目标可以直接使用 ```paddlenlp.Taskflow```实现零样本（zero-shot）抽取，对于细分场景我们推荐使用轻定制功能（标注少量数据进行模型微调）以进一步提升效果。下面通过`报销工单信息抽取`的例子展示如何通过几十条训练数据进行 PP-UIE 模型微调。

<a name="代码结构"></a>

#### 3.1 代码结构

```shell
.
├── utils.py          # 数据处理工具
├── doccano.py        # 数据标注脚本
├── doccano.md        # 数据标注文档
└── README.md
```

<a name="数据标注"></a>

#### 3.2 数据标注

我们推荐使用数据标注平台[doccano](https://github.com/doccano/doccano) 进行数据标注，本示例也打通了从标注到训练的通道，即 doccano 导出数据后可通过[doccano.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/application/information_extraction/doccano.py)脚本轻松将数据转换为输入模型时需要的形式，实现无缝衔接。标注方法的详细介绍请参考[doccano 数据标注指南](doccano.md)。

原始数据示例：

```text
深大到双龙28块钱4月24号交通费
```

抽取的目标(schema)为：

```python
schema = ['出发地', '目的地', '费用', '时间']
```

标注步骤如下：

- 在 doccano 平台上，创建一个类型为``序列标注``的标注项目。
- 定义实体标签类别，上例中需要定义的实体标签有``出发地``、``目的地``、``费用``和``时间``。
- 使用以上定义的标签开始标注数据，下面展示了一个 doccano 标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167336891-afef1ad5-8777-456d-805b-9c65d9014b80.png height=100 hspace='10'/>
</div>

- 标注完成后，在 doccano 平台上导出文件，并将其重命名为``doccano_ext.json``后，放入``./data``目录下。

- 这里我们提供预先标注好的文件[doccano_ext.json](https://bj.bcebos.com/paddlenlp/datasets/uie/doccano_ext.json)，可直接下载并放入`./data`目录。执行以下脚本进行数据转换，执行后会在`./data`目录下生成训练/验证/测试集文件。

```shell
python doccano.py \
    --doccano_file ./data/doccano_ext.json \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --schema_lang ch
```


可配置参数说明：

- ``doccano_file``: 从 doccano 导出的数据标注文件。
- ``save_dir``: 训练数据的保存目录，默认存储在``data``目录下。
- ``negative_ratio``: 最大负例比例，该参数只对抽取类型任务有效，适当构造负例可提升模型效果。负例数量和实际的标签数量有关，最大负例数量 = negative_ratio * 正例数量。
- ``splits``: 划分数据集时训练集、验证集所占的比例。默认为[0.8, 0.1, 0.1]表示按照``8:1:1``的比例将数据划分为训练集、验证集和测试集。
- ``task_type``: 选择任务类型，目前只有信息抽取`ie`这一种任务。
- ``is_shuffle``: 是否对数据集进行随机打散，默认为 False。
- ``seed``: 随机种子，默认为1000.
- ``schema_lang``: 选择 schema 的语言，可选有`ch`和`en`。默认为`ch`，英文数据集请选择`en`。

备注：
- 默认情况下 doccano.py 脚本会按照比例将数据划分为 train/dev/test 数据集
- 每次执行 doccano.py 脚本，将会覆盖已有的同名数据文件
- 在模型训练阶段我们推荐构造一些负例以提升模型效果，在数据转换阶段我们内置了这一功能。可通过`negative_ratio`控制自动构造的负样本比例；负样本数量 = negative_ratio * 正样本数量。
- 对于从 doccano 导出的文件，默认文件中的每条数据都是经过人工正确标注的。


<a name="模型微调"></a>

#### 3.3 模型微调

推荐使用 [大模型精调](../../docs/finetune.md) 对模型进行微调。只需输入模型、数据集等就可以高效快速地进行微调和模型压缩等任务，可以一键启动多卡训练、混合精度训练、梯度累积、断点重启、日志显示等功能，并且针对训练过程的通用训练配置做了封装，比如：优化器、学习率调度等。

使用下面的命令，使用 `paddlenlp/PP-UIE-0.5B` 作为预训练模型进行模型微调，将微调后的模型保存至指定路径中。

如果在 GPU 环境中使用，可以指定 gpus 参数进行多卡训练：

```shell
# 返回 PaddleNLP/llm 目录
python -u  -m paddle.distributed.launch --gpus "0,1" run_finetune.py ./config/qwen/sft_argument.json
```

`sft_argument.json` 的参考配置如下：
```shell
{
    "model_name_or_path": "paddlenlp/PP-UIE-0.5B",
    "dataset_name_or_path": "./application/information_extraction/data",
    "output_dir": "./checkpoints/ie_ckpts",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "per_device_eval_batch_size": 1,
    "eval_accumulation_steps":8,
    "num_train_epochs": 3,
    "learning_rate": 3e-05,
    "warmup_steps": 30,
    "logging_steps": 1,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "src_length": 1024,
    "max_length": 2048,
    "fp16": true,
    "fp16_opt_level": "O2",
    "do_train": true,
    "do_eval": true,
    "disable_tqdm": true,
    "load_best_model_at_end": true,
    "eval_with_do_generation": false,
    "metric_for_best_model": "accuracy",
    "recompute": false,
    "save_total_limit": 1,
    "tensor_parallel_degree": 1,
    "pipeline_parallel_degree": 1,
    "sharding": "stage2",
    "zero_padding": false,
    "unified_checkpoint": true,
    "use_flash_attention": false
  }
```
更多 `sft_argument.json` 配置文件说明，请参考[大模型精调](../../docs/finetune.md)


<a name="定制模型一键预测"></a>

#### 3.4 定制模型一键预测

使用 PaddleNLP 的高性能 predictor 进行快速推理
- 内置全环节融合算子策略
- 支持 Weight Only INT8及 INT4推理，支持权重、激活、Cache KV 进行 INT8、FP8量化的推理
- 支持动态图推理和静态图推理两种方式

在推理之前，推荐编译安装 PaddleNLP 大模型高性能自定义推理算子。使用这些高性能算子，可以大幅提升大模型推理速度。详细的安装教程请参考[大模型高性能推理算子安装教程](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/csrc/README.md)

安装完之后，可按照下列指令，进行高性能推理。

```shell
# PaddleNLP/llm目录下
python predict/predictor.py \
    --model_name_or_path ./checkpoints/ie_ckpts \
    --dtype float16 \
    --data_file ./application/information_extraction/data/test.json \
    --output_file ./output.json \
    --src_length  512 \
    --max_length  1024 \
    --batch_size  4 \
    --inference_model 1 \
    --quant_type weight_only_int8
```

可配置参数说明：

- ``model_name_or_path``: 必需，预训练模型名称或者本地的模型路径，用于热启模型和分词器，默认为 None。
- ``src_length``: 模型输入上下文最大 token 长度，默认为1024。
- ``max_length``: 模型输入（上下文+生成内容）的最大 token 长度, 默认为2048。
- ``inference_model``: 是否使用 Inference Model 推理，默认值为 False。Inference Model 内置动态插入和全环节算子融合策略，开启后性能更优。**如果没有编译安装 PaddleNLP 大模型高性能自定义推理算子，只能设置为False**
- ``quant_type``: 是否使用量化推理，默认值为 None。可选的数值有weight_only_int8、weight_only_int4、a8w8和a8w8_fp8。**如果没有编译安装 PaddleNLP 大模型高性能自定义推理算子，只能设置为None**

更多关于 `predictor.py` 的配置参数说明，请参考[大模型推理教程](../../docs/predict/inference.md)



<a name="实验指标"></a>

#### 3.5 实验指标

我们在通用测试集和医疗、新闻、对话与金融等垂类测试集上进行了实验：

<table>
<tr><td>模型名称</td><td>数据集名称</td><td>CMeEE-V2</td><td>Boson</td><td>CLUENER</td><td>CCIR2021-NER</td><td>任务对话2018-NER</td><td>银行借贷2021-NER</td><td>SKE2019</td><td>Avg</td></tr>
<tr><td></td><td>数据集领域</td><td>医疗领域</td><td>通用领域</td><td>通用领域</td><td>新闻领域</td><td>对话领域</td><td>金融领域</td><td>金融领域</td><td></td></tr>
<tr><td>PP-UIE-0.5B</td><td>F1(0-shot)</td><td>0.479</td><td>0.638</td><td>0.593</td><td>0.773</td><td>0.723</td><td>0.361</td><td>0.782</td><td>0.621</td></tr>
<tr><td>PP-UIE-1.5B</td><td>F1(0-shot)</td><td>0.485</td><td>0.688</td><td>0.61</td><td>0.799</td><td>0.768</td><td>0.444</td><td>0.803</td><td>0.657</td></tr>
<tr><td></td><td>F1(5-shot)</td><td>0.52</td><td>0.694</td><td>0.625</td><td>0.812</td><td>0.812</td><td>0.466</td><td>0.801</td><td>0.676</td></tr>
<tr><td>PP-UIE-7B</td><td>F1(0-shot)</td><td>0.521</td><td>0.696</td><td>0.615</td><td>0.826</td><td>0.807</td><td>0.434</td><td>0.812</td><td>0.673</td></tr>
<tr><td></td><td>F1(5-shot)</td><td>0.527</td><td>0.705</td><td>0.626</td><td>0.826</td><td>0.861</td><td>0.483</td><td>0.801</td><td>0.69</td></tr>
<tr><td>PP-UIE-14B</td><td>F1(0-shot)</td><td>0.556</td><td>0.712</td><td>0.637</td><td>0.841</td><td>0.843</td><td>0.488</td><td>0.832</td><td>0.701</td></tr>
<tr><td></td><td>F1(5-shot)</td><td>0.588</td><td>0.729</td><td>0.67</td><td>0.837</td><td>0.865</td><td>0.576</td><td>0.832</td><td>0.728</td></tr>
</table>


0-shot 表示无训练数据直接通过模型进行预测，5-shot 表示预测时使用五个数据样例作为提示。**实验表明 PP-UIE 在垂类场景可以通过少量数据（few-shot）进一步提升效果**。


同时，我们测试了PP-UI系列模型在不同数据集，分别在纯动态图、开启融合算子（infernce_model = True）和win8(开启Int8量化)在batch size为[1，2，4，8，16，32，64]时的运行速度（Tokens Per Second）和预测精度（F1）。

**PP-UIE-0.5B**

<table><tr><td>模型名称</td><td></td><td></td><td>数据集名称</td><td>CMeEE-V2</td><td>Boson</td><td>CLUENER</td><td>CCIR2021-NER</td><td>任务对话2018-NER</td><td>银行借贷2021-NER</td><td>SKE2019</td></tr><tr><td></td><td></td><td>batch size</td><td>数据集领域</td><td>医疗领域</td><td>通用领域</td><td>通用领域</td><td>新闻领域</td><td>对话领域</td><td>金融领域</td><td>金融领域</td></tr><tr><td>PP-UIE-0.5B</td><td>动态图</td><td>1</td><td>F1</td><td>0.508</td><td>0.623</td><td>0.593</td><td>0.784</td><td>0.723</td><td>0.332</td><td>0.787</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>30.269</td><td>30.515</td><td>30.403</td><td>30.901</td><td>29.922</td><td>30.823</td><td>30.662</td></tr><tr><td></td><td></td><td>2</td><td>F1</td><td>0.504</td><td>0.617</td><td>0.591</td><td>0.78</td><td>0.721</td><td>0.337</td><td>0.785</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>56.906</td><td>56.696</td><td>57.726</td><td>56.205</td><td>58.576</td><td>56.472</td><td>57.674</td></tr><tr><td></td><td></td><td>4</td><td>F1  </td><td>0.494</td><td>0.609</td><td>0.591</td><td>0.774</td><td>0.721</td><td>0.335</td><td>0.784</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>109.094</td><td>109.307</td><td>107.597</td><td>106.739</td><td>106.243</td><td>107.37</td><td>108.95</td></tr><tr><td></td><td></td><td>8</td><td>F1  </td><td>0.482</td><td>0.607</td><td>0.587</td><td>0.765</td><td>0.712</td><td>0.333</td><td>0.784</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>199.777</td><td>199.373</td><td>199.513</td><td>201.492</td><td>200.301</td><td>197.366</td><td>198.628</td></tr><tr><td></td><td></td><td>16</td><td>F1  </td><td>0.461</td><td>0.594</td><td>0.588</td><td>0.75</td><td>0.718</td><td>0.332</td><td>0.771</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>342.747</td><td>338.052</td><td>333.88</td><td>339.824</td><td>325.661</td><td>319.512</td><td>339.599</td></tr><tr><td></td><td></td><td>32</td><td>F1  </td><td>0.425</td><td>0.584</td><td>0.587</td><td>0.725</td><td>0.714</td><td>0.33</td><td>0.751</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>500.259</td><td>495.871</td><td>478.906</td><td>508.637</td><td>483.591</td><td>480.621</td><td>504.758</td></tr><tr><td></td><td></td><td>64</td><td>F1  </td><td>0.36</td><td>0.564</td><td>0.585</td><td>0.685</td><td>0.713</td><td>0.317</td><td>0.738</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>714.742</td><td>701.403</td><td>661.534</td><td>705.949</td><td>668.907</td><td>671.853</td><td>718.122</td></tr><tr><td></td><td> fuse_mt</td><td>1</td><td>F1</td><td>0.497</td><td>0.618</td><td>0.585</td><td>0.78</td><td>0.72</td><td>0.326</td><td>0.784</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>88.024</td><td>71.534</td><td>78.178</td><td>63.195</td><td>51.87</td><td>48.631</td><td>56.048</td></tr><tr><td></td><td></td><td>2</td><td>F1</td><td>0.495</td><td>0.617</td><td>0.591</td><td>0.781</td><td>0.721</td><td>0.327</td><td>0.78</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>188.091</td><td>157.388</td><td>140.698</td><td>146.544</td><td>111.774</td><td>102.06</td><td>133.236</td></tr><tr><td></td><td></td><td>4</td><td>F1  </td><td>0.495</td><td>0.609</td><td>0.594</td><td>0.781</td><td>0.715</td><td>0.332</td><td>0.784</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>395.09</td><td>318.009</td><td>294.731</td><td>298.858</td><td>220.825</td><td>207.682</td><td>304.236</td></tr><tr><td></td><td></td><td>8</td><td>F1  </td><td>0.497</td><td>0.619</td><td>0.592</td><td>0.78</td><td>0.719</td><td>0.321</td><td>0.787</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>784.377</td><td>695.807</td><td>587.066</td><td>612.309</td><td>456.107</td><td>428.209</td><td>538.053</td></tr><tr><td></td><td></td><td>16</td><td>F1  </td><td>0.493</td><td>0.625</td><td>0.585</td><td>0.775</td><td>0.724</td><td>0.319</td><td>0.789</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>1456.824</td><td>1260.593</td><td>1092.222</td><td>1189.585</td><td>895.154</td><td>822.057</td><td>1134.441</td></tr><tr><td></td><td></td><td>32</td><td>F1  </td><td>0.495</td><td>0.621</td><td>0.591</td><td>0.778</td><td>0.721</td><td>0.321</td><td>0.788</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>2619.044</td><td>2241.112</td><td>1957.307</td><td>2130.925</td><td>1668.488</td><td>1533.073</td><td>2311.613</td></tr><tr><td></td><td></td><td>64</td><td>F1  </td><td>0.496</td><td>0.613</td><td>0.587</td><td>0.781</td><td>0.719</td><td>0.322</td><td>0.788</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>4279.335</td><td>3571.327</td><td>2775.013</td><td>3692.86</td><td>2709.238</td><td>2724.1</td><td>3918.789</td></tr><tr><td></td><td>WINT8</td><td>1</td><td>F1  </td><td>0.5</td><td>0.619</td><td>0.589</td><td>0.774</td><td>0.71</td><td>0.333</td><td>0.787</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>102.626</td><td>82.016</td><td>65.701</td><td>67.226</td><td>53.328</td><td>53.327</td><td>57.867</td></tr><tr><td></td><td></td><td>2</td><td>F1  </td><td>0.502</td><td>0.613</td><td>0.585</td><td>0.779</td><td>0.72</td><td>0.331</td><td>0.789</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>199.294</td><td>169.8</td><td>142.026</td><td>147.443</td><td>111.743</td><td>102.999</td><td>121.712</td></tr><tr><td></td><td></td><td>4</td><td>F1  </td><td>0.499</td><td>0.628</td><td>0.591</td><td>0.777</td><td>0.714</td><td>0.327</td><td>0.788</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>390.208</td><td>340.839</td><td>299.54</td><td>299.343</td><td>256.566</td><td>268.026</td><td>258.988</td></tr><tr><td></td><td></td><td>8</td><td>F1  </td><td>0.502</td><td>0.622</td><td>0.588</td><td>0.779</td><td>0.712</td><td>0.323</td><td>0.784</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>821.311</td><td>713.367</td><td>597.427</td><td>656.373</td><td>439.528</td><td>466.009</td><td>532.473</td></tr><tr><td></td><td></td><td>16</td><td>F1  </td><td>0.499</td><td>0.621</td><td>0.587</td><td>0.779</td><td>0.72</td><td>0.327</td><td>0.784</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>1547.189</td><td>1335.012</td><td>1194.904</td><td>1289.993</td><td>875.995</td><td>936.525</td><td>1052.361</td></tr><tr><td></td><td></td><td>32</td><td>F1  </td><td>0.501</td><td>0.619</td><td>0.593</td><td>0.781</td><td>0.721</td><td>0.318</td><td>0.788</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>2981.043</td><td>2176.571</td><td>2193.828</td><td>2260.412</td><td>1517.517</td><td>1516.653</td><td>1937.827</td></tr><tr><td></td><td></td><td>64</td><td>F1  </td><td>0.499</td><td>0.623 </td><td>0.589</td><td>0.778</td><td>0.722</td><td>0.339</td><td>0.785</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>5288.722</td><td>3643.228</td><td>2646.107</td><td>3674.814</td><td>2748.316</td><td>2478.676</td><td>3510.926</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>


**PP-UIE-1.5B**

<table><tr><td>模型名称</td><td></td><td></td><td>数据集名称</td><td>CMeEE-V2</td><td>Boson</td><td>CLUENER</td><td>CCIR2021-NER</td><td>任务对话2018-NER</td><td>银行借贷2021-NER</td><td>SKE2019</td></tr><tr><td></td><td></td><td>batch size</td><td>数据集领域</td><td>医疗领域</td><td>通用领域</td><td>通用领域</td><td>新闻领域</td><td>对话领域</td><td>金融领域</td><td>金融领域</td></tr><tr><td>PP-UIE-1.5B</td><td>动态图</td><td>1</td><td>F1</td><td>0.52</td><td>0.695</td><td>0.626</td><td>0.818</td><td>0.766</td><td>0.456</td><td>0.8</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>26.47</td><td>26.141</td><td>25.449</td><td>25.612</td><td>25.37</td><td>26.084</td><td>25.999</td></tr><tr><td></td><td></td><td>2</td><td>F1</td><td>0.509</td><td>0.69</td><td>0.626</td><td>0.814</td><td>0.764</td><td>0.46</td><td>0.799</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>48.545</td><td>48.5</td><td>47.851</td><td>47.639</td><td>47.627</td><td>46.602</td><td>48.196</td></tr><tr><td></td><td></td><td>4</td><td>F1  </td><td>0.492</td><td>0.689</td><td>0.624</td><td>0.81</td><td>0.765</td><td>0.456</td><td>0.795</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>91.653</td><td>91.684</td><td>91.34</td><td>91.48</td><td>91.156</td><td>89.752</td><td>90.967</td></tr><tr><td></td><td></td><td>8</td><td>F1  </td><td>0.468</td><td>0.676</td><td>0.626</td><td>0.8</td><td>0.763</td><td>0.458</td><td>0.791</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>169.458</td><td>169.043</td><td>165.25</td><td>170.523</td><td>169.804</td><td>164.427</td><td>171.739</td></tr><tr><td></td><td></td><td>16</td><td>F1  </td><td>0.428</td><td>0.664</td><td>0.626</td><td>0.785</td><td>0.763</td><td>0.456</td><td>0.78</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>296.315</td><td>292.382</td><td>290.317</td><td>295.588</td><td>281.144</td><td>281.313</td><td>295.661</td></tr><tr><td></td><td></td><td>32</td><td>F1  </td><td>0.379</td><td>0.632</td><td>0.625</td><td>0.755</td><td>0.763</td><td>0.452</td><td>0.758</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>481.643</td><td>476.989</td><td>458.55</td><td>477.239</td><td>464.775</td><td>451.335</td><td>487.228</td></tr><tr><td></td><td></td><td>64</td><td>F1  </td><td>0.328</td><td>0.576</td><td>0.625</td><td>0.707</td><td>0.761</td><td>0.451</td><td>0.72</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>689.837</td><td>682.329</td><td>606.592</td><td>663.469</td><td>645.239</td><td>636.379</td><td>660.386</td></tr><tr><td></td><td> fuse_mt</td><td>1</td><td>F1</td><td>0.509</td><td>0.681</td><td>0.621</td><td>0.813</td><td>0.765</td><td>0.454</td><td>0.798</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>83.865</td><td>68.819</td><td>78.828</td><td>61.905</td><td>55.432</td><td>56.596</td><td>57.138</td></tr><tr><td></td><td></td><td>2</td><td>F1</td><td>0.511</td><td>0.686</td><td>0.625</td><td>0.811</td><td>0.768</td><td>0.439</td><td>0.799</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>182.739</td><td>150.276</td><td>131.843</td><td>150.973</td><td>107.681</td><td>99.068</td><td>138.105</td></tr><tr><td></td><td></td><td>4</td><td>F1  </td><td>0.509</td><td>0.686</td><td>0.618</td><td>0.811</td><td>0.769</td><td>0.448</td><td>0.799</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>389.224</td><td>316.024</td><td>274.555</td><td>276.531</td><td>236.939</td><td>204.266</td><td>300.587</td></tr><tr><td></td><td></td><td>8</td><td>F1  </td><td>0.509</td><td>0.682</td><td>0.619</td><td>0.812</td><td>0.762</td><td>0.45</td><td>0.798</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>751.196</td><td>627.038</td><td>554.42</td><td>583.324</td><td>435.373</td><td>412.652</td><td>618.796</td></tr><tr><td></td><td></td><td>16</td><td>F1  </td><td>0.504</td><td>0.683</td><td>0.618</td><td>0.815</td><td>0.763</td><td>0.443</td><td>0.798</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>1367.616</td><td>1139.204</td><td>1023.104</td><td>1079.171</td><td>859.398</td><td>789.85</td><td>1224.739</td></tr><tr><td></td><td></td><td>32</td><td>F1  </td><td>0.51</td><td>0.687</td><td>0.615</td><td>0.812</td><td>0.763</td><td>0.448</td><td>0.8</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>2346.183</td><td>1862.637</td><td>1721.626</td><td>1873.001</td><td>1446.156</td><td>1358.769</td><td>2174.648</td></tr><tr><td></td><td></td><td>64</td><td>F1  </td><td>0.505</td><td>0.686</td><td>0.612</td><td>0.811</td><td>0.764</td><td>0.45</td><td>0.799</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>3435.418</td><td>2807.375</td><td>2642.186</td><td>2862.773</td><td>2201.76</td><td>2086.964</td><td>3377.49</td></tr><tr><td></td><td>WINT8</td><td>1</td><td>F1  </td><td>0.516</td><td>0.685</td><td>0.63</td><td>0.81</td><td>0.776</td><td>0.451</td><td>0.795</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>74.782</td><td>68.263</td><td>58.323</td><td>64.345</td><td>50.213</td><td>47.894</td><td>48.872</td></tr><tr><td></td><td></td><td>2</td><td>F1  </td><td>0.515</td><td>0.689</td><td>0.626</td><td>0.809</td><td>0.765</td><td>0.44</td><td>0.793</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>197.449</td><td>151.655</td><td>139.386</td><td>140.525</td><td>116.931</td><td>97.83</td><td>131.507</td></tr><tr><td></td><td></td><td>4</td><td>F1  </td><td>0.515</td><td>0.692</td><td>0.622</td><td>0.809</td><td>0.769</td><td>0.443</td><td>0.797</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>356.658</td><td>291.106</td><td>267.558</td><td>272.57</td><td>207.656</td><td>198.878</td><td>251.976</td></tr><tr><td></td><td></td><td>8</td><td>F1  </td><td>0.515</td><td>0.684</td><td>0.623</td><td>0.812</td><td>0.762</td><td>0.442</td><td>0.798</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>709.983</td><td>575.773</td><td>522.708</td><td>543.154</td><td>431.868</td><td>429.064</td><td>518.811</td></tr><tr><td></td><td></td><td>16</td><td>F1  </td><td>0.515</td><td>0.682</td><td>0.618</td><td>0.814</td><td>0.772</td><td>0.453</td><td>0.799</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>1318.79</td><td>1031.525</td><td>935.156</td><td>983.182</td><td>765.36</td><td>714.785</td><td>1065.399</td></tr><tr><td></td><td></td><td>32</td><td>F1  </td><td>0.515</td><td>0.69</td><td>0.629</td><td>0.811</td><td>0.762</td><td>0.448</td><td>0.798</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>2366.751</td><td>1744.833</td><td>1543</td><td>1757.031</td><td>1264.179</td><td>1177.245</td><td>1816.415</td></tr><tr><td></td><td></td><td>64</td><td>F1  </td><td>0.515</td><td>0.681</td><td>0.622</td><td>0.811</td><td>0.764</td><td>0.444</td><td>0.797</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>3799.326</td><td>2567.648</td><td>2265.59</td><td>2650.271</td><td>1906.524</td><td>1761.032</td><td>3083.406</td></tr></table>


**PP-UIE-7B**

<table><tr><td>模型名称</td><td></td><td></td><td>数据集名称</td><td>CMeEE-V2</td><td>Boson</td><td>CLUENER</td><td>CCIR2021-NER</td><td>任务对话2018-NER</td><td>银行借贷2021-NER</td><td>SKE2019</td></tr><tr><td></td><td></td><td>batch size</td><td>数据集领域</td><td>医疗领域</td><td>通用领域</td><td>通用领域</td><td>新闻领域</td><td>对话领域</td><td>金融领域</td><td>金融领域</td></tr><tr><td>PP-UIE-7B</td><td>动态图</td><td>1</td><td>F1</td><td>0.528</td><td>0.703</td><td>0.615</td><td>0.827</td><td>0.786</td><td>0.431</td><td>0.813</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>24.971</td><td>24.263</td><td>24.935</td><td>24.201</td><td>24.43</td><td>24.59</td><td>24.579</td></tr><tr><td></td><td></td><td>2</td><td>F1</td><td>0.524</td><td>0.702</td><td>0.615</td><td>0.827</td><td>0.786</td><td>0.433</td><td>0.812</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>47.833</td><td>46.968</td><td>47.388</td><td>48.066</td><td>47.8</td><td>47.521</td><td>48.033</td></tr><tr><td></td><td></td><td>4</td><td>F1  </td><td>0.519</td><td>0.704</td><td>0.616</td><td>0.827</td><td>0.784</td><td>0.433</td><td>0.813</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>88.69</td><td>87.364</td><td>87.516</td><td>88.941</td><td>89.231</td><td>90.045</td><td>90.196</td></tr><tr><td></td><td></td><td>8</td><td>F1  </td><td>0.514</td><td>0.704</td><td>0.615</td><td>0.826</td><td>0.785</td><td>0.433</td><td>0.813</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>169.087</td><td>161.141</td><td>162.046</td><td>164.154</td><td>164.776</td><td>153.411</td><td>161.853</td></tr><tr><td></td><td></td><td>16</td><td>F1  </td><td>0.501</td><td>0.703</td><td>0.614</td><td>0.826</td><td>0.785</td><td>0.432</td><td>0.813</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>288.043</td><td>268.144</td><td>264.288</td><td>270.323</td><td>260.654</td><td>252.396</td><td>270.884</td></tr><tr><td></td><td></td><td>32</td><td>F1  </td><td>0.479</td><td>0.703</td><td>0.615</td><td>0.823</td><td>0.784</td><td>0.432</td><td>0.12</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>439.281</td><td>400.6</td><td>385.3381</td><td>406.698</td><td>379.117</td><td>366.518</td><td>399.546</td></tr><tr><td></td><td></td><td>64</td><td>F1  </td><td>0.441</td><td>0.702</td><td>0.614</td><td>0.816</td><td>0.783</td><td>0.432</td><td>0.808</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>613.321</td><td>593.829</td><td>515.7</td><td>574.011</td><td>504.845</td><td>506.382</td><td>556.177</td></tr><tr><td></td><td> fuse_mt</td><td>1</td><td>F1</td><td>0.517</td><td>0.702</td><td>0.623</td><td>0.823</td><td>0.788</td><td>0.423</td><td>0.811</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>51.74</td><td>47.895</td><td>41.021</td><td>43.369</td><td>37.08</td><td>37.437</td><td>41.661</td></tr><tr><td></td><td></td><td>2</td><td>F1</td><td>0.516</td><td>0.699</td><td>0.613</td><td>0.82</td><td>0.788</td><td>0.427</td><td>0.812</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>105.843</td><td>97.718</td><td>84.915</td><td>89.266</td><td>74.252</td><td>66.249</td><td>80.974</td></tr><tr><td></td><td></td><td>4</td><td>F1  </td><td>0.514</td><td>0.696</td><td>0.609</td><td>0.823</td><td>0.783</td><td>0.434</td><td>0.808</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>216.985</td><td>189.58</td><td>180.078</td><td>187.1</td><td>146.36</td><td>131.028</td><td>172.963</td></tr><tr><td></td><td></td><td>8</td><td>F1  </td><td>0.518</td><td>0.701</td><td>0.618</td><td>0.821</td><td>0.787</td><td>0.428</td><td>0.809</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>391.686</td><td>355.544</td><td>334.309</td><td>349.757</td><td>291.318</td><td>249.223</td><td>348.771</td></tr><tr><td></td><td></td><td>16</td><td>F1  </td><td>0.515</td><td>0.695</td><td>0.611</td><td>0.823</td><td>0.788</td><td>0.426</td><td>0.809</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>736.629</td><td>642.235</td><td>568.576</td><td>628.74</td><td>489.87</td><td>458.587</td><td>610.345</td></tr><tr><td></td><td></td><td>32</td><td>F1  </td><td>0.514</td><td>0.701</td><td>0.609</td><td>0.826</td><td>0.782</td><td>0.423</td><td>0.812</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>1230.591</td><td>1050.501</td><td>927.891</td><td>1001.303</td><td>781.299</td><td>734.324</td><td>1055.442</td></tr><tr><td></td><td></td><td>64</td><td>F1  </td><td>0.517</td><td>0.697</td><td>0.613</td><td>0.823</td><td>0.788</td><td>0.424</td><td>0.81</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>1819.105</td><td>1579.228</td><td>1336.426</td><td>1514.931</td><td>1161.161</td><td>1121.559</td><td>1594.559</td></tr><tr><td></td><td>WINT8</td><td>1</td><td>F1  </td><td>0.535</td><td>0.699</td><td>0.623</td><td>0.824</td><td>0.782</td><td>0.444</td><td>0.812</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>65.298</td><td>46.819</td><td>40.873</td><td>43.83</td><td>35.281</td><td>32.287</td><td>35.705</td></tr><tr><td></td><td></td><td>2</td><td>F1  </td><td>0.522</td><td>0.7</td><td>0.61</td><td>0.824</td><td>0.78</td><td>0.418</td><td>0.812</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>127.689</td><td>90.929</td><td>78.174</td><td>85.999</td><td>67.124</td><td>59.175</td><td>82.49</td></tr><tr><td></td><td></td><td>4</td><td>F1  </td><td>0.525</td><td>0.695</td><td>0.614</td><td>0.826</td><td>0.779</td><td>0.425</td><td>0.81</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>234.016</td><td>193.467</td><td>165.158</td><td>179.821</td><td>141.483</td><td>129.085</td><td>158.411</td></tr><tr><td></td><td></td><td>8</td><td>F1  </td><td>0.522</td><td>0.696</td><td>0.618</td><td>0.824</td><td>0.781</td><td>0.431</td><td>0.811</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>497.447</td><td>372.414</td><td>319.802</td><td>334.657</td><td>274.958</td><td>236.714</td><td>341.587</td></tr><tr><td></td><td></td><td>16</td><td>F1  </td><td>0.522</td><td>0.703</td><td>0.613</td><td>0.824</td><td>0.776</td><td>0.429</td><td>0.812</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>897.135</td><td>695.732</td><td>604.092</td><td>635.239</td><td>478.883</td><td>423.663</td><td>596.289</td></tr><tr><td></td><td></td><td>32</td><td>F1  </td><td>0.522</td><td>0.703</td><td>0.615</td><td>0.827</td><td>0.784</td><td>0.427</td><td>0.812</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>1468.647</td><td>1049.653</td><td>890.938</td><td>1017.609</td><td>816.842</td><td>708.418</td><td>992.633</td></tr><tr><td></td><td></td><td>64</td><td>F1  </td><td>0.526</td><td>0.702</td><td>0.62</td><td>0.822</td><td>0.786</td><td>0.423</td><td>0.809</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>2152.035</td><td>1432.949</td><td>1237.672</td><td>1477.637</td><td>1066.383</td><td>954.065</td><td>1503.071</td></tr></table>


**PP-UIE-14B**
<table><tr><td>模型名称</td><td></td><td></td><td>数据集名称</td><td>CMeEE-V2</td><td>Boson</td><td>CLUENER</td><td>CCIR2021-NER</td><td>任务对话2018-NER</td><td>银行借贷2021-NER</td><td>SKE2019</td></tr><tr><td></td><td></td><td>batch size</td><td>数据集领域</td><td>医疗领域</td><td>通用领域</td><td>通用领域</td><td>新闻领域</td><td>对话领域</td><td>金融领域</td><td>金融领域</td></tr><tr><td>PP-UIE-14B</td><td>动态图</td><td>1</td><td>F1</td><td>0.532</td><td>0.715</td><td>0.637</td><td>0.844</td><td>0.826</td><td>0.49</td><td>0.828</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>14.685</td><td>14.837</td><td>14.751</td><td>14.698</td><td>14.329</td><td>14.212</td><td>14.261</td></tr><tr><td></td><td></td><td>2</td><td>F1</td><td>0.53</td><td>0.713</td><td>0.637</td><td>0.843</td><td>0.827</td><td>0.489</td><td>0.828</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>29.062</td><td>29.146</td><td>28.734</td><td>29.245</td><td>29.56</td><td>29.205</td><td>28.972</td></tr><tr><td></td><td></td><td>4</td><td>F1  </td><td>0.526</td><td>0.711</td><td>0.637</td><td>0.843</td><td>0.826</td><td>0.488</td><td>0.829</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>55.025</td><td>54.938</td><td>54.633</td><td>54.532</td><td>55.626</td><td>53.934</td><td>54.969</td></tr><tr><td></td><td></td><td>8</td><td>F1  </td><td>0.52</td><td>0.708</td><td>0.636</td><td>0.842</td><td>0.827</td><td>0.489</td><td>0.828</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>102.478</td><td>99.568</td><td>99.376</td><td>100.461</td><td>100.831</td><td>95.64</td><td>98.418</td></tr><tr><td></td><td></td><td>16</td><td>F1  </td><td>0.51</td><td>0.706</td><td>0.635</td><td>0.841</td><td>0.825</td><td>0.489</td><td>0.827</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>185.198</td><td>171.829</td><td>170.281</td><td>174.586</td><td>164.299</td><td>152.416</td><td>178.834</td></tr><tr><td></td><td></td><td>32</td><td>F1  </td><td>0.49</td><td>0.711</td><td>0.634</td><td>0.836</td><td>0.822</td><td>0.489</td><td>0.827</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>309.815</td><td>268.985</td><td>267.216</td><td>285.568</td><td>253.737</td><td>227.749</td><td>293.794</td></tr><tr><td></td><td></td><td>64</td><td>F1  </td><td>0.449</td><td>0.712</td><td>0.633</td><td>0.832</td><td>0.822</td><td>0.488</td><td>0.826</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>459.762</td><td>428.323</td><td>376.201</td><td>427.951</td><td>343.526</td><td>355.367</td><td>459.668</td></tr><tr><td></td><td> fuse_mt</td><td>1</td><td>F1</td><td>0.523</td><td>0.706</td><td>0.637</td><td>0.839</td><td>0.822</td><td>0.484</td><td>0.829</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>34.494</td><td>31.347</td><td>29.996</td><td>30.977</td><td>28.478</td><td>25.846</td><td>29.052</td></tr><tr><td></td><td></td><td>2</td><td>F1</td><td>0.519</td><td>0.708</td><td>0.631</td><td>0.84</td><td>0.827</td><td>0.48</td><td>0.826</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>67.869</td><td>62.088</td><td>61.321</td><td>60.017</td><td>54.291</td><td>50.911</td><td>57.019</td></tr><tr><td></td><td></td><td>4</td><td>F1  </td><td>0.522</td><td>0.701</td><td>0.638</td><td>0.84</td><td>0.823</td><td>0.476</td><td>0.826</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>131.164</td><td>122.297</td><td>112.31</td><td>115.559</td><td>103.088</td><td>96.69</td><td>110.686</td></tr><tr><td></td><td></td><td>8</td><td>F1  </td><td>0.52</td><td>0.708</td><td>0.63</td><td>0.842</td><td>0.825</td><td>0.478</td><td>0.827</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>245.615</td><td>229.256</td><td>215.212</td><td>220.401</td><td>196.891</td><td>175.141</td><td>210.526</td></tr><tr><td></td><td></td><td>16</td><td>F1  </td><td>0.518</td><td>0.714</td><td>0.634</td><td>0.842</td><td>0.82</td><td>0.477</td><td>0.827</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>440.587</td><td>399.806</td><td>368.995</td><td>375.22</td><td>329.55</td><td>303.739</td><td>364.359</td></tr><tr><td></td><td>WINT8</td><td>1</td><td>F1  </td><td>0.524</td><td>0.712</td><td>0.634</td><td>0.842</td><td>0.825</td><td>0.477</td><td>0.827</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>41.234</td><td>38.762</td><td>34.777</td><td>35.233</td><td>34.728</td><td>30.871</td><td>32.129</td></tr><tr><td></td><td></td><td>2</td><td>F1  </td><td>0.525</td><td>0.706</td><td>0.633</td><td>0.842</td><td>0.82</td><td>0.478</td><td>0.826</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>80.323</td><td>73.722</td><td>63.893</td><td>64.988</td><td>87.728</td><td>58.289</td><td>67.826</td></tr><tr><td></td><td></td><td>4</td><td>F1  </td><td>0.524</td><td>0.708</td><td>0.637</td><td>0.844</td><td>0.824</td><td>0.478</td><td>0.826</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>162.169</td><td>141.365</td><td>127.283</td><td>130.367</td><td>120.714</td><td>112.827</td><td>132.525</td></tr><tr><td></td><td></td><td>8</td><td>F1  </td><td>0.525</td><td>0.701</td><td>0.63</td><td>0.842</td><td>0.818</td><td>0.477</td><td>0.826</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>332.437</td><td>281.661</td><td>238.875</td><td>266.18</td><td>209.635</td><td>175.688</td><td>267.373</td></tr><tr><td></td><td></td><td>16</td><td>F1  </td><td>0.524</td><td>0.712</td><td>0.634</td><td>0.843</td><td>0.821</td><td>0.487</td><td>0.828</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>545.886</td><td>472.752</td><td>391.256</td><td>424.774</td><td>353.743</td><td>288.442</td><td>420.256</td></tr><tr><td></td><td></td><td>32</td><td>F1  </td><td>0.524</td><td>0.707</td><td>0.635</td><td>0.841</td><td>0.819</td><td>0.478</td><td>0.823</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>787.417</td><td>640.262</td><td>545.588</td><td>602.611</td><td>460.068</td><td>424.596</td><td>607.832</td></tr><tr><td></td><td></td><td>64</td><td>F1  </td><td>0.526</td><td>0.707</td><td>0.637</td><td>0.839</td><td>0.831</td><td>0.481</td><td>0.827</td></tr><tr><td></td><td></td><td></td><td>TPS</td><td>1261.826</td><td>941.326</td><td>794.079</td><td>894.79</td><td>658.474</td><td>639.901</td><td>924.28</td></tr></table>

**以上实验均在单卡A100 80G运行**
