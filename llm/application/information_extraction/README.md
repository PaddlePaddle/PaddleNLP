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
                  model='paddlenlp/PP-UIE-0.5B')
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
* `model`：选择任务使用的模型，默认为`paddlenlp/PP-UIE-0.5B`，可选有`paddlenlp/PP-UIE-0.5B`, `paddlenlp/PP-UIE-1.5B`, `paddlenlp/PP-UIE-7B`, `paddlenlp/PP-UIE-14B`。
* `precision`：选择模型精度，默认为`float16`，可选有`float16`、`bfloat16`和`float32`和。如果选择`float16`，在 GPU 硬件环境下，请先确保机器正确安装 NVIDIA 相关驱动和基础软件，**确保 CUDA>=11.2，cuDNN>=8.1.1**，初次使用需按照提示安装相关依赖。其次，需要确保 GPU 设备的 CUDA 计算能力（CUDA Compute Capability）大于7.0，典型的设备包括 V100、T4、A10、A100、GTX 20系列和30系列显卡等。如果选择`bfloat16`，能有效加速处理大模型和批量数据，尤其与混合精度结合使用时性能表现更优。但需确保硬件和软件环境支持该精度。支持 `bfloat16`的硬件包括 NVIDIA A100 和 H100 GPU，同时需要确保使用 CUDA>=11.2、cuDNN>=8.1.1 等软件环境。更多关于 CUDA Compute Capability 和精度支持情况请参考 NVIDIA 文档：[GPU 硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)。


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
    "如有单位或个人对公示人员申请廉租住房保障资格有异议的，可以信件和电话的形式向市住建局举报，监督电话：5641079",
    "姓名：张三，年龄：30岁，手机：13854488452，性别：男，家庭住址：北京市海淀区西北旺",
    "张三,30岁,13854488452,男,北京市海淀区西北旺",
]

prompts = [
    "电话号码",
    "姓名，年龄，手机号码，性别，地址",
    "姓名",
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
    ind1 = text.find("\n**回答结束**\n\n")
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

我们推荐使用数据标注平台[doccano](https://github.com/doccano/doccano) 进行数据标注，本示例也打通了从标注到训练的通道，即 doccano 导出数据后可通过[doccano.py](./doccano.py)脚本轻松将数据转换为输入模型时需要的形式，实现无缝衔接。标注方法的详细介绍请参考[doccano 数据标注指南](doccano.md)。

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
    --splits 0.8 0.2 0 \
    --schema_lang ch
```


可配置参数说明：

- ``doccano_file``: 从 doccano 导出的数据标注文件。
- ``save_dir``: 训练数据的保存目录，默认存储在``data``目录下。
- ``negative_ratio``: 最大负例比例，该参数只对抽取类型任务有效，适当构造负例可提升模型效果。负例数量和实际的标签数量有关，最大负例数量 = negative_ratio * 正例数量。
- ``splits``: 划分数据集时训练集、验证集所占的比例。默认为[0.8, 0.1, 0.1]表示按照``8:1:1``的比例将数据划分为训练集、验证集和测试集。
- ``task_type``: 选择任务类型，目前只有信息抽取这一种任务。
- ``is_shuffle``: 是否对数据集进行随机打散，默认为 False。
- ``seed``: 随机种子，默认为1000.
- ``schema_lang``: 选择 schema 的语言，可选有`ch`和`en`。默认为`ch`，英文数据集请选择`en`。

备注：
- 默认情况下 [doccano.py](./doccano.py) 脚本会按照比例将数据划分为 train/dev/test 数据集
- 每次执行 [doccano.py](./doccano.py) 脚本，将会覆盖已有的同名数据文件
- 在模型训练阶段我们推荐构造一些负例以提升模型效果，在数据转换阶段我们内置了这一功能。可通过`negative_ratio`控制自动构造的负样本比例；负样本数量 = negative_ratio * 正样本数量。
- 对于从 doccano 导出的文件，默认文件中的每条数据都是经过人工正确标注的。


<a name="模型微调"></a>

#### 3.3 模型微调

推荐使用 [大模型精调](../../docs/finetune.md) 对模型进行微调。只需输入模型、数据集等就可以高效快速地进行微调和模型压缩等任务，可以一键启动多卡训练、混合精度训练、梯度累积、断点重启、日志显示等功能，并且针对训练过程的通用训练配置做了封装，比如：优化器、学习率调度等。

使用下面的命令，使用 `paddlenlp/PP-UIE-0.5B` 作为预训练模型进行模型微调，将微调后的模型保存至指定路径中。

如果在 GPU 环境中使用，可以指定 gpus 参数进行多卡训练：

```shell
cd ../../
# 返回llm目录
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
更多 sft_argument.json 配置文件说明，请参考[大模型精调](../../docs/finetune.md)


<a name="定制模型一键预测"></a>

#### 3.4 定制模型一键预测

1. 使用 PaddleNLP的高性能 predictor进行快速推理
- 内置全环节融合算子策略
- 支持 Weight Only INT8及 INT4推理，支持权重、激活、Cache KV 进行 INT8、FP8量化的推理
- 支持动态图推理和静态图推理两种方式

```shell
# llm目录下
python predict/predictor.py \
    --model_name_or_path ./checkpoints/ie_ckpts \
    --dtype float16 \
    --data_file ./application/information_extraction/data/test.json \
    --output_file ./output.json \
    --src_length  512 \
    --max_length  20 \
    --batch_size  4 \
```
更多关于 `predictor.py` 的配置参数说明，请参考[大模型推理教程](../../docs/predict/inference.md)

2. 使用 taskflow进行快速推理
`paddlenlp.Taskflow`支持装载定制模型，通过`task_path`指定模型权重文件的路径，路径下需要包含训练好的模型权重文件

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow

>>> schema = ['出发地', '目的地', '费用', '时间']
# 设定抽取目标和定制化模型权重路径
>>> my_ie = Taskflow("information_extraction", schema=schema, model='paddlenlp/PP-UIE-0.5B',precision = "float16", task_path='./checkpoints/ie_ckpts')
>>> pprint(my_ie("城市内交通费7月5日金额114广州至佛山"))
[{'出发地': [{'text': '广州'}],
  '时间': [{'text': '7月5日'}],
  '目的地': [{'text': '佛山'}],
  '费用': [{'text': '114'}]}]
```



<a name="实验指标"></a>

#### 3.5 实验指标

我们在通用测试集和医疗、新闻、对话与金融等垂类测试集上进行了实验：

<!-- <table>
<tr><th row_span='2'><th colspan='2'>金融<th colspan='2'>医疗<th colspan='2'>互联网
<tr><td><th>0-shot<th>5-shot<th>0-shot<th>5-shot<th>0-shot<th>5-shot
<tr><td>uie-base (12L768H)<td>46.43<td>70.92<td><b>71.83</b><td>85.72<td>78.33<td>81.86
<tr><td>uie-medium (6L768H)<td>41.11<td>64.53<td>65.40<td>75.72<td>78.32<td>79.68
<tr><td>uie-mini (6L384H)<td>37.04<td>64.65<td>60.50<td>78.36<td>72.09<td>76.38
<tr><td>uie-micro (4L384H)<td>37.53<td>62.11<td>57.04<td>75.92<td>66.00<td>70.22
<tr><td>uie-nano (4L312H)<td>38.94<td>66.83<td>48.29<td>76.74<td>62.86<td>72.35
<tr><td>uie-m-large (24L1024H)<td><b>49.35</b><td><b>74.55</b><td>70.50<td><b>92.66</b><td><b>78.49</b><td><b>83.02</b>
<tr><td>uie-m-base (12L768H)<td>38.46<td>74.31<td>63.37<td>87.32<td>76.27<td>80.13
</table> -->

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