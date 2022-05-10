# 通用信息抽取 Universal Information Extraction (UIE)

 **目录**

* [1. 模型简介](#模型简介)
* [2. 应用场景](#应用场景)
* [3. 开箱即用](#开箱即用)
* [4. 轻定制功能](#轻定制功能)

<a name="模型简介"></a>

## 1. 模型简介

[Universal Information Extraction (UIE)](https://arxiv.org/pdf/2203.12277.pdf)：Yaojie Lu等人提出了开放域信息抽取的统一框架，这一框架在实体抽取、关系抽取、事件抽取、情感分析等任务上都有着良好的泛化效果。本示例基于这篇工作的prompt设计思想，提供了以ERNIE为底座的阅读理解型信息抽取模型，用于关键信息抽取。同时，针对不同场景，支持通过构造小样本数据来优化模型效果，快速适配特定的关键信息配置。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167236006-66ed845d-21b8-4647-908b-e1c6e7613eb1.png height=400 hspace='10'/>
    <p>图1 模型结构图 <p/>
</div>

#### UIE的优势

- **使用简单**：用户可以使用自然语言自定义抽取目标，无需训练即可统一抽取输入文本中的对应信息。**实现开箱即用，并满足各类信息抽取需求**。

- **降本增效**：以往的信息抽取技术需要大量标注数据才能保证信息抽取的效果，为了提高开发过程中的开发效率，减少不必要的重复工作时间，开放域信息抽取可以实现零样本（zero-shot）或者少样本（few-shot）抽取，**大幅度降低标注数据依赖，在降低成本的同时，还提升了效果**。

- **效果领先**：开放域信息抽取在多种场景，多种任务上，均有不俗的表现。

<a name="应用场景"></a>

## 2. 应用场景

UIE可以从自然语言文本中，抽取出结构化的关键字段信息，以下是UIE在医疗、金融等领域的应用示例。

#### 医疗

在医疗场景下，医生需要从病历中快速重要信息以便分析病人病情，UIE可将专病信息进行结构化处理，快速抽取病历内容中的检查内容、炎症部位、结节大小等信息，大幅提升医务人员对患者的诊断效率以及准确率，协助医务人员高效诊断病情。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/166708474-8f05bdba-143f-4d11-8bd5-8ce26ab7c987.png height=300 hspace='10'/>
    <p>图2 医疗场景示例<p/>
</div>

#### 金融

在金融场景下，工作人员想要整理一份资产评估证明，UIE可以根据抽取内容自定义抽取目标，大幅提升工作人员的工作效率及准确率，协助工作人员对数据进行整理和调研。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/166708694-e2671e29-3c02-4e29-9823-9fbdfd117eb6.png height=400 hspace='10'/>
    <p>图3 金融场景示例<p/>
</div>

<a name="开箱即用"></a>

## 3. 开箱即用

```paddlenlp.Taskflow```提供通用信息抽取、评价观点抽取等能力，可抽取多种类型的信息，包括但不限于命名实体识别（如人名、地名、机构名等）、关系（如电影的导演、歌曲的发行时间等）、事件（如某路口发生车祸、某地发生地震等）、以及评价维度、观点词、情感倾向等信息。用户可以使用自然语言自定义抽取目标，无需训练即可统一抽取输入文本中的对应信息。**实现开箱即用，并满足各类信息抽取需求**

```python
>>> from paddlenlp import Taskflow

>>> schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
>>> ie = Taskflow('information_extraction', schema=schema)
>>> ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")
[{'时间': [{'text': '2月8日上午', 'start': 0, 'end': 6, 'probability': 0.9907337794563702}], '选手': [{'text': '谷爱凌', 'start': 28, 'end': 31, 'probability': 0.8914310308098763}], '赛事名称': [{'text': '北京冬奥会自由式滑雪女子大跳台决赛', 'start': 6, 'end': 23, 'probability': 0.8944207860063003}]}]
```

更多不同任务的使用方法请参考[Taskflow信息抽取](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/taskflow.md#%E4%BF%A1%E6%81%AF%E6%8A%BD%E5%8F%96)

<a name="轻定制功能"></a>

## 4. 轻定制功能

对于简单的抽取目标可以直接使用```paddlenlp.Taskflow```实现零样本（zero-shot）抽取，对于细分场景我们推荐使用轻定制功能（标注少量数据进行模型微调）以进一步提升效果。下面通过`报销工单信息抽取`的例子展示如何通过5条训练数据进行UIE模型微调。

#### 代码结构

```shell
.
├── utils.py          # 数据处理工具
├── model.py          # 模型组网脚本
├── doccano.py        # 数据标注脚本
├── doccano.md        # 数据标注文档
├── finetune.py       # 模型微调脚本
├── evaluate.py       # 模型评估脚本
└── README.md
```

#### 数据标注

我们推荐使用数据标注平台[doccano](https://github.com/doccano/doccano) 进行数据标注，本示例也打通了从标注到训练的通道，即doccano导出数据后可通过[doccano.py](./doccano.py)脚本轻松将数据转换为输入模型时需要的形式，实现无缝衔接。

原始数据示例：

```text
深大到双龙28块钱4月24号交通费
```

抽取的目标(schema)为：

```python
schema = ['出发地', '目的地', '费用', '时间']
```

标注步骤如下：

- 在doccano平台上，创建一个类型为``序列标注``的标注项目。
- 定义实体标签类别，上例中需要定义的实体标签有``出发地``、``目的地``、``费用``和``时间``。
- 使用以上定义的标签开始标注数据，下面展示了一个doccano标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167336891-afef1ad5-8777-456d-805b-9c65d9014b80.png height=100 hspace='10'/>
    <p>图4 标注示例图<p/>
</div>

- 标注完成后，在doccano平台上导出文件，并将其重命名为``doccano_ext.json``后，放入``./data``目录下。

- 这里我们提供预先标注好的文件[doccano_ext.json](https://bj.bcebos.com/paddlenlp/datasets/uie/doccano_ext.json)，可直接下载并放入`./data`目录。执行以下脚本进行数据转换，执行后会在`./data`目录下生成训练/验证/测试集文件。

```shell
python doccano.py \
    --doccano_file ./data/doccano_ext.json \
    --task_type "ext" \
    --save_dir ./data \
    --splits 0.1 0.9 0
```

可配置参数说明：

- ``doccano_file``: 从doccano导出的数据标注文件。
- ``save_dir``: 训练数据的保存目录，默认存储在``data``目录下。
- ``negative_ratio``: 负样本与正样本的比例，该参数只对抽取类型任务有效。使用负样本策略可提升模型效果，负样本数量 = negative_ratio * 正样本数量。
- ``splits``: 划分数据集时训练集、验证集所占的比例。默认为[0.8, 0.1, 0.1]表示按照``8:1:1``的比例将数据划分为训练集、验证集和测试集。
- ``task_type``: 选择任务类型，可选有抽取和分类两种类型的任务。
- ``options``: 指定分类任务的类别标签，该参数只对分类类型任务有效。
- ``prompt_prefix``: 声明分类任务的prompt前缀信息，该参数只对分类类型任务有效。
- ``is_shuffle``: 是否对数据集进行随机打散，默认为True。
- ``seed``: 随机种子，默认为1000.

更多不同类型任务（关系抽取、事件抽取、评价观点抽取等）的标注规则及参数说明，请参考[doccano数据标注指南](doccano.md)。

#### 模型微调

通过运行以下命令进行模型微调：

```shell
python finetune.py \
    --train_path "./data/train.txt" \
    --dev_path "./data/dev.txt" \
    --save_dir "./checkpoint" \
    --learning_rate 1e-5 \
    --batch_size 16 \
    --max_seq_len 512 \
    --num_epochs 100 \
    --model "uie-base" \
    --seed 1000 \
    --logging_steps 10 \
    --valid_steps 100 \
    --device "gpu"
```

可配置参数说明：

- `train_path`: 训练集文件路径。
- `dev_path`: 验证集文件路径。
- `save_dir`: 模型存储路径，默认为`./checkpoint`。
- `learning_rate`: 学习率，默认为1e-5。
- `batch_size`: 批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数，默认为16。
- `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
- `num_epochs`: 训练轮数，默认为100。
- `model`: 选择模型，程序会基于选择的模型进行模型微调，可选有`uie-base`和`uie-tiny`。
- `seed`: 随机种子，默认为1000.
- `logging_steps`: 日志打印的间隔steps数，默认10。
- `valid_steps`: evaluate的间隔steps数，默认100。
- `device`: 选用什么设备进行训练，可选cpu或gpu。

#### 模型评估

通过运行以下命令进行模型评估：

```shell
python evaluate.py \
    --model_path "./checkpoint/model_best/model_state.pdparams" \
    --test_path "./data/dev.txt" \
    --model "uie-base" \
    --batch_size 16 \
    --max_seq_len 512
```

可配置参数说明：

- `model_path`: 进行评估的模型权重文件。
- `test_path`: 进行评估的测试集文件。
- `batch_size`: 批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数，默认为16。
- `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
- `model`: 进行效果评估的模型类型，可选有`uie-base`和`uie-tiny`。

#### 定制模型一键预测

`paddlenlp.Taskflow`装载定制模型，通过`task_path`指定模型权重文件的路径，路径下需要包含训练好的模型权重文件`model_state.pdparams`。

```python
>>> from paddlenlp import Taskflow

>>> schema = ['出发地', '目的地', '费用', '时间']
# 设定抽取目标和定制化模型权重路径
>>> my_ie = Taskflow("information_extraction", schema=schema, task_path='./checkpoint/model_best')
>>> my_ie("城市内交通费7月5日金额114广州至佛山")
[{'出发地': [{'text': '广州', 'start': 15, 'end': 17, 'probability': 0.9975287467835301}], '目的地': [{'text': '佛山', 'start': 18, 'end': 20, 'probability': 0.9998511131226735}], '费用': [{'text': '114', 'start': 12, 'end': 15, 'probability': 0.9994474579292856}], '时间': [{'text': '7月5日', 'start': 6, 'end': 10, 'probability': 0.9999476678061399}]}]
```

#### Few-Shot实验

我们在互联网、医疗、金融三大垂类自建测试集上进行了实验：

<table>
<tr><th row_span='2'><th colspan='2'>互联网<th colspan='2'>医疗<th colspan='2'>金融
<tr><td><th>0-shot<th>5-shot<th>0-shot<th>5-shot<th>0-shot<th>5-shot
<tr><td>uie-tiny<td>75.92<td>78.45<td>63.34<td>74.65<td>42.03<td>65.78
<tr><td>uie-base<td>80.13<td>81.53<td>66.71<td>79.94<td>41.29<td>70.91
</table>

0-shot表示无训练数据直接通过```paddlenlp.Taskflow```进行预测，5-shot表示基于5条标注数据进行模型微调。实验表明UIE在垂类场景可以通过少量数据（few-shot）进一步提升效果。

## References
- **[Unified Structure Generation for Universal Information Extraction](https://arxiv.org/pdf/2203.12277.pdf)**
