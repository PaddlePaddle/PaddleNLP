# 通用信息抽取UIE

 **目录**

* [1. 功能简介](#功能简介)
* [2. 模型简介](#模型简介)
* [3. 应用场景](#应用场景)
* [4. 开箱即用](#开箱即用)
* [5. 轻定制功能](#轻定制功能)

<a name="模型简介"></a>

## 1. 功能简介

开放域信息抽取是信息抽取的一种全新范式，主要思想是减少人工参与，利用单一模型支持多种类型的开放抽取任务，用户可以使用自然语言自定义抽取目标，在实体、关系类别等未定义的情况下抽取输入文本中的信息片段。

<a name="模型简介"></a>

## 2. 模型简介

[Universal Information Extraction(UIE)](https://arxiv.org/pdf/2203.12277.pdf)开放域信息抽取的统一框架。本示例基于这篇工作的prompt设计思想，提供了以ERNIE为底座的阅读理解型信息抽取模型，用于关键数据抽取。同时，针对不同场景，支持通过构造小样本数据来优化模型效果，快速适配特定的关键信息配置。

#### UIE的优势

- **使用简单**：用户可以使用自然语言自定义抽取目标，无需训练即可统一抽取输入文本中的对应信息。**实现开箱即用，并满足各类信息抽取需求**。

- **降本增效**：以往的信息抽取技术需要大量标注数据才能保证信息抽取的效果，为了提高开发过程中的开发效率，减少不必要的重复工作时间，开放域信息抽取可以实现零样本（zero-shot）或者少样本（few-shot）抽取，**大幅度降低标注数据依赖，在降低成本的同时，还提升了效果**。

- **效果领先**：开放域信息抽取在多种场景，多种任务上，均有不俗的表现。

<a name="应用场景"></a>

## 3. 应用场景

UIE可以从自然语言文本中，抽取出结构化的关键字段信息，以下是UIE在医疗、金融等领域的应用示例。

#### 医疗

在医疗场景下，医生需要从病历中快速重要信息以便分析病人病情，UIE可将专病信息进行结构化处理，快速抽取病历内容中的检查内容、炎症部位、结节大小等信息，大幅提升医务人员对患者的诊断效率以及准确率，协助医务人员高效诊断病情。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/166708474-8f05bdba-143f-4d11-8bd5-8ce26ab7c987.png height=300 hspace='10'/>
    <p>图1 医疗场景示例<p/>
</div>

#### 金融

在金融场景下，工作人员想要整理一份资产评估证明，UIE可以根据抽取内容自定义抽取目标，大幅提升工作人员的工作效率及准确率，协助工作人员对数据进行整理和调研。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/166708694-e2671e29-3c02-4e29-9823-9fbdfd117eb6.png height=400 hspace='10'/>
    <p>图2 金融场景示例<p/>
</div>

<a name="开箱即用"></a>

## 4. 开箱即用

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

## 5. 轻定制功能

对于简单的抽取目标可以直接使用```paddlenlp.Taskflow```实现零样本（zero-shot）抽取。对于复杂目标可以标注少量数据进行模型微调以进一步提升效果。我们在互联网、医疗、金融三大垂类自建测试集上进行了实验：

<table>
<tr><th row_span='2'><th colspan='2'>互联网<th colspan='2'>医疗<th colspan='2'>金融
<tr><td><th>0-shot<th>5-shot<th>0-shot<th>5-shot<th>0-shot<th>5-shot
<tr><td>uie-tiny<td>75.92<td>78.45<td>63.34<td>74.65<td>42.03<td>65.78
<tr><td>uie-base<td>80.13<td>81.53<td>66.71<td>79.94<td>41.29<td>70.91
</table>

0-shot表示无训练数据直接通过```paddlenlp.Taskflow```进行预测，5-shot表示基于5条标注数据进行模型微调。

#### 代码结构

```shell
.
├── utils.py          # 数据处理工具
├── model.py          # 模型组网脚本
├── doccano.py        # 数据标注脚本
├── train.py          # 模型训练脚本
├── evaluate.py       # 模型评估脚本
└── README.md
```

#### 数据标注

我们推荐使用数据标注平台[doccano](https://github.com/doccano/doccano) 进行数据标注，本案例也打通了从标注到训练的通道，即doccano导出数据后可通过[doccano.py](./doccano.py)脚本轻松将数据转换为输入模型时需要的形式，实现无缝衔接。为达到这个目的，您需要按以下标注规则在doccano平台上标注数据：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/164374314-9beea9ad-08ed-42bc-bbbc-9f68eb8a40ee.png />
    <p>图3 数据标注样例图<p/>
</div>

- 在doccano平台上，定义实体标签类别和关系标签类别，上例中需要定义的实体标签有`作品名`、`机构名`和`人物名`，关系标签有`出版社名称`和`作者`。
- 使用以上定义的标签开始标注数据，图2展示了一个标注样例。
- 当标注完成后，在 doccano 平台上导出 `jsonl` 形式的文件，并将其重命名为 `doccano.json` 后，放入 `./data` 目录下。
- 通过 [doccano.py](./doccano.py) 脚本进行数据形式转换，然后便可以开始进行相应模型训练。

```shell
python doccano.py \
    --doccano_file ./data/doccano.json \
    --save_dir ./data/ext_data \
    --negative_ratio 5
```

**备注：**
- 默认情况下 [doccano.py](./doccano.py) 脚本会按照比例将数据划分为 train/dev/test 数据集
- 每次执行 [doccano.py](./doccano.py) 脚本，将会覆盖已有的同名数据文件
- 在模型训练阶段我们推荐构造一些负例以提升模型效果，在数据转换阶段我们内置了这一功能。可通过`negative_ratio`控制自动构造的负样本比例；负样本数量 = negative_ratio * 正样本数量。

#### 模型训练

通过运行以下命令进行自定义UIE模型训练：

```shell
python train.py \
    --train_path "./data/ext_data/train.txt" \
    --dev_path "./data/ext_data/dev.txt" \
    --save_dir "./checkpoint" \
    --learning_rate 1e-5 \
    --batch_size 16 \
    --max_seq_len 512 \
    --num_epochs 50 \
    --model "uie-base" \
    --seed 1000 \
    --logging_steps 10 \
    --valid_steps 100 \
    --device "gpu"
```

#### 模型评估

通过运行以下命令进行模型评估：

```shell
python evaluate.py \
    --model_path "./checkpoint/model_best/model_state.pdparams" \
    --test_path "./data/ext_data/test.txt" \
    --batch_size 16 \
    --max_seq_len 512
```

#### Taskflow装载定制模型

通过`schema`自定义抽取目标，`task_path`指定使用标注数据训练的UIE模型。

```python
from paddlenlp import Taskflow

schema = [{"作品名": ["作者", "出版社名称"]}]

# 为任务实例设定抽取目标和定制化模型权重路径
my_ie = Taskflow("information_extraction", schema=schema, task_path='./checkpoint/model_best')
```

## References
- **[Unified Structure Generation for Universal Information Extraction](https://arxiv.org/pdf/2203.12277.pdf)**
