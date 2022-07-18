# doccano

 **目录**

* [1. 安装](#安装)
* [2. 项目创建](#项目创建)
* [3. 数据上传](#数据上传)
* [4. 标签构建](#标签构建)
* [5. 任务标注](#任务标注)
* [6. 数据导出](#数据导出)
* [7. 数据转换](#数据转换)

<a name="安装"></a>

## 1. 安装

参考[doccano官方文档](https://github.com/doccano/doccano) 完成doccano的安装与初始配置。

**以下标注示例用到的环境配置：**

- doccano 1.6.2

<a name="项目创建"></a>

## 2. 项目创建

UIE支持抽取与分类两种类型的任务，根据实际需要创建一个新的项目：

#### 2.1 抽取式任务项目创建

创建项目时选择**序列标注**任务，并勾选**Allow overlapping entity**及**Use relation Labeling**。适配**命名实体识别、关系抽取、事件抽取、评价观点抽取**等任务。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167249142-44885510-51dc-4359-8054-9c89c9633700.png height=230 hspace='15'/>
</div>

#### 2.2 分类式任务项目创建

创建项目时选择**文本分类**任务。适配**文本分类、句子级情感倾向分类**等任务。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167249258-48fb4f0c-f68c-4c9a-ab84-5c555ddcf427.png height=230 hspace='15'/>
</div>

<a name="数据上传"></a>

## 3. 数据上传

上传的文件为txt格式，每一行为一条待标注文本，示例:

```text
2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌
第十四届全运会在西安举办
```

上传数据类型**选择TextLine**:

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167247061-d5795c26-7a6f-4cdb-88ad-107a3cae5446.png height=300 hspace='15'/>
</div>

**NOTE**：doccano支持`TextFile`、`TextLine`、`JSONL`和`CoNLL`四种数据上传格式，UIE定制训练中**统一使用TextLine**这一文件格式，即上传的文件需要为txt格式，且在数据标注时，该文件的每一行待标注文本显示为一页内容。

<a name="标签构建"></a>

## 4. 标签构建

#### 4.1 构建抽取式任务标签

抽取式任务包含**Span**与**Relation**两种标签类型，Span指**原文本中的目标信息片段**，如实体识别中某个类型的实体，事件抽取中的触发词和论元；Relation指**原文本中Span之间的关系**，如关系抽取中两个实体（Subject&Object）之间的关系，事件抽取中论元和触发词之间的关系。

Span类型标签构建示例:

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248034-afa3f637-65c5-4038-ada0-344ffbd776a2.png height=300 hspace='15'/>
</div>

Relation类型标签构建示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248307-916c77f6-bf80-4d6b-aa71-30c719f68257.png height=260 hspace='16'/>
</div>

#### 4.2 构建分类式任务标签

添加分类类别标签：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167249484-2b5f6338-8a91-48f3-8d56-edc2b26b41d7.png height=160 hspace='15'/>
</div>

<a name="任务标注"></a>

## 5. 任务标注

#### 5.1 命名实体识别

命名实体识别（Named Entity Recognition，简称NER），是指识别文本中具有特定意义的实体。在开放域信息抽取中，**抽取的类别没有限制，用户可以自己定义**。

标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248557-f1da3694-1063-465a-be9a-1bb811949530.png height=200 hspace='20'/>
</div>

示例中定义了`时间`、`选手`、`赛事名称`和`得分`四种Span类型标签。

```text
schema = [
    '时间',
    '选手',
    '赛事名称',
    '得分'
]
```

#### 5.2 关系抽取

关系抽取（Relation Extraction，简称RE），是指从文本中识别实体并抽取实体之间的语义关系，即抽取三元组（实体一，关系类型，实体二）。

标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248502-16a87902-3878-4432-b5b8-9808bd8d4de5.png height=200 hspace='20'/>
</div>

示例中定义了`作品名`、`人物名`和`时间`三种Span类型标签，以及`歌手`、`发行时间`和`所属专辑`三种Relation标签。Relation标签**由Subject对应实体指向Object对应实体**。

该标注示例对应的schema为：

```text
schema = {
    '作品名': [
        '歌手',
        '发行时间',
        '所属专辑'
    ]
}
```

#### 5.3 事件抽取

事件抽取 (Event Extraction, 简称EE)，是指从自然语言文本中抽取事件并识别事件类型和事件论元的技术。UIE所包含的事件抽取任务，是指根据已知事件类型，抽取该事件所包含的事件论元。

标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248793-138a1e37-43c9-4933-bf89-f3ac7228bf9c.png height=200 hspace='20'/>
</div>

示例中定义了`地震触发词`（触发词）、`等级`（事件论元）和`时间`（事件论元）三种Span标签，以及`时间`和`震级`两种Relation标签。触发词标签**统一格式为`XX触发词`**，`XX`表示具体事件类型，上例中的事件类型是`地震`，则对应触发词为`地震触发词`。Relation标签**由触发词指向对应的事件论元**。

该标注示例对应的schema为：

```text
schema = {
    '地震触发词': [
        '时间',
        '震级'
    ]
}
```

#### 5.4 评价观点抽取

评论观点抽取，是指抽取文本中包含的评价维度、观点词。

标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167249035-6c16c68e-d94e-4a37-8489-111ee65924a3.png height=190 hspace='20'/>
</div>

示例中定义了`评价维度`和`观点词`两种Span标签，以及`观点词`一种Relation标签。Relation标签**由评价维度指向观点词**。

该标注示例对应的schema为：

```text
schema = {
    '评价维度': '观点词'
}
```

#### 5.5 句子级分类任务

标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167249572-48a04c4f-ab79-47ef-a138-798f4243f520.png height=100 hspace='20'/>
</div>

示例中定义了`正向`和`负向`两种类别标签对文本的情感倾向进行分类。

该标注示例对应的schema为：

```text
schema = '情感倾向[正向，负向]'
```

#### 5.6 实体/评价维度级分类任务

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/172628328-878923d7-8c5d-4667-a0e2-b92bce89b47c.png height=200 hspace='20'/>
</div>

标注示例：

示例中定义了`评价维度##正向`，`评价维度##负向`和`观点词`三种Span标签以及`观点词`一种Relation标签。其中，`##`是实体类别/评价维度与分类标签的分隔符（可通过doccano.py中的separator参数自定义）。

该标注示例对应的schema为：

```text
schema = {
    '评价维度': [
        '观点词',
        '情感倾向[正向，负向]'
    ]
}
```

<a name="数据导出"></a>

## 6. 数据导出

#### 6.1 导出抽取式和实体/评价维度级分类任务数据

选择导出的文件类型为``JSONL(relation)``，导出数据示例：

```text
{
    "id": 38,
    "text": "百科名片你知道我要什么，是歌手高明骏演唱的一首歌曲，1989年发行，收录于个人专辑《丛林男孩》中",
    "relations": [
        {
            "id": 20,
            "from_id": 51,
            "to_id": 53,
            "type": "歌手"
        },
        {
            "id": 21,
            "from_id": 51,
            "to_id": 55,
            "type": "发行时间"
        },
        {
            "id": 22,
            "from_id": 51,
            "to_id": 54,
            "type": "所属专辑"
        }
    ],
    "entities": [
        {
            "id": 51,
            "start_offset": 4,
            "end_offset": 11,
            "label": "作品名"
        },
        {
            "id": 53,
            "start_offset": 15,
            "end_offset": 18,
            "label": "人物名"
        },
        {
            "id": 54,
            "start_offset": 42,
            "end_offset": 46,
            "label": "作品名"
        },
        {
            "id": 55,
            "start_offset": 26,
            "end_offset": 31,
            "label": "时间"
        }
    ]
}
```

标注数据保存在同一个文本文件中，每条样例占一行且存储为``json``格式，其包含以下字段
- ``id``: 样本在数据集中的唯一标识ID。
- ``text``: 原始文本数据。
- ``entities``: 数据中包含的Span标签，每个Span标签包含四个字段：
    - ``id``: Span在数据集中的唯一标识ID。
    - ``start_offset``: Span的起始token在文本中的下标。
    - ``end_offset``: Span的结束token在文本中下标的下一个位置。
    - ``label``: Span类型。
- ``relations``: 数据中包含的Relation标签，每个Relation标签包含四个字段：
    - ``id``: (Span1, Relation, Span2)三元组在数据集中的唯一标识ID，不同样本中的相同三元组对应同一个ID。
    - ``from_id``: Span1对应的标识ID。
    - ``to_id``: Span2对应的标识ID。
    - ``type``: Relation类型。

#### 6.2 导出句子级分类任务数据

选择导出的文件类型为``JSONL``，导出数据示例：

```text
{
    "id": 41,
    "data": "大年初一就把车前保险杠给碰坏了，保险杠和保险公司 真够倒霉的，我决定步行反省。",
    "label": [
        "负向"
    ]
}
```

标注数据保存在同一个文本文件中，每条样例占一行且存储为``json``格式，其包含以下字段
- ``id``: 样本在数据集中的唯一标识ID。
- ``data``: 原始文本数据。
- ``label``: 文本对应类别标签。

<a name="数据转换"></a>

## 7.数据转换

该章节详细说明如何通过`doccano.py`脚本对doccano平台导出的标注数据进行转换，一键生成训练/验证/测试集。

#### 7.1 抽取式任务数据转换

- 当标注完成后，在 doccano 平台上导出 `JSONL(relation)` 形式的文件，并将其重命名为 `doccano_ext.json` 后，放入 `./data` 目录下。
- 通过 [doccano.py](./doccano.py) 脚本进行数据形式转换，然后便可以开始进行相应模型训练。

```shell
python doccano.py \
    --doccano_file ./data/doccano_ext.json \
    --task_type "ext" \
    --save_dir ./data \
    --negative_ratio 5
```

#### 7.2 句子级分类任务数据转换

- 当标注完成后，在 doccano 平台上导出 `JSON` 形式的文件，并将其重命名为 `doccano_cls.json` 后，放入 `./data` 目录下。
- 在数据转换阶段，我们会自动构造用于模型训练的prompt信息。例如句子级情感分类中，prompt为``情感倾向[正向,负向]``，可以通过`prompt_prefix`和`options`参数进行声明。
- 通过 [doccano.py](./doccano.py) 脚本进行数据形式转换，然后便可以开始进行相应模型训练。

```shell
python doccano.py \
    --doccano_file ./data/doccano_cls.json \
    --task_type "cls" \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --prompt_prefix "情感倾向" \
    --options "正向" "负向"
```

#### 7.3 实体/评价维度级分类任务数据转换

- 当标注完成后，在 doccano 平台上导出 `JSONL(relation)` 形式的文件，并将其重命名为 `doccano_ext.json` 后，放入 `./data` 目录下。
- 在数据转换阶段，我们会自动构造用于模型训练的prompt信息。例如评价维度级情感分类中，prompt为``XXX的情感倾向[正向,负向]``，可以通过`prompt_prefix`和`options`参数进行声明。
- 通过 [doccano.py](./doccano.py) 脚本进行数据形式转换，然后便可以开始进行相应模型训练。

```shell
python doccano.py \
    --doccano_file ./data/doccano_ext.json \
    --task_type "ext" \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --prompt_prefix "情感倾向" \
    --options "正向" "负向" \
    --separator "##"
```

可配置参数说明：

- ``doccano_file``: 从doccano导出的数据标注文件。
- ``save_dir``: 训练数据的保存目录，默认存储在``data``目录下。
- ``negative_ratio``: 最大负例比例，该参数只对抽取类型任务有效，适当构造负例可提升模型效果。负例数量和实际的标签数量有关，最大负例数量 = negative_ratio * 正例数量。该参数只对训练集有效，默认为5。为了保证评估指标的准确性，验证集和测试集默认构造全负例。
- ``splits``: 划分数据集时训练集、验证集所占的比例。默认为[0.8, 0.1, 0.1]表示按照``8:1:1``的比例将数据划分为训练集、验证集和测试集。
- ``task_type``: 选择任务类型，可选有抽取和分类两种类型的任务。
- ``options``: 指定分类任务的类别标签，该参数只对分类类型任务有效。默认为["正向", "负向"]。
- ``prompt_prefix``: 声明分类任务的prompt前缀信息，该参数只对分类类型任务有效。默认为"情感倾向"。
- ``is_shuffle``: 是否对数据集进行随机打散，默认为True。
- ``seed``: 随机种子，默认为1000.
- ``separator``: 实体类别/评价维度与分类标签的分隔符，该参数只对实体/评价维度级分类任务有效。默认为"##"。

备注：
- 默认情况下 [doccano.py](./doccano.py) 脚本会按照比例将数据划分为 train/dev/test 数据集
- 每次执行 [doccano.py](./doccano.py) 脚本，将会覆盖已有的同名数据文件
- 在模型训练阶段我们推荐构造一些负例以提升模型效果，在数据转换阶段我们内置了这一功能。可通过`negative_ratio`控制自动构造的负样本比例；负样本数量 = negative_ratio * 正样本数量。
- 对于从doccano导出的文件，默认文件中的每条数据都是经过人工正确标注的。

## References
- **[doccano](https://github.com/doccano/doccano)**
