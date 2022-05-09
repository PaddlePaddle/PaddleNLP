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

<a name="项目创建"></a>

## 2. 项目创建

#### 抽取任务

创建项目时选择``序列标注``任务，并勾选``Use Relation Labeling``

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167249142-44885510-51dc-4359-8054-9c89c9633700.png height=180 hspace='10'/>
</div>

#### 分类任务

创建项目时选择``文本分类``任务

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167249258-48fb4f0c-f68c-4c9a-ab84-5c555ddcf427.png height=180 hspace='10'/>
</div>

<a name="数据上传"></a>

## 3. 数据上传

上传的文件为txt格式，每一行为一条待标注文本，示例：

```text
2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌
第十四届全运会在西安举办
```

上传数据类型选择``TextLine``

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167247061-d5795c26-7a6f-4cdb-88ad-107a3cae5446.png height=300 hspace='10'/>
</div>

<a name="标签构建"></a>

## 4. 标签构建

#### 抽取任务

添加Span类型标签

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248034-afa3f637-65c5-4038-ada0-344ffbd776a2.png height=300 hspace='10'/>
</div>

添加Relation类型标签

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248307-916c77f6-bf80-4d6b-aa71-30c719f68257.png height=260 hspace='10'/>
</div>

#### 分类任务

添加类别标签

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167249484-2b5f6338-8a91-48f3-8d56-edc2b26b41d7.png height=160 hspace='10'/>
</div>

<a name="任务标注"></a>

## 5. 任务标注

#### 命名实体识别

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248557-f1da3694-1063-465a-be9a-1bb811949530.png height=200 hspace='10'/>
</div>

#### 关系抽取

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248502-16a87902-3878-4432-b5b8-9808bd8d4de5.png height=200 hspace='10'/>
</div>

#### 事件抽取

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248793-138a1e37-43c9-4933-bf89-f3ac7228bf9c.png height=200 hspace='10'/>
</div>

#### 评价观点抽取

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167249035-6c16c68e-d94e-4a37-8489-111ee65924a3.png height=190 hspace='10'/>
</div>


#### 文本分类

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167249572-48a04c4f-ab79-47ef-a138-798f4243f520.png height=100 hspace='10'/>
</div>

<a name="数据导出"></a>

## 6. 数据导出

#### 抽取任务

选择导出的文件类型为``JSONL(relation)``，导出数据示例：

```text
{
    "id": 36,
    "text": "2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌",
    "relations": [],
    "entities": [
        {
            "id": 47,
            "start_offset": 0,
            "end_offset": 6,
            "label": "时间"
        },
        {
            "id": 48,
            "start_offset": 6,
            "end_offset": 23,
            "label": "赛事名称"
        },
        {
            "id": 49,
            "start_offset": 28,
            "end_offset": 31,
            "label": "选手"
        },
        {
            "id": 50,
            "start_offset": 32,
            "end_offset": 39,
            "label": "得分"
        }
    ]
}
```

#### 分类任务

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

<a name="数据转换"></a>

## 7.数据转换

抽取任务：

- 当标注完成后，在 doccano 平台上导出 `JSONL(relation)` 形式的文件，并将其重命名为 `doccano_ext.json` 后，放入 `./data` 目录下。
- 通过 [doccano.py](./doccano.py) 脚本进行数据形式转换，然后便可以开始进行相应模型训练。

```shell
python doccano.py \
    --doccano_file ./data/doccano_ext.json \
    --task_type "ext" \
    --save_dir ./data \
    --negative_ratio 5
```

分类任务：

- 当标注完成后，在 doccano 平台上导出 `JSON` 形式的文件，并将其重命名为 `doccano_cls.json` 后，放入 `./data` 目录下。
- 在数据转换阶段，我们会自动构造用于模型训练需要的prompt信息。例如句子级情感分类中，prompt为``情感倾向[正向,负向]``，可以通过`prompt_prefix`和`options`参数进行声明。
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

备注：
- 默认情况下 [doccano.py](./doccano.py) 脚本会按照比例将数据划分为 train/dev/test 数据集
- 每次执行 [doccano.py](./doccano.py) 脚本，将会覆盖已有的同名数据文件
- 在模型训练阶段我们推荐构造一些负例以提升模型效果，在数据转换阶段我们内置了这一功能。可通过`negative_ratio`控制自动构造的负样本比例；负样本数量 = negative_ratio * 正样本数量。

## References
- **[doccano](https://github.com/doccano/doccano)**
