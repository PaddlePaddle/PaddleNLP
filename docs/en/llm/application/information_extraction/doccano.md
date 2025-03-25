# doccano

**Table of Contents**

* [1. Installation](#安装)
* [2. Project Creation](#项目创建)
* [3. Data Upload](#数据上传)
* [4. Label Construction](#标签构建)
* [5. Task Annotation](#任务标注)
* [6. Data Export](#数据导出)
* [7. Data Conversion](#数据转换)

<a name="安装"></a>

## 1. Installation

Refer to the [doccano official documentation](https://github.com/doccano/doccano) to complete installation and initial configuration.

**Environment configuration for the following annotation examples:**

- doccano 1.6.2

<a name="项目创建"></a>

## 2. Project Creation

PP-UIE supports extraction tasks. Create a new project according to actual needs:

#### 2.1 Extraction Task Project Creation

When creating a project, select **Sequence Labeling** task, and check **Allow overlapping entity** and **Use relation Labeling**. Adapts to **Named Entity Recognition, Relation Extraction, Event Extraction**, etc.

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167249142-44885510-51dc-4359-8054-9c89c9633700.png height=230 hspace='15'/>
</div>

<a name="数据上传"></a>

## 3. Data Upload

Upload files in txt format, with each line containing one text to be annotated. Example:

```text
2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌
第十四届全运会在西安举办
```

Select **TextLine** as the data type:

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167247061-d5795c26-7a6f-4cdb-88ad-107a3cae5446.png height=300 hspace='15'/>
</div>

**NOTE**: doccano supports `TextFile`, `TextLine`, `JSONL` and `CoNLL`
In PP-UIE's custom training, there are four data formats, and **TextLine is uniformly used** for annotation. This means uploaded files must be in txt format, where each line of text in the file is displayed as one page during data annotation.

<a name="Label Construction"></a>

## 4. Label Construction

#### 4.1 Constructing Extraction Task Labels

Extraction tasks include two label types: **Span** and **Relation**. Span refers to **target information fragments in the original text**, such as entities in NER or triggers/arguments in event extraction. Relation denotes **relationships between Spans in the text**, like relationships between entities (Subject & Object) in relation extraction, or relationships between arguments and triggers in event extraction.

Span-type label construction example:

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248034-afa3f637-65c5-4038-ada0-344ffbd776a2.png height=300 hspace='15'/>
</div>

Relation-type label construction example:

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248307-916c77f6-bf80-4d6b-aa71-30c719f68257.png height=260 hspace='16'/>
</div>

## 5. Task Annotation

#### 5.1 Named Entity Recognition

Named Entity Recognition (NER) identifies entities with specific meanings in text. In open-domain information extraction, **the categories are not limited and can be user-defined**.

Annotation example:

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248557-f1da3694-1063-465a-be9a-1bb811949530.png height=200 hspace='20'/>
</div>

This example defines four Span-type labels: `Time`, `Player`, `Event Name`, and `Score`.

```text
schema = [
    '时间',
    '选手',
    '赛事名称',
    '得分'
]
```

#### 5.2 Relation Extraction

Relation Extraction (RE) identifies entities and extracts semantic relationships between them, i.e., extracting triples (Entity1, RelationType, Entity2).

Annotation example:

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248502-16a87902-3878-4432-b5b8-9808bd8d4de5.png height=200 hspace='20'/>
</div>

This example defines three Span-type labels: `Work Name`, `Person Name`, and `Time`, along with three Relation labels: `Singer`, `Release Date`, and `Album`. Relation labels **go from Subject entities to Object entities**.

The corresponding schema for this annotation example is:
```text
schema = {
    'Track name': [
        'Artist',
        'Release date',
        'Album'
    ]
}
```

#### 5.3 Event Extraction

Event Extraction (EE) refers to the technique of extracting events from natural language text and identifying event types and arguments. The event extraction task included in UIE involves extracting event arguments based on predefined event types.

Annotation example:

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248793-138a1e37-43c9-4933-bf89-f3ac7228bf9c.png height=200 hspace='20'/>
</div>

This example defines three Span labels: `earthquake trigger word` (trigger), `magnitude` (event argument), and `time` (event argument), as well as two Relation labels: `time` and `magnitude`. The trigger label format must follow `XX trigger word` where `XX` represents the specific event type. In this example, the event type is `earthquake`, thus the trigger is labeled `earthquake trigger word`. Relation labels point from triggers to corresponding event arguments.

The corresponding schema for this annotation example is:

```text
schema = {
    'earthquake trigger word': [
        'time',
        'magnitude'
    ]
}
```

<a name="数据导出"></a>

## 6. Data Export

#### 6.1 Exporting Extraction Task Data

When selecting the export file type as ``JSONL(relation)``, the exported data example would be:
```text
{
    "id": 38,
    "text": "Encyclopedia Entry: Do You Know What I Want, a song performed by singer Gao Mingjun, released in 1989, included in the personal album Jungle Boy",
    "relations": [
        {
            "id": 20,
            "from_id": 51,
            "to_id": 53,
            "type": "Singer"
        },
        {
            "id": 21,
            "from_id": 51,
            "to_id": 55,
            "type": "Release Date"
        },
        {
            "id": 22,
            "from_id": 51,
            "to_id": 54,
            "type": "Album"
        }
    ],
    "entities": [
        {
            "id": 51,
            "start_offset": 4,
            "end_offset": 11,
            "label": "Work Title"
        },
        {
            "id": 53,
            "start_offset": 15,
            "end_offset": 18,
            "label": "Person Name"
        },
        {
            "id": 54,
            "start_offset": 42,
            "end_offset": 46,
            "label": "Work Title"
        },
        {
            "id": 55,
            "start_offset": 26,
            "end_offset": 31,
            "label": "Time"
        }
    ]
}
```

The annotated data is saved in the same text file, with each sample occupying one line stored in ``json`` format, containing the following fields:
- ``id``: Unique identifier ID for the sample in the dataset.
- ``text``: Original text data.
- ``entities``: Span labels contained in the data, each Span label includes four fields:
    - ``id``: Unique identifier ID for the Span in the dataset.
    - ``start_offset``: Start offset of the Span's initial token in the text.
    - ``end_offset``: Next position after the end offset of the Span's final token in the text.
    - ``label``: Span type.
- ``relations``: Relation labels contained in the data, each Relation label includes four fields:
    - ``id``: Unique identifier ID for the (Span1, Relation, Span2) triplet in the dataset. Identical triplets in different samples share the same ID.
    - ``from_id``
<a name="Data-Conversion"></a>

## 7. Data Conversion

This section details how to use the `doccano.py` script to convert annotated data exported from the doccano platform, enabling one-click generation of training/validation/test sets.

#### 7.1 Data Conversion for Extractive Tasks

- After annotation, export the file in `JSONL(relation)` format from the doccano platform, rename it to `doccano_ext.json`, and place it in the `./data` directory.
- Use the [doccano.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/application/information_extraction/doccano.py) script for data format conversion, after which model training can commence.

```shell
python doccano.py \
    --doccano_file ./data/doccano_ext.json \
    --save_dir ./data \
    --negative_ratio 1
```

Configurable parameters:

- ``doccano_file``: Annotated data file exported from doccano.
- ``save_dir``: Directory to save training data (default: ``data``).
- ``negative_ratio``: Maximum negative sampling ratio (applies only to extraction tasks). Proper negative sampling enhances model performance. Number of negatives = negative_ratio * number of positive samples.
- ``splits``: Dataset split ratios for training/validation/test sets. Default [0.8, 0.1, 0.1] corresponds to 8:1:1 split.
- ``task_type``: Task type (currently only information extraction supported).
- ``is_shuffle``: Whether to shuffle the dataset (default: True).
- ``seed``: Random seed (default: 1000).
- ``schema_lang``: Schema language choice (``ch`` or ``en``, default: ``ch``).

Notes:
- By default, the doccano.py script splits data into train/dev/test sets.
- Running doccano.py will overwrite existing files with the same name.
- We recommend negative sampling during training. This is implemented via the ``negative_ratio`` parameter during data conversion.
- Each entry in the exported doccano file is assumed to be correctly human-annotated.

## References
- **[doccano](https://github.com/doccano/doccano)**
