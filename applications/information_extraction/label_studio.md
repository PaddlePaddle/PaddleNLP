# 信息抽取任务Label Studio使用指南

 **目录**

- [1. 安装](#1)
- [2. 文本抽取任务标注](#2)
    - [2.1 项目创建](#21)
    - [2.2 数据上传](#22)
    - [2.3 标签构建](#23)
    - [2.4 任务标注](#24)
    - [2.5 数据导出](#25)
    - [2.6 数据转换](#26)
- [3. 文档抽取任务标注](#3)
    - [3.1 项目创建](#31)
    - [3.2 数据上传](#32)
    - [3.3 标签构建](#33)
    - [3.4 任务标注](#34)
    - [3.5 数据导出](#35)
    - [3.6 数据转换](#36)
- [4. 更多配置](#4)

<a name="1"></a>

## 1. 安装
**以下标注示例用到的环境配置：**

- Python 3.8+
- label-studio == 1.6.0
- paddleocr >= 2.6.0.1

在终端(terminal)使用pip安装label-studio：

```shell
pip install label-studio==1.6.0
```

安装完成后，运行以下命令行：
```shell
label-studio start
```

在浏览器打开[http://localhost:8080/](http://127.0.0.1:8080/)，输入用户名和密码登录，开始使用label-studio进行标注。

<a name="2"></a>

## 2. 文档抽取任务标注

<a name="21"></a>

#### 2.1 项目创建

点击创建（Create）开始创建一个新的项目，填写项目名称、描述，然后选择``Object Detection with Bounding Boxes``。

- 填写项目名称、描述

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199661377-d9664165-61aa-4462-927d-225118b8535b.png height=230 width=1200 />
</div>

- **命名实体识别、关系抽取、事件抽取、实体/评价维度分类**任务选择``Relation Extraction`。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199661638-48a870eb-a1df-4db5-82b9-bc8e985f5190.png height=350 width=1200 />
</div>

- **文本分类、句子级情感倾向分类**任务选择``Relation Extraction`。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199661638-48a870eb-a1df-4db5-82b9-bc8e985f5190.png height=350 width=1200 />
</div>

- 添加标签(也可跳过后续在Setting/Labeling Interface中配置)

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199662737-ed996a2c-7a24-4077-8a36-239c4bfb0a16.png height=380 width=1200 />
</div>

图中展示了实体类型标签的构建，其他类型标签的构建可参考[2.3标签构建](#23)

<a name="22"></a>

#### 2.2 数据上传

先从本地上传txt格式文件，选择``List of tasks``，然后选择导入本项目。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199667670-1b8f6755-b41f-41c4-8afc-06bb051690b6.png height=210 width=1200 />
</div>

<a name="23"></a>

#### 2.3 标签构建

- Span类型标签

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199667941-04e300c5-3cd7-4b8e-aaf5-561415414891.png height=480 width=1200 />
</div>

- Relation类型标签

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199725229-f5e998bf-367c-4449-b83a-c799f1e3de00.png height=620 width=1200 />
</div>

Relation XML模板：

```xml
  <Relations>
    <Relation value="歌手"/>
    <Relation value="发行时间"/>
    <Relation value="所属专辑"/>
  </Relations>
```

- 分类类别标签

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199724082-ee82dceb-dab0-496d-a930-a8ecb284d8b2.png height=370 width=1200 />
</div>


<a name="24"></a>

#### 2.4 任务标注

- 实体抽取

标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199879957-aeec9d17-d342-4ea0-a840-457b49f6066e.png height=140 width=1000 />
</div>

该标注示例对应的schema为：

```text
schema = [
    '时间',
    '选手',
    '赛事名称',
    '得分'
]
```

- 关系抽取

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199879866-03c1ecac-1828-4f35-af70-9ae61701c303.png height=230 width=1200 />
</div>

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

- 事件抽取

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199879776-75abbade-9bea-44dc-ac36-322fecdc03e0.png height=220 width=1200 />
</div>

该标注示例对应的schema为：

```text
schema = {
    '地震触发词': [
        '时间',
        '震级'
    ]
}
```

- 句子级分类

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199879672-c3f286fe-a217-4888-950f-d4ee45b19f5a.png height=210 width=1000 />
</div>


该标注示例对应的schema为：

```text
schema = '情感倾向[正向，负向]'
```

- 实体/评价维度分类

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199879586-8c6e4826-a3b0-49e0-9920-98ca062dccff.png height=240 width=1200 />
</div>

该标注示例对应的schema为：

```text
schema = {
    '评价维度': [
        '观点词',
        '情感倾向[正向，负向]'
    ]
}
```

<a name="25"></a>

#### 2.5 数据导出

勾选已标注文本ID，选择导出的文件类型为``JSON``，导出数据：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199891344-023736e2-6f9d-454b-b72a-dec6689f8436.png height=180 width=1200 />
</div>

<a name="26"></a>

#### 2.6 数据转换

将导出的文件重命名为``label_studio.json``后，放入``./data``目录下。通过[label_studio.py](./label_studio.py)脚本可转为UIE的数据格式。

- 抽取式任务

```shell
python label_studio.py \
    --label_studio_file ./data/label_studio.json \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --task_type ext
```

- 句子级分类任务

在数据转换阶段，我们会自动构造用于模型训练的prompt信息。例如句子级情感分类中，prompt为``情感倾向[正向,负向]``，可以通过`prompt_prefix`和`options`参数进行配置。

```shell
python label_studio.py \
    --label_studio_file ./data/label_studio.json \
    --task_type cls \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --prompt_prefix "情感倾向" \
    --options "正向" "负向"
```

- 实体/评价维度分类任务

在数据转换阶段，我们会自动构造用于模型训练的prompt信息。例如评价维度情感分类中，prompt为``XXX的情感倾向[正向,负向]``，可以通过`prompt_prefix`和`options`参数进行声明。

```shell
python label_studio.py \
    --label_studio_file ./data/label_studio.json \
    --task_type ext \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --prompt_prefix "情感倾向" \
    --options "正向" "负向" \
    --separator "##"
```

<a name="3"></a>

## 3. 文档抽取任务标注

<a name="31"></a>

#### 3.1 项目创建

点击创建（Create）开始创建一个新的项目，填写项目名称、描述，然后选择``Object Detection with Bounding Boxes``。

- 填写项目名称、描述

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199445809-1206f887-2782-459e-9001-fbd790d59a5e.png height=300 width=1200 />
</div>

- **命名实体识别、关系抽取、事件抽取、实体/评价维度分类**任务选择``Object Detection with Bounding Boxes`

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199660090-d84901dd-001d-4620-bffa-0101a4ecd6e5.png height=400 width=1200 />
</div>

- **文档分类**任务选择``Image Classification`

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199729973-53a994d8-da71-4ab9-84f5-83297e19a7a1.png height=400 width=1200 />
</div>

- 添加标签(也可跳过后续在Setting/Labeling Interface中添加)

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199450930-4c0cd189-6085-465a-aca0-6ba6f52a0c0d.png height=600 width=1200 />
</div>

图中展示了实体类型标签的构建，其他类型标签的构建可参考[3.3标签构建](#33)

<a name="32"></a>

#### 3.2 数据上传

先从本地或HTTP链接上传图片，然后选择导入本项目。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199452007-2d45f7ba-c631-46b4-b21f-729a2ed652e9.png height=270 width=1200 />
</div>

<a name="33"></a>

#### 3.3 标签构建

- Span类型标签

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199456432-ce601ab0-7d6c-458f-ac46-8839dbc4d013.png height=500 width=1200 />
</div>


- Relation类型标签

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199877621-f60e00c7-81ae-42e1-b498-8ebc5b5bd0fd.png height=650 width=1200 />
</div>

Relation XML模板：

```xml
  <Relations>
    <Relation value="单位"/>
    <Relation value="数量"/>
    <Relation value="金额"/>
  </Relations>
```


- 分类类别标签

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199891626-cc995783-18d2-41dc-88de-260b979edc56.png height=500 width=1200 />
</div>

<a name="34"></a>

#### 3.4 任务标注

- 实体抽取

标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199879427-82806ffc-dc60-4ec7-bda5-e16419ee9d15.png height=650 width=800 />
</div>

该标注示例对应的schema为：

```text
schema = ['开票日期', '名称', '纳税人识别号', '地址、电话', '开户行及账号', '金额', '税额', '价税合计', 'No', '税率']
```

- 关系抽取

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199879032-88896a00-85ca-4bb0-a8e8-305a47bbaf78.png height=450 width=1000 />
</div>


```text
schema = {
    '名称及规格': [
        '金额',
        '单位',
        '数量'
    ]
}
```

- 文档分类

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199879238-b8b41d4a-7e77-47cd-8def-2fc8ba89442f.png height=650 width=800 />
</div>

该标注示例对应的schema为：

```text
schema = '文档类别[发票，报关单]'
```


<a name="35"></a>

#### 3.5 数据导出

勾选已标注图片ID，选择导出的文件类型为``JSON``，导出数据：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199890897-b33ede99-97d8-4d44-877a-2518a87f8b67.png height=200 width=1200 />
</div>

<a name="36"></a>

#### 3.6 数据转换

将导出的文件重命名为``label_studio.json``后，放入``./data``目录下，并将对应的标注图片放入``./data/images``目录下（图片的文件名需与上传到label studio时的命名一致）。通过[label_studio.py](./label_studio.py)脚本可转为UIE的数据格式。

- 路径示例

```shell
data/
├── images # 图片目录
│   ├── b0.jpg # 原始图片（文件名需与上传到label studio时的命名一致）
│   └── b1.jpg
└── label_studio.json # 从label studio导出的标注文件
```

- 抽取式任务

```shell
python label_studio.py \
    --label_studio_file ./data/label_studio.json \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --task_type ext
```

- 文档分类任务

```shell
python label_studio.py \
    --label_studio_file ./data/label_studio.json \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --task_type cls \
    --prompt_prefix "文档类别" \
    --options "发票" "报关单"
```

<a name="4"></a>

## 4. 更多配置

- ``label_studio_file``: 从label studio导出的数据标注文件。
- ``save_dir``: 训练数据的保存目录，默认存储在``data``目录下。
- ``negative_ratio``: 最大负例比例，该参数只对抽取类型任务有效，适当构造负例可提升模型效果。负例数量和实际的标签数量有关，最大负例数量 = negative_ratio * 正例数量。该参数只对训练集有效，默认为5。为了保证评估指标的准确性，验证集和测试集默认构造全负例。
- ``splits``: 划分数据集时训练集、验证集所占的比例。默认为[0.8, 0.1, 0.1]表示按照``8:1:1``的比例将数据划分为训练集、验证集和测试集。
- ``task_type``: 选择任务类型，可选有抽取和分类两种类型的任务。
- ``options``: 指定分类任务的类别标签，该参数只对分类类型任务有效。默认为["正向", "负向"]。
- ``prompt_prefix``: 声明分类任务的prompt前缀信息，该参数只对分类类型任务有效。默认为"情感倾向"。
- ``is_shuffle``: 是否对数据集进行随机打散，默认为True。
- ``seed``: 随机种子，默认为1000.
- ``separator``: 实体类别/评价维度与分类标签的分隔符，该参数只对实体/评价维度分类任务有效。默认为"##"。

备注：
- 默认情况下 [label_studio.py](./label_studio.py) 脚本会按照比例将数据划分为 train/dev/test 数据集
- 每次执行 [label_studio.py](./label_studio.py) 脚本，将会覆盖已有的同名数据文件
- 在模型训练阶段我们推荐构造一些负例以提升模型效果，在数据转换阶段我们内置了这一功能。可通过`negative_ratio`控制自动构造的负样本比例；负样本数量 = negative_ratio * 正样本数量。
- 对于从label_studio导出的文件，默认文件中的每条数据都是经过人工正确标注的。


## References
- **[Label Studio](https://labelstud.io/)**
