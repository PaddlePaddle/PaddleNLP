# 文本抽取任务 Label Studio 使用指南

 **目录**

- [1. 安装](#1)
- [2. 文本抽取任务标注](#2)
    - [2.1 项目创建](#21)
    - [2.2 数据上传](#22)
    - [2.3 标签构建](#23)
    - [2.4 任务标注](#24)
    - [2.5 数据导出](#25)
    - [2.6 数据转换](#26)
    - [2.7 更多配置](#27)

<a name="1"></a>

## 1. 安装
**以下标注示例用到的环境配置：**

- Python 3.8+
- label-studio == 1.6.0
- paddleocr >= 2.6.0.1

在终端(terminal)使用 pip 安装 label-studio：

```shell
pip install label-studio==1.6.0
```

安装完成后，运行以下命令行：
```shell
label-studio start
```

在浏览器打开[http://localhost:8080/](http://127.0.0.1:8080/)，输入用户名和密码登录，开始使用 label-studio 进行标注。

<a name="2"></a>

## 2. 文本抽取任务标注

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

- **文本分类、句子级情感倾向分类**任务选择``Text Classification``。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/212617773-34534e68-4544-4b24-8f39-ae7f9573d397.png height=420 width=1200 />
</div>

- 添加标签(也可跳过后续在 Setting/Labeling Interface 中配置)

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199662737-ed996a2c-7a24-4077-8a36-239c4bfb0a16.png height=380 width=1200 />
</div>

图中展示了实体类型标签的构建，其他类型标签的构建可参考[2.3标签构建](#23)

<a name="22"></a>

#### 2.2 数据上传

先从本地上传 txt 格式文件，选择``List of tasks``，然后选择导入本项目。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199667670-1b8f6755-b41f-41c4-8afc-06bb051690b6.png height=210 width=1200 />
</div>

<a name="23"></a>

#### 2.3 标签构建

- Span 类型标签

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199667941-04e300c5-3cd7-4b8e-aaf5-561415414891.png height=480 width=1200 />
</div>

- Relation 类型标签

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199725229-f5e998bf-367c-4449-b83a-c799f1e3de00.png height=620 width=1200 />
</div>

Relation XML 模板：

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

该标注示例对应的 schema 为：

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

对于关系抽取，其 P 的类型设置十分重要，需要遵循以下原则

“{S}的{P}为{O}”需要能够构成语义合理的短语。比如对于三元组(S, 父子, O)，关系类别为父子是没有问题的。但按照 UIE 当前关系类型 prompt 的构造方式，“S 的父子为 O”这个表达不是很通顺，因此 P 改成孩子更好，即“S 的孩子为 O”。**合理的 P 类型设置，将显著提升零样本效果**。

该标注示例对应的 schema 为：

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

该标注示例对应的 schema 为：

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


该标注示例对应的 schema 为：

```text
schema = '情感倾向[正向，负向]'
```

- 实体/评价维度分类

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199879586-8c6e4826-a3b0-49e0-9920-98ca062dccff.png height=240 width=1200 />
</div>

该标注示例对应的 schema 为：

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

勾选已标注文本 ID，选择导出的文件类型为``JSON``，导出数据：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199891344-023736e2-6f9d-454b-b72a-dec6689f8436.png height=180 width=1200 />
</div>

<a name="26"></a>

#### 2.6 数据转换

将导出的文件重命名为``label_studio.json``后，放入``./data``目录下。通过[label_studio.py](./label_studio.py)脚本可转为 UIE 的数据格式。

- 抽取式任务

```shell
python label_studio.py \
    --label_studio_file ./data/label_studio.json \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --task_type ext
```

- 句子级分类任务

在数据转换阶段，我们会自动构造用于模型训练的 prompt 信息。例如句子级情感分类中，prompt 为``情感倾向[正向,负向]``，可以通过`prompt_prefix`和`options`参数进行配置。

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

在数据转换阶段，我们会自动构造用于模型训练的 prompt 信息。例如评价维度情感分类中，prompt 为``XXX 的情感倾向[正向,负向]``，可以通过`prompt_prefix`和`options`参数进行声明。

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

<a name="27"></a>

#### 2.7 更多配置

- ``label_studio_file``: 从 label studio 导出的数据标注文件。
- ``save_dir``: 训练数据的保存目录，默认存储在``data``目录下。
- ``negative_ratio``: 最大负例比例，该参数只对抽取类型任务有效，适当构造负例可提升模型效果。负例数量和实际的标签数量有关，最大负例数量 = negative_ratio * 正例数量。该参数只对训练集有效，默认为5。为了保证评估指标的准确性，验证集和测试集默认构造全负例。
- ``splits``: 划分数据集时训练集、验证集所占的比例。默认为[0.8, 0.1, 0.1]表示按照``8:1:1``的比例将数据划分为训练集、验证集和测试集。
- ``task_type``: 选择任务类型，可选有抽取和分类两种类型的任务。
- ``options``: 指定分类任务的类别标签，该参数只对分类类型任务有效。默认为["正向", "负向"]。
- ``prompt_prefix``: 声明分类任务的 prompt 前缀信息，该参数只对分类类型任务有效。默认为"情感倾向"。
- ``is_shuffle``: 是否对数据集进行随机打散，默认为 True。
- ``seed``: 随机种子，默认为1000.
- ``schema_lang``：选择 schema 的语言，将会应该训练数据 prompt 的构造方式，可选有`ch`和`en`。默认为`ch`。
- ``separator``: 实体类别/评价维度与分类标签的分隔符，该参数只对实体/评价维度分类任务有效。默认为"##"。

备注：
- 默认情况下 [label_studio.py](./label_studio.py) 脚本会按照比例将数据划分为 train/dev/test 数据集
- 每次执行 [label_studio.py](./label_studio.py) 脚本，将会覆盖已有的同名数据文件
- 在模型训练阶段我们推荐构造一些负例以提升模型效果，在数据转换阶段我们内置了这一功能。可通过`negative_ratio`控制自动构造的负样本比例；负样本数量 = negative_ratio * 正样本数量。
- 对于从 label_studio 导出的文件，默认文件中的每条数据都是经过人工正确标注的。


## References
- **[Label Studio](https://labelstud.io/)**
