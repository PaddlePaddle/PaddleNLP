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

<a name="1"></a>

## 1. 安装
**以下标注示例用到的环境配置：**

- Python 3.8+
- label-studio 1.6.0

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

- **命名实体识别、关系抽取、事件抽取、实体/评价维度级分类**任务选择``Relation Extraction`。

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
    <img src=https://user-images.githubusercontent.com/40840292/199879957-aeec9d17-d342-4ea0-a840-457b49f6066e.png height=160 width=1000 />
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
    <img src=https://user-images.githubusercontent.com/40840292/199879866-03c1ecac-1828-4f35-af70-9ae61701c303.png height=240 width=1200 />
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
    <img src=https://user-images.githubusercontent.com/40840292/199879776-75abbade-9bea-44dc-ac36-322fecdc03e0.png height=230 width=1200 />
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
    <img src=https://user-images.githubusercontent.com/40840292/199879672-c3f286fe-a217-4888-950f-d4ee45b19f5a.png height=200 width=1000 />
</div>


该标注示例对应的schema为：

```text
schema = '情感倾向[正向，负向]'
```

- 实体/评价维度级分类

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
    <img src=https://user-images.githubusercontent.com/40840292/199891344-023736e2-6f9d-454b-b72a-dec6689f8436.png height=200 width=1200 />
</div>

<a name="26"></a>

#### 2.6 数据转换

<a name="3"></a>

## 3. 文档抽取任务标注

<a name="31"></a>

#### 3.1 项目创建

点击创建（Create）开始创建一个新的项目，填写项目名称、描述，然后选择``Object Detection with Bounding Boxes``。

- 填写项目名称、描述

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199445809-1206f887-2782-459e-9001-fbd790d59a5e.png height=300 width=1200 />
</div>

- **命名实体识别、关系抽取、事件抽取、实体/评价维度级分类**任务选择``Object Detection with Bounding Boxes`

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



## References
- **[Label Studio](https://labelstud.io/)**
