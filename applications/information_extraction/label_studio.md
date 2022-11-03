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

<a name="22"></a>

#### 2.2 数据上传

<a name="23"></a>

#### 2.3 标签构建

- 构建实体标签

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199456432-ce601ab0-7d6c-458f-ac46-8839dbc4d013.png height=350 hspace='15' />
</div>


- 构建关系标签

- 构建事件标签

- 构建文档分类任务标签

- 构建实体/评价维度级分类任务标签


<a name="24"></a>

#### 2.4 任务标注

- 实体抽取

标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199459737-fab20c95-cb7e-42e2-b4f1-06865d58be95.png height=350 hspace='15' />
</div>

该标注示例对应的schema为：

```text
schema = ['开票日期', '名称', '纳税人识别号', '地址、电话', '开户行及账号', '金额', '税额', '价税合计', 'No', '税率']
```

- 关系抽取

- 事件抽取

- 文档分类

- 实体/评价维度级分类

<a name="25"></a>

#### 2.5 数据导出


<a name="26"></a>

#### 2.6 数据转换

<a name="3"></a>

## 3. 文档抽取任务标注

<a name="31"></a>

#### 3.1 项目创建

点击创建（Create）开始创建一个新的项目，填写项目名称、描述，然后选择``Object Detection with Bounding Boxes``。点击创建成功创建一个label-studio项目。

填写项目名称、描述

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199445809-1206f887-2782-459e-9001-fbd790d59a5e.png height=200 hspace='15' />
</div>

选择``Object Detection with Bounding Boxes`


添加标签

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199450930-4c0cd189-6085-465a-aca0-6ba6f52a0c0d.png height=400 hspace='15' />
</div>


<a name="32"></a>

#### 3.2 数据上传

先从本地或HTTP链接上传图片，然后选择导入本项目。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199452007-2d45f7ba-c631-46b4-b21f-729a2ed652e9.png height=160 hspace='15' />
</div>

<a name="33"></a>

#### 3.3 标签构建

- 构建实体标签

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199456432-ce601ab0-7d6c-458f-ac46-8839dbc4d013.png height=350 hspace='15' />
</div>


- 构建关系标签

- 构建事件标签

- 构建文档分类任务标签

- 构建实体/评价维度级分类任务标签


<a name="34"></a>

#### 3.4 任务标注

- 实体抽取

标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199459737-fab20c95-cb7e-42e2-b4f1-06865d58be95.png height=350 hspace='15' />
</div>

该标注示例对应的schema为：

```text
schema = ['开票日期', '名称', '纳税人识别号', '地址、电话', '开户行及账号', '金额', '税额', '价税合计', 'No', '税率']
```

- 关系抽取

- 事件抽取

- 文档分类

- 实体/评价维度级分类

<a name="35"></a>

#### 3.5 数据导出


<a name="36"></a>

#### 3.6 数据转换



## References
- **[Label Studio](https://labelstud.io/)**
