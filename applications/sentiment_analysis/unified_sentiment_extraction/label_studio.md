# 情感分析任务Label Studio使用指南

 **目录**

- [1. 安装](#1)
- [2. 情感分析任务标注](#2)
    - [2.1 项目创建](#2.1)
    - [2.2 数据上传](#2.2)
    - [2.3 标签构建](#2.3)
    - [2.4 任务标注](#2.4)
    - [2.5 数据导出](#2.5)
    - [2.6 数据转换](#2.6)
- [4. 更多配置](#4)

<a name="1"></a>

## 1. 安装
**以下标注示例用到的环境配置：**

- Python 3.8+
- label-studio == 1.6.0

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

<a name="2.1"></a>

### 2.1 项目创建

点击创建（Create）开始创建一个新的项目，填写项目名称、描述。

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/202995157-9caa0b26-202d-46d2-832a-f1cdf3f9a9b6.png />
</div>

开始导入数据

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/202995686-954cc001-4478-46e1-8329-ab3ab02e8a35.png />
</div>

配置标注任务，点击 `Natural Language Processing`，然后选择待标注任务。

- 语句级情感分类任务请选择``Text Classification`。

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/202996231-a4cf809d-000e-4693-b7c8-70ff2fae22ae.png />
</div>

- A-O、A-S、A-S-O等属性级的信息抽取任务请选择``Relation Extraction`。

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/202997005-e8b0e865-584e-460e-8e68-a41532b6ef1b.png />
</div>

最后点击保存即可。


<a name="2.2"></a>

### 2.2 标签构建及标注

#### 2.2.1 语句级情感分类任务
这里需要配置任务类型为`Text Classification`，该任务对应的schema为：

```text
schema = '情感倾向[正向，负向]'
```
首先需要设定正向和负向的标签，然后保存即可。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199724082-ee82dceb-dab0-496d-a930-a8ecb284d8b2.png />
</div>

设定好标签后，即可开始进行标注，选择正向或负向，最后点击提交，便标注好一条数据。
<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/203003559-f51f9b89-5065-4c5b-8a0c-1c69d308bf24.png />
</div>

#### 2.2.2 属性+情感极性+观点抽取任务

以属性-情感极性-观点词(A-S-O)三元组抽取为例，这里需要配置任务类型为`Relation Extraction`。假设我们定义如下Schema以对应A-S-O三元组抽取：

```text
schema = {
    '评价维度': [
        '观点词',
        '情感倾向[正向，负向]'
    ]
}
```

可以看到，其标注内容涉及两类标签：Span 类型标签和 Relation 类型标签。其中Span标签用于定位评价维度、观点词和情感倾向三类信息，Relation类型标签用于设置评价维度和观点词、情感倾向之间的关系。

- Span类型标签
这里为方便标注和后续处理，可以设定`评价维度##正向`用于定位情感倾向为正向的属性，`评价维度##负向`用于定位情感倾向为负向的属性，利用`观点词`标签定位语句中的观点词。

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/202999690-c76948cf-45ba-42a2-85ed-ee55e6a0907f.png />
</div>

- Relation类型标签
点击Code，然后配置关系名称，最后点击保存即可

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/203000684-c7ce1483-6e1c-4399-9d43-369eae2f8684.png />
</div>

在设置好Span类型和Rlation标签之后，便可以开始进行标注数据了

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/203001847-8e41709b-0f5a-4673-8aca-5c4fb7705d4a.png />
</div>


<a name="2.3"></a>

### 2.3 数据导出

勾选已标注文本ID，选择导出的文件类型为``JSON``，导出数据：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199891344-023736e2-6f9d-454b-b72a-dec6689f8436.png />
</div>


## References
- **[Label Studio](https://labelstud.io/)**
