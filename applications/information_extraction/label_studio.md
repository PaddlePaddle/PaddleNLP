# 信息抽取任务Label Studio使用指南

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

<a name="项目创建"></a>

## 2. 项目创建

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


<a name="数据上传"></a>

## 3. 数据上传

先从本地或HTTP链接上传图片，然后选择导入本项目。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199452007-2d45f7ba-c631-46b4-b21f-729a2ed652e9.png height=160 hspace='15' />
</div>

<a name="标签构建"></a>

## 4. 标签构建

#### 4.1 构建实体标签

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199456432-ce601ab0-7d6c-458f-ac46-8839dbc4d013.png height=350 hspace='15' />
</div>


#### 4.2 构建关系标签

#### 4.3 构建文档分类任务标签

#### 4.4 构建实体/评价维度级分类任务标签


<a name="任务标注"></a>

## 5. 任务标注

#### 5.1 实体抽取

标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199459737-fab20c95-cb7e-42e2-b4f1-06865d58be95.png height=350 hspace='15' />
</div>

该标注示例对应的schema为：

```text
schema = ['开票日期', '名称', '纳税人识别号', '地址、电话', '开户行及账号', '金额', '税额', '价税合计', 'No', '税率']
```

#### 5.2 关系抽取/属性抽取

#### 5.3 文档分类

#### 5.4 实体/评价维度级分类

<a name="数据导出"></a>

## 6. 数据导出


<a name="数据转换"></a>

## 7.数据转换



## References
- **[Label Studio](https://labelstud.io/)**
