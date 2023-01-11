# 情感分析任务Label Studio使用指南

 **目录**

- [1. label-studio 安装](#1)
- [2. label-studio 项目创建](#2)
- [3. 情感分析任务标注](#3)
    - [3.1 语句级情感分类任务](#3.1)
    - [3.2 属性级情感分析任务](#3.2)
      - [3.2.1 属性-情感极性-观点词抽取](#3.2.1)
      - [3.2.2 属性-情感极性抽取](#3.2.2)
      - [3.2.3 属性-观点词抽取](#3.2.3)
      - [3.2.4 属性抽取](#3.2.4)
      - [3.2.5 观点词抽取](#3.2.5)
- [4. 导出标注数据](#4)
- [5. References](#5)

<a name="1"></a>

## **1. label-studio 安装**
本内容在以下环境进行测试安装：
- python == 3.9.12
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

## **2. label-studio 项目创建**

创建项目之前，需要先确定标注的任务类型以及需要标注哪些内容，然后点击创建（Create）开始创建一个新的项目，填写项目名称、描述。

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/202995157-9caa0b26-202d-46d2-832a-f1cdf3f9a9b6.png />
</div>

如果数据已经准备好，可以在此进行导入数据。

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/202995686-954cc001-4478-46e1-8329-ab3ab02e8a35.png />
</div>


接下来，根据需要标注的任务类型，选择适合的任务。在本项目中，默认会包含两种类型的任务：语句级情感分类任务和属性级情感分析任务。由于这两者都属于自然语言处理（NLP）任务，因此可以点击 `Natural Language Processing` 选项，在该选项下面进行选择相应的子项任务。

- 如果标注语句级情感分类任务，请选择`Text Classification`。

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/202996231-a4cf809d-000e-4693-b7c8-70ff2fae22ae.png />
</div>

- 如果标注属性级情感分析任务，比如属性-观点词-情感极性三元组的信息抽取，请选择`Relation Extraction`。

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/202997005-e8b0e865-584e-460e-8e68-a41532b6ef1b.png />
</div>

最后点击保存即可。

<a name="3"></a>

## **3. 情感分析任务标注**

<a name="3.1"></a>

### **3.1 语句级情感分类任务**
这里对应的任务类型为`Text Classification`，在标注之前，需要设定`正向`和`负向`的标签，然后保存即可。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199724082-ee82dceb-dab0-496d-a930-a8ecb284d8b2.png />
</div>

设定好标签后，即可开始进行标注，选择正向或负向，最后点击提交，便标注好一条数据。
<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/210329413-deff1eb7-3472-463e-aef8-bc9b0456504a.png />
</div>

<a name="3.2"></a>

### **3.2 属性级情感分析任务**

在本项目中，属性级的情感分析需要配置的标注任务类型为`Relation Extraction`，包括属性抽取、观点抽取、属性-观点抽取、属性-情感极性抽取、属性-情感极性-观点词三元组抽取等任务。其中属性-情感极-观点词(A-S-O)三元组抽取是最常见的任务之一，下面优先讲解该任务的标注规则。

<a name="3.2.1"></a>

#### **3.2.1 属性-情感极性-观点词抽取**
属性-情感极性-观点词(A-S-O)三元组抽取标注内容涉及两类标签：Span 类型标签和 Relation 类型标签。其中Span标签用于定位文本批评中属性、观点词和情感极性三类信息，Relation类型标签用于设置评价维度和观点词、情感倾向之间的关系。


#### **（1）Span类型标签**
这里需要定位属性、情感极性、观点词三类信息，在标注时，需要将属性和情感极性进行组合，形成复合标签。具体来讲，设定`评价维度##正向`用于定位情感倾向为正向的属性，`评价维度##负向`用于定位情感倾向为负向的属性。另外，利用标注标签`观点词`定位语句中的观点词。

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/202999690-c76948cf-45ba-42a2-85ed-ee55e6a0907f.png />
</div>

#### **（2）Relation类型标签**
这里只涉及到1中Relation类型标签，即`评价维度`到`观点词`的映射关系。这里可以设置一下两者关系的名称，即点击Code，然后配置关系名称（这里将两者关系设置为`观点词`），最后点击保存即可。

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/203000684-c7ce1483-6e1c-4399-9d43-369eae2f8684.png />
</div>

在设置好Span类型和Relation标签之后，便可以开始进行标注数据了。

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/203001847-8e41709b-0f5a-4673-8aca-5c4fb7705d4a.png />
</div>

<a name="3.2.2"></a>

#### **3.2.2 属性-情感极性抽取**
如3.2.1所述，本项目中针对属性-情感极性(A-S)抽取任务，采用`Span`的形式进行标注。设定`评价维度##正向`用于定位情感倾向为正向的属性，`评价维度##负向`用于定位情感倾向为负向的属性。下图展示了关于属性-情感极性抽取任务的标注示例。

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/210720997-950bae9c-407c-431c-bdec-d8289a831920.png />
</div>

<a name="3.2.3"></a>

#### **3.2.3 属性-观点词抽取**
针对属性-观点词(A-O)抽取任务，采用`Relation`的形式进行标注。这需要将属性对应标注标签设定为`评价维度`，观点词设定为`观点词`。下图展示了关于属性-观点词抽取任务的标注示例。

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/210721569-9462145c-14ee-459d-a11b-72a8cd7b9a45.png />
</div>

<a name="3.2.4"></a>

#### **3.2.4 属性抽取**
针对属性(A)抽取任务，采用`Span`的形式进行标注。 这需要将属性对应的标注标签设定为`评价维度`。下图展示了关于属性抽取任务的标注示例。

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/210722135-cc697497-cd46-444a-aad6-d31d9adfeaed.png />
</div>

<a name="3.2.5"></a>

#### **3.2.4 观点词抽取**
针对观点词(O)抽取任务，采用`Span`的形式进行标注。 这需要将观点词对应的标注标签设定为`观点词`。下图展示了关于观点词抽取任务的标注示例。

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/210722331-64865a9a-522a-41aa-aaab-2bdf9de85263.png />
</div>


<a name="4"></a>

## **4. 导出标注数据**

勾选已标注文本ID，点击Export按钮，选择导出的文件类型为`JSON`，导出数据：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/199891344-023736e2-6f9d-454b-b72a-dec6689f8436.png />
</div>

<a name="5"></a>

## **5. References**
- **[Label Studio 官网](https://labelstud.io/)**
