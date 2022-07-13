# 文本分类任务doccano使用指南

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
- doccano 1.6.2

在终端(terminal)使用pip安装doccano：

```shell
pip install doccano==1.6.2
```
安装完成后，运行以下命令行：
```shell
# Initialize database.
doccano init
# Create a super user.
doccano createuser --username admin --password pass
# Start a web server.
doccano webserver --port 8000
```
在新的终端(terminal)运行如下命令行：
```shell
# Start the task queue to handle file upload/download.
doccano task
```
在浏览器打开[http://127.0.0.1:8000/](http://127.0.0.1:8000/)，输入用户名和密码登录，开始使用doccano进行标注。doccano支持中文版本，可以点击右上角选择ZH(中文)。

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/176856052-bde31dd7-6317-49d9-8ae6-572c821f4e3d.png height=230 hspace='15'/>
</div>

doccano还支持PostgreSQL、Docker、Docker Compose等安装方式，详情请参考[doccano官方文档](https://github.com/doccano/doccano) 完成doccano的安装与初始配置。


<a name="项目创建"></a>

## 2. 项目创建

文本分类支持多分类、多标签、层次分类三种类型的文本分类任务。

点击创建（Create）开始创建一个新的项目，选择文本分类，然后填写项目名称、描述、Tags等项目信息。如果是多分类任务或者是单路径层次分类任务，勾选 `Allow single label` ，勾选后标签标注只允许选择一个标签进行标注。点击创建成功创建一个doccano项目。
<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/176857646-b324d075-e281-4e9f-9f42-50fa2645b271.png height=400 hspace='15'/>
</div>


<a name="数据上传"></a>

## 3. 数据上传

点击数据集-操作-导入数据集，开始导入本地待标注数据集：
<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/176858888-79b781f9-2ce0-4348-ba14-faac19031867.png height=300 hspace='15'/>
</div>

doccano支持`TextFile`、`TextLine`、`JSONL`和`CoNLL`四种数据上传格式，文本分类本地数据集定制训练中**统一使用TextLine**这一文件格式，即上传的文件需要为txt等格式，且在数据标注时，该文件的每一行待标注文本显示为一页内容。
上传的文件为txt等格式，每一行为一条待标注文本，示例:

```text
黑苦荞茶的功效与作用及食用方法
交界痣会凸起吗
检查是否能怀孕挂什么科
鱼油怎么吃咬破吃还是直接咽下去
幼儿挑食的生理原因是
...
```

上传数据类型**选择TextLine**，选择待标注文本或拖拽文本导入doccano项目中，点击导入，导入待标注数据集。

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/176859861-b790288f-32d7-4ab0-8b5f-b30e97f8c306.png height=300 hspace='15'/>
</div>


<a name="标签构建"></a>

## 4. 标签构建

点击标签-操作-创建标签，开始添加分类类别标签：
<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/176860972-eb9cacf1-199a-4cec-9940-6858434cfb94.png height=300 hspace='15'/>
</div>
填入分类类别标签，选择标签颜色，建议不同标签选择不同颜色，最后点击保存或保存并添加下一个，保存标签：
<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/176860977-55292e2a-8bf8-4316-a0f8-b925872e5023.png height=300 hspace='15'/>
</div>
文本分类标签构建示例：
<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/176860996-542cd1f7-9770-4b22-9586-a5bf0e802970.png height=300 hspace='15'/>
</div>

**NOTE:**
我们默认层次分类标签不同层的标签之间具有关联性，以下图为例一个样本具有标签美短虎斑，我们默认还包含美国短毛猫和猫两个标签。

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/175248039-ce1673f1-9b03-4804-b1cb-29e4b4193f86.png height=300 hspace='15'/>
</div>
对于层次分类任务的分类标签我们建议使用标签层次结构中叶结点标签路径作为标签，以上图的标签结构为例，我们建议使用`##`作为分隔符，分隔不同层之间的标签：

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/177095794-0acb9665-3862-4de9-8771-8f424fd4f7b0.png height=300 hspace='15'/>
</div>
<a name="任务标注"></a>

## 5. 任务标注

标注示例，选择对应的分类类别标签，输入回车（Enter）键确认：

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/176872684-4a19f592-be5c-4b86-8adf-eb0a7d7aa375.png height=200 hspace='10'/>
</div>


<a name="数据导出"></a>

## 6. 数据导出

选择数据集-操作-导出数据集，将标注好的数据导出，我们默认所有数据集已经标注完成且正确：
<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/176874195-d21615f4-8d53-4033-8f53-2106ebdf21f8.png height=250 hspace='20'/>
</div>

选择导出的文件类型为``JSONL``，导出数据：

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/176873347-fd995e4e-5baf-4d13-92b9-800cabd1f0b1.png height=300 hspace='20'/>
</div>

导出数据示例：
```text
{"id": 23, "data": "黑苦荞茶的功效与作用及食用方法", "label": ["功效作用"]}
{"id": 24, "data": "交界痣会凸起吗", "label": ["疾病表述"]}
{"id": 25, "data": "检查是否能怀孕挂什么科", "label": ["就医建议"]}
{"id": 26, "data": "鱼油怎么吃咬破吃还是直接咽下去", "label": ["其他"]}
{"id": 27, "data": "幼儿挑食的生理原因是", "label": ["病因分析"]}
```

标注数据保存在同一个文本文件中，每条样例占一行且存储为``jsonl``格式，其包含以下字段
- ``id``: 样本在数据集中的唯一标识ID。
- ``data``: 原始文本数据。
- ``label``: 文本对应类别标签。

<a name="数据转换"></a>

## 7.数据转换

该章节详细说明如何通过`doccano.py`脚本对doccano平台导出的标注数据进行转换，一键生成训练/验证/测试集。当标注完成后，在 doccano 平台上导出 `JSON` 形式的文件，并将其重命名为 `doccano.jsonl`。


### 7.1 多分类任务
通过 [doccano.py](./doccano.py) 脚本进行数据形式转换，然后便可以按照[多分类文本任务指南](multi_class/README.md)进行相应模型训练。
运行
```shell
python doccano.py \
    --doccano_file doccano.jsonl \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --task_type "multi_class"
```

### 7.1 多标签任务
通过 [doccano.py](./doccano.py) 脚本进行数据形式转换，然后便可以按照[多标签文本分类任务指南](multi_label/README.md)进行相应模型训练。
运行
```shell
python doccano.py \
    --doccano_file doccano.jsonl \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --task_type "multi_label"
```

### 7.1 多分类任务
通过 [doccano.py](./doccano.py) 脚本进行数据形式转换，然后便可以按照[层次文本分类任务指南](hierarchical/README.md)进行相应模型训练。
运行
```shell
python doccano.py \
    --doccano_file doccano.jsonl \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --task_type "hierarchical" \
    --separator "##"
```

可配置参数说明：

- ``doccano_file``: 从doccano导出的数据标注文件。
- ``save_dir``: 训练数据的保存目录，默认存储在``data``目录下。
- ``splits``: 划分数据集时训练集、验证集所占的比例。默认为[0.8, 0.1, 0.1]表示按照``8:1:1``的比例将数据划分为训练集、验证集和测试集。
- ``task_type``: 可选，选择任务类型,有多分类，多标签，层次分类三种类型的任务。
- ``is_shuffle``: 是否对数据集进行随机打散，默认为True。
- ``seed``: 随机种子，默认为1000.
- ``separator``: 不同层标签之间的分隔符，该参数只对层次文本分类任务有效。默认为"##"。

转化后的doccano标注数据目录结构如下：
```text
data/
├── train.txt # 训练数据集文件
├── dev.txt # 开发数据集文件
├── test.txt # 测试训练集文件（可选，数据划分为 train/dev/test 数据集）
├── label.txt # 分类标签文件
└── data.txt # 待预测数据文件
```

备注：
- 默认情况下 [doccano.py](./doccano.py) 脚本会按照比例将数据划分为 train/dev/test 数据集，也可以划分成train/dev 数据集。
- 如果数据划分为 train/dev/test 数据集，data.txt则为test数据集无标签数据；如果数据划分为 train/dev 数据集，data.txt则为dev数据集无标签数据。
- 每次执行 [doccano.py](./doccano.py) 脚本，将会覆盖已有的同名数据文件
- 对于从doccano导出的文件，默认文件中的每条数据都是经过人工正确标注的。
## References
- **[doccano](https://github.com/doccano/doccano)**
