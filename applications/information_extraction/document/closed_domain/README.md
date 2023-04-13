# 封闭域信息抽取

## **目录**
- [1. 封闭域信息抽取简介](#1)
- [2. 快速开始](#2)
    - [2.1 标签体系构建](#21)
    - [2.2 数据标注](#22)
        - [2.2.1 数据标注及训练样本构建](#221)
        - [2.2.2 通过UIE生成更多标注数据](#222)
    - [2.3 模型训练](#22)
    - [2.4 模型评估](#23)
    - [2.5 模型预测及效果展示](#24)
    - [2.6 模型部署](#25)

<a name="1"></a>

## **1. 封闭域建模技术方案**

封闭域信息抽取，目标对于给定的自然语言句子，根据预先定义的 schema 集合，从自然语言句子中提取出所有满足 schema 约束的结构化知识。在封闭域中，**schema标签体系已经完善并且固定，推理性能和模型效果的提升更为重要**。

本项目以 ERNIE Encoder + Global Pointer 作为训练底座，提供了通用封闭域信息抽取建模方案，覆盖命名实体识别、实体关系联合抽取等主流信息抽取任务。封闭域信息抽取中对推理性能要求较高，因此我们建议如果**在业务场景上的数据积累足够多，且需求场景已经相对固定的情况下，可以从UIE切换为封闭域建模**。本项目提供了封闭域场景下的**数据标注、模型训练、模型部署全流程解决方案**。此外，封闭域的数据结构设计与 UIE 完全一致，开发者可以实现从开放域到封闭域建模及部署的无缝迁移。

<a name="2"></a>

## **2. 快速开始**

<a name="21"></a>

## **2.1 标签体系构建**

封闭域场景下的schema延用了UIE的设计方式，关于不同任务的schema构造可以参考[这里](../../taskflow_text.md)的例子。

`data_convert.py`中默认的schema为：

```python
["开票日期", "名称", "纳税人识别号", "开户行及账号", "金额", "价税合计", "No", "税率", "地址、电话", "税额", "开票人"]
```

**NOTE：** `data_convert.py`脚本中的schema需要根据实际使用场景修改，且schema需要包含标注数据中的所有标签类型。

<a name="22"></a>

## **2.2 数据标注**

通过`data_converter.py`构造训练样本，训练样本构建分为以下两类情况：

1）标注数据规模足够大且质量较高，可直接参考[数据标注及训练样本构建](#221)生成训练样本。

2）标注数据规模较小或标注难度较大，可[通过微调后的UIE模型生成更多标注数据](#222)。

<a name="221"></a>

### **2.2.1 数据标注及训练样本构建 (适合标注数据足够多的情况)**

本项目建议用户使用label-studio平台标注数据，标注规则与UIE一致，可以参考[文本信息抽取label-studio标注指南](../../label_studio_text.md)获取更多信息。同时本项目提供了从 label-studio 标注平台到转换为封闭域模型输入形式数据的流程，即支持用户在基于 label_studio 标注业务侧数据后，通过label-studio 导出标注好的json数据，然后利用本项目提供的 ``data_convert.py`` 脚本，可以将导出数据一键转换为模型训练数据。

```shell
python data_convert.py \
    --label_studio_file ../data/label_studio.json \
    --splits 0.8 0.2 0
```

<a name="222"></a>

### **2.2.2 通过UIE生成更多标注数据 (适合标注数据较少的情况)**

当实际标注数据较少时，可以通过微调后的UIE模型产生更多标注数据。其原理是通过数据作为桥梁，将UIE模型的知识迁移到封闭域信息抽取小模型，以达到精度损失较小的情况下却能达到大幅度预测速度提升的效果。

具体流程如下：

#### 2.2.2.1 数据准备

通过label-studio标注平台进行数据标注，并准备与标注数据同源的无监督文档/图片。

#### 2.2.2.2 微调UIE模型

参考[UIE微调](../README.md)得到``../checkpoint/model_best``。

#### 2.2.2.3 构建封闭域训练样本

```shell
python data_convert.py \
    --label_studio_file ../data/label_studio.json \
    --sample_index_file ../data/sample_index.json \
    --unlabeled_images_path ../data/unlabeled_images \
    --save_dir ./data \
    --synthetic_ratio 10 \
    --model_path ../checkpoint/model_best
```

可配置参数说明：

- `label_studio_file`: label-studio标注数据路径。
- `sample_index_file`: UIE微调中的样本位置，确保数据集的训练/验证/测试划分与UIE一致，该文件在UIE微调阶段自动生成，如未指定，标注样本将按照splits中指定的比例随机划分。
- `unlabeled_images_path`: 无监督图片存放路径。
- ``splits``: 划分数据集时训练集、验证集所占的比例。默认为[0.8, 0.1, 0.1]表示按照``8:1:1``的比例将数据划分为训练集、验证集和测试集。该参数在指定sample_index_file时失效。
- `model_path`: 训练好的UIE定制模型路径。
- `save_dir`: 封闭域抽取训练数据保存路径。
- `synthetic_ratio`: 控制合成数据的比例。最大合成数据数量=synthetic_ratio*标注数据数量。
- `seed`: 随机种子，默认为1000。

<a name="23"></a>

## **2.3 模型训练**

```shell
python train.py \
    --train_path data/train_data.json \
    --dev_path data/dev_data.json \
    --label_maps_path data/label_maps.json \
    --num_epochs 50 \
    --model_name_or_path ernie-3.0-base-zh
```

可配置参数说明：

- `train_path`: 训练集文件路径。
- `dev_path`: 验证集文件路径。
- `batch_size`: 批处理大小，默认为16。
- `learning_rate`: 学习率，默认为3e-5。
- `save_dir`: 模型存储路径，默认为`./checkpoint`。
- `max_seq_len`: 最大文本长度，默认为512。
- `doc_stride`: 滑窗大小，默认为256.
- `weight_decay`: 表示AdamW优化器中使用的 weight_decay 的系数。
- `warmup_proportion`: 学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0.0。
- `num_epochs`: 训练轮数，默认为50。
- `seed`: 随机种子，默认为1000。
- `model_name_or_path`: 选择封闭域模型的编码器，默认为`ernie-layoutx-base-uncased`。
- `logging_steps`: 日志打印的间隔steps数，默认10。
- `eval_steps`: evaluate的间隔steps数，默认200。
- `device`: 选用什么设备进行训练，可选cpu或gpu。
- `init_from_ckpt`: 可选，模型参数路径，热启动模型训练；默认为None。

<a name="24"></a>

## **2.4 模型评估**

```shell
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path data/dev_data.json \
    --label_maps_path data/label_maps.json
```

可配置参数说明：

- `model_path`: 训练好的UIE定制模型路径。
- `test_path`: 测试数据集路径。
- `label_maps_path`: 学生模型标签字典。
- `batch_size`: 批处理大小，默认为8。
- `max_seq_len`: 最大文本长度，默认为512。
- `doc_stride`: 滑窗大小，默认为256.

## **2.5 模型预测及效果展示**

- 通过Taskflow一键部署封闭域信息抽取模型，`task_path`为学生模型路径。

```python
from pprint import pprint
from paddlenlp import Taskflow
from paddlenlp.utils.doc_parser import DocParser

my_ie = Taskflow("information_extraction", model="global-pointer", task_path="checkpoint/model_best/") # Schema is fixed in closed-domain information extraction
```

我们可以根据设置的`schema`，对指定的`doc_path`文档进行信息抽取并进行可视化：

```python
doc_path = "../data/images/b199.jpg"
results = my_ie({"doc": doc_path})
pprint(results)

# 结果可视化
DocParser.write_image_with_results(
    doc_path,
    result=results[0],
    save_path="./image_show.png")
```

<a name="26"></a>

## **2.6 模型部署**

参考[封闭域文本抽取HTTP部署指南](./deploy/simple_serving/README.md)


# References

- **[GlobalPointer](https://kexue.fm/search/globalpointer/)**

- **[GPLinker](https://kexue.fm/archives/8888)**

- **[JunnYu/GPLinker_pytorch](https://github.com/JunnYu/GPLinker_pytorch)**

- **[CBLUE](https://github.com/CBLUEbenchmark/CBLUE)**
