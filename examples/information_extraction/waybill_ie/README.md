# 快递单信息抽取 (Waybill Information Extraction)

## 简介

本示例将通过BiGRU-CRF和ERNIE + FC两类模型，演示如何从用户提供的快递单中，抽取姓名、电话、省、市、区、详细地址等内容，形成结构化信息。辅助物流行业从业者进行有效信息的提取，从而降低客户填单的成本。

## 快速开始

### 数据准备

执行以下命令，下载并解压示例数据集：

```bash
python download.py --data_dir ./waybill_ie
```

数据示例如下：

```
1^B6^B6^B2^B0^B2^B0^B0^B0^B7^B7^B宣^B荣^B嗣^B甘^B肃^B省^B白^B银^B市^B会^B宁^B县^B河^B畔^B镇^B十^B字^B街^B金^B海^B超^B市^B西^B行^B5^B0^B米    T-B^BT-I^BT-I^BT-I^BT-I^BT-I^BT-I^BT-I^BT-I^BT-I^BT-I^BP-B^BP-I^BP-I^BA1-B^BA1-I^BA1-I^BA2-B^BA2-I^BA2-I^BA3-B^BA3-I^BA3-I^BA4-B^BA4-I^BA4-I^BA4-I^BA4-I^BA4-I^BA4-I^BA4-I^BA4-I^BA4-I^BA4-I^BA4-I^BA4-I^BA4-I^BA4-I
1^B3^B5^B5^B2^B6^B6^B4^B3^B0^B7^B姜^B骏^B炜^B云^B南^B省^B德^B宏^B傣^B族^B景^B颇^B族^B自^B治^B州^B盈^B江^B县^B平^B原^B镇^B蜜^B回^B路^B下^B段    T-B^BT-I^BT-I^BT-I^BT-I^BT-I^BT-I^BT-I^BT-I^BT-I^BT-I^BP-B^BP-I^BP-I^BA1-B^BA1-I^BA1-I^BA2-B^BA2-I^BA2-I^BA2-I^BA2-I^BA2-I^BA2-I^BA2-I^BA2-I^BA2-I^BA3-B^BA3-I^BA3-I^BA4-B^BA4-I^BA4-I^BA4-I^BA4-I^BA4-I^BA4-I^BA4-I
```
数据集中以特殊字符"\t"分隔文本、标签，以特殊字符"\002"(示例中显示为"^B")分隔每个字。标签的定义如下：

| 标签 | 定义 |  标签 | 定义 |
| -------- | -------- |-------- | -------- |
| P-B | 姓名起始位置 | P-I | 姓名中间位置或结束位置 |
| T-B | 电话起始位置 | T-I | 电话中间位置或结束位置 |
| A1-B | 省份起始位置 | A1-I | 省份中间位置或结束位置 |
| A2-B | 城市起始位置 | A2-I | 城市中间位置或结束位置 |
| A3-B | 县区起始位置 | A3-I | 县区中间位置或结束位置 |
| A4-B | 详细地址起始位置 | A4-I | 详细地址中间位置或结束位置 |
| O | 无关字符 | | |

数据标注采用**BIO模式**。其中 B(begin) 表示一个标签类别的开头，比如 P-B 指的是姓名的开头；相应的，I(inside) 表示一个标签的延续。O表示Outside无关字符。更多标注模式介绍请参考[Inside–outside–beginning (tagging)](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))

### 启动训练

本项目提供了两种模型结构，一种是BiGRU+CRF结构，另一种是ERNIE+FC结构，前者显存占用小，推理速度快；后者能够在更快收敛并取得更高的精度，但推理速度较慢。

#### 启动BiGRU + CRF训练

```bash
export CUDA_VISIBLE_DEVICES=0
python run_bigru_crf.py
```

#### 启动ERNIE + FC训练

```bash
export CUDA_VISIBLE_DEVICES=0
python run_ernie.py
```
##### 模型导出
使用动态图训练结束之后，还可以将动态图参数导出成静态图参数，具体代码见export_model.py。静态图参数保存在output_path指定路径中。 运行方式：

`python export_model.py --params_path ernie_ckpt/model_80/model_state.pdparams --output_path=./output`

其中`params_path`是指动态图训练保存的参数路径，`output_path`是指静态图参数导出路径。

导出模型之后，可以用于部署，deploy/python/predict.py文件提供了python部署预测示例。运行方式：

`python deploy/python/predict.py --model_dir ./output`


#### 启动ERNIE + CRF训练

```bash
export CUDA_VISIBLE_DEVICES=0
python run_ernie_crf.py
```

## 更多详细教程请参考：

[基于Bi-GRU+CRF的快递单信息抽取](https://aistudio.baidu.com/aistudio/projectdetail/1317771)

[使用预训练模型ERNIE优化快递单信息抽取](https://aistudio.baidu.com/aistudio/projectdetail/1329361)
