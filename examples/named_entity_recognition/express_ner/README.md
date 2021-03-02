# 快递单信息抽取

## 1. 简介

本项目将演示如何从用户提供的快递单中，抽取姓名、电话、省、市、区、详细地址等内容，形成结构化信息。辅助物流行业从业者进行有效信息的提取，从而降低客户填单的成本。

## 2. 快速开始

### 2.1 环境配置

- Python >= 3.6

- paddlepaddle >= 2.0.0，安装方式请参考 [快速安装](https://www.paddlepaddle.org.cn/install/quick)。

- paddlenlp >= 2.0.0rc4, 安装方式：`pip install paddlenlp\>=2.0.0rc4`


### 2.2 数据准备

数据集已经保存在data目录中，示例如下

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

注意每个标签的结果只有 B、I、O 三种，这种标签的定义方式叫做 BIO 体系。其中 B 表示一个标签类别的开头，比如 P-B 指的是姓名的开头；相应的，I 表示一个标签的延续。

### 2.3 启动训练

本项目提供了两种模型结构，一种是BiGRU+CRF结构，另一种是ERNIE+FC结构，前者显存占用小，推理速度快；后者能够在更快收敛并取得更高的精度，但推理速度较慢。

#### 2.3.1 启动BiGRU + CRF训练

```bash
export CUDA_VISIBLE_DEVICES=0 # 只支持单卡训练
python run_bigru_crf.py
```

详细介绍请参考教程：[基于Bi-GRU+CRF的快递单信息抽取](https://aistudio.baidu.com/aistudio/projectdetail/1317771)

#### 2.3.2 启动ERNIE + FC训练

```bash
export CUDA_VISIBLE_DEVICES=0 # 只支持单卡训练
python run_ernie.py
```

详细介绍请参考教程：[使用PaddleNLP预训练模型ERNIE优化快递单信息抽取](https://aistudio.baidu.com/aistudio/projectdetail/1329361)
