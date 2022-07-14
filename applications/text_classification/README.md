# 文本分类任务指南

**目录**
   * [文本分类任务介绍](#文本分类任务介绍)
   * [基于预训练模型的文本分类任务微调](#基于预训练模型的文本分类任务微调)
   * [文本分类应用全流程介绍](#文本分类应用全流程介绍)
   * [文本分类指南](#文本分类指南)

## 文本分类任务介绍

文本分类任务是自然语言处理中最常见的任务，文本分类任务简单来说就是对给定的一个句子或一段文本使用文本分类器进行分类。文本分类任务广泛应用于**长短文本分类、情感分析、新闻分类、事件类别分类、政务数据分类、商品信息分类、商品类目预测、文章分类、论文类别分类、专利分类、案件描述分类、罪名分类、意图分类、论文专利分类、邮件自动标签、评论正负识别、药物反应分类、对话分类、税种识别、来电信息自动分类、投诉分类、广告检测、敏感违法内容检测、内容安全检测、舆情分析、话题标记**等各类日常或专业领域中。

文本分类任务可以根据标签类型分为多分类（multi class）、多标签（multi label）、层次分类（hierarchical）等三类任务，接下来我们将以下图的新闻文本分类为例介绍三种分类任务的区别。

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/178486882-bcf797a8-5a07-420c-acbb-837bef5c80b5.jpg />
</div>

- **多分类**（Multi class）是最常见的文本分类类型，多分类数据集的标签集含有两个或两个以上的类别，所有输入句子/文本有且只有一个标签。在文本多分类任务中，我们需要预测输入句子/文本最可能来自 `n` 个标签类别中的哪一个类别。以上图多分类中新闻文本为例，该新闻分类具有一个标签为 `娱乐`。

- **多标签**（Multi label）数据集的标签集含有两个或两个以上的类别，输入句子/文本具有一个或多个标签。在文本多标签任务中，我们需要预测输入句子/文本可能来自 `n` 个标签类别中的哪几个类别。以上图多标签中新闻文本为例，该新闻分类具有 `相机` 和 `芯片` 两个标签。

- **层次分类**（Hierarchical）数据集的标签集具有多级标签且标签之间具有层级结构关系，输入句子/文本具有一个或多个标签。在文本层次分类任务中，我们需要预测输入句子/文本可能来自于不同级标签类别中的某一个或几个类别。以上图层次分类中新闻文本为例（新闻为根节点），该新闻一级分类标签为 `体育`，二级分类标签为 `足球`。并且由于标签间存在层次结构关系，如果具有二级分类标签 `足球`，那么必然存在二级分类标签的父节点 `体育`。

## 基于预训练模型的文本分类任务
训练一个文本分类器对待预测的句子或文本进行分类，目前最常用、效果最好的文本分类方法是对预训练语言模型（Pre-trained Language Model, PLM）进行微调得到文本分类器。

预训练模型是在超大规模的语料采用无监督（unsupervised）或者弱监督（weak-supervised）的方式训练模型，期望模型能够获得语言相关的知识，比如句法，语法知识等等，预训练模型编码得到[CLS]输出向量被视为输入文本语义表示。然后再利用预训练模型[CLS]输出层后叠加线性层去训练不同的文本分类任务，使得预训练模型”更懂”这个任务。例如，利用预训练好的模型继续训练多分类任务，将会获得比较好的一个分类结果，直观地想，预训练模型已经懂得了语言的知识，在这些知识基础上去学习文本分类任务将会事半功倍。

PaddleNLP采用`AutoModelForSequenceClassification`, `AutoTokenizer`提供了方便易用的接口，可指定模型名或模型参数文件路径通过`from_pretrained()`方法加载不同网络结构的预训练模型,并在输出层上叠加一层线性层，且相应预训练模型权重下载速度快、稳定。[Transformer预训练模型汇总](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer)包含了如 `ERNIE`、`BERT`、`RoBERTa`等40多个主流预训练模型，500多个模型权重。下面以ERNIE 3.0 中文base模型为例，演示如何加载预训练模型和分词器：

```shell
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
num_classes = 10
model_name = "ernie-3.0-base-zh"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=num_classes)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```
在中文领域内 **ERNIE 3.0** 系列的模型在模型效果和模型计算效率都是相对比较突出的，因此在文本分类的预训练模型选择主要是ERNIE 3.0 系列模型为主，下面是 ERNIE 3.0 batch_size=32 和 1，预测精度为 FP16 时，GPU 下的效果-时延图，具体的测评细节可以见[**ERNIE 3.0 效果和性能测评文档**](../../model_zoo/ernie-3.0)：
<img width="1030" alt="image" src="https://user-images.githubusercontent.com/16698950/178405140-4b5885ee-dcb8-4d67-8cd6-9aa1fdb98e92.png">


## 文本分类应用全流程介绍

接下来，我们将按数据准备、训练、性能优化部署等三个阶段对文本分类应用的全流程进行介绍。

<div align="center">
    <img width="1238" alt="image" src="https://user-images.githubusercontent.com/16698950/178186513-565e29ec-95d4-4368-8382-cab59b27d94c.png">
</div>

1. **数据准备**
- 如果没有已标注的数据集，我们推荐doccano数据标注工具，如何使用doccano进行数据标注并转化成指定格式本地数据集详见[文本分类任务doccano使用指南](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/applications/text_classification/doccano.md)。如果已有标注好的本地数据集，我们需要根据不同任务要求将数据集整理为文档要求的格式：[多分类数据集格式要求](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_class#%E4%BB%8E%E6%9C%AC%E5%9C%B0%E6%96%87%E4%BB%B6%E5%88%9B%E5%BB%BA%E6%95%B0%E6%8D%AE%E9%9B%86)、[多标签数据集格式要求](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_label#%E4%BB%8E%E6%9C%AC%E5%9C%B0%E6%96%87%E4%BB%B6%E5%88%9B%E5%BB%BA%E6%95%B0%E6%8D%AE%E9%9B%86)、[层次分类数据集格式要求](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/hierarchical#%E4%BB%A5%E5%86%85%E7%BD%AE%E6%95%B0%E6%8D%AE%E9%9B%86%E6%A0%BC%E5%BC%8F%E8%AF%BB%E5%8F%96%E6%9C%AC%E5%9C%B0%E6%95%B0%E6%8D%AE%E9%9B%86)。
- 准备好数据集后，我们可以根据现有的数据集规模或训练后模型表现选择是否使用数据增强策略进行数据集扩充。（目前数据增强工具正在开发中，敬请期待）

2. **模型训练**

- 数据准备完成后，可以开始使用我们的数据集对预训练模型进行微调训练。我们可以根据任务需求，调整可配置参数，选择使用GPU或CPU进行模型训练，脚本默认保存在开发集最佳表现模型。中文任务默认使用"ernie-3.0-base-zh"模型，英文任务默认使用"ernie-2.0-base-en"模型，ERNIE 3.0还支持多个轻量级中文模型，详见[ERNIE模型汇总](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers/ERNIE/contents.html)，可以根据任务和设备需求进行选择。
- 首先我们需要根据场景选择不同的任务目录，具体可以见
 [多分类任务点击这里](./multi_class)
 [多标签任务点击这里](./multi_label)
 [层次分类任务点击这里](./hierarchical)

- 训练结束后，我们可以加载保存的最佳模型进行模型测试，打印模型预测结果。

3. **模型预测**

- 在现实部署场景中，我们通常不仅对模型的精度表现有要求，也需要考虑模型性能上的表现。我们可以使用模型裁剪进一步压缩模型体积，文本分类应用已提供裁剪API对上一步微调后的模型进行裁剪，模型裁剪之后会默认导出静态图模型。

- 模型部署需要将保存的最佳模型参数（动态图）导出成静态图参数，用于后续的推理部署。

- 文本分类应用提供了基于ONNXRuntime的本地部署predictor，并且支持在GPU设备使用FP16，在CPU设备使用动态量化的低精度加速推理。

- 文本分类应用同时基于Paddle Serving的服务端部署方案。（目前低精度加速推理正在开发中，敬请期待）

## 文本分类指南

更多文本分类应用具体使用细节详见：

- [文本多分类任务指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_class#%E6%96%87%E6%9C%AC%E5%A4%9A%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1%E6%8C%87%E5%8D%97)

- [文本多标签分类任务指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_label#%E5%A4%9A%E6%A0%87%E7%AD%BE%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1)

- [文本层次分类任务指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/hierarchical#%E5%A4%9A%E6%A0%87%E7%AD%BE%E5%B1%82%E6%AC%A1%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1)
