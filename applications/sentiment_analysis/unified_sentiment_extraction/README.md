# 通用情感信息抽取

**目录**
- [1. 应用简介](#1)
- [2. 快速开始](#2)
  - [2.1 运行环境](#2.1) 
  - [2.2 代码结构](#2.2) 
  - [2.3 快速开始：从输入数据到分析结果可视化](#2.3) 
    - [2.3.1 数据描述](#2.3.1)
    - [2.3.2 批量情感分析](#2.3.2)
    - [2.3.3 情感分析结果可视化](#2.3.3)
  - [2.4 通用情感分析能力](#2.4)
  - [2.5 情感分析可视化使用介绍](#2.5)
  - [2.6 面向垂域定制情感分析](#2.6) 
    - [2.6.1 数据标注](#2.6.1) 
    - [2.6.2 模型训练](#2.6.2) 
  - [2.7 模型部署](#2.7)
  
<a name="1"></a>

## 1. 应用简介

PaddleNLP情感分析应用立足真实企业用户对情感分析方面的需求，同时针对情感分析领域的痛点和难点，基于前沿模型开源了细粒度的情感分析解决方案，助力开发者快速分析业务相关产品或服务的用户感受。本项目以通用信息抽取模型UIE为训练底座，同时利用大量情感分析数据进行训练，增强了模型对于情感知识的处理能力，并通过信息抽取的方式解决情感分析相应问题。

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/199965793-f0933baa-5b82-47da-9271-ba36642119f8.png" />
</div>
<br>


**方案亮点**：

- **提供强大训练基座，覆盖情感分析多项基础能力🏃**： 本项目以通用信息抽取模型UIE为训练底座，并基于大量情感分析数据进行训练，增强了模型对于情感知识的处理能力，同时支持常见的基础情感分析能力。
- **用户友好的情感分析方案，从输入数据直达分析结果可视化✊**： 打通了从数据输入到情感分析结果可视化的流程，帮助开发者可以更轻松地获取情感分析结果，更聚焦于业务分析。
- **支持定制面向垂域的情感分析能力，解决同义属性聚合以及隐性观点抽取👶**： 在某项产品/服务垂域场景中，业务方可能比较关注的属性有限的。本项目支持根据预先给定的属性集进行情感分析，同时支持解决常见情感分析过程中的同义属性聚合以及隐性观点抽取功能。

<a name="2"></a>

## 2. 快速开始

<a name="2.1"></a>

### 2.1 运行环境
- python >= 3.6
- paddlepaddle >= 2.3
- paddlenlp >= 2.4
- wordcloud >= 1.8.2

**安装PaddlePaddle**：

环境中paddlepaddle-gpu或paddlepaddle版本应大于或等于2.3, 请参见[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)根据自己需求选择合适的PaddlePaddle下载命令。

**安装PaddleNLP**：
安装PaddleNLP可以开启百度镜像源来加速下载，更多关于PaddleNLP安装的详细教程请查见[PaddleNLP快速安装](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/get_started/installation.rst)。

```shell
python3 -m pip install --upgrade paddlenlp -i https://mirror.baidu.com/pypi/simple
```

**安装wordcloud**：

```shell
python3 -m pip install wordcloud==1.8.2.2
```

<a name="2.2"></a>

### 2.2 代码结构
```
unified_sentiment_extraction/  
├── predict # 模型预测
│   └── predictor.py # 模型预测核心脚本
│   ├── predict.py # 模型预测Demo脚本
│   ├── batch_predict.py # 模型批量预测脚本
│   ├── predict_with_aspect.py # 根据给定评价属性进行预测Demo脚本
│   └── batch_predict_with_aspect # 根据给定评价属性进行批量预测脚本
├── train.py # 训练评估脚本
├── model.py # 模型定义脚本
├── finetune.py # 模型微调脚本
├── predict.py # 预测脚本
├── export_model.py # 静态图模型导出脚本
├── utils.py # 工具函数脚本
├── visual_analysis.py # 情感分析结果可视化脚本
└── README.md # 使用说明
```

<a name="2.3"></a>
### 2.3 快速开始：从输入数据到分析结果可视化

为增强通用信息抽取模型UIE对于情感分析知识的处理能力，本项目基于大量情感分析数据进行了训练，以更好地支持常见的基础情感分析能力，可以点击[这里](https://paddlenlp.bj.bcebos.com/applications/sentiment_analysis/sentiment_uie.tar.gz)进行下载，下载解压后，可将模型放至当前目录`./checkpoint/model_best`。

另外，在分析结果可视化时，如果需要在词云中显示中文，需要指定字体路径，这里可以使用黑体进行显示，点击[这里](https://paddlenlp.bj.bcebos.com/applications/sentiment_analysis/SimHei.ttf)进行下载。

<a name="2.3.1"></a>

#### 2.3.1 数据描述
输入数据如下方式进行组织，每行表示一个文本评论。

```
非常好的酒店 不枉我们爬了近一个小时的山，另外 大厨手艺非常棒 竹筒饭 竹筒鸡推荐入住的客人必须要点，
房间隔音效果不好，楼下KTV好吵的
酒店的房间很大，干净舒适，服务热情
怎么说呢，早上办理入住的，一进房间闷热的一股怪味，很臭，不能开热风，好多了，虽然房间小，但是合理范围
总台服务很差，房间一般
```

<a name="2.3.2"></a>

#### 2.3.2 快速情感分析

可使用`predict/batch_predict.py`文件进行情感预测，默认情况下，会自动分析文本评论中的属性、观点词和情感极性，分析完成后会保存情感分析结果，用以后续可视化。

```
python predict/batch_predict.py \
    --ckpt_dir "./checkpoint/model_best" \
    --test_set_path "./data/test_hotel.txt" \
    --save_path "./outputs/test_hotel.json" \
    --position_prob 0.5 \
    --max_seq_len 512 \
    --batch_size 8 
````

**参数说明**：
- ckpt_dir: 用于加载模型的保存目录，可以在上述模型下载后，进行解压，然后传入该模型目录。
- test_set_path： 指定测试集文件路径。
- save_path： 在进行情感分析预测后，分析结果的保存地址，该结果可用于后续数据可视化。
- position_prob：模型对于span的起始位置/终止位置的结果概率 0~1 之间，返回结果去掉小于这个阈值的结果，默认为 0.5，span 的最终概率输出为起始位置概率和终止位置概率的乘积。
- max_seq_len: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为 512。
- batch_size: 批处理大小，请结合机器情况进行调整，默认为 4。

<a name="2.3.3"></a>
#### 2.3.3 情感分析结果可视化
基于以上生成的情感分析结果，可以使用`visual_analysis.py`脚本对情感分析结果进行可视化，命令如下。

```
python visual_analysis.py \
    --file_path "./outputs/test_hotel.json" \
    --save_dir "./images" \
    --font_path "./SimHei.ttf" \
    --sentiment_name "情感倾向[正向,负向]"
```
**参数说明**：
- file_path: 指定情感分析结果的保存路径。
- save_dir: 指定图片的保存目录。
- font_path: 指定字体文件的路径，用以在生成的wordcloud图片中辅助显示中文。
- sentiment_name: 情感分析的Prompt文本，如果在预先给定的属性集上进行情感分析，则需要指定为"情感倾向[正向,负向,未提及]"，如果不预先指定属性，则需要指定为"情感倾向[正向,负向]"。

在执行完成后，可以在`save_dir`指定的目录下查看可视化结果，部分结果示例如下。

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/200259473-434888f7-c0ac-4253-ab23-ede1628e6ba2.png" />
</div>
<br>

<a name="2.4"></a>
### 2.4 通用情感分析能力
本项目以通用信息抽取模型UIE为训练底座，并基于大量情感分析数据进行训练，增强了模型对于情感知识的处理能力，同时支持常见的基础情感分析能力。从使用方式来看，可分为两类：预先给定属性集和不给定属性集，如果预先给定了属性集，则只会在该属性集上进行情感分析。默认情况下可不给定属性集。

<a name="2.4.1"></a>

#### 2.4.1 不给定属性集

对给定的文本评论，直接进行情感分析。可以在`predict/predict.py`或`predict/batch_predict.py`文件中，通过设置不同的Schema进行相应信息的抽取。其中`predict/predict.py`即时运行情感分析功能，`predict/batch_predict.py`会接收文件，同时将结果保存相应文件中。

**（1）整句情感分析**  
整句情感分析功能当前支持二分类：正向和负向。可以设置其schema为：

```python
schema =  ['情感倾向[正向，负向]']
```
在`predict/predict.py`文件中设置schema后，可以通过如下代码进行运行。

```shell
python predict/predict.py \
    --ckpt_dir "./checkpoint/model_best" \
    --position_prob 0.5 \
    --max_seq_len 512 \
    --batch_size 8 
```
**参数说明**：
- ckpt_dir: 用于加载模型的保存目录，可以在上述模型下载后，进行解压，然后传入该模型目录。
- position_prob：模型对于span的起始位置/终止位置的结果概率 0~1 之间，返回结果去掉小于这个阈值的结果，默认为 0.5，span 的最终概率输出为起始位置概率和终止位置概率的乘积。
- max_seq_len: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为 512。
- batch_size: 批处理大小，请结合机器情况进行调整，默认为 4。、

**（2）属性级情感分析**  
除整句情感分析之外，本项目同时支持属性级情感分析，包括属性抽取（Aspect Term Extraction）、观点抽取（Opinion Term Extraction）、属性级情感分析（Aspect Based Sentiment Classification）等等。可以通过设置相应的schema进行对应信息的抽取。

```python
# Aspect Term Extraction
schema =  ["评价维度"]
# Aspect - Opinion Extraction
schema =  [{"评价维度":["观点词"]}]
# Aspect - Sentiment Extraction
schema =  [{"评价维度":["情感倾向[正向，负向]"]}]
# Aspect - Sentiment - Opinion Extraction
schema =  [{"评价维度":["观点词", "情感倾向[正向，负向]"]}]
```

在设置好schema之后，可以通过`predict/predict.py`文件进行情感预测。


<a name="2.4.2"></a>

#### 2.4.2 预先给定属性集

本项目支持在预先给定的属性集上进行情感分析，需要注意的是，如果预先给定了属性集，则只会在该属性集上进行情感分析，分析和抽取该属性级中各个属性的信息。可以通过`predict/predict_with_aspect.py`或`predict/batch_predict_with_aspect.py`文件定义的预测方式进行实现。其中，在`predict/predict_with_aspect.py`定义的属性级和schema如下。

```python
    # define schema for pre-defined aspects, schema and initializing UIEPredictor
    schema = ["观点词", "情感倾向[正向,负向,未提及]"]
    aspects = ["房间","服务", "环境", "位置", "隔音", "价格"]
```
其表示对指定数据集，分析"房间","服务", "环境", "位置", "隔音" 和 "价格"的观点词和情感倾向。 在给定属性级的模式下，在进行情感倾向预测是需要设置prompt为`"情感倾向[正向,负向,未提及]"`，其中通过`未提及`来指明某些属性在当前文本评论中并未涉及。在设置完成后，可通过如下命令进行情感分析预测。

```shell
python predict/predict_with_aspect.py \
    --ckpt_dir "./checkpoint/model_best" \
    --position_prob 0.5 \
    --max_seq_len 512 \
    --batch_size 8 
```
其可配置参数解释同`predict/predict.py`。


<a name="2.5"></a>

### 2.5 情感分析可视化使用介绍

基于情感分析的预测结果，本项目提供了结果可视化功能。默认情况下，可视化功能支持围绕属性、观点、属性+观点、属性+情感、固定属性+观点分析功能。在各项分析中，均支持词云和直方图两类图像展示。以下各项功能介绍中，以酒店场景数据为例进行展示。

 **(1) 属性分析**  
通过属性信息，可以查看客户对于产品/服务的重点关注方面. 可以通过`plot_aspect_with_frequency`函数对属性进行可视化，当前可通过参数`image_type`分别指定`wordcloud`和'histogram'，通过词云和直方图的形式进行可视化。

```python
# define VisualSentiment to help visualization
vs = VisualSentiment(font_path=args.font_path)

# visualization for aspect
save_path = os.path.join(args.save_dir, "aspect_wc.png")
vs.plot_aspect_with_frequency(sr.aspect_frequency, save_path, image_type="wordcloud")
save_path = os.path.join(args.save_dir, "aspect_hist.png")
vs.plot_aspect_with_frequency(sr.aspect_frequency, save_path, image_type="histogram")
```

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/200250669-7e06c742-ce62-4d2d-90f4-89efd7f6298c.png" />
</div>
<br>

 **(2) 观点分析**   
通过观点信息，可以查看客户对于产品/服务整体的直观印象。可以通过`plot_opinion_with_frequency`函数对观点进行可视化。

```python
# visualization for opinion
save_path = os.path.join(args.save_dir, "opinion_wc.png")
vs.plot_opinion_with_frequency(sr.opinion_frequency, save_path, image_type="wordcloud")
```

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/200251285-741881b5-8910-4152-a5c1-df34affaed42.png" />
</div>
<br>

**(3) 属性+观点分析**  
结合属性和观点两者信息，可以更加具体的展现客户对于产品/服务的详细观点，分析某个属性的优劣，从而能够帮助商家更有针对性地改善或提高自己的产品/服务质量。可以通过`plot_aspect_with_opinion`函数对属性+观点进行可视化，同时可通过设置参数`sentiment`按照情感倾向展示不同分析结果，以更好进行情感分析，若设置为`all`，则会展示正向和负向所有的属性；若为`positive`，则会仅展示正向的属性；若为`negative`，则会仅展示负向的属性。如果在绘制直方图时，通过设置参数`top_n`，可以展示频率最高的top n个属性。

```python
# visualization for aspect + opinion
save_path = os.path.join(args.save_dir, "aspect_opinion_wc.png")
vs.plot_aspect_with_opinion(sr.aspect_opinion, save_path, image_type="wordcloud", sentiment="all")
save_path = os.path.join(args.save_dir, "aspect_opinion_hist.png")
vs.plot_aspect_with_opinion(sr.aspect_opinion, save_path, image_type="histogram", sentiment="all", top_n=8)
```

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/199974942-8e55aabd-6c35-48ec-8f6d-3270b67b299c.png"/>
</div>


 **(4) 属性+情感分析**  
挖掘客户对于产品/服务针对属性的情感极性，帮助商家直观地查看客户对于产品/服务的某些属性的印象。可以通过`plot_aspect_with_sentiment`函数对属性+情感进行可视化。如果在绘制直方图时，通过设置参数`top_n`，可以展示频率最高的top n个属性。

```python
# visualization for aspect + sentiment
save_path = os.path.join(args.save_dir, "aspect_sentiment_wc.png")
vs.plot_aspect_with_sentiment(sr.aspect_sentiment, save_path, image_type="wordcloud")
save_path = os.path.join(args.save_dir, "aspect_sentiment_hist.png")
vs.plot_aspect_with_sentiment(sr.aspect_sentiment, save_path, image_type="histogram", top_n=15, descend_aspects=sr.descend_aspects)
```

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/200213177-0342bec4-5955-4ab9-9e98-5e4ef8e1a35e.png"/>
</div>

**(5) 对给定属性进行观点分析**  
通过指定属性，更加细致查看客户对于产品/服务某个属性的观点。可以帮助商家更加细粒度地分析客户对于产品/服务的某个属性的印象。下面图片示例中，展示了客户对于属性"房间"的观点。可以通过`plot_opinion_with_aspect`函数，对给定的属性进行观点分析。默认情况下，不会自动生成该类图像，需要开发者手动调用`plot_opinion_with_aspect`进行可视化分析。

```python
aspect = "房间"
save_path = os.path.join(args.save_dir, "opinions_for_aspect_wc.png")
vs.plot_opinion_with_aspect(aspect, sr.aspect_opinion, save_path, image_type="wordcloud")
save_path = os.path.join(args.save_dir, "opinions_for_aspect_hist.png")
vs.plot_opinion_with_aspect(aspect, sr.aspect_opinion, save_path, image_type="histogram")
```

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/200213998-e646c422-7ab5-48ae-9e28-d6068cdf7b8f.png"/>
</div>



<a name="2.6"></a>

### 2.6 支持定制面向垂域的情感分析能力，解决同义属性聚合以及隐性观点抽取

<a name="2.6.1"></a>

#### 2.6.1 数据标注

<a name="2.6.2"></a>

#### 2.6.2 模型训练



<a name="2.7"></a>

### 2.7 模型部署


