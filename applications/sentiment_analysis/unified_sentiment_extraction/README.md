# 通用情感信息抽取

**目录**
- [1. 应用简介](#1)
- [2. 快速开始](#2)
  - [2.1 运行环境](#2.1)
  - [2.2 代码结构](#2.2)
  - [2.3 开箱即用：从输入数据到分析结果可视化](#2.3)
    - [2.3.1 数据描述](#2.3.1)
    - [2.3.2 批量情感分析](#2.3.2)
    - [2.3.3 情感分析结果可视化](#2.3.3)
  - [2.4 通用情感分析能力](#2.4)
  - [2.5 情感分析可视化使用介绍](#2.5)
  - [2.6 面向垂域定制情感分析，解决同义属性聚合以及隐性观点抽取](#2.6)
    - [2.6.1 打通数据标注到训练样本构建](#2.6.1)
    - [2.6.2 模型训练](#2.6.2)
    - [2.6.3 模型测试](#2.6.3)
    - [2.6.4 预测及效果展示](#2.6.4)
  - [2.7 模型部署](#2.7)

<a name="1"></a>

## **1. 应用简介**

PaddleNLP情感分析应用立足真实企业用户对情感分析方面的需求，同时针对情感分析领域的痛点和难点，基于前沿模型开源了细粒度的情感分析解决方案，助力开发者快速分析业务相关产品或服务的用户感受。本项目以通用信息抽取模型UIE为训练底座，同时利用大量情感分析数据进行训练，增强了模型对于情感知识的处理能力，并通过信息抽取的方式解决情感分析相应问题。

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/199965793-f0933baa-5b82-47da-9271-ba36642119f8.png" />
</div>

**方案亮点**：

- **提供强大训练基座，覆盖情感分析多项基础能力🏃**： 本项目以通用信息抽取模型UIE为训练底座，并基于大量情感分析数据进行训练，增强了模型对于情感知识的处理能力，同时支持常见的基础情感分析能力。
- **用户友好的情感分析方案，从输入数据直达分析结果可视化✊**： 打通了从数据输入到情感分析结果可视化的流程，帮助开发者可以更轻松地获取情感分析结果，更聚焦于业务分析。
- **支持定制面向垂域的情感分析能力，解决同义属性聚合以及隐性观点抽取👶**： 在某项产品/服务垂域场景中，业务方可能比较关注的属性有限的。本项目支持根据预先给定的属性集进行情感分析，同时支持解决常见情感分析过程中的同义属性聚合以及隐性观点抽取功能。

<a name="2"></a>

## **2. 快速开始**

<a name="2.1"></a>

### **2.1 运行环境**
- python >= 3.7
- paddlepaddle >= 2.3
- paddlenlp >= 2.4.4
- wordcloud >= 1.8.2

**安装PaddlePaddle**：

环境中paddlepaddle-gpu或paddlepaddle版本应大于或等于2.3, 具体可以参见[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)根据自己需求选择合适的PaddlePaddle下载命令。如下命令可以安装linux系统，CUDA版本为10.2环境下的paddlepaddle，具体版本号为支持GPU的2.3.2版本。

```shell
conda install paddlepaddle-gpu==2.3.2 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```

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

### **2.2 代码结构**
```
unified_sentiment_extraction/
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
### **2.3 开箱即用：从输入数据到分析结果可视化**

为增强通用信息抽取模型UIE对于情感分析知识的处理能力，本项目基于大量情感分析数据进行了训练，以更好地支持常见的基础情感分析能力。同时该模型集成到了 Taskflow 中，可以通过 TaskFlow 提供的API直接进行情感分析。

另外，在分析结果可视化时，如果需要在词云中显示中文，需要指定字体路径，这里可以使用黑体进行显示，点击[这里](https://paddlenlp.bj.bcebos.com/applications/sentiment_analysis/SimHei.ttf)进行下载。

<a name="2.3.1"></a>

#### **2.3.1 数据描述**
输入数据如下方式进行组织，每行表示一个文本评论。

```
非常好的酒店 不枉我们爬了近一个小时的山，另外 大厨手艺非常棒 竹筒饭 竹筒鸡推荐入住的客人必须要点，
房间隔音效果不好，楼下KTV好吵的
酒店的房间很大，干净舒适，服务热情
怎么说呢，早上办理入住的，一进房间闷热的一股怪味，很臭，不能开热风，好多了，虽然房间小，但是合理范围
总台服务很差，房间一般
```

<a name="2.3.2"></a>

#### **2.3.2 开箱即用：情感分析**

基于UIE的情感分析功能已经集成到了 Taskflow，可以通过 Taskflow 直接进行情感分析预测。为自动分析文本评论中的属性、观点词和情感极性，定义 schema 如下：
```
[{'评价维度': ['观点词', '情感倾向[正向,负向,未提及]']}]
```

调用示例如下：

```python
>>> from paddlenlp import Taskflow

>>> schema = [{'评价维度': ['观点词', '情感倾向[正向,负向,未提及]']}]
>>> senta = Taskflow("sentiment_analysis", model="uie-base", schema=schema)
>>> print(senta('蛋糕味道不错，店家服务也很好'))

[
    {
        '评价维度': [
            {
                'end': 11,
                'probability': 0.9978359659431995,
                'relations': {
                    '情感倾向[正向,负向,未提及]': [
                        {
                            'probability': 0.9999209657300412,
                            'text': '正向'
                        }
                    ],
                    '观点词': [
                        {
                            'end': 14,
                            'probability': 0.9998056980106895,
                            'start': 13,
                            'text': '好'
                        }
                    ]
                },
                'start': 9,
                'text': '服务'
            },
            {
                'end': 4,
                'probability': 0.9152585827976623,
                'relations': {
                    '情感倾向[正向,负向,未提及]': [
                        {
                            'probability': 0.9998750722759127,
                            'text': '正向'
                        }
                    ],
                    '观点词': [
                        {
                            'end': 6,
                            'probability': 0.9855414084471477,
                            'start': 4,
                            'text': '不错'
                        }
                    ]
                },
                'start': 2,
                'text': '味道'
            }
        ]
    }
]
```

当数据规模较大时，可以以文件的形式传入数据，同时将分析结果保存至指定的文件中，文件内容可以按照如下方式进行组织，每行表示一条样本:
```
酒店环境很好，安静，周边餐饮很方便
环境美，房间不大，但舒适温馨，老板服务周到贴心
房间整洁，设施完好，客服人员不错
```

支持以文件形式批量传入的代码示例如下：
```python
>>> from paddlenlp import Taskflow

>>> file_path = './data/test_hotel.txt'
>>> save_path = './outputs/test_hotel.json'
>>> schema = [{'评价维度': ['观点词', '情感倾向[正向,负向]']}]
>>> senta = Taskflow("sentiment_analysis", model="uie-base", schema=schema)
>>> senta(file_path, save_path)

```

在执行完以上命令后，情感分析的结果将被保存至 `save_path` 指定的文件中。


<a name="2.3.3"></a>

#### **2.3.3 情感分析结果可视化**

基于以上生成的情感分析结果，可以使用`visual_analysis.py`脚本对情感分析结果进行可视化，命令如下。

```
python visual_analysis.py \
    --file_path "./outputs/test_hotel.json" \
    --save_dir "./outputs/images" \
    --font_path "./SimHei.ttf" \
    --sentiment_name "情感倾向[正向,负向]"
```
**参数说明**：
- file_path: 指定情感分析结果的保存路径。
- save_dir: 指定图片的保存目录。
- font_path: 指定字体文件的路径，用以在生成的wordcloud图片中辅助显示中文。
- sentiment_name: 情感分析的Prompt文本，默认为"情感倾向[正向,负向,未提及]"。

在执行完成后，可以在`save_dir`指定的目录下查看可视化结果，部分结果示例如下。

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/200259473-434888f7-c0ac-4253-ab23-ede1628e6ba2.png" />
</div>
<br>

<a name="2.4"></a>

### **2.4 通用情感分析能力**

本项目以通用信息抽取模型UIE为训练底座，并基于大量情感分析数据进行训练，增强了模型对于情感知识的处理能力，同时支持常见的基础情感分析能力。从使用方式来看，可分为两类：预先给定属性集和不给定属性集，如果预先给定了属性集，则只会在该属性集上进行情感分析。默认情况下可不给定属性集。

<a name="2.4.1"></a>

#### **2.4.1 不给定属性集**

对给定的文本评论，直接进行情感分析。可以在`predict/predict.py`或`predict/batch_predict.py`文件中，通过设置不同的Schema进行相应信息的抽取。其中`predict/predict.py`即时运行情感分析功能，`predict/batch_predict.py`会接收文件，同时将结果保存相应文件中。

**（1）整句情感分析**  
整句情感分析功能当前支持二分类：正向和负向，调用示例如下：

```python
>>> from paddlenlp import Taskflow

>>> schema = ['情感倾向[正向，负向]']
>>> senta = Taskflow("sentiment_analysis", model="uie-base", schema=schema)
>>> print(senta('蛋糕味道不错，店家服务也很好'))
```

**（2）属性级情感分析**  
除整句情感分析之外，本项目同时支持属性级情感分析，包括属性抽取（Aspect Term Extraction）、观点抽取（Opinion Term Extraction）、属性级情感分析（Aspect Based Sentiment Classification）等等。可以通过设置相应的schema进行对应信息的抽取，其调用示例如下。

```python
>>> from paddlenlp import Taskflow

>>> # Aspect Term Extraction
>>> # schema =  ["评价维度"]
>>> # Aspect - Opinion Extraction
>>> # schema =  [{"评价维度":["观点词"]}]
>>> # Aspect - Sentiment Extraction
>>> # schema =  [{"评价维度":["情感倾向[正向，负向]"]}]
>>> # Aspect - Sentiment - Opinion Extraction
>>> schema =  [{"评价维度":["观点词", "情感倾向[正向，负向]"]}]

>>> senta = Taskflow("sentiment_analysis", model="uie-base", schema=schema)
>>> print(senta('蛋糕味道不错，店家服务也很好'))
```

<a name="2.4.2"></a>

#### **2.4.2 预先给定属性集**

本项目支持在预先给定的属性集上进行情感分析，需要注意的是，如果预先给定了属性集，则只会在该属性集上进行情感分析，分析和抽取该属性级中各个属性的信息。在给定属性级的模式下，在进行情感倾向预测是需要设置prompt为`"情感倾向[正向,负向,未提及]"`，其中通过`未提及`来指明某些属性在当前文本评论中并未涉及。

```python
>>> # define schema for pre-defined aspects, schema
>>> schema = ["观点词", "情感倾向[正向,负向,未提及]"]
>>> aspects = ["房间", "位置", "隔音"]
>>> # set aspects for Taskflow
>>> senta = Taskflow("sentiment_analysis", model="uie-base", schema=schema, aspects=aspects)
>>> senta("这家点的房间很大，店家服务也很热情，就是房间隔音不好")
```

<a name="2.5"></a>

### **2.5 情感分析可视化使用介绍**

基于情感分析的预测结果，本项目提供了结果可视化功能。默认情况下，可视化功能支持围绕属性、观点、属性+观点、属性+情感、固定属性+观点分析功能。在各项分析中，均支持词云和直方图两类图像展示。以下各项功能介绍中，以酒店场景数据为例进行展示。

**(1) 属性分析**
通过属性信息，可以查看客户对于产品/服务的重点关注方面. 可以通过`plot_aspect_with_frequency`函数对属性进行可视化，当前可通过参数`image_type`分别指定`wordcloud`和'histogram'，通过词云和直方图的形式进行可视化。

```python
# define SentimentResult to process the result of sentiment result.
sr = SentimentResult(args.file_path, sentiment_name=args.sentiment_name)
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

### **2.6 支持定制面向垂域的情感分析能力，解决同义属性聚合以及隐性观点抽取**
考虑到用户在对业务数据进行情感分析时，往往聚焦于某个特定场景或领域，为满足用户更高的情感分析要求，本项目除了预先设定的通用情感分析能力之外，同时支持进一步地微调，以在当前业务侧获取更好的效果。

本节以酒店场景为例，讲解定制酒店垂域的情感分析能力。接下来，将从数据标注及样本构建 - 模型训练 - 模型测试 - 模型预测及效果展示等全流程展开介绍。

<a name="2.6.1"></a>

#### **2.6.1 打通数据标注到训练样本构建**
本项目打通了标注平台 label-studio， 支持用户自己标注业务侧数据进行模型训练，同时支持将label-studio平台导出数据一键转换成模型训练样本形式，如下图所示。如果对label-studio数据标注规则尚不清楚，请参考[情感分析任务Label Studio使用指南](./label_studio.md)。

在利用 label-studio 导出标注好的json数据之后，本项目提供了`label_studio.py`文件，用于将导出数据一键转换为模型训练数据。

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/203001847-8e41709b-0f5a-4673-8aca-5c4fb7705d4a.png  />
</div>


#### **2.6.1.1 语句级情感分类任务**

对于语句级情感分类任务，可以配置参数`prompt_prefix`和`options`，通过以下命令构造相关训练数据。

```shell
python label_studio.py \
    --label_studio_file ./data/label_studio.json \
    --task_type cls \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --prompt_prefix "情感倾向" \
    --options "正向" "负向"
```

#### **2.6.1.2 属性抽取相关任务**


针对属性抽取式的任务，比如属性、观点抽取、属性分类任务等，本项目提供了基础和升级的功能构造相关训练数据。即除了基础的属性相关信息抽取能力之外，本项目还支持属性聚合，以及加强了对隐性观点抽取的升级功能。

**基础使用方式**：  
可以使用如下命令将label-studio导出数据转换为模型训练数据：

```shell
python label_studio.py \
    --label_studio_file ./data/label_studio.json \
    --task_type ext \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --prompt_prefix "情感倾向" \
    --options "正向" "负向" "未提及" \
    --separator "##" \
    --negative_ratio 5 \
    --is_shuffle True \
    --seed 1000
```

参数介绍：  
- ``label_studio_file``: 从label studio导出的数据标注文件。
- ``task_type``: 选择任务类型，可选有抽取和分类两种类型的任务。
- ``save_dir``: 训练数据的保存目录，默认存储在``data``目录下。
- ``splits``: 划分数据集时训练集、验证集所占的比例。默认为[0.8, 0.1, 0.1]表示按照``8:1:1``的比例将数据划分为训练集、验证集和测试集。
- ``prompt_prefix``: 声明分类任务的prompt前缀信息，该参数只对分类类型任务有效。默认为"情感倾向"。
- ``options``: 指定分类任务的类别标签，该参数只对分类类型任务有效。默认为["正向", "负向", "未提及"]。
- ``separator``: 实体类别/评价维度与分类标签的分隔符，该参数只对实体/评价维度分类任务有效。默认为"##"。
- ``negative_ratio``: 最大负例比例，该参数只对抽取类型任务有效，适当构造负例可提升模型效果。负例数量和实际的标签数量有关，最大负例数量 = negative_ratio * 正例数量。该参数只对训练集有效，默认为5。为了保证评估指标的准确性，验证集和测试集默认构造全负例。
- ``is_shuffle``: 是否对数据集进行随机打散，默认为True。
- ``seed``: 随机种子，默认为1000.


**升级1：支持属性聚合能力**: 在用户对产品或服务进行评论时，对某一些属性可能会有不同的说法，这会在后续对属性分析时可能会带来困扰。如以下示例中的"价格","价钱"和"费用"。

```
蛋糕味道不错，外观很漂亮，而且价格比较便宜
蛋糕味道不错，外观很漂亮，而且价钱比较便宜
蛋糕味道不错，外观很漂亮，而且费用比较便宜
```

本项目通过以下两点，支持对属性聚合能力的建设。
-  支持针对用户给定属性进行观点或情感极性分析，例如当用户给出属性"价格"时，期望能够从以上示例中，均能抽取出其观点词"便宜"
- 支持用户提供属性的同义词表，用来加强模型对用户领域属性同义词的理解能力。

以下给出了酒店场景的示例，每行代表1类同义词，不同词之间以"空格"隔开。

```
房间 屋子 房子
位置 地理位置
隔音 隔声
价格 价钱 费用
```

**升级2：加强隐性观点抽取功能**: 为提高模型效果，本项目加强了对隐性观点功能抽取功能的支持。本项目中定义隐性观点是指没有对应属性的纯观点词，如以下示例中的"比较便宜"便是隐性观点。

```
蛋糕味道不错，外观很漂亮，而且比较便宜
```

本项目支持用户提供一个隐性观点映射文件，用户可以根据自己的业务场景定义隐性观点词，以下给出了酒店场景的示例。其格式为，第1个单词为隐性观点对应的属性，后续按照情感情感倾向对隐性观点词进行了归类，同一类的以"[ ]"方式放到一块。

```
价格, 正向[实惠 便宜 超划算 划算 物超所值 物有所值 不贵], 负向[贵 不便宜 不划算]
卫生, 正向[干净], 负向[很脏 很臭 不干净]
隔音, 负向[好吵]
位置, 负向[不太好找]
```

可以分别通过参数"synonym_file"和"implicit_file"分别将同义词文件和隐性观点文件传入以下命令中，进行相关数据构建。

```shell
python label_studio.py \
    --label_studio_file ./data/label_studio.json \
    --synonym_file ./data/synonyms.json \
    --implicit_file ./data/implicit_opinions.json \
    --task_type ext \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --prompt_prefix "情感倾向" \
    --options "正向" "负向" "未提及" \
    --separator "##" \
    -- negative_ratio 5 \
    --is_shuffle True \
    --seed 1000
```

备注：
- 默认情况下 [label_studio.py](./label_studio.py) 脚本会按照比例将数据划分为 train/dev/test 数据集
- 每次执行 [label_studio.py](./label_studio.py) 脚本，将会覆盖已有的同名数据文件
- 在模型训练阶段推荐构造一些负例以提升模型效果，在数据转换阶段内置了这一功能。可通过`negative_ratio`控制自动构造的负样本比例；负样本数量 = negative_ratio * 样本数量。
- 对于从label_studio导出的文件，默认文件中的每条数据都是经过人工正确标注的。


<a name="2.6.2"></a>

#### **2.6.2 模型微调**
在生成酒店场景的训练数据后，可以通过以下命令启动模型微调。

```shell
python -u -m paddle.distributed.launch --gpus "0" finetune.py \
  --train_path ./data/train.txt \
  --dev_path ./data/dev.txt \
  --save_dir ./checkpoint \
  --learning_rate 1e-5 \
  --batch_size 16 \
  --max_seq_len 512 \
  --num_epochs 10 \
  --model uie-base \
  --seed 1000 \
  --logging_steps 10 \
  --valid_steps 100 \
  --device gpu
```

可配置参数说明：

* `train_path`：必须，训练集文件路径。
* `dev_path`：必须，验证集文件路径。
* `save_dir`：模型 checkpoints 的保存目录，默认为"./checkpoint"。
* `learning_rate`：训练最大学习率，UIE 推荐设置为 1e-5；默认值为1e-5。
* `batch_size`：训练集训练过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为 16。
* `max_seq_len`：模型支持处理的最大序列长度，默认为512。
* `num_epochs`：模型训练的轮次，可以视任务情况进行调整，默认为10。
* `model`：训练使用的预训练模型。可选择的有"uie-base"、 "uie-medium", "uie-mini", "uie-micro", "uie-nano", "uie-m-base", "uie-m-large"。
* `logging_steps`: 训练过程中日志打印的间隔 steps 数，默认10。
* `valid_steps`: 训练过程中模型评估的间隔 steps 数，默认100。
* `seed`：全局随机种子，默认为 42。
* `device`: 训练设备，可选择 'cpu'、'gpu' 其中的一种；默认为 GPU 训练。


#### **2.6.3 模型测试**

通过运行以下命令进行对酒店场景的测试集进行评估：

```
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/test.txt \
    --batch_size 16 \
    --max_seq_len 512
```

可配置参数说明：

* `model_path`：必须，用以数据集测试的模型路径。
* `test_path`：必须，测试集文件路径。
* `batch_size`：训练集训练过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为 16。
* `max_seq_len`：模型支持处理的最大序列长度，默认为512。


#### **2.6.4 定制预测及效果展示**

paddlenlp.Taskflow装载定制模型，通过task_path指定模型权重文件的路径，路径下需要包含训练好的模型权重文件model_state.pdparams。

```python
>>> # define schema
>>> schema = [{'评价维度': ['观点词', '情感倾向[正向,负向,未提及]']}]
>>> senta = Taskflow("sentiment_analysis", model="uie-base", schema=schema, task_path="./checkpoint/model_best")
>>> senta("这家点的房间很大，店家服务也很热情，就是房间隔音不好")
[
    {
        '评价维度': [
            {
                'text': '服务',
                'start': 11,
                'end': 13,
                'probability': 0.9600759151746807,
                'relations': {
                    '观点词': [
                        {
                            'text': '热情',
                            'start': 15,
                            'end': 17,
                            'probability': 0.9995151134519027
                        }
                    ],
                    '情感倾向[正向,负向,未提及]': [
                        {
                            'text': '正向',
                            'probability': 0.9998306104766073
                        }
                    ]
                }
            },
            {
                'text': '隔音',
                'start': 22,
                'end': 24,
                'probability': 0.9993525950520166,
                'relations': {
                    '观点词': [
                        {
                            'text': '不好',
                            'start': 24,
                            'end': 26,
                            'probability': 0.9992370362201655
                        }
                    ],
                    '情感倾向[正向,负向,未提及]': [
                        {
                            'text': '负向',
                            'probability': 0.9842680108546062
                        }
                    ]
                }
            },
            {
                'text': '房间',
                'start': 4,
                'end': 6,
                'probability': 0.9991784415865368,
                'relations': {
                    '观点词': [
                        {
                            'text': '很大',
                            'start': 6,
                            'end': 8,
                            'probability': 0.8359714693985723
                        }
                    ],
                    '情感倾向[正向,负向,未提及]': [
                        {
                            'text': '正向',
                            'probability': 0.997688853839179
                        }
                    ]
                }
            }
        ]
    }
]
```

**属性聚合样本预测**  
由于在构造样本时，引入酒店场景的部分属性的同义词，可以看到对于"隔音"与"隔声"、"位置"与"所处位置", "价格"、"价钱"与"费用"等各项内容均能识别出正确的观点和情感倾向。

对于"隔音"与"隔声"的调用示例：
```python
>>> schema = [{'评价维度': ['观点词', '情感倾向[正向,负向,未提及]']}]
>>> aspects = ["隔声", "隔音"]
>>> senta = Taskflow("sentiment_analysis", model="uie-base", schema=schema, task_path="./checkpoint/model_best", aspects=aspects)
>>> senta("这家点的房间很大，店家服务也很热情，就是房间隔音不好")
[
    {
        '评价维度': [
            {
                'text': '隔声',
                'relations': {
                    '观点词': [
                        {
                            'text': '不好',
                            'start': 24,
                            'end': 26,
                            'probability': 0.9990923567964849
                        }
                    ],
                    '情感倾向[正向,负向,未提及]': [
                        {
                            'text': '负向',
                            'probability': 0.9840917780791472
                        }
                    ]
                }
            },
            {
                'text': '隔音',
                'relations': {
                    '观点词': [
                        {
                            'text': '不好',
                            'start': 24,
                            'end': 26,
                            'probability': 0.9992370362201655
                        }
                    ],
                    '情感倾向[正向,负向,未提及]': [
                        {
                            'text': '负向',
                            'probability': 0.9842680108546062
                        }
                    ]
                }
            }
        ]
    }
]

```

下图展示了关于模型对于属性聚合能力支持的样本：

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/203913660-ac95caad-c5e2-43c5-b291-6208babd58d3.png />
</div>

<a name="2.7"></a>

**隐性观点词抽取样本预测**  
由于在构造样本时，引入了酒店场景的部分高频隐性观点，因此模型可以很好地支持对于相关隐性观点的抽取。

对于"价格"的调用示例：
```python
>>> schema = [{'评价维度': ['观点词', '情感倾向[正向,负向,未提及]']}]
>>> aspects = ["价格"]
>>> senta = Taskflow("sentiment_analysis", model="uie-base", schema=schema, task_path="./checkpoint/model_best", aspects=aspects)
>>> senta("这家点的房间很大，店家服务也很热情，而且还很便宜")
[
    {
        '评价维度': [
            {
                'text': '价格',
                'relations': {
                    '观点词': [
                        {
                            'text': '便宜',
                            'start': 22,
                            'end': 24,
                            'probability': 0.9994707157967682
                        }
                    ],
                    '情感倾向[正向,负向,未提及]': [
                        {
                            'text': '正向',
                            'probability': 0.9950765734397677
                        }
                    ]
                }
            }
        ]
    }
]
```

下图展示了关于模型对于隐性观点抽取的样本：

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/203913490-a6fbf0aa-1f9c-476d-83c7-ea4604ab94d0.png />
</div>


### **2.7 模型部署**

本项目支持基于PaddleNLP SimpleServing进行服务化部署，可以在`deploy`目录下执行以下命令启动服务和请求。

**启动服务**
```
paddlenlp server server:app --workers 1 --host 0.0.0.0 --port 8189
```
**Client发送请求**

服务启动后， 通过 `client.py` 脚本发送请求：
```
python client.py
```

**多卡服务化预测**

PaddleNLP SimpleServing 支持多卡负载均衡预测，主要在服务化注册的时候，注册两个Taskflow的task即可，代码示例如下：

```python
senta1 = Taskflow("sentiment_analysis", schema=schema, model="uie-senta-base", device_id=0)
senta2 = Taskflow("sentiment_analysis", schema=schema, model="uie-senta-base", device_id=1)

app.register_taskflow('senta', [senta1, senta2])
```

