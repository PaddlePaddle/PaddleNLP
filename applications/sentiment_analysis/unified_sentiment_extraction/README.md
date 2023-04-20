# 通用情感信息抽取

## **目录**
## **目录**
- [1. 情感分析应用简介](#1)
- [2. 特色介绍](#2)
- [3. 运行环境](#3)
- [4. 整体功能介绍与Taskflow快速体验](#4)
  - [4.1 开箱即用的情感分析能力](#4.1)
    - [4.1.1 语句级情感分析](#4.1.1)
    - [4.1.2 属性级情感分析](#4.1.2)
    - [4.1.3 多版本模型选择](#4.1.3)
  - [4.2 批量处理：从数据到情感分析可视化](#4.2)
    - [4.2.1 数据描述](#4.2.1)
    - [4.2.2 批量情感分析](#4.2.2)
    - [4.2.3 情感分析可视化](#4.2.3)
      - [4.2.3.1 一键生成情感分析结果](#4.2.3.1)
      - [4.2.3.2 情感分析详细展示](#4.2.3.2)
- [5. 更进一步：结合业务分析经验，定制情感分析](#5)
  - [5.1 打通数据标注到训练样本构建](#5.1)
    - [5.1.1 样本构建：语句级情感分类任务](#5.1.1)
    - [5.1.2 样本构建：属性抽取相关任务](#5.1.2)
    - [5.1.3 样本构建升级1：加强属性聚合能力](#5.1.3)
    - [5.1.4 样本构建升级2：加强隐性观点抽取能力](#5.1.4)
  - [5.2 模型训练](#5.2)
  - [5.3 模型测试](#5.3)
  - [5.4 模型预测及效果展示](#5.4)
    - [5.4.1 使用训练后的模型进行预测](#5.4.1)
    - [5.4.2 属性聚合预测和分析](#5.4.2)
    - [5.4.3 隐性观点词抽取预测和分析](#5.4.3)
- [6. 模型部署](#6)
  - [6.1 基于SimpleServer进行服务化部署](#6.1)
  - [6.2 基于Pipeline进行部署](#6.2)

<a name="1"></a>

## **1. 情感分析应用简介**

PaddleNLP情感分析应用立足真实企业用户对情感分析方面的需求，针对情感分析领域的痛点和难点，提供基于前沿模型的情感分析解决方案，助力开发者快速分析业务相关产品或服务的用户感受。

本项目以通用信息抽取模型UIE为训练底座，提供了语句级情感分析和属性级情感分析能力、覆盖情感分类、属性抽取、观点抽取等常用情感分析能力，如下图所示。同时提供了可视化能力，支持从输入数据到情感分析结果可视化，帮助用户快速分析业务数据。更进一步地，本项目同时支持基于业务数据进行定制训练，同时支持引入业务侧积累的经验和知识，包括同义属性和隐性观点词表，加强模型进行属性聚合和隐性观点抽取的能力，进一步提高模型对于业务场景数据的分析能力。

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/199965793-f0933baa-5b82-47da-9271-ba36642119f8.png" />
</div>

<a name="2"></a>

## **2. 特色介绍**

- **功能丰富🎓**：提供情感分析训练模型作为底座，支持语句级情感分析和属性级情感分析，覆盖情感分类、属性与观点抽取、同义属性聚合、隐性观点抽取、可视化分析等常见情感分析任务。
- **效果领先✊**： 以通用信息抽取模型UIE为训练底座，具有较强的零样本预测和小样本微调能力。
- **开箱即用👶**：打通Taskflow使用流程，3行代码获取分析结果，同时提供了情感分析结果可视化能力。
- **定制模型🏃**：支持针对特定业务场景进行全流程定制，包括数据标注、样本构建、模型训练和模型测试，同时通过融合业务相关的同义属性词和隐性观点词，可进一步提高模型针对业务场景的情感分析能力。


<a name="3"></a>

## **3. 运行环境**

**代码结构**
```
unified_sentiment_extraction/
├── batch_predict.py # 以文件的形式输入，进行批量预测的脚本
├── evaluate.py # 模型评估脚本
├── finetune.py # 模型微调脚本
├── label_studio.py # 将label-studio导出数据转换为模型输入数据的脚本
├── label_studio.md # 将label-studio标注说明
├── utils.py # 工具函数脚本
├── visual_analysis.py # 情感分析结果可视化脚本
└── README.md # 使用说明
```

**安装依赖**

- python == 3.9.12
- paddlepaddle == 2.3.2
- paddlenlp == 2.4.5
- wordcloud == 1.8.2.2

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

<a name="4"></a>

## **4. 整体功能介绍与Taskflow快速体验**

本项目以通用信息抽取模型UIE为训练底座，基于大量情感分析数据进一步训练，增强了模型对于情感知识的处理能力，支持语句级情感分类、属性抽取、观点词抽取、属性级情感分类等基础情感分析能力。下表展示了通用UIE `uie-base` 和情感知识增强的UIE `uie-senta-base` 在测试集上的效果对比。

|  模型 | Precision | Recall | F1 |
|  :---: | :--------: | :--------: | :--------: |
| `uie-base` | 0.86759 | 0.83696 | 0.85200 |
| `uie-senta-base` | 0.93403 | 0.92795 | 0.93098 |

另外，为方便用户体验和使用，本项目提供的情感分析能力已经集成到了 Taskflow，可以通过Taskflow开箱即用的能力快速体验情感分析的功能。

<a name="4.1"></a>

### **4.1 开箱即用的情感分析能力**

<a name="4.1.1"></a>

#### **4.1.1 语句级情感分析**
整句情感分析功能当前支持二分类：正向和负向，调用示例如下：

```python
>>> from paddlenlp import Taskflow

>>> schema = ['情感倾向[正向，负向]']
>>> senta = Taskflow("sentiment_analysis", model="uie-senta-base", schema=schema)
>>> print(senta('蛋糕味道不错，店家服务也很好'))

[
    {
        '情感倾向[正向,负向]': [
            {
                'text': '正向',
                'probability': 0.996646058824652
            }
        ]
    }
]
```

<a name="4.1.2"></a>

#### **4.1.2 属性级情感分析**

除语句级情感分析之外，本项目同时支持属性级情感分析，包括属性抽取（Aspect Term Extraction）、观点抽取（Opinion Term Extraction）、属性级情感分析（Aspect Based Sentiment Classification）等等。可以通过设置相应的schema进行对应信息的抽取，其调用示例如下。

```python
>>> from paddlenlp import Taskflow

>>> # Aspect Term Extraction
>>> # schema =  ["评价维度"]
>>> # Aspect - Opinion Extraction
>>> # schema =  [{"评价维度":["观点词"]}]
>>> # Aspect - Sentiment Extraction
>>> # schema =  [{"评价维度":["情感倾向[正向,负向,未提及]"]}]
>>> # Aspect - Sentiment - Opinion Extraction
>>> schema =  [{"评价维度":["观点词", "情感倾向[正向,负向,未提及]"]}]

>>> senta = Taskflow("sentiment_analysis", model="uie-senta-base", schema=schema)
>>> print(senta('蛋糕味道不错，店家服务也很热情'))

[
    {
        '评价维度': [
            {
                'text': '服务',
                'start': 9,
                'end': 11,
                'probability': 0.9709093024793489,
                'relations': {
                    '观点词': [
                        {
                            'text': '热情',
                            'start': 13,
                            'end': 15,
                            'probability': 0.9897222206316556
                        }
                    ],
                    '情感倾向[正向,负向,未提及]': [
                        {
                            'text': '正向',
                            'probability': 0.9999327669598301
                        }
                    ]
                }
            },
            {
                'text': '味道',
                'start': 2,
                'end': 4,
                'probability': 0.9105472387838915,
                'relations': {
                    '观点词': [
                        {
                            'text': '不错',
                            'start': 4,
                            'end': 6,
                            'probability': 0.9946981266891619
                        }
                    ],
                    '情感倾向[正向,负向,未提及]': [
                        {
                            'text': '正向',
                            'probability': 0.9998829392709467
                        }
                    ]
                }
            }
        ]
    }
]
```

更进一步地，在某些业务场景中，特别是一些垂域场景，用户可能比较关注固定的某些属性。在这种情况下，可以预先提供相应的属性集合，则本项目将只会在该属性集上进行情感分析，分析和抽取该集合中各个属性的信息。

针对固定属性的情感分析示例如下，需要将属性集合传入参数 `aspects` 中，后续将只针对这些属性进行分析。可以看到在示例中，传入了属性 `房间`，`位置` 和 `价格`，针对 `房间` 和 `价格` 均分析到了观点词和情感倾向，但是`位置`由于在样本中并未提及，因此相应观点词为空，情感倾向为 `未提及`。

```python
>>> # define schema for pre-defined aspects, schema
>>> schema = ["观点词", "情感倾向[正向,负向,未提及]"]
>>> aspects = ["房间", "位置", "价格"]
>>> # set aspects for Taskflow
>>> senta = Taskflow("sentiment_analysis", model="uie-senta-base", schema=schema, aspects=aspects)
>>> print(senta("这家店的房间很大，店家服务也很热情，就是价格有点贵"))

[
    {
        '评价维度': [
            {
                'text': '房间',
                'relations': {
                    '观点词': [
                        {
                            'text': '大',
                            'start': 7,
                            'end': 8,
                            'probability': 0.9998772175681552
                        }
                    ],
                    '情感倾向[正向,负向,未提及]': [
                        {
                            'text': '正向',
                            'probability': 0.9999312170965595
                        }
                    ]
                }
            },
            {
                'text': '位置',
                'relations': {
                    '情感倾向[正向,负向,未提及]': [
                        {
                            'text': '未提及',
                            'probability': 0.9999939203353847
                        }
                    ]
                }
            },
            {
                'text': '价格',
                'relations': {
                    '观点词': [
                        {
                            'text': '贵',
                            'start': 24,
                            'end': 25,
                            'probability': 0.998841669863026
                        }
                    ],
                    '情感倾向[正向,负向,未提及]': [
                        {
                            'text': '负向',
                            'probability': 0.9997340617174757
                        }
                    ]
                }
            }
        ]
    }
]
```

<a name="4.1.3"></a>

#### **4.1.3 多版本模型选择**
为方便用户实际业务应用情况，本项目多个版本的模型，可以根据业务对于精度和速度方面的要求进行选择，下表展示了不同版本模型的结构以及在测试集上的指标。

|  模型 |  结构  | Precision | Recall | F1 |
|  :---: | :--------: | :--------: | :--------: | :--------: |
|  `uie-senta-base` (默认) | 12-layers, 768-hidden, 12-heads | 0.93403 | 0.92795 | 0.93098 |
| `uie-senta-medium` | 6-layers, 768-hidden, 12-heads | 0.93146 | 0.92137 | 0.92639 |
| `uie-senta-mini` | 6-layers, 384-hidden, 12-heads | 0.91799 | 0.92028 | 0.91913 |
| `uie-senta-micro` | 4-layers, 384-hidden, 12-heads | 0.91542 | 0.90957 | 0.91248 |
| `uie-senta-nano` | 4-layers, 312-hidden, 12-heads | 0.90817 | 0.90878 | 0.90847 |

在Taskflow中，可以直接指定相应模型名称进行使用，使用`uie-senta-mini`版本的示例如下：

```python
>>> from paddlenlp import Taskflow

>>> schema =  [{"评价维度":["观点词", "情感倾向[正向,负向,未提及]"]}]
>>> senta = Taskflow("sentiment_analysis", model="uie-senta-mini", schema=schema)
```

<a name="4.2"></a>

### **4.2 批量处理：从数据到情感分析可视化**

为方便使用，本项目提供了批量处理的功能，支持以文件形式输入，批量进行情感分析。同时打通了从数据到情感分析结果可视化的流程，帮助用户可以更加快速获取情感分析结果，聚焦于业务分析方面。

<a name="4.2.1"></a>

#### **4.2.1 数据描述**
输入数据如下方式进行组织，每行表示一个文本评论。可以点击[这里](https://paddlenlp.bj.bcebos.com/datasets/sentiment_analysis/hotel/test_hotel.tar.gz)下载酒店场景的测试数据进行分析。

```
非常好的酒店 不枉我们爬了近一个小时的山，另外 大厨手艺非常棒 竹筒饭 竹筒鸡推荐入住的客人必须要点，
房间隔音效果不好，楼下KTV好吵的
酒店的房间很大，干净舒适，服务热情
怎么说呢，早上办理入住的，一进房间闷热的一股怪味，很臭，不能开热风，好多了，虽然房间小，但是合理范围
总台服务很差，房间一般
```

<a name="4.2.2"></a>

#### **4.2.2 批量情感分析**

通过脚本 `batch_predict.py` 批量进行情感分析，通过 `file_path` 指定要进行情感分析的文件路径，处理完后，结果将会保存在 `save_path` 指定的文件中，示例如下：

```shell
python batch_predict.py \
    --file_path "./data/test_hotel.txt" \
    --save_path "./data/sentiment_analysis.json" \
    --model "uie-senta-base" \
    --schema "[{'评价维度': ['观点词', '情感倾向[正向,负向,未提及]']}]" \
    --batch_size 4 \
    --max_seq_len 512
```

参数说明：
- ``file_path``: 用于进行情感分析的文件路径。
- ``save_path``: 情感分析结果的保存路径。
- ``model``: 进行情感分析的模型名称，可以在这些模型中进行选择：['uie-senta-base', 'uie-senta-medium', 'uie-senta-mini', 'uie-senta-micro', 'uie-senta-nano']。
- ``load_from_dir``: 指定需要加载的离线模型目录，比如训练后保存的模型，如果不进行指定，则默认根据 `model` 指定的模型名称自动下载相应模型。
- ``schema``: 基于UIE模型进行信息抽取的Schema描述。
- ``batch_size``: 预测过程中的批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为 16。
- ``max_seq_len``: 模型支持处理的最大序列长度，默认为512。
- ``aspects``: 预先给定的属性，如果设置，模型将只针对这些属性进行情感分析，比如分析这些属性的观点词。

<a name="4.2.3"></a>

#### **4.2.3 情感分析可视化**

在情感分析处理之后，可以根据情感分析的保存结果进行可视化展示，帮助用户更友好地分析业务特点。默认情况下，可视化功能支持围绕属性、观点、属性+观点、属性+情感、指定属性+观点分析功能。在各项分析中，均支持词云和直方图两类图像展示。

下面将以酒店场景数据为例进行展示。

<a name="4.2.3.1"></a>

**4.2.3.1 一键生成情感分析结果**

基于以上生成的情感分析结果，可以使用`visual_analysis.py`脚本对情感分析结果进行可视化，最终可视化结果将会被保存在 `save_dir` 指定的目录下。 使用时需要指定情感分析可视化的结果的任务类型，若是语句级的情感分类，则将task_type指定为``cls``，若是属性级的情感分析，则将task_type指定为``ext``，示例如下：

```
python visual_analysis.py \
    --file_path "./outputs/test_hotel.json" \
    --save_dir "./outputs/images" \
    --task_type "ext"
```

可配置参数说明：
- ``file_path``: 指定情感分析结果的保存路径。
- ``save_dir``: 指定图片的保存目录。
- ``task_type``: 指定任务类型，语句级情感分类请指定为``cls``，属性级情感分析请指定为``ext``，默认为``ext``。
- ``font_path``: 指定字体文件的路径，用以在生成的wordcloud图片中辅助显示中文，如果为空，则会自动下载黑体字，用以展示中文字体。

**备注**：在`visual_analysis.py`脚本启动时，默认会删除当前已经存在的`save_dir`目录以及其中文件，然后在该目录下重新生成相应的可视化图片。

下图展示了对酒店场景数据分析后的部分图片：

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/200259473-434888f7-c0ac-4253-ab23-ede1628e6ba2.png" />
</div>
<br>

<a name="4.2.3.2"></a>

**4.2.3.2 情感分析详细展示**

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


<a name="5"></a>

## **5. 更进一步：结合业务分析经验，定制情感分析**

考虑到用户在对业务数据进行情感分析时，往往聚焦于某个特定场景或领域，为满足用户更高的情感分析要求，本项目支持从以下方面协助用户，结合业务经验，进一步定制情感分析能力，提高模型对业务数据的理解和分析能力。

- 数据层面：打通 label-studio 平台，定制了情感信息的标注规则，支持根据标注数据自动转换为模型输入样本。
- 属性聚合：结合业务经验，支持传入同义的属性集合，可以增强模型对于数据聚合的能力。
- 隐性观点抽取：结合业务经验，支持自定义隐性观点词表，可以增强模型对于隐性观点的抽取能力。

下面以酒店场景为例，讲解定制酒店垂域的情感分析能力。具体地，将从数据标注及样本构建 - 模型训练 - 模型测试 - 模型预测及效果展示等全流程展开介绍。

<a name="5.1"></a>

### **5.1 打通数据标注到训练样本构建**

本项目建议用户使用 label-studio 平台标注数据，同时提供了一套用于情感信息标注的规则，可以参考[情感分析任务Label Studio使用指南](./label_studio.md)获取更多信息，这里不再赘述。同时本项目打通了从 label-studio 标注平台到转换为模型输入形式数据的流程， 即支持用户在基于 label_studio 标注业务侧数据后，通过label-studio 导出标注好的json数据， 然后利用本项目提供的  `label_studio.py` 脚本，可以将导出数据一键转换为模型训练数据。

在利用 `label_studio.py` 脚本进行数据转换时，需要考虑任务类型的不同，选择相应的样本构建方式，整体可以分为 `分类` 和 `抽取` 任务。

<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/203001847-8e41709b-0f5a-4673-8aca-5c4fb7705d4a.png  />
</div>

为方便用户使用，本项目提供了300+条酒店场景的标注数据，可点击[这里](https://paddlenlp.bj.bcebos.com/datasets/sentiment_analysis/hotel/label_studio.tar.gz)进行下载，请注意该数据仅适合用于 `抽取` 类型的任务。


<a name="5.1.1"></a>

#### **5.1.1 样本构建：语句级情感分类任务**

对于语句级情感分类任务，默认支持2分类：``正向`` 和 ``负向``，可以通过如下命令构造相关训练数据。

```shell
python label_studio.py \
    --label_studio_file ./data/label_studio.json \
    --task_type cls \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --options "正向" "负向" \
    --is_shuffle True \
    --seed 1000
```

参数介绍：
- ``label_studio_file``: 从label studio导出的语句级情感分类的数据标注文件。
- ``task_type``: 选择任务类型，可选有抽取和分类两种类型的任务，其中前者需要设置为``ext``，后者需要设置为``cls``。由于此处为语句级情感分类任务，因此需要设置为``cls``。
- ``save_dir``: 训练数据的保存目录，默认存储在``data``目录下。
- ``splits``: 划分数据集时训练集、验证集所占的比例。默认为[0.8, 0.1, 0.1]表示按照``8:1:1``的比例将数据划分为训练集、验证集和测试集。
- ``options``: 情感极性分类任务的选项设置。对于语句级情感分类任务，默认支持2分类：``正向`` 和 ``负向``；对于属性级情感分析任务，默认支持3分类：``正向``, ``负向``和 ``未提及``，其中``未提及``表示要分析的属性在原文本评论中未提及，因此无法分析情感极性。如果业务需要其他情感极性选项，可以通过``options``字段进行设置，需要注意的是，如果定制了``options``，参数``label_studio_file``指定的文件需要包含针对新设置的选项的标注数据。
- ``is_shuffle``: 是否对数据集进行随机打散，默认为True。
- ``seed``: 随机种子，默认为1000.

**备注**：参数``options``可以不进行手动指定，如果这么做，则采用默认的设置。针对语句级情感分类任务，其默认将被设置为：``"正向" "负向"``；对于属性级情感分析任务，默认将被设置为：``"正向" "负向" "未提及"``。

<a name="5.1.2"></a>

#### **5.1.2 样本构建：属性抽取相关任务**

针对抽取式的任务，比如属性-观点抽取、属性-情感极性-观点词抽取、属性分类任务等，可以使用如下命令将label-studio导出数据转换为模型训练数据。

```shell
python label_studio.py \
    --label_studio_file ./data/label_studio.json \
    --task_type ext \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --options "正向" "负向" "未提及" \
    --negative_ratio 5 \
    --is_shuffle True \
    --seed 1000
```

重点参数介绍：
- ``label_studio_file``: 从label studio导出的属性抽取相关的数据标注文件。
- ``task_type``: 选择任务类型，可选有抽取和分类两种类型的任务，其中前者需要设置为``ext``，后者需要设置为``cls``。由于此处为属性抽取相关任务，因此需要设置为``ext``。
- ``negative_ratio``表示对于一个样本，为每个子任务（属性级的观点抽取，属性级的情感分类）最多生成``negative_ratio``个负样本。如果额外提供了属性同义词标或隐性观点抽取词表，将结合两者信息生成更多的负样本，以增强属性聚合和隐性观点抽取能力。
其他参数解释同上，这里不再赘述。

<a name="5.1.3"></a>

#### **5.1.3 样本构建升级1：加强属性聚合能力**

在用户对产品或服务进行评论时，对某一些属性可能会有不同的说法，这会在后续对属性分析时可能会带来困扰。如以下示例中的"价格","价钱"和"费用"。

```
蛋糕味道不错，外观很漂亮，而且价格比较便宜
蛋糕味道不错，外观很漂亮，而且价钱比较便宜
蛋糕味道不错，外观很漂亮，而且费用比较便宜
```

针对这种情况，针对属性相关任务，本项目同时支持用户结合业务经验，通过设置同义的属性词表，加强模型的属性聚合能力。具体来讲，本项目期望通过以下两点，支持对属性聚合能力的建设。

- 支持针对用户给定的属性进行情感分析
- 支持用户提供同义的属性词表，用以加强模型对用户领域属性同义词的理解能力

以下给出了酒店场景的示例，每行代表1类同义词，不同词之间以"空格"隔开。

```
房间 屋子 房子
位置 地理位置
隔音 隔声
价格 价钱 费用
```

可以通过以下命令，使用synonym_file指定凝聚业务经验的同义属性集合，利用同义属性生成对应的数据样本：

```shell
python label_studio.py \
    --label_studio_file ./data/label_studio.json \
    --synonym_file ./data/synonyms.txt \
    --task_type ext \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --options "正向" "负向" "未提及" \
    --negative_ratio 5 \
    --is_shuffle True \
    --seed 1000
```

<a name="5.1.4"></a>

#### **5.1.4 样本构建升级2：加强隐性观点抽取能力**

另外，本项目同时支持加强对隐性观点功能抽取的能力，这里需要说明一点，本项目中定义隐性观点是指没有对应属性的纯观点词，如以下示例中的"比较便宜"便是隐性观点。

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

可以通过参数"implicit_file"指定凝聚业务经验的隐性观点词表，生成对应的隐性观点数据样本：

```shell
python label_studio.py \
    --label_studio_file ./data/label_studio.json \
    --implicit_file ./data/implicit_opinions.txt \
    --task_type ext \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --options "正向" "负向" "未提及" \
    --negative_ratio 5 \
    --is_shuffle True \
    --seed 1000
```

<a name="5.2"></a>

### **5.2 模型训练**
在生成酒店场景的训练数据后，可以通过以下命令启动模型训练：

```shell
python -u -m paddle.distributed.launch --gpus "0" finetune.py \
  --train_path ./data/train.json \
  --dev_path ./data/dev.json \
  --save_dir ./checkpoint \
  --learning_rate 1e-5 \
  --batch_size 16 \
  --max_seq_len 512 \
  --num_epochs 3 \
  --model uie-senta-base \
  --seed 1000 \
  --logging_steps 10 \
  --valid_steps 100 \
  --device gpu
```

可配置参数说明：

* ``train_path``：必须，训练集文件路径。
* ``dev_path``：必须，验证集文件路径。
* ``save_dir``：模型 checkpoints 的保存目录，默认为"./checkpoint"。
* ``learning_rate``：训练最大学习率，UIE 推荐设置为 1e-5；默认值为1e-5。
* ``batch_size``：训练集训练过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为 16。
* ``max_seq_len``：模型支持处理的最大序列长度，默认为512。
* ``num_epochs``：模型训练的轮次，可以视任务情况进行调整，默认为10。
* ``model``：训练使用的预训练模型。可选择的有`uie-senta-base`, `uie-senta-medium`, `uie-senta-mini`, `uie-senta-micro`, `uie-senta-nano`，默认为`uie-senta-base`。
* ``logging_steps``: 训练过程中日志打印的间隔 steps 数，默认10。
* ``valid_steps``: 训练过程中模型评估的间隔 steps 数，默认100。
* ``seed``：全局随机种子，默认为 42。
* ``device``: 训练设备，可选择 'cpu'、'gpu' 其中的一种；默认为 GPU 训练。

<a name="5.3"></a>

### **5.3 模型测试**
通过运行以下命令进行对酒店场景的测试集进行评估：

```
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/test.json \
    --batch_size 16 \
    --max_seq_len 512
```

可配置参数说明：

* ``model_path``：必须，进行评估的模型文件夹路径，路径下需包含模型权重文件model_state.pdparams及配置文件model_config.json。
* ``test_path``：必须，进行评估的测试集文件。
* ``batch_size``：训练集训练过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为 16。
* ``max_seq_len``：文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
* ``debug``： 是否开启debug模式对每个正例类别分别进行评估，该模式仅用于模型调试，默认关闭。

在构造样本过程中，如果设置了最大负例比例negative_ratio，会在样本中添加一定数量的负样本，模型测试默认会对正样本和负样本共同进行评估。特别地，当开启debug模式后，会对每个正例类别分别进行评估，该模式仅用于模型调试：
```
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/test.json \
    --batch_size 16 \
    --max_seq_len 512 \
    --debug
```

输出打印示例：
```
[2022-12-12 05:20:06,152] [    INFO] - -----------------------------
[2022-12-12 05:20:06,152] [    INFO] - Class Name: 评价维度
[2022-12-12 05:20:06,152] [    INFO] - Evaluation Precision: 0.89655 | Recall: 0.89655 | F1: 0.89655
[2022-12-12 05:20:06,553] [    INFO] - -----------------------------
[2022-12-12 05:20:06,553] [    INFO] - Class Name: 观点词
[2022-12-12 05:20:06,553] [    INFO] - Evaluation Precision: 0.81159 | Recall: 0.86154 | F1: 0.83582
[2022-12-12 05:20:07,610] [    INFO] - -----------------------------
[2022-12-12 05:20:07,611] [    INFO] - Class Name: X的观点词
[2022-12-12 05:20:07,611] [    INFO] - Evaluation Precision: 0.92222 | Recall: 0.90217 | F1: 0.91209
[2022-12-12 05:20:08,331] [    INFO] - -----------------------------
[2022-12-12 05:20:08,331] [    INFO] - Class Name: X的情感倾向[未提及,正向,负向]
[2022-12-12 05:20:08,331] [    INFO] - Evaluation Precision: 0.81481 | Recall: 0.81481 | F1: 0.81481
```

<a name="5.4"></a>

### **5.4 模型预测及效果展示**

<a name="5.4.1"></a>

#### **5.4.1 使用训练后的模型进行预测**
paddlenlp.Taskflow装载定制模型，通过task_path指定模型权重文件的路径，路径下需要包含训练好的模型权重文件model_state.pdparams。

```python
>>> # define schema
>>> schema = [{'评价维度': ['观点词', '情感倾向[正向,负向,未提及]']}]
>>> senta = Taskflow("sentiment_analysis", model="uie-senta-base", schema=schema, task_path="./checkpoint/model_best")
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

<a name="5.4.2"></a>

#### **5.4.2 属性聚合预测和分析**

下面就 `隔音` 与 `价格` 两个属性进行分析，抽取样本中与这两个属性相关的情感信息，代码如下：

```python
>>> schema = [{'评价维度': ['观点词', '情感倾向[正向,负向,未提及]']}]
>>> aspects = ["隔音", "价格"]
>>> senta = Taskflow("sentiment_analysis", model="uie-senta-base", schema=schema, task_path="./checkpoint/model_best", aspects=aspects)
>>> senta("这家点的房间很大，店家服务也很热情，就是房间隔音不好")
```

下图展示了关于模型对于属性聚合能力支持的样本，在分析之前设定固定的属性集合`["隔音", "价格"]`，可以看到尽管语料中同时出现了`隔音`、`隔声`、`价格`、`价钱`和`费用`，但是经过定制后的情感分析模型依然能够准确识别出对于属性 `隔音` 和 `价格`的情感信息，从而起到属性聚合的效果。

| 样本 | 属性 | 观点词 | 情感倾向 |
| :----: |:----: |:----: |:----: |
|这家店的房间很大，隔音效果不错，而且价格很便宜|隔音|不错|正向|
|房间比较小，隔声也不太好，设施还是挺齐全的|隔音|不太好|负向|
|房间还不错，有免费的矿泉水，而且价格很实惠|价格|实惠|正向|
|房间很大，店家也挺热情，很棒，就是价钱有点贵|价格|贵|负向|
|酒店不错，房间面积大，住的也舒适，而且价格很划算|价格|划算|正向|
|房间好大呀，而且这边还挺安静的，不过整体还是很好的，很宽敞，而且价格很便宜|价格|便宜|正向|

<a name="5.4.3"></a>

#### **5.4.3 隐性观点词抽取预测和分析**

下面就`价格` 和 `卫生` 两个属性进行分析隐性观点，抽取样本中与这两个属性相关的情感信息，代码如下：

对于"价格"的调用示例：
```python
>>> schema = [{'评价维度': ['观点词', '情感倾向[正向,负向,未提及]']}]
>>> aspects = ["价格"]
>>> senta = Taskflow("sentiment_analysis", model="uie-senta-base", schema=schema, task_path="./checkpoint/model_best", aspects=aspects)
>>> senta("这家店的房间很大，店家服务也很热情，而且很便宜")
```

下图展示了关于模型对于隐性观点抽取的样本，可以看到，虽然以下这些样本中，并未出现`价格` 和 `卫生`，但模型依然正确识别除了这两个属性的情感信息。

| 样本 | 属性 | 观点词 | 情感倾向 |
| :----: |:----: |:----: |:----: |
|房间比较大，就是感觉贵了点，不太划算|价格|贵、不太划算|负向|
|这家店的房间很大，店家服务也很热情，而且很便宜|价格|便宜|正向|
|这次来荆州给我的房间小的无语了，所幸比较实惠|价格|实惠|正向|
|酒店不大，有点不干净|卫生|不干净|负向|
|老板人很好，房间虽然很大，但有点脏|卫生|脏|负向|
|房间不大，很温暖，也很干净|卫生|干净|正向|


<a name="6"></a>

## **6. 模型部署**

<a name="6.1"></a>

### **6.1 基于SimpleServer进行服务化部署**
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

<a name="6.2"></a>

### **6.2 基于Pipeline进行部署**

本项目支持基于Pipeline的方式进行部署，用户只需要上传测试文件，即可获取对应的情感分析可视化结果，更多信息请参考[情感分析Pipeline](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/examples/sentiment_analysis)。
