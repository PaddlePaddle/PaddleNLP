# PaddleNLP一键预测功能：Taskflow API



<p align="left">
    <a href="https://pypi.org/project/paddlenlp/"><img src="https://img.shields.io/pypi/v/paddlenlp.svg?label=pip&logo=PyPI&logoColor=white"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleNLP?color=ffa"></a>
    <a href="https://pypi.org/project/paddlenlp/"><img src="https://img.shields.io/pypi/pyversions/paddlenlp"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg"></a>
    <a href="../../LICENSE"><img src="https://img.shields.io/github/license/paddlepaddle/paddlenlp"></a>
</p>


<h4 align="left">
  <a href=#QuickStart> QuickStart </a> |
  <a href=#社区交流> 社区交流 </a> |
  <a href=#详细使用> 一键预测&定制训练 </a> |
  <a href=#FAQ> FAQ </a>
</h4>


------------------------------------------------------------------------------------------

## 特性
PaddleNLP提供**开箱即用**的产业级NLP预置任务能力，无需训练，一键预测。
- 最全的中文任务：覆盖自然语言理解与自然语言生成两大核心应用；
- 极致的产业级效果：在多个中文场景上提供产业级的精度与预测性能；
- 统一的应用范式：通过`paddlenlp.Taskflow`调用，简捷易用。

| 任务名称                           | 调用方式                         | 一键预测 | 单条输入 | 多条输入 | 文档级输入 | 定制化训练 | 其它特性                                               |
| :--------------------------------- | -------------------------------- | -------- | -------- | -------- | ---------- | ---------- | ------------------------------------------------------ |
| [中文分词](#中文分词)              | `Taskflow("word_segmentation")`  | ✅        | ✅        | ✅        | ✅          | ✅          | 多种分词模式，满足快速切分和实体粒度精准切分           |
| [信息抽取](#信息抽取)           | `Taskflow("information_extraction")`| ✅        | ✅        | ✅        | ✅         | ✅          | 适配多场景的开放域通用信息抽取工具                     |
| [文本纠错](#文本纠错)              | `Taskflow("text_correction")`    | ✅        | ✅        | ✅        | ✅          | ✅          | 融合拼音特征的端到端文本纠错模型ERNIE-CSC              |
| [文本相似度](#文本相似度)          | `Taskflow("text_similarity")`    | ✅        | ✅        | ✅        |            |            | 基于百万量级Dureader Retrieval数据集训练RocketQA并达到前沿文本相似效果|
| [情感分析](#情感分析)      | `Taskflow("sentiment_analysis")`  | ✅        | ✅        | ✅        |            | ✅          | 集成BiLSTM、SKEP、UIE等模型，支持评论维度、观点抽取、情感极性分类等情感分析任务             |
| [模型特征提取](#模型特征提取)      | `Taskflow("feature_extraction")`  | ✅        | ✅        | ✅        |     ✅       |          | 集成文本，图片的特征抽取工具       |

## QuickStart

**环境依赖**
  - python >= 3.6
  - paddlepaddle >= 2.3.0
  - paddlenlp >= 2.3.4

![taskflow1](https://user-images.githubusercontent.com/11793384/159693816-fda35221-9751-43bb-b05c-7fc77571dd76.gif)

可进入 Jupyter Notebook 环境，在线体验 👉🏻  [进入在线运行环境](https://aistudio.baidu.com/aistudio/projectdetail/3696243)

PaddleNLP Taskflow API 支持任务持续丰富中，我们将根据开发者反馈，灵活调整功能建设优先级，可通过Issue或[问卷](https://iwenjuan.baidu.com/?code=44amg8)反馈给我们。

## 社区交流👬

- 微信扫描二维码并填写问卷之后，加入交流群领取福利
  - 获取5月18-19日每晚20:30《产业级通用信息抽取技术UIE+ERNIE轻量级模型》直播课链接
  - 10G重磅NLP学习大礼包：

  <div align="center">
  <img src="https://user-images.githubusercontent.com/11793384/168411900-d9f3d777-99ab-4b5c-8cdc-ef747a48b864.jpg" width="188" height="188" />
  </div>

## 详细使用

## PART Ⅰ &emsp; 一键预测

### 中文分词

<details><summary>&emsp;（可展开详情）多种分词模式，满足快速切分和实体粒度精准切分 </summary><div>

#### 三种分词模式，满足各类分词需求

```python
from paddlenlp import Taskflow

# 默认模式————实体粒度分词，在精度和速度上的权衡，基于百度LAC
>>> seg = Taskflow("word_segmentation")
>>> seg("近日国家卫健委发布第九版新型冠状病毒肺炎诊疗方案")
['近日', '国家卫健委', '发布', '第九版', '新型', '冠状病毒肺炎', '诊疗', '方案']

# 快速模式————最快：实现文本快速切分，基于jieba中文分词工具
>>> seg_fast = Taskflow("word_segmentation", mode="fast")
>>> seg_fast("近日国家卫健委发布第九版新型冠状病毒肺炎诊疗方案")
['近日', '国家', '卫健委', '发布', '第九版', '新型', '冠状病毒', '肺炎', '诊疗', '方案']

# 精确模式————最准：实体粒度切分准确度最高，基于百度解语
# 精确模式基于预训练模型，更适合实体粒度分词需求，适用于知识图谱构建、企业搜索Query分析等场景中
>>> seg_accurate = Taskflow("word_segmentation", mode="accurate")
>>> seg_accurate("近日国家卫健委发布第九版新型冠状病毒肺炎诊疗方案")
['近日', '国家卫健委', '发布', '第九版', '新型冠状病毒肺炎', '诊疗', '方案']
```

#### 批量样本输入，平均速度更快

输入为多个句子组成的list，平均速度会更快。

```python
>>> from paddlenlp import Taskflow
>>> seg = Taskflow("word_segmentation")
>>> seg(["第十四届全运会在西安举办", "三亚是一个美丽的城市"])
[['第十四届', '全运会', '在', '西安', '举办'], ['三亚', '是', '一个', '美丽', '的', '城市']]
```

#### 自定义词典

你可以通过传入`user_dict`参数，装载自定义词典来定制分词结果。
在默认模式和精确模式下，词典文件每一行由一个或多个自定义item组成。词典文件`user_dict.txt`示例：
```text
平原上的火焰
上 映
```

在快速模式下，词典文件每一行为一个自定义item+"\t"+词频（词频可省略，词频省略则自动计算能保证分出该词的词频），暂时不支持黑名单词典（即通过设置”年“、”末“，以达到切分”年末“的目的）。词典文件`user_dict.txt`示例：

```text
平原上的火焰  10
```

加载自定义词典及输出结果示例：
```python
>>> from paddlenlp import Taskflow
>>> seg = Taskflow("word_segmentation")
>>> seg("平原上的火焰宣布延期上映")
['平原', '上', '的', '火焰', '宣布', '延期', '上映']
>>> seg = Taskflow("word_segmentation", user_dict="user_dict.txt")
>>> seg("平原上的火焰宣布延期上映")
['平原上的火焰', '宣布', '延期', '上', '映']
```
#### 参数说明
* `mode`：指定分词模式，默认为None。
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `user_dict`：自定义词典文件路径，默认为None。
* `task_path`：自定义任务路径，默认为None。
</div></details>


### 信息抽取
<details><summary>&emsp; 适配多场景的开放域通用信息抽取工具 </summary><div>

开放域信息抽取是信息抽取的一种全新范式，主要思想是减少人工参与，利用单一模型支持多种类型的开放抽取任务，用户可以使用自然语言自定义抽取目标，在实体、关系类别等未定义的情况下抽取输入文本中的信息片段。

#### 实体抽取

  命名实体识别（Named Entity Recognition，简称NER），是指识别文本中具有特定意义的实体。在开放域信息抽取中，抽取的类别没有限制，用户可以自己定义。

  - 例如抽取的目标实体类型是"时间"、"选手"和"赛事名称", schema构造如下：

    ```text
    ['时间', '选手', '赛事名称']
    ```

    调用示例：

    ```python
    >>> from pprint import pprint
    >>> from paddlenlp import Taskflow

    >>> schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
    >>> ie = Taskflow('information_extraction', schema=schema)
    >>> pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")) # Better print results using pprint
    [{'时间': [{'end': 6,
              'probability': 0.9857378532924486,
              'start': 0,
              'text': '2月8日上午'}],
      '赛事名称': [{'end': 23,
                'probability': 0.8503089953268272,
                'start': 6,
                'text': '北京冬奥会自由式滑雪女子大跳台决赛'}],
      '选手': [{'end': 31,
              'probability': 0.8981548639781138,
              'start': 28,
              'text': '谷爱凌'}]}]
    ```

  - 例如抽取的目标实体类型是"肿瘤的大小"、"肿瘤的个数"、"肝癌级别"和"脉管内癌栓分级", schema构造如下：

    ```text
    ['肿瘤的大小', '肿瘤的个数', '肝癌级别', '脉管内癌栓分级']
    ```

    在上例中我们已经实例化了一个`Taskflow`对象，这里可以通过`set_schema`方法重置抽取目标。

    调用示例：

    ```python
    >>> schema = ['肿瘤的大小', '肿瘤的个数', '肝癌级别', '脉管内癌栓分级']
    >>> ie.set_schema(schema)
    >>> pprint(ie("（右肝肿瘤）肝细胞性肝癌（II-III级，梁索型和假腺管型），肿瘤包膜不完整，紧邻肝被膜，侵及周围肝组织，未见脉管内癌栓（MVI分级：M0级）及卫星子灶形成。（肿物1个，大小4.2×4.0×2.8cm）。"))
    [{'肝癌级别': [{'end': 20,
                'probability': 0.9243267447402701,
                'start': 13,
                'text': 'II-III级'}],
      '肿瘤的个数': [{'end': 84,
                'probability': 0.7538413804059623,
                'start': 82,
                'text': '1个'}],
      '肿瘤的大小': [{'end': 100,
                'probability': 0.8341128043459491,
                'start': 87,
                'text': '4.2×4.0×2.8cm'}],
      '脉管内癌栓分级': [{'end': 70,
                  'probability': 0.9083292325934664,
                  'start': 67,
                  'text': 'M0级'}]}]
    ```

  - 例如抽取的目标实体类型是"person"和"organization"，schema构造如下：

    ```text
    ['person', 'organization']
    ```

    英文模型调用示例：

    ```python
    >>> from pprint import pprint
    >>> from paddlenlp import Taskflow
    >>> schema = ['Person', 'Organization']
    >>> ie_en = Taskflow('information_extraction', schema=schema, model='uie-base-en')
    >>> pprint(ie_en('In 1997, Steve was excited to become the CEO of Apple.'))
    [{'Organization': [{'end': 53,
                        'probability': 0.9985840259877357,
                        'start': 48,
                        'text': 'Apple'}],
      'Person': [{'end': 14,
                  'probability': 0.999631971804547,
                  'start': 9,
                  'text': 'Steve'}]}]
    ```

#### 关系抽取

  关系抽取（Relation Extraction，简称RE），是指从文本中识别实体并抽取实体之间的语义关系，进而获取三元组信息，即<主体，谓语，客体>。

  - 例如以"竞赛名称"作为抽取主体，抽取关系类型为"主办方"、"承办方"和"已举办次数", schema构造如下：

    ```text
    {
      '竞赛名称': [
        '主办方',
        '承办方',
        '已举办次数'
      ]
    }
    ```

    调用示例：

    ```python
    >>> schema = {'竞赛名称': ['主办方', '承办方', '已举办次数']} # Define the schema for relation extraction
    >>> ie.set_schema(schema) # Reset schema
    >>> pprint(ie('2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。'))
    [{'竞赛名称': [{'end': 13,
                'probability': 0.7825402622754041,
                'relations': {'主办方': [{'end': 22,
                                      'probability': 0.8421710521379353,
                                      'start': 14,
                                      'text': '中国中文信息学会'},
                                      {'end': 30,
                                      'probability': 0.7580801847701935,
                                      'start': 23,
                                      'text': '中国计算机学会'}],
                              '已举办次数': [{'end': 82,
                                        'probability': 0.4671295049136148,
                                        'start': 80,
                                        'text': '4届'}],
                              '承办方': [{'end': 39,
                                      'probability': 0.8292706618236352,
                                      'start': 35,
                                      'text': '百度公司'},
                                      {'end': 72,
                                      'probability': 0.6193477885474685,
                                      'start': 56,
                                      'text': '中国计算机学会自然语言处理专委会'},
                                      {'end': 55,
                                      'probability': 0.7000497331473241,
                                      'start': 40,
                                      'text': '中国中文信息学会评测工作委员会'}]},
                'start': 0,
                'text': '2022语言与智能技术竞赛'}]}]
    ```

  - 例如以"person"作为抽取主体，抽取关系类型为"Company"和"Position", schema构造如下：

    ```text
    {
      'Person': [
        'Company',
        'Position'
      ]
    }
    ```

    英文模型调用示例：

    ```python
    >>> schema = [{'Person': ['Company', 'Position']}]
    >>> ie_en.set_schema(schema)
    >>> pprint(ie_en('In 1997, Steve was excited to become the CEO of Apple.'))
    [{'Person': [{'end': 14,
                  'probability': 0.999631971804547,
                  'relations': {'Company': [{'end': 53,
                                            'probability': 0.9960158209451642,
                                            'start': 48,
                                            'text': 'Apple'}],
                                'Position': [{'end': 44,
                                              'probability': 0.8871063806420736,
                                              'start': 41,
                                              'text': 'CEO'}]},
                  'start': 9,
                  'text': 'Steve'}]}]
    ```

#### 事件抽取

  事件抽取 (Event Extraction, 简称EE)，是指从自然语言文本中抽取预定义的事件触发词(Trigger)和事件论元(Argument)，组合为相应的事件结构化信息。

  - 例如抽取的目标是"地震"事件的"地震强度"、"时间"、"震中位置"和"震源深度"这些信息，schema构造如下：

    ```text
    {
      '地震触发词': [
        '地震强度',
        '时间',
        '震中位置',
        '震源深度'
      ]
    }
    ```

    触发词的格式统一为`触发词`或``XX触发词`，`XX`表示具体事件类型，上例中的事件类型是`地震`，则对应触发词为`地震触发词`。

    调用示例：

    ```python
    >>> schema = {'地震触发词': ['地震强度', '时间', '震中位置', '震源深度']} # Define the schema for event extraction
    >>> ie.set_schema(schema) # Reset schema
    >>> ie('中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。')
    [{'地震触发词': [{'text': '地震', 'start': 56, 'end': 58, 'probability': 0.9987181623528585, 'relations': {'地震强度': [{'text': '3.5级', 'start': 52, 'end': 56, 'probability': 0.9962985320905915}], '时间': [{'text': '5月16日06时08分', 'start': 11, 'end': 22, 'probability': 0.9882578028575182}], '震中位置': [{'text': '云南临沧市凤庆县(北纬24.34度，东经99.98度)', 'start': 23, 'end': 50, 'probability': 0.8551415716584501}], '震源深度': [{'text': '10千米', 'start': 63, 'end': 67, 'probability': 0.999158304648045}]}}]}]
    ```

  - 英文模型zero-shot方式**暂不支持事件抽取**，如有英文事件抽取相关语料请进行训练定制。

#### 评论观点抽取

  评论观点抽取，是指抽取文本中包含的评价维度、观点词。

  - 例如抽取的目标是文本中包含的评价维度及其对应的观点词和情感倾向，schema构造如下：

    ```text
    {
      '评价维度': [
        '观点词',
        '情感倾向[正向，负向]'
      ]
    }
    ```

    调用示例：

    ```python
    >>> schema = {'评价维度': ['观点词', '情感倾向[正向，负向]']} # Define the schema for opinion extraction
    >>> ie.set_schema(schema) # Reset schema
    >>> pprint(ie("店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队")) # Better print results using pprint
    [{'评价维度': [{'end': 20,
                'probability': 0.9817040258681473,
                'relations': {'情感倾向[正向，负向]': [{'probability': 0.9966142505350533,
                                              'text': '正向'}],
                              '观点词': [{'end': 22,
                                      'probability': 0.957396472711558,
                                      'start': 21,
                                      'text': '高'}]},
                'start': 17,
                'text': '性价比'},
              {'end': 2,
                'probability': 0.9696849569741168,
                'relations': {'情感倾向[正向，负向]': [{'probability': 0.9982153274927796,
                                              'text': '正向'}],
                              '观点词': [{'end': 4,
                                      'probability': 0.9945318044652538,
                                      'start': 2,
                                      'text': '干净'}]},
                'start': 0,
                'text': '店面'}]}]
    ```

  - 英文模型schema构造如下：

    ```text
    {
      'Aspect': [
        'Opinion',
        'Sentiment classification [negative, positive]'
      ]
    }
    ```

    英文模型调用示例：

    ```python
    >>> schema = [{'Aspect': ['Opinion', 'Sentiment classification [negative, positive]']}]
    >>> ie_en.set_schema(schema)
    >>> pprint(ie_en("The teacher is very nice."))
    [{'Aspect': [{'end': 11,
                  'probability': 0.4301476415932193,
                  'relations': {'Opinion': [{'end': 24,
                                            'probability': 0.9072940447883724,
                                            'start': 15,
                                            'text': 'very nice'}],
                                'Sentiment classification [negative, positive]': [{'probability': 0.9998571920670685,
                                                                                  'text': 'positive'}]},
                  'start': 4,
                  'text': 'teacher'}]}]
    ```

#### 情感分类

  - 句子级情感倾向分类，即判断句子的情感倾向是“正向”还是“负向”，schema构造如下：

    ```text
    '情感倾向[正向，负向]'
    ```

    调用示例：

    ```python
    >>> schema = '情感倾向[正向，负向]' # Define the schema for sentence-level sentiment classification
    >>> ie.set_schema(schema) # Reset schema
    >>> ie('这个产品用起来真的很流畅，我非常喜欢')
    [{'情感倾向[正向，负向]': [{'text': '正向', 'probability': 0.9988661643929895}]}]
    ```

    英文模型schema构造如下：

    ```text
    '情感倾向[正向，负向]'
    ```

    英文模型调用示例：

    ```python
    >>> schema = 'Sentiment classification [negative, positive]'
    >>> ie_en.set_schema(schema)
    >>> ie_en('I am sorry but this is the worst film I have ever seen in my life.')
    [{'Sentiment classification [negative, positive]': [{'text': 'negative', 'probability': 0.9998415771287057}]}]
    ```

#### 跨任务抽取

  - 例如在法律场景同时对文本进行实体抽取和关系抽取，schema可按照如下方式进行构造：

    ```text
    [
      "法院",
      {
          "原告": "委托代理人"
      },
      {
          "被告": "委托代理人"
      }
    ]
    ```

    调用示例：

    ```python
    >>> schema = ['法院', {'原告': '委托代理人'}, {'被告': '委托代理人'}]
    >>> ie.set_schema(schema)
    >>> pprint(ie("北京市海淀区人民法院\n民事判决书\n(199x)建初字第xxx号\n原告：张三。\n委托代理人李四，北京市 A律师事务所律师。\n被告：B公司，法定代表人王五，开发公司总经理。\n委托代理人赵六，北京市 C律师事务所律师。")) # Better print results using pprint
    [{'原告': [{'end': 37,
              'probability': 0.9949814024296764,
              'relations': {'委托代理人': [{'end': 46,
                                      'probability': 0.7956844697990384,
                                      'start': 44,
                                      'text': '李四'}]},
              'start': 35,
              'text': '张三'}],
      '法院': [{'end': 10,
              'probability': 0.9221074192336651,
              'start': 0,
              'text': '北京市海淀区人民法院'}],
      '被告': [{'end': 67,
              'probability': 0.8437349536631089,
              'relations': {'委托代理人': [{'end': 92,
                                      'probability': 0.7267121388225029,
                                      'start': 90,
                                      'text': '赵六'}]},
              'start': 64,
              'text': 'B公司'}]}]
    ```

#### 模型选择

- 多模型选择，满足精度、速度要求

  | 模型 |  结构  | 语言 |
  | :---: | :--------: | :--------: |
  | `uie-base` (默认)| 12-layers, 768-hidden, 12-heads | 中文 |
  | `uie-base-en` | 12-layers, 768-hidden, 12-heads | 英文 |
  | `uie-medical-base` | 12-layers, 768-hidden, 12-heads | 中文 |
  | `uie-medium`| 6-layers, 768-hidden, 12-heads | 中文 |
  | `uie-mini`| 6-layers, 384-hidden, 12-heads | 中文 |
  | `uie-micro`| 4-layers, 384-hidden, 12-heads | 中文 |
  | `uie-nano`| 4-layers, 312-hidden, 12-heads | 中文 |
  | `uie-m-large`| 24-layers, 1024-hidden, 16-heads | 中、英文 |
  | `uie-m-base`| 12-layers, 768-hidden, 12-heads | 中、英文 |

- `uie-nano`调用示例：

  ```python
  >>> from paddlenlp import Taskflow

  >>> schema = ['时间', '选手', '赛事名称']
  >>> ie = Taskflow('information_extraction', schema=schema, model="uie-nano")
  >>> ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")
  [{'时间': [{'text': '2月8日上午', 'start': 0, 'end': 6, 'probability': 0.6513581678349247}], '选手': [{'text': '谷爱凌', 'start': 28, 'end': 31, 'probability': 0.9819330659468051}], '赛事名称': [{'text': '北京冬奥会自由式滑雪女子大跳台决赛', 'start': 6, 'end': 23, 'probability': 0.4908131110420939}]}]
  ```

- `uie-m-base`和`uie-m-large`支持中英文混合抽取，调用示例：

  ```python
  >>> from pprint import pprint
  >>> from paddlenlp import Taskflow

  >>> schema = ['Time', 'Player', 'Competition', 'Score']
  >>> ie = Taskflow('information_extraction', schema=schema, model="uie-m-base", schema_lang="en")
  >>> pprint(ie(["2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！", "Rafael Nadal wins French Open Final!"]))
  [{'Competition': [{'end': 23,
                    'probability': 0.9373889907291257,
                    'start': 6,
                    'text': '北京冬奥会自由式滑雪女子大跳台决赛'}],
    'Player': [{'end': 31,
                'probability': 0.6981119555336441,
                'start': 28,
                'text': '谷爱凌'}],
    'Score': [{'end': 39,
              'probability': 0.9888507878270296,
              'start': 32,
              'text': '188.25分'}],
    'Time': [{'end': 6,
              'probability': 0.9784080036931151,
              'start': 0,
              'text': '2月8日上午'}]},
  {'Competition': [{'end': 35,
                    'probability': 0.9851549932171295,
                    'start': 18,
                    'text': 'French Open Final'}],
    'Player': [{'end': 12,
                'probability': 0.9379371275888104,
                'start': 0,
                'text': 'Rafael Nadal'}]}]
  ```

#### 定制训练

对于简单的抽取目标可以直接使用```paddlenlp.Taskflow```实现零样本（zero-shot）抽取，对于细分场景我们推荐使用[定制训练](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie)（标注少量数据进行模型微调）以进一步提升效果。

我们在互联网、医疗、金融三大垂类自建测试集上进行了实验：

<table>
<tr><th row_span='2'><th colspan='2'>金融<th colspan='2'>医疗<th colspan='2'>互联网
<tr><td><th>0-shot<th>5-shot<th>0-shot<th>5-shot<th>0-shot<th>5-shot
<tr><td>uie-base (12L768H)<td>46.43<td>70.92<td><b>71.83</b><td>85.72<td>78.33<td>81.86
<tr><td>uie-medium (6L768H)<td>41.11<td>64.53<td>65.40<td>75.72<td>78.32<td>79.68
<tr><td>uie-mini (6L384H)<td>37.04<td>64.65<td>60.50<td>78.36<td>72.09<td>76.38
<tr><td>uie-micro (4L384H)<td>37.53<td>62.11<td>57.04<td>75.92<td>66.00<td>70.22
<tr><td>uie-nano (4L312H)<td>38.94<td>66.83<td>48.29<td>76.74<td>62.86<td>72.35
<tr><td>uie-m-large (24L1024H)<td><b>49.35</b><td><b>74.55</b><td>70.50<td><b>92.66</b><td><b>78.49</b><td><b>83.02</b>
<tr><td>uie-m-base (12L768H)<td>38.46<td>74.31<td>63.37<td>87.32<td>76.27<td>80.13
</table>

0-shot表示无训练数据直接通过```paddlenlp.Taskflow```进行预测，5-shot表示每个类别包含5条标注数据进行模型微调。**实验表明UIE在垂类场景可以通过少量数据（few-shot）进一步提升效果**。

#### 可配置参数说明

* `schema`：定义任务抽取目标，可参考开箱即用中不同任务的调用示例进行配置。
* `schema_lang`：设置schema的语言，默认为`zh`, 可选有`zh`和`en`。因为中英schema的构造有所不同，因此需要指定schema的语言。该参数只对`uie-m-base`和`uie-m-large`模型有效。
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `model`：选择任务使用的模型，默认为`uie-base`，可选有`uie-base`, `uie-medium`, `uie-mini`, `uie-micro`, `uie-nano`, `uie-medical-base`, `uie-base-en`。
* `position_prob`：模型对于span的起始位置/终止位置的结果概率0~1之间，返回结果去掉小于这个阈值的结果，默认为0.5，span的最终概率输出为起始位置概率和终止位置概率的乘积。
* `precision`：选择模型精度，默认为`fp32`，可选有`fp16`和`fp32`。`fp16`推理速度更快。如果选择`fp16`，请先确保机器正确安装NVIDIA相关驱动和基础软件，**确保CUDA>=11.2，cuDNN>=8.1.1**，初次使用需按照提示安装相关依赖(主要是**确保安装onnxruntime-gpu**)。其次，需要确保GPU设备的CUDA计算能力（CUDA Compute Capability）大于7.0，典型的设备包括V100、T4、A10、A100、GTX 20系列和30系列显卡等。更多关于CUDA Compute Capability和精度支持情况请参考NVIDIA文档：[GPU硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)。
</div></details>


### 文本纠错
<details><summary>&emsp;融合拼音特征的端到端文本纠错模型ERNIE-CSC</summary><div>


#### 支持单条、批量预测

```python
>>> from paddlenlp import Taskflow
>>> corrector = Taskflow("text_correction")
# 单条输入
>>> corrector('遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇。')
[{'source': '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇。', 'target': '遇到逆境时，我们必须勇于面对，而且要愈挫愈勇。', 'errors': [{'position': 3, 'correction': {'竟': '境'}}]}]

# 批量预测
>>> corrector(['遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇。', '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。'])
[{'source': '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇。', 'target': '遇到逆境时，我们必须勇于面对，而且要愈挫愈勇。', 'errors': [{'position': 3, 'correction': {'竟': '境'}}]}, {'source': '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。', 'target': '人生就是如此，经过磨练才能让自己更加茁壮，才能使自己更加乐观。', 'errors': [{'position': 18, 'correction': {'拙': '茁'}}]}]
```

#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `task_path`：自定义任务路径，默认为None。
</div></details>

### 文本相似度
<details><summary>&emsp;基于百万量级Dureader Retrieval数据集训练RocketQA并达到前沿文本相似效果</summary><div>

#### 单条输入

+ Query-Query的相似度匹配

```python
>>> from paddlenlp import Taskflow
>>> similarity = Taskflow("text_similarity")
>>> similarity([["春天适合种什么花？", "春天适合种什么菜？"]])
[{'text1': '春天适合种什么花？', 'text2': '春天适合种什么菜？', 'similarity': 0.83402544}]
```

+ Query-Passage的相似度匹配

```python
>>> similarity = Taskflow("text_similarity", model='rocketqa-base-cross-encoder')
>>> similarity([["国家法定节假日共多少天?", "现在法定假日是元旦1天，春节3天，清明节1天，五一劳动节1天，端午节1天，国庆节3天，中秋节1天，共计11天。法定休息日每年52个周末总共104天。合到一起总计115天。"]])
[{'text1': '国家法定节假日共多少天?', 'text2': '现在法定假日是元旦1天，春节3天，清明节1天，五一劳动节1天，端午节1天，国庆节3天，中秋节1天，共计11天。法定休息日每年52个周末总共104天。合到一起总计115天。', 'similarity': 0.7174624800682068}]
```

#### 批量样本输入，平均速度更快

+ Query-Query的相似度匹配

```python
>>> from paddlenlp import Taskflow
>>> similarity = Taskflow("text_similarity")
>>> similarity([['春天适合种什么花？','春天适合种什么菜？'],['谁有狂三这张高清的','这张高清图，谁有']])
[{'text1': '春天适合种什么花？', 'text2': '春天适合种什么菜？', 'similarity': 0.83402544}, {'text1': '谁有狂三这张高清的', 'text2': '这张高清图，谁有', 'similarity': 0.6540646}]
```

+ Query-Passage的相似度匹配

```python
>>> similarity = Taskflow("text_similarity", model='rocketqa-base-cross-encoder')
>>> similarity([["国家法定节假日共多少天?", "现在法定假日是元旦1天，春节3天，清明节1天，五一劳动节1天，端午节1天，国庆节3天，中秋节1天，共计11天。法定休息日每年52个周末总共104天。合到一起总计115天。"],["衡量酒水的价格的因素有哪些?", "衡量酒水的价格的因素很多的，酒水的血统(也就是那里产的，采用什么工艺等）；存储的时间等等，酒水是一件很难标准化得商品，只要你敢要价，有买的那就值那个钱。"]])
[{'text1': '国家法定节假日共多少天?', 'text2': '现在法定假日是元旦1天，春节3天，清明节1天，五一劳动节1天，端午节1天，国庆节3天，中秋节1天，共计11天。法定休息日每年52个周末总共104天。合到一起总计115天。', 'similarity': 0.7174624800682068}, {'text1': '衡量酒水的价格的因素有哪些?', 'text2': '衡量酒水的价格的因素很多的，酒水的血统(也就是那里产的，采用什么工艺等）；存储的时间等等，酒水是一件很难标准化得商品，只要你敢要价，有买的那就值那个钱。', 'similarity': 0.9069755673408508}]

```

#### 模型选择

- 多模型选择，满足精度、速度要求

  | 模型 |  结构  | 语言 |
  | :---: | :--------: | :--------: |
  | `rocketqa-zh-dureader-cross-encoder`       | 12-layers, 768-hidden, 12-heads | 中文 |
  | `simbert-base-chinese` (默认)               | 12-layers, 768-hidden, 12-heads | 中文 |
  | `rocketqa-base-cross-encoder`              | 12-layers, 768-hidden, 12-heads | 中文 |
  | `rocketqa-medium-cross-encoder`            | 6-layers, 768-hidden, 12-heads | 中文 |
  | `rocketqa-mini-cross-encoder`              | 6-layers, 384-hidden, 12-heads | 中文 |
  | `rocketqa-micro-cross-encoder`             | 4-layers, 384-hidden, 12-heads | 中文 |
  | `rocketqa-nano-cross-encoder`              | 4-layers, 312-hidden, 12-heads | 中文 |
  | `rocketqav2-en-marco-cross-encoder`        | 12-layers, 768-hidden, 12-heads | 英文 |


#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `max_seq_len`：最大序列长度，默认为384。
* `task_path`：自定义任务路径，默认为None。
</div></details>

### 情感分析
<details><summary>&emsp;集成BiLSTM、SKEP、UIE等模型，支持评论维度、观点抽取、情感极性分类等情感分析任务 </summary><div>

#### 支持不同模型，速度快和精度高两种模式

```python
>>> from paddlenlp import Taskflow
# 默认使用bilstm模型进行预测，速度快
>>> senta = Taskflow("sentiment_analysis")
>>> senta("这个产品用起来真的很流畅，我非常喜欢")
[{'text': '这个产品用起来真的很流畅，我非常喜欢', 'label': 'positive', 'score': 0.9938690066337585}]

# 使用SKEP情感分析预训练模型进行预测，精度高
>>> senta = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch")
>>> senta("作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。")
[{'text': '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。', 'label': 'positive', 'score': 0.984320878982544}]

# 使用UIE模型进行情感分析，具有较强的样本迁移能力
# 1. 语句级情感分析
>>> schema = ['情感倾向[正向，负向]']
>>> senta = Taskflow("sentiment_analysis", model="uie-senta-base", schema=schema)
>>> senta('蛋糕味道不错，店家服务也很好')
[{'情感倾向[正向,负向]': [{'text': '正向', 'probability': 0.996646058824652}]}]

# 2. 评价维度级情感分析
>>> # Aspect Term Extraction
>>> # schema =  ["评价维度"]
>>> # Aspect - Opinion Extraction
>>> # schema =  [{"评价维度":["观点词"]}]
>>> # Aspect - Sentiment Extraction
>>> # schema =  [{"评价维度":["情感倾向[正向,负向,未提及]"]}]
>>> # Aspect - Sentiment - Opinion Extraction
>>> schema =  [{"评价维度":["观点词", "情感倾向[正向,负向,未提及]"]}]

>>> senta = Taskflow("sentiment_analysis", model="uie-senta-base", schema=schema)
>>> senta('蛋糕味道不错，店家服务也很热情')
[{'评价维度': [{'text': '服务', 'start': 9, 'end': 11, 'probability': 0.9709093024793489, 'relations': { '观点词': [{'text': '热情', 'start': 13, 'end': 15, 'probability': 0.9897222206316556}], '情感倾向[正向,负向,未提及]': [{'text': '正向', 'probability': 0.9999327669598301}]}}, {'text': '味道', 'start': 2, 'end': 4, 'probability': 0.9105472387838915, 'relations': {'观点词': [{'text': '不错', 'start': 4, 'end': 6, 'probability': 0.9946981266891619}], '情感倾向[正向,负向,未提及]': [{'text': '正向', 'probability': 0.9998829392709467}]}}]}]
```

#### 批量样本输入，平均速度更快
```python
>>> from paddlenlp import Taskflow
>>> schema =  [{"评价维度":["观点词", "情感倾向[正向,负向,未提及]"]}]
>>> senta = Taskflow("sentiment_analysis", model="uie-senta-base", schema=schema)
>>> senta(["房间不大，很干净", "老板服务热情，价格也便宜"])
[{'评价维度': [{'text': '房间', 'start': 0, 'end': 2, 'probability': 0.998526653966298, 'relations': {'观点词': [{'text': '干净', 'start': 6, 'end': 8, 'probability': 0.9899580841973474}, {'text': '不大', 'start': 2, 'end': 4, 'probability': 0.9945525066163512}], '情感倾向[正向,负向,未提及]': [{'text': '正向', 'probability': 0.6077412795680956}]}}]}, {'评价维度': [{'text': '服务', 'start': 2, 'end': 4, 'probability': 0.9913965811617516, 'relations': {'观点词': [{'text': '热情', 'start': 4, 'end': 6, 'probability': 0.9995530034336753}], '情感倾向[正向,负向,未提及]': [{'text': '正向', 'probability': 0.9956709542206106}]}}, {'text': '价格', 'start': 7, 'end': 9, 'probability': 0.9970075537913772, 'relations': {'观点词': [{'text': '便宜', 'start': 10, 'end': 12, 'probability': 0.9991568497876635}], '情感倾向[正向,负向,未提及]': [{'text': '正向', 'probability': 0.9943191048602245}]}}]}]
```

#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `model`：选择任务使用的模型，可选有`bilstm`,`skep_ernie_1.0_large_ch`,`uie-senta-base`,`uie-senta-medium`,`uie-senta-mini`,`uie-senta-micro`,`uie-senta-nano`。
* `task_path`：自定义任务路径，默认为None。
</div></details>

### 模型特征提取

<details><summary>&emsp; 基于百度自研中文图文跨模态预训练模型ERNIE-ViL 2.0</summary><div>

#### 多模态特征提取

```python
>>> from paddlenlp import Taskflow
>>> from PIL import Image
>>> import paddle.nn.functional as F
>>> vision_language= Taskflow("feature_extraction")
# 单条输入
>>> image_embeds = vision_language(Image.open("demo/000000039769.jpg"))
>>> image_embeds["features"]
Tensor(shape=[1, 768], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[-0.59475428, -0.69795364,  0.22144008,  0.88066685, -0.58184201,
# 单条输入
>>> text_embeds = vision_language("猫的照片")
>>> text_embeds['features']
Tensor(shape=[1, 768], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[ 0.04250504, -0.41429776,  0.26163983,  0.29910022,  0.39019185,
         -0.41884750, -0.19893740,  0.44328332,  0.08186490,  0.10953025,
         ......

# 多条输入
>>> image_embeds = vision_language([Image.open("demo/000000039769.jpg")])
>>> image_embeds["features"]
Tensor(shape=[1, 768], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[-0.59475428, -0.69795364,  0.22144008,  0.88066685, -0.58184201,
       ......
# 多条输入
>>> text_embeds = vision_language(["猫的照片","狗的照片"])
>>> text_embeds["features"]
Tensor(shape=[2, 768], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[ 0.04250504, -0.41429776,  0.26163983, ...,  0.26221892,
          0.34387422,  0.18779707],
        [ 0.06672225, -0.41456309,  0.13787819, ...,  0.21791610,
          0.36693242,  0.34208685]])
>>> image_features = image_embeds["features"]
>>> text_features = text_embeds["features"]
>>> image_features /= image_features.norm(axis=-1, keepdim=True)
>>> text_features /= text_features.norm(axis=-1, keepdim=True)
>>> logits_per_image = 100 * image_features @ text_features.t()
>>> probs = F.softmax(logits_per_image, axis=-1)
>>> probs
Tensor(shape=[1, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[0.99833173, 0.00166824]])
```
#### 模型选择

- 多模型选择，满足精度、速度要求

  | 模型 |  视觉| 文本  | 语言 |
  | :---: | :--------: | :--------: | :--------: |
  | `PaddlePaddle/ernie_vil-2.0-base-zh` (默认) | ViT | ERNIE | 中文 |
  | `OFA-Sys/chinese-clip-vit-base-patch16`                     | ViT-B/16 |RoBERTa-wwm-Base| 中文 |
  | `OFA-Sys/chinese-clip-vit-large-patch14`            | ViT-L/14 | RoBERTa-wwm-Base | 中文 |
  | `OFA-Sys/chinese-clip-vit-large-patch14-336px`              | ViT-L/14 | RoBERTa-wwm-Base | 中文 |


#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `_static_mode`：静态图模式，默认开启。
* `model`：选择任务使用的模型，默认为`PaddlePaddle/ernie_vil-2.0-base-zh`。

#### 文本特征提取

```python
>>> from paddlenlp import Taskflow
>>> import paddle.nn.functional as F
>>> text_encoder = Taskflow("feature_extraction", model='rocketqa-zh-base-query-encoder')
>>> text_embeds = text_encoder(['春天适合种什么花？','谁有狂三这张高清的?'])
>>> text_features1 = text_embeds["features"]
>>> text_features1
Tensor(shape=[2, 768], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[ 0.27640465, -0.13405125,  0.00612330, ..., -0.15600294,
         -0.18932408, -0.03029604],
        [-0.12041329, -0.07424965,  0.07895312, ..., -0.17068857,
          0.04485796, -0.18887770]])
>>> text_embeds = text_encoder('春天适合种什么菜？')
>>> text_features2 = text_embeds["features"]
>>> text_features2
Tensor(shape=[1, 768], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[ 0.32578075, -0.02398480, -0.18929179, -0.18639392, -0.04062131,
       ......
>>> probs = F.cosine_similarity(text_features1, text_features2)
>>> probs
Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [0.86455142, 0.41222256])
```

#### 模型选择

- 多模型选择，满足精度、速度要求

  | 模型 |  层数| 维度  | 语言|
  | :---: | :--------: | :--------: | :--------: |
  | `rocketqa-zh-dureader-query-encoder`  | 12 | 768 | 中文|
  | `rocketqa-zh-dureader-para-encoder`  | 12 | 768 | 中文|
  | `rocketqa-zh-base-query-encoder`  | 12 | 768 | 中文|
  | `rocketqa-zh-base-para-encoder`  | 12 | 768 | 中文|
  | `moka-ai/m3e-base`  | 12 | 768 | 中文|
  | `rocketqa-zh-medium-query-encoder`  | 6 | 768 | 中文|
  | `rocketqa-zh-medium-para-encoder`  | 6 | 768 | 中文|
  | `rocketqa-zh-mini-query-encoder`  | 6 | 384 | 中文|
  | `rocketqa-zh-mini-para-encoder`  | 6 | 384 | 中文|
  | `rocketqa-zh-micro-query-encoder`  | 4 | 384 | 中文|
  | `rocketqa-zh-micro-para-encoder`  | 4 | 384 | 中文|
  | `rocketqa-zh-nano-query-encoder`  | 4 | 312 | 中文|
  | `rocketqa-zh-nano-para-encoder`  | 4 | 312 | 中文|
  | `rocketqav2-en-marco-query-encoder`  | 12 | 768 | 英文|
  | `rocketqav2-en-marco-para-encoder`  | 12 | 768 | 英文|
  | `ernie-search-base-dual-encoder-marco-en"`  | 12 | 768 | 英文|

#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `max_seq_len`：文本序列的最大长度，默认为128。
* `return_tensors`: 返回的类型，有pd和np，默认为pd。
* `model`：选择任务使用的模型，默认为`PaddlePaddle/ernie_vil-2.0-base-zh`。
* `pooling_mode`：选择句向量获取方式，有'max_tokens','mean_tokens','mean_sqrt_len_tokens','cls_token'，默认为'cls_token'（`moka-ai/m3e-base`）。

</div></details>

## PART Ⅱ &emsp; 定制化训练

<details><summary>适配任务列表</summary><div>

如果你有自己的业务数据集，可以对模型效果进一步调优，支持定制化训练的任务如下：

|                           任务名称                           |                           默认路径                           |                                                              |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|         `Taskflow("word_segmentation", mode="base")`         |             `$HOME/.paddlenlp/taskflow/lac`                  | [示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/lexical_analysis) |
|       `Taskflow("word_segmentation", mode="accurate")`       |             `$HOME/.paddlenlp/taskflow/wordtag`              | [示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/ernie-ctm) |
|              `Taskflow("information_extraction", model="uie-base")`              |             `$HOME/.paddlenlp/taskflow/information_extraction/uie-base`              | [示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie) |
|              `Taskflow("information_extraction", model="uie-tiny")`              |             `$HOME/.paddlenlp/taskflow/information_extraction/uie-tiny`              | [示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie) |
|     `Taskflow("text_correction", model="ernie-csc")`     |  `$HOME/.paddlenlp/taskflow/text_correction/ernie-csc`   | [示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_correction/ernie-csc) |
| `Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch")` | `$HOME/.paddlenlp/taskflow/sentiment_analysis/skep_ernie_1.0_large_ch` | [示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/sentiment_analysis/skep) |

</div></details>


<details><summary>定制化训练示例</summary><div>

这里我们以命名实体识别`Taskflow("ner", mode="accurate")`为例，展示如何定制自己的模型。

调用`Taskflow`接口后，程序自动将相关文件下载到`$HOME/.paddlenlp/taskflow/wordtag/`，该默认路径包含以下文件:

```text
$HOME/.paddlenlp/taskflow/wordtag/
├── model_state.pdparams # 默认模型参数文件
├── model_config.json # 默认模型配置文件
└── tags.txt # 默认标签文件
```

* 参考上表中对应[示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/ernie-ctm)准备数据集和标签文件`tags.txt`，执行相应训练脚本得到自己的`model_state.pdparams`和`model_config.json`。

* 根据自己数据集情况，修改标签文件`tags.txt`。

* 将以上文件保存到任意路径中，自定义路径下的文件需要和默认路径的文件一致:

```text
custom_task_path/
├── model_state.pdparams # 定制模型参数文件
├── model_config.json # 定制模型配置文件
└── tags.txt # 定制标签文件
```
* 通过`task_path`指定自定义路径，使用Taskflow加载自定义模型进行一键预测：

```python
from paddlenlp import Taskflow
my_ner = Taskflow("ner", mode="accurate", task_path="./custom_task_path/")
```
</div></details>

## 模型算法

<details><summary>模型算法说明</summary><div>

<table>
  <tr><td>任务名称<td>模型<td>模型详情<td>训练集
  <tr><td rowspan="3">中文分词<td>默认模式: BiGRU+CRF<td>  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/lexical_analysis"> 训练详情 <td> 百度自建数据集，包含近2200万句子，覆盖多种场景
  <tr><td>快速模式：Jieba<td> - <td> -
  <tr><td>精确模式：WordTag<td> <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/ernie-ctm"> 训练详情 <td> 百度自建数据集，词类体系基于TermTree构建
  <tr><td>信息抽取<td> UIE <td> <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie"> 训练详情 <td> 百度自建数据集
  <tr><td>文本纠错<td>ERNIE-CSC<td> <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_correction/ernie-csc"> 训练详情 <td> SIGHAN简体版数据集及 <a href="https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml"> Automatic Corpus Generation生成的中文纠错数据集
  <tr><td>文本相似度<td>SimBERT<td> - <td> 收集百度知道2200万对相似句组
  <tr><td rowspan="3">情感分析<td> BiLSTM <td> - <td> 百度自建数据集
  <tr><td> SKEP <td> <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/sentiment_analysis/skep"> 训练详情 <td> 百度自建数据集
  <tr><td> UIE <td> <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/sentiment_analysis/unified_sentiment_extraction"> 训练详情 <td> 百度自建数据集
</table>

</div></details>

## FAQ

<details><summary><b>Q：</b>Taskflow如何修改任务保存路径？</summary><div>

**A:** Taskflow默认会将任务相关模型等文件保存到`$HOME/.paddlenlp`下，可以在任务初始化的时候通过`home_path`自定义修改保存路径。示例：
```python
from paddlenlp import Taskflow

ner = Taskflow("ner", home_path="/workspace")
```
通过以上方式即可将ner任务相关文件保存至`/workspace`路径下。
</div></details>


<details><summary><b>Q：</b>下载或调用模型失败，多次下载均失败怎么办？</summary><div>

**A:** Taskflow默认会将任务相关模型等文件保存到`$HOME/.paddlenlp/taskflow`下，如果下载或调用失败，可删除相应路径下的文件，重新尝试即可

</div></details>

<details><summary><b>Q：</b>Taskflow如何提升预测速度？</summary><div>

**A:** 可以结合设备情况适当调整batch_size，采用批量输入的方式来提升平均速率。示例：
```python
from paddlenlp import Taskflow

# 精确模式模型体积较大，可结合机器情况适当调整batch_size，采用批量样本输入的方式。
seg_accurate = Taskflow("word_segmentation", mode="accurate", batch_size=32)

# 批量样本输入，输入为多个句子组成的list，预测速度更快
texts = ["热梅茶是一道以梅子为主要原料制作的茶饮", "《孤女》是2010年九州出版社出版的小说，作者是余兼羽"]
seg_accurate(texts)
```
通过上述方式进行分词可以大幅提升预测速度。

</div></details>

<details><summary><b>Q：</b>后续会增加更多任务支持吗？</summary><div>

**A:** Taskflow支持任务持续丰富中，我们将根据开发者反馈，灵活调整功能建设优先级，可通过Issue或[问卷](https://wenjuan.baidu-int.com/manage/?r=survey/pageEdit&sid=85827)反馈给我们。

</div></details>


## 附录

<details><summary><b>参考资料</b> </summary><div>

1. [fxsjy/jieba](https://github.com/fxsjy/jieba)
2. [ZhuiyiTechnology/simbert](https://github.com/ZhuiyiTechnology/simbert)
3. [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://arxiv.org/abs/2012.00413)

</div></details>
