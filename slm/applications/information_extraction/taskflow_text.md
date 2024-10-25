# 文本抽取任务 UIE Taskflow 使用指南

**目录**
- [1. 功能简介](#1)
- [2. 应用示例](#2)
- [3. 文本信息抽取](#3)
  - [3.1 实体抽取](#31)
  - [3.2 关系抽取](#32)
  - [3.3 事件抽取](#33)
  - [3.4 评论观点抽取](#34)
  - [3.5 情感分类](#35)
  - [3.6 跨任务抽取](#36)
  - [3.7 模型选择](#37)
  - [3.8 更多配置](#38)

<a name="1"></a>

## 1. 功能简介

```paddlenlp.Taskflow```提供纯文本的通用信息抽取、评价观点抽取等能力，可抽取多种类型的信息，包括但不限于命名实体识别（如人名、地名、机构名等）、关系（如电影的导演、歌曲的发行时间等）、事件（如某路口发生车祸、某地发生地震等）、以及评价维度、观点词、情感倾向等信息。用户可以使用自然语言自定义抽取目标，无需训练即可统一抽取输入文本中的对应信息。**实现开箱即用，并满足各类信息抽取需求**

<a name="2"></a>

## 2. 应用示例

UIE 不限定行业领域和抽取目标，以下是一些通过 Taskflow 实现开箱即用的行业示例：

- 医疗场景-专病结构化

![image](https://user-images.githubusercontent.com/40840292/169017581-93c8ee44-856d-4d17-970c-b6138d10f8bc.png)

- 法律场景-判决书抽取

![image](https://user-images.githubusercontent.com/40840292/169017863-442c50f1-bfd4-47d0-8d95-8b1d53cfba3c.png)

- 金融场景-收入证明、招股书抽取

![image](https://user-images.githubusercontent.com/40840292/169017982-e521ddf6-d233-41f3-974e-6f40f8f2edbc.png)

- 公安场景-事故报告抽取

![image](https://user-images.githubusercontent.com/40840292/169018340-31efc1bf-f54d-43f7-b62a-8f7ce9bf0536.png)

- 旅游场景-宣传册、手册抽取

![image](https://user-images.githubusercontent.com/40840292/169018113-c937eb0b-9fd7-4ecc-8615-bcdde2dac81d.png)

<a name="3"></a>

## 3. 文本信息抽取

<a name="31"></a>

#### 3.1 实体抽取

  实体抽取，又称命名实体识别（Named Entity Recognition，简称 NER），是指识别文本中具有特定意义的实体。在开放域信息抽取中，抽取的类别没有限制，用户可以自己定义。

  - 例如抽取的目标实体类型是"时间"、"选手"和"赛事名称", schema 构造如下：

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

  - 例如抽取的目标实体类型是"肿瘤的大小"、"肿瘤的个数"、"肝癌级别"和"脉管内癌栓分级", schema 构造如下：

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

  - 例如抽取的目标实体类型是"person"和"organization"，schema 构造如下：

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

<a name="32"></a>

#### 3.2 关系抽取

  关系抽取（Relation Extraction，简称 RE），是指从文本中识别实体并抽取实体之间的语义关系，进而获取三元组信息，即<主体，谓语，客体>。

  - 例如以"竞赛名称"作为抽取主体，抽取关系类型为"主办方"、"承办方"和"已举办次数", schema 构造如下：

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

  - 例如以"person"作为抽取主体，抽取关系类型为"Company"和"Position", schema 构造如下：

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

<a name="33"></a>

#### 3.3 事件抽取

  事件抽取 (Event Extraction, 简称 EE)，是指从自然语言文本中抽取预定义的事件触发词(Trigger)和事件论元(Argument)，组合为相应的事件结构化信息。

  - 例如抽取的目标是"地震"事件的"地震强度"、"时间"、"震中位置"和"震源深度"这些信息，schema 构造如下：

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

    触发词的格式统一为`触发词`或``XX 触发词`，`XX`表示具体事件类型，上例中的事件类型是`地震`，则对应触发词为`地震触发词`。

    调用示例：

    ```python
    >>> schema = {'地震触发词': ['地震强度', '时间', '震中位置', '震源深度']} # Define the schema for event extraction
    >>> ie.set_schema(schema) # Reset schema
    >>> ie('中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。')
    [{'地震触发词': [{'text': '地震', 'start': 56, 'end': 58, 'probability': 0.9987181623528585, 'relations': {'地震强度': [{'text': '3.5级', 'start': 52, 'end': 56, 'probability': 0.9962985320905915}], '时间': [{'text': '5月16日06时08分', 'start': 11, 'end': 22, 'probability': 0.9882578028575182}], '震中位置': [{'text': '云南临沧市凤庆县(北纬24.34度，东经99.98度)', 'start': 23, 'end': 50, 'probability': 0.8551415716584501}], '震源深度': [{'text': '10千米', 'start': 63, 'end': 67, 'probability': 0.999158304648045}]}}]}]
    ```

  - 英文模型**暂不支持事件抽取**，如有需要可使用英文事件数据集进行定制。

<a name="34"></a>

#### 3.4 评论观点抽取

  评论观点抽取，是指抽取文本中包含的评价维度、观点词。

  - 例如抽取的目标是文本中包含的评价维度及其对应的观点词和情感倾向，schema 构造如下：

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

  - 英文模型 schema 构造如下：

    ```text
    {
      'Aspect': [
        'Opinion',
        'Sentiment classification [negative, positive]'
      ]
    }
    ```

    调用示例：

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

<a name="35"></a>

#### 3.5 情感分类

  - 句子级情感倾向分类，即判断句子的情感倾向是“正向”还是“负向”，schema 构造如下：

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

    英文模型 schema 构造如下：

    ```text
    'Sentiment classification [negative, positive]'
    ```

    英文模型调用示例：

    ```python
    >>> schema = 'Sentiment classification [negative, positive]'
    >>> ie_en.set_schema(schema)
    >>> ie_en('I am sorry but this is the worst film I have ever seen in my life.')
    [{'Sentiment classification [negative, positive]': [{'text': 'negative', 'probability': 0.9998415771287057}]}]
    ```

<a name="36"></a>

#### 3.6 跨任务抽取

  - 例如在法律场景同时对文本进行实体抽取和关系抽取，schema 可按照如下方式进行构造：

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

<a name="37"></a>

#### 3.7 模型选择

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

<a name="38"></a>

#### 3.8 更多配置

```python
>>> from paddlenlp import Taskflow

>>> ie = Taskflow('information_extraction',
                  schema="",
                  schema_lang="ch",
                  batch_size=16,
                  model='uie-base',
                  position_prob=0.5,
                  precision='fp32')
```

* `schema`：定义任务抽取目标，可参考开箱即用中不同任务的调用示例进行配置。
* `schema_lang`：设置 schema 的语言，默认为`ch`, 可选有`ch`和`en`。因为中英 schema 的构造有所不同，因此需要指定 schema 的语言。该参数只对`uie-x-base`,`uie-m-base`和`uie-m-large`模型有效。
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为16。
* `model`：选择任务使用的模型，默认为`uie-base`，可选有`uie-base`, `uie-medium`, `uie-mini`, `uie-micro`, `uie-nano`和`uie-medical-base`, `uie-base-en`，`uie-x-base`。
* `position_prob`：模型对于 span 的起始位置/终止位置的结果概率在0~1之间，返回结果去掉小于这个阈值的结果，默认为0.5，span 的最终概率输出为起始位置概率和终止位置概率的乘积。
* `precision`：选择模型精度，默认为`fp32`，可选有`fp16`和`fp32`。`fp16`推理速度更快，支持 GPU 和 NPU 硬件环境。如果选择`fp16`，在 GPU 硬件环境下，请先确保机器正确安装 NVIDIA 相关驱动和基础软件，**确保 CUDA>=11.2，cuDNN>=8.1.1**，初次使用需按照提示安装相关依赖。其次，需要确保 GPU 设备的 CUDA 计算能力（CUDA Compute Capability）大于7.0，典型的设备包括 V100、T4、A10、A100、GTX 20系列和30系列显卡等。更多关于 CUDA Compute Capability 和精度支持情况请参考 NVIDIA 文档：[GPU 硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)。
