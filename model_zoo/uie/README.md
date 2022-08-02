# 通用信息抽取 UIE(Universal Information Extraction)

 **目录**

- [1. 模型简介](#模型简介)
- [2. 应用示例](#应用示例)
- [3. 开箱即用](#开箱即用)
  - [3.1 实体抽取](#实体抽取)
  - [3.2 关系抽取](#关系抽取)
  - [3.3 事件抽取](#事件抽取)
  - [3.4 评论观点抽取](#评论观点抽取)
  - [3.5 情感分类](#情感分类)
  - [3.6 跨任务抽取](#跨任务抽取)
  - [3.7 模型选择](#模型选择)
  - [3.8 更多配置](#更多配置)
- [4. 训练定制](#训练定制)
  - [4.1 代码结构](#代码结构)
  - [4.2 数据标注](#数据标注)
  - [4.3 模型微调](#模型微调)
  - [4.4 模型评估](#模型评估)
  - [4.5 定制模型一键预测](#定制模型一键预测)
  - [4.6 实验指标](#实验指标)
  - [4.7 模型部署](#模型部署)
- [5. CCKS比赛](#CCKS比赛)

<a name="模型简介"></a>

## 1. 模型简介

[UIE(Universal Information Extraction)](https://arxiv.org/pdf/2203.12277.pdf)：Yaojie Lu等人在ACL-2022中提出了通用信息抽取统一框架UIE。该框架实现了实体抽取、关系抽取、事件抽取、情感分析等任务的统一建模，并使得不同任务间具备良好的迁移和泛化能力。为了方便大家使用UIE的强大能力，PaddleNLP借鉴该论文的方法，基于ERNIE 3.0知识增强预训练模型，训练并开源了首个中文通用信息抽取模型UIE。该模型可以支持不限定行业领域和抽取目标的关键信息抽取，实现零样本快速冷启动，并具备优秀的小样本微调能力，快速适配特定的抽取目标。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167236006-66ed845d-21b8-4647-908b-e1c6e7613eb1.png height=400 hspace='10'/>
</div>

#### UIE的优势

- **使用简单**：用户可以使用自然语言自定义抽取目标，无需训练即可统一抽取输入文本中的对应信息。**实现开箱即用，并满足各类信息抽取需求**。

- **降本增效**：以往的信息抽取技术需要大量标注数据才能保证信息抽取的效果，为了提高开发过程中的开发效率，减少不必要的重复工作时间，开放域信息抽取可以实现零样本（zero-shot）或者少样本（few-shot）抽取，**大幅度降低标注数据依赖，在降低成本的同时，还提升了效果**。

- **效果领先**：开放域信息抽取在多种场景，多种任务上，均有不俗的表现。

<a name="应用示例"></a>

## 2. 应用示例

UIE不限定行业领域和抽取目标，以下是一些零样本行业示例：

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

<a name="开箱即用"></a>

## 3. 开箱即用

```paddlenlp.Taskflow```提供通用信息抽取、评价观点抽取等能力，可抽取多种类型的信息，包括但不限于命名实体识别（如人名、地名、机构名等）、关系（如电影的导演、歌曲的发行时间等）、事件（如某路口发生车祸、某地发生地震等）、以及评价维度、观点词、情感倾向等信息。用户可以使用自然语言自定义抽取目标，无需训练即可统一抽取输入文本中的对应信息。**实现开箱即用，并满足各类信息抽取需求**

<a name="实体抽取"></a>

#### 3.1 实体抽取

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

<a name="关系抽取"></a>

#### 3.2 关系抽取

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
    >>> ie_en('In 1997, Steve was excited to become the CEO of Apple.')
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

<a name="事件抽取"></a>

#### 3.3 事件抽取

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

  - 英文模型**暂不支持事件抽取**

<a name="评论观点抽取"></a>

#### 3.4 评论观点抽取

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

    调用示例：

    ```python
    >>> schema = [{'Comment object': ['Opinion', 'Sentiment classification [negative, positive]']}]
    >>> ie_en.set_schema(schema)
    >>> ie_en("overall i 'm happy with my toy.")
    [{'Comment object': [{'end': 30,
                          'probability': 0.9774399346859042,
                          'relations': {'Opinion': [{'end': 18,
                                                    'probability': 0.6168918705033555,
                                                    'start': 13,
                                                    'text': 'happy'}],
                                        'Sentiment classification [negative, positive]': [{'probability': 0.9999556545777182,
                                                                                          'text': 'positive'}]},
                          'start': 24,
                          'text': 'my toy'}]}]
    ```

<a name="情感分类"></a>

#### 3.5 情感分类

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
    >>> schema = [{'Person': ['Company', 'Position']}]
    >>> ie_en.set_schema(schema)
    >>> ie_en('I am sorry but this is the worst film I have ever seen in my life.')
    [{'Sentiment classification [negative, positive]': [{'text': 'negative', 'probability': 0.9998415771287057}]}]
    ```

<a name="跨任务抽取"></a>

#### 3.6 跨任务抽取

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

<a name="模型选择"></a>

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


- `uie-nano`调用示例

  ```python
  >>> from paddlenlp import Taskflow

  >>> schema = ['时间', '选手', '赛事名称']
  >>> ie = Taskflow('information_extraction', schema=schema, model="uie-nano")
  >>> ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")
  [{'时间': [{'text': '2月8日上午', 'start': 0, 'end': 6, 'probability': 0.6513581678349247}], '选手': [{'text': '谷爱凌', 'start': 28, 'end': 31, 'probability': 0.9819330659468051}], '赛事名称': [{'text': '北京冬奥会自由式滑雪女子大跳台决赛', 'start': 6, 'end': 23, 'probability': 0.4908131110420939}]}]
  ```

<a name="更多配置"></a>

#### 3.8 更多配置

```python
>>> from paddlenlp import Taskflow

>>> ie = Taskflow('information_extraction',
                  schema="",
                  batch_size=1,
                  model='uie-base',
                  position_prob=0.5,
                  precision='fp32')
```

* `schema`：定义任务抽取目标，可参考开箱即用中不同任务的调用示例进行配置。
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `model`：选择任务使用的模型，默认为`uie-base`，可选有`uie-base`, `uie-medium`, `uie-mini`, `uie-micro`, `uie-nano`和`uie-medical-base`, `uie-base-en`。
* `position_prob`：模型对于span的起始位置/终止位置的结果概率在0~1之间，返回结果去掉小于这个阈值的结果，默认为0.5，span的最终概率输出为起始位置概率和终止位置概率的乘积。
* `precision`：选择模型精度，默认为`fp32`，可选有`fp16`和`fp32`。`fp16`推理速度更快。如果选择`fp16`，请先确保机器正确安装NVIDIA相关驱动和基础软件，**确保CUDA>=11.2，cuDNN>=8.1.1**，初次使用需按照提示安装相关依赖。其次，需要确保GPU设备的CUDA计算能力（CUDA Compute Capability）大于7.0，典型的设备包括V100、T4、A10、A100、GTX 20系列和30系列显卡等。更多关于CUDA Compute Capability和精度支持情况请参考NVIDIA文档：[GPU硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)。

<a name="训练定制"></a>

## 4. 训练定制

对于简单的抽取目标可以直接使用```paddlenlp.Taskflow```实现零样本（zero-shot）抽取，对于细分场景我们推荐使用轻定制功能（标注少量数据进行模型微调）以进一步提升效果。下面通过`报销工单信息抽取`的例子展示如何通过5条训练数据进行UIE模型微调。

<a name="代码结构"></a>

#### 4.1 代码结构

```shell
.
├── utils.py          # 数据处理工具
├── model.py          # 模型组网脚本
├── doccano.py        # 数据标注脚本
├── doccano.md        # 数据标注文档
├── finetune.py       # 模型微调脚本
├── evaluate.py       # 模型评估脚本
└── README.md
```

<a name="数据标注"></a>

#### 4.2 数据标注

我们推荐使用数据标注平台[doccano](https://github.com/doccano/doccano) 进行数据标注，本示例也打通了从标注到训练的通道，即doccano导出数据后可通过[doccano.py](./doccano.py)脚本轻松将数据转换为输入模型时需要的形式，实现无缝衔接。标注方法的详细介绍请参考[doccano数据标注指南](doccano.md)。

原始数据示例：

```text
深大到双龙28块钱4月24号交通费
```

抽取的目标(schema)为：

```python
schema = ['出发地', '目的地', '费用', '时间']
```

标注步骤如下：

- 在doccano平台上，创建一个类型为``序列标注``的标注项目。
- 定义实体标签类别，上例中需要定义的实体标签有``出发地``、``目的地``、``费用``和``时间``。
- 使用以上定义的标签开始标注数据，下面展示了一个doccano标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167336891-afef1ad5-8777-456d-805b-9c65d9014b80.png height=100 hspace='10'/>
</div>

- 标注完成后，在doccano平台上导出文件，并将其重命名为``doccano_ext.json``后，放入``./data``目录下。

- 这里我们提供预先标注好的文件[doccano_ext.json](https://bj.bcebos.com/paddlenlp/datasets/uie/doccano_ext.json)，可直接下载并放入`./data`目录。执行以下脚本进行数据转换，执行后会在`./data`目录下生成训练/验证/测试集文件。

```shell
python doccano.py \
    --doccano_file ./data/doccano_ext.json \
    --task_type ext \
    --save_dir ./data \
    --splits 0.8 0.2 0
```


可配置参数说明：

- ``doccano_file``: 从doccano导出的数据标注文件。
- ``save_dir``: 训练数据的保存目录，默认存储在``data``目录下。
- ``negative_ratio``: 最大负例比例，该参数只对抽取类型任务有效，适当构造负例可提升模型效果。负例数量和实际的标签数量有关，最大负例数量 = negative_ratio * 正例数量。该参数只对训练集有效，默认为5。为了保证评估指标的准确性，验证集和测试集默认构造全负例。
- ``splits``: 划分数据集时训练集、验证集所占的比例。默认为[0.8, 0.1, 0.1]表示按照``8:1:1``的比例将数据划分为训练集、验证集和测试集。
- ``task_type``: 选择任务类型，可选有抽取和分类两种类型的任务。
- ``options``: 指定分类任务的类别标签，该参数只对分类类型任务有效。默认为["正向", "负向"]。
- ``prompt_prefix``: 声明分类任务的prompt前缀信息，该参数只对分类类型任务有效。默认为"情感倾向"。
- ``is_shuffle``: 是否对数据集进行随机打散，默认为True。
- ``seed``: 随机种子，默认为1000.
- ``separator``: 实体类别/评价维度与分类标签的分隔符，该参数只对实体/评价维度级分类任务有效。默认为"##"。

备注：
- 默认情况下 [doccano.py](./doccano.py) 脚本会按照比例将数据划分为 train/dev/test 数据集
- 每次执行 [doccano.py](./doccano.py) 脚本，将会覆盖已有的同名数据文件
- 在模型训练阶段我们推荐构造一些负例以提升模型效果，在数据转换阶段我们内置了这一功能。可通过`negative_ratio`控制自动构造的负样本比例；负样本数量 = negative_ratio * 正样本数量。
- 对于从doccano导出的文件，默认文件中的每条数据都是经过人工正确标注的。

更多**不同类型任务（关系抽取、事件抽取、评价观点抽取等）的标注规则及参数说明**，请参考[doccano数据标注指南](doccano.md)。

<a name="模型微调"></a>

#### 4.3 模型微调

单卡启动：

```shell
python finetune.py \
    --train_path ./data/train.txt \
    --dev_path ./data/dev.txt \
    --save_dir ./checkpoint \
    --learning_rate 1e-5 \
    --batch_size 16 \
    --max_seq_len 512 \
    --num_epochs 100 \
    --model uie-base \
    --seed 1000 \
    --logging_steps 10 \
    --valid_steps 100 \
    --device gpu
```

多卡启动：

```shell
python -u -m paddle.distributed.launch --gpus "0,1" finetune.py \
  --train_path ./data/train.txt \
  --dev_path ./data/dev.txt \
  --save_dir ./checkpoint \
  --learning_rate 1e-5 \
  --batch_size 16 \
  --max_seq_len 512 \
  --num_epochs 100 \
  --model uie-base \
  --seed 1000 \
  --logging_steps 10 \
  --valid_steps 100 \
  --device gpu
```

可配置参数说明：

- `train_path`: 训练集文件路径。
- `dev_path`: 验证集文件路径。
- `save_dir`: 模型存储路径，默认为`./checkpoint`。
- `learning_rate`: 学习率，默认为1e-5。
- `batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
- `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
- `num_epochs`: 训练轮数，默认为100。
- `model`: 选择模型，程序会基于选择的模型进行模型微调，可选有`uie-base`, `uie-medium`, `uie-mini`, `uie-micro`和`uie-nano`，默认为`uie-base`。
- `seed`: 随机种子，默认为1000.
- `logging_steps`: 日志打印的间隔steps数，默认10。
- `valid_steps`: evaluate的间隔steps数，默认100。
- `device`: 选用什么设备进行训练，可选cpu或gpu。

<a name="模型评估"></a>

#### 4.4 模型评估

通过运行以下命令进行模型评估：

```shell
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --batch_size 16 \
    --max_seq_len 512
```

评估方式说明：采用单阶段评价的方式，即关系抽取、事件抽取等需要分阶段预测的任务对每一阶段的预测结果进行分别评价。验证/测试集默认会利用同一层级的所有标签来构造出全部负例。

可开启`debug`模式对每个正例类别分别进行评估，该模式仅用于模型调试：

```shell
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --debug
```

输出打印示例：

```text
[2022-06-23 08:25:23,017] [    INFO] - -----------------------------
[2022-06-23 08:25:23,017] [    INFO] - Class name: 时间
[2022-06-23 08:25:23,018] [    INFO] - Evaluation precision: 1.00000 | recall: 1.00000 | F1: 1.00000
[2022-06-23 08:25:23,145] [    INFO] - -----------------------------
[2022-06-23 08:25:23,146] [    INFO] - Class name: 目的地
[2022-06-23 08:25:23,146] [    INFO] - Evaluation precision: 0.64286 | recall: 0.90000 | F1: 0.75000
[2022-06-23 08:25:23,272] [    INFO] - -----------------------------
[2022-06-23 08:25:23,273] [    INFO] - Class name: 费用
[2022-06-23 08:25:23,273] [    INFO] - Evaluation precision: 0.11111 | recall: 0.10000 | F1: 0.10526
[2022-06-23 08:25:23,399] [    INFO] - -----------------------------
[2022-06-23 08:25:23,399] [    INFO] - Class name: 出发地
[2022-06-23 08:25:23,400] [    INFO] - Evaluation precision: 1.00000 | recall: 1.00000 | F1: 1.00000
```

可配置参数说明：

- `model_path`: 进行评估的模型文件夹路径，路径下需包含模型权重文件`model_state.pdparams`及配置文件`model_config.json`。
- `test_path`: 进行评估的测试集文件。
- `batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
- `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
- `model`: 选择所使用的模型，可选有`uie-base`, `uie-medium`, `uie-mini`, `uie-micro`和`uie-nano`，默认为`uie-base`。
- `debug`: 是否开启debug模式对每个正例类别分别进行评估，该模式仅用于模型调试，默认关闭。

<a name="定制模型一键预测"></a>

#### 4.5 定制模型一键预测

`paddlenlp.Taskflow`装载定制模型，通过`task_path`指定模型权重文件的路径，路径下需要包含训练好的模型权重文件`model_state.pdparams`。

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow

>>> schema = ['出发地', '目的地', '费用', '时间']
# 设定抽取目标和定制化模型权重路径
>>> my_ie = Taskflow("information_extraction", schema=schema, task_path='./checkpoint/model_best')
>>> pprint(my_ie("城市内交通费7月5日金额114广州至佛山"))
[{'出发地': [{'end': 17,
           'probability': 0.9975287467835301,
           'start': 15,
           'text': '广州'}],
  '时间': [{'end': 10,
          'probability': 0.9999476678061399,
          'start': 6,
          'text': '7月5日'}],
  '目的地': [{'end': 20,
           'probability': 0.9998511131226735,
           'start': 18,
           'text': '佛山'}],
  '费用': [{'end': 15,
          'probability': 0.9994474579292856,
          'start': 12,
          'text': '114'}]}]
```

<a name="实验指标"></a>

#### 4.6 实验指标

我们在互联网、医疗、金融三大垂类自建测试集上进行了实验：

<table>
<tr><th row_span='2'><th colspan='2'>金融<th colspan='2'>医疗<th colspan='2'>互联网
<tr><td><th>0-shot<th>5-shot<th>0-shot<th>5-shot<th>0-shot<th>5-shot
<tr><td>uie-base (12L768H)<td><b>46.43</b><td><b>70.92</b><td><b>71.83</b><td><b>85.72</b><td><b>78.33</b><td><b>81.86</b>
<tr><td>uie-medium (6L768H)<td>41.11<td>64.53<td>65.40<td>75.72<td>78.32<td>79.68
<tr><td>uie-mini (6L384H)<td>37.04<td>64.65<td>60.50<td>78.36<td>72.09<td>76.38
<tr><td>uie-micro (4L384H)<td>37.53<td>62.11<td>57.04<td>75.92<td>66.00<td>70.22
<tr><td>uie-nano (4L312H)<td>38.94<td>66.83<td>48.29<td>76.74<td>62.86<td>72.35
</table>

0-shot表示无训练数据直接通过```paddlenlp.Taskflow```进行预测，5-shot表示基于5条标注数据进行模型微调。**实验表明UIE在垂类场景可以通过少量数据（few-shot）进一步提升效果**。

<a name="模型部署"></a>

#### 4.7 模型部署

以下是UIE Python端的部署流程，包括环境准备、模型导出和使用示例。

- 环境准备
  UIE的部署分为CPU和GPU两种情况，请根据你的部署环境安装对应的依赖。

  - CPU端

    CPU端的部署请使用如下命令安装所需依赖

    ```shell
    pip install -r deploy/python/requirements_cpu.txt
    ```

  - GPU端

    为了在GPU上获得最佳的推理性能和稳定性，请先确保机器已正确安装NVIDIA相关驱动和基础软件，确保**CUDA >= 11.2，cuDNN >= 8.1.1**，并使用以下命令安装所需依赖

    ```shell
    pip install -r deploy/python/requirements_gpu.txt
    ```

    如需使用半精度（FP16）部署，请确保GPU设备的CUDA计算能力 (CUDA Compute Capability) 大于7.0，典型的设备包括V100、T4、A10、A100、GTX 20系列和30系列显卡等。
    更多关于CUDA Compute Capability和精度支持情况请参考NVIDIA文档：[GPU硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)


- 模型导出

  将训练后的动态图参数导出为静态图参数：

  ```shell
  python export_model.py --model_path ./checkpoint/model_best --output_path ./export
  ```

  可配置参数说明：

  - `model_path`: 动态图训练保存的参数路径，路径下包含模型参数文件`model_state.pdparams`和模型配置文件`model_config.json`。
  - `output_path`: 静态图参数导出路径，默认导出路径为`./export`。

- 推理

  - CPU端推理样例

    在CPU端，请使用如下命令进行部署

    ```shell
    python deploy/python/infer_cpu.py --model_path_prefix export/inference
    ```

    可配置参数说明：

    - `model_path_prefix`: 用于推理的Paddle模型文件路径，需加上文件前缀名称。例如模型文件路径为`./export/inference.pdiparams`，则传入`./export/inference`。
    - `position_prob`：模型对于span的起始位置/终止位置的结果概率0~1之间，返回结果去掉小于这个阈值的结果，默认为0.5，span的最终概率输出为起始位置概率和终止位置概率的乘积。
    - `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
    - `batch_size`: 批处理大小，请结合机器情况进行调整，默认为4。

  - GPU端推理样例

    在GPU端，请使用如下命令进行部署

    ```shell
    python deploy/python/infer_gpu.py --model_path_prefix export/inference --use_fp16
    ```

    可配置参数说明：

    - `model_path_prefix`: 用于推理的Paddle模型文件路径，需加上文件前缀名称。例如模型文件路径为`./export/inference.pdiparams`，则传入`./export/inference`。
    - `use_fp16`: 是否使用FP16进行加速，默认关闭。
    - `position_prob`：模型对于span的起始位置/终止位置的结果概率0~1之间，返回结果去掉小于这个阈值的结果，默认为0.5，span的最终概率输出为起始位置概率和终止位置概率的乘积。
    - `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
    - `batch_size`: 批处理大小，请结合机器情况进行调整，默认为4。

<a name="CCKS比赛"></a>

## 5.CCKS比赛

为了进一步探索通用信息抽取的边界，我们举办了**CCKS 2022 千言通用信息抽取竞赛评测**（2022/03/30 - 2022/07/31）。

- [报名链接](https://aistudio.baidu.com/aistudio/competition/detail/161/0/introduction)
- [基线代码](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction/DuUIE)

## References
- **[Unified Structure Generation for Universal Information Extraction](https://arxiv.org/pdf/2203.12277.pdf)**
