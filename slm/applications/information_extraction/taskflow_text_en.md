# UIE Taskflow User Guide - Text Information Extraction

**Table of contents**
- [1. Introduction](#1)
- [2. Examples](#2)
- [3. Text Information Extraction](#3)
   - [3.1 Entity Extraction](#31)
   - [3.2 Relation Extraction](#32)
   - [3.3 Event Extraction](#33)
   - [3.4 Opinion Extraction](#34)
   - [3.5 Sentiment Classification](#35)
   - [3.6 Multi-task Extraction](#36)
   - [3.7 Available Models](#37)
   - [3.8 More Configuration](#38)

<a name="1"></a>

## 1. Introduction
```paddlenlp.Taskflow``` provides general information extraction of text and documents, evaluation opinion extraction and other capabilities, and can extract various types of information, including but not limited to named entities (such as person name, place name, organization name, etc.), relations (such as the director of the movie, the release time of the song, etc.), events (such as a car accident at a certain intersection, an earthquake in a certain place, etc.), and information such as product reviews, opinions, and sentiments. Users can use natural language to customize the extraction target, and can uniformly extract the corresponding information in the input text or document without training.

<a name="2"></a>

## 2. Examples

UIE does not limit industry fields and extraction targets. The following are some industry examples implemented out of the box by Taskflow:

- Medical scenarios - specialized disease structure

![image](https://user-images.githubusercontent.com/40840292/169017581-93c8ee44-856d-4d17-970c-b6138d10f8bc.png)

- Legal scene - Judgment extraction

![image](https://user-images.githubusercontent.com/40840292/169017863-442c50f1-bfd4-47d0-8d95-8b1d53cfba3c.png)

- Financial scenarios - proof of income, extraction of prospectus

![image](https://user-images.githubusercontent.com/40840292/169017982-e521ddf6-d233-41f3-974e-6f40f8f2edbc.png)

- Public security scene - accident report extraction

![image](https://user-images.githubusercontent.com/40840292/169018340-31efc1bf-f54d-43f7-b62a-8f7ce9bf0536.png)

- Tourism scene - brochure, manual extraction

![image](https://user-images.githubusercontent.com/40840292/169018113-c937eb0b-9fd7-4ecc-8615-bcdde2dac81d.png)

<a name="3"></a>

## 3. Text information extraction

<a name="31"></a>

#### 3.1 Entity Extraction

   Entity extraction, also known as Named Entity Recognition (NER for short), refers to identifying entities with specific meanings in text. In the open domain information extraction, the extracted categories are not limited, and users can define them by themselves.

   - For example, the extracted target entity types are "person" and "organization", and the schema defined as follows:

     ```text
     ['person', 'organization']
     ```

     Example:

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

#### 3.2 Relationship Extraction

   Relation Extraction refers to identifying entities from text and extracting the semantic relationship between entities, and then obtaining triple information, namely <subject, predicate, object>.

   - For example, if "person" is used as the extraction subject, and the extraction relationship types are "Company" and "Position", the schema structure is as follows:

     ```text
     {
       'Person': [
         'Company',
         'Position'
       ]
     }
     ```

     Example:

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

#### 3.3 Event extraction

   Event Extraction refers to extracting predefined event trigger words (Trigger) and event arguments (Argument) from natural language texts, and combining them into corresponding event structured information.

   - The English model** does not support event extraction**, if necessary, it can be customized using the English event dataset.

<a name="34"></a>

#### 3.4 Opinion Extraction

   Opinion extraction refers to the extraction of evaluation dimensions and opinion words contained in the text.

   - For example, the target of extraction is the evaluation dimension contained in the text and its corresponding opinion words and emotional tendencies. The schema structure is as follows:

     ```text
     {
       'Aspect': [
         'Opinion',
         'Sentiment classification [negative, positive]'
       ]
     }
     ```

     Example:

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

#### 3.5 Sentiment Classification

   - Sentence-level sentiment classification, that is, to judge whether the emotional orientation of a sentence is "positive" or "negative". The schema structure is as follows:

     ```text
     'Sentiment classification [negative, positive]'
     ```

     Example:

     ```python
     >>> schema = 'Sentiment classification [negative, positive]'
     >>> ie_en.set_schema(schema)
     >>> ie_en('I am sorry but this is the worst film I have ever seen in my life.')
     [{'Sentiment classification [negative, positive]': [{'text': 'negative', 'probability': 0.9998415771287057}]}]
     ```

#### 3.6 Multi-Task Extraction

   - For example, in the legal scene, entity extraction and relation extraction are performed on the text at the same time, and the schema can be constructed as follows:

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

    Example:

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

#### 3.7 Available Model

- A variety of models to different accuracy and speed requirements

   | Model | Structure | Language |
   | :---: | :--------: | :--------: |
   | `uie-base` (default)| 12-layers, 768-hidden, 12-heads | Chinese |
   | `uie-base-en` | 12-layers, 768-hidden, 12-heads | English |
   | `uie-medical-base` | 12-layers, 768-hidden, 12-heads | Chinese |
   | `uie-medium`| 6-layers, 768-hidden, 12-heads | Chinese |
   | `uie-mini`| 6-layers, 384-hidden, 12-heads | Chinese |
   | `uie-micro`| 4-layers, 384-hidden, 12-heads | Chinese |
   | `uie-nano`| 4-layers, 312-hidden, 12-heads | Chinese |
   | `uie-m-large`| 24-layers, 1024-hidden, 16-heads | Chinese and English |
   | `uie-m-base`| 12-layers, 768-hidden, 12-heads | Chinese and English |


- `uie-nano` call example:

  ```python
  >>> from paddlenlp import Taskflow

  >>> schema = ['时间', '选手', '赛事名称']
  >>> ie = Taskflow('information_extraction', schema=schema, model="uie-nano")
  >>> ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")
  [{'时间': [{'text': '2月8日上午', 'start': 0, 'end': 6, 'probability': 0.6513581678349247}], '选手': [{'text': '谷爱凌', 'start': 28, 'end': 31, 'probability': 0.9819330659468051}], '赛事名称': [{'text': '北京冬奥会自由式滑雪女子大跳台决赛', 'start': 6, 'end': 23, 'probability': 0.4908131110420939}]}]
  ```

- `uie-m-base` and `uie-m-large` support extraction of both Chinese and English, call example:

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

#### 3.8 More Configuration

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

* `schema`: Define the task extraction target, which can be configured by referring to the calling examples of different tasks in the out-of-the-box.
* `schema_lang`: Set the language of the schema, the default is `ch`, optional `ch` and `en`. Because the structure of the Chinese and English schemas is different, the language of the schema needs to be specified. This parameter is only valid for `uie-x-base`, `uie-m-base` and `uie-m-large` models.
* `batch_size`: batch size, please adjust according to the machine situation, the default is 16.
* `model`: select the model used by the task, the default is `uie-base`, optional `uie-base`, `uie-medium`, `uie-mini`, `uie-micro`, `uie-nano` and `uie-medical-base`, `uie-base-en`, `uie-x-base`.
* `position_prob`: The result probability of the model for the start position/end position of the span is between 0 and 1, and the returned result removes the results less than this threshold, the default is 0.5, and the final probability output of the span is the start position probability and end position The product of the position probabilities.
* `precision`: select the model precision, the default is `fp32`, optional `fp16` and `fp32`. `fp16` inference is faster, support GPU and NPU hardware. If you choose `fp16` and GPU hardware, please ensure that the machine is correctly installed with NVIDIA-related drivers and basic software. **Ensure that CUDA>=11.2, cuDNN>=8.1.1**. For the first time use, you need to follow the prompts to install the relevant dependencies. Secondly, it is necessary to ensure that the CUDA Compute Capability of the GPU device is greater than 7.0. Typical devices include V100, T4, A10, A100, GTX 20 series and 30 series graphics cards, etc. For more information about CUDA Compute Capability and precision support, please refer to NVIDIA documentation: [GPU Hardware and Supported Precision Comparison Table](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix).
