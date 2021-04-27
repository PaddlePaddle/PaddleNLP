# 解语 - 中文文本知识标注工具（wordtag）

中文文本知识标注工具是首个覆盖中文所有词类的知识标注工具，旨在提供全面、丰富的知识标注结果，可以应用于挖掘模板生成，数据挖掘、词类细化、新词发现、关系挖掘等应用中。

![image-20210427182133542](../doc/img/image-20210427182133542.png)

## 特性

- **覆盖中文全词类，更丰富的知识标注结果**
  - 中文文本知识标注工具使用的词类体系为覆盖中文全部词汇的词类体系，包括各类实体词与非实体词（如概念、专名、语法词等），考虑到一些类单使用文本无法细分（如人物类、作品类、品牌名等），词类体系中不做区分，一些可细分类别（如医院、学校等），划分了较细类别。
- **整合TermTree linking结果，获得更丰富的标注知识**
  - 如上图示例所示，各个切分标注结果中，除词类标注外，还整合了TermTree的linking结果，用户可以结合[TermTree](../termtree)数据共同使用，如利用TermTree中的subtype获得更细的上位粒度；利用TermTree中term的关系获得更加丰富的知识等；同时，TermTree数据是可插拔的，用户可将自定义的词表按照TermTree字段定义好，挂接到TermTree体系中，即可使用自己的Term数据展开下游任务。

## 应用示例

### 新词发现

直接使用wordtag模型的标注结果，可以发现各种类别的未收录词汇，如组织机构、人物、作品等。例如下面的标注结果：

```json
{
    "text": "贵州黔劲房地产开发有限公司董事长，纳雍瑞祥精神病专科医院有限责任公司监事、赫章德康精神病专科医院有限责任公司执行董事、金沙中科房地产开发有限公司执行董事兼总经理。",
    "items": [
        {
            "item": "贵州黔劲房地产开发有限公司",
            "offset": 0,
            "wordtag_label": "组织机构类_企事业单位",
            "length": 13
        },
        {
            "item": "董事长",
            "offset": 13,
            "wordtag_label": "人物类_概念",
            "length": 3,
            "termid": "人物_cb_董事长"
        },
        {
            "item": "，",
            "offset": 16,
            "wordtag_label": "w",
            "length": 1
        },
        {
            "item": "纳雍瑞祥精神病专科医院有限责任公司",
            "offset": 17,
            "wordtag_label": "组织机构类_企事业单位",
            "length": 17
        },
        {
            "item": "监事",
            "offset": 34,
            "wordtag_label": "人物类_概念",
            "length": 2,
            "termid": "人物_cb_监事"
        },
        {
            "item": "、",
            "offset": 36,
            "wordtag_label": "w",
            "length": 1
        },
        {
            "item": "赫章德康精神病专科医院有限责任公司",
            "offset": 37,
            "wordtag_label": "组织机构类_企事业单位",
            "length": 17
        },
        {
            "item": "执行董事",
            "offset": 54,
            "wordtag_label": "人物类_概念",
            "length": 4,
            "termid": "人物_cb_执行董事"
        },
        {
            "item": "、",
            "offset": 58,
            "wordtag_label": "w",
            "length": 1
        },
        {
            "item": "金沙中科房地产开发有限公司",
            "offset": 59,
            "wordtag_label": "组织机构类_企事业单位",
            "length": 13
        },
        {
            "item": "执行董事",
            "offset": 72,
            "wordtag_label": "人物类_概念",
            "length": 4,
            "termid": "人物_cb_执行董事"
        },
        {
            "item": "兼",
            "offset": 76,
            "wordtag_label": "连词",
            "length": 1,
            "termid": "连词_cb_兼"
        },
        {
            "item": "总经理",
            "offset": 77,
            "wordtag_label": "人物类_概念",
            "length": 3,
            "termid": "人物_cb_总经理"
        },
        {
            "item": "。",
            "offset": 80,
            "wordtag_label": "w",
            "length": 1
        }
    ]
}

```

### 挖掘模板生成：基于知识的关系抽取

使用文本知识标注的结果，可以获得更加全面、准确的挖掘模板，精准地从文本中挖掘到想要的东西，例如关系抽取任务中，可以利用标注结果生成关系挖掘pattern，挖掘文本中存在的关系三元组，例如：

**示例文本**：《孤女》是2010年九州出版社出版的小说，作者是余兼羽。

如想要从该文本中挖掘到“作者”关系，则可以如下操作：

1. 输入到wordtag模型中，得到的标注结果为：

   ```json
   {
       "text": "《孤女》是2010年九州出版社出版的小说，作者是余兼羽。",
       "items": [
           {
               "item": "《",
               "offset": 0,
               "wordtag_label": "w",
               "length": 1
           },
           {
               "item": "孤女",
               "offset": 1,
               "wordtag_label": "作品类_实体",
               "length": 2,
               "termid": "作品与出版物_eb_孤女"
           },
           {
               "item": "》",
               "offset": 3,
               "wordtag_label": "w",
               "length": 1
           },
           {
               "item": "是",
               "offset": 4,
               "wordtag_label": "肯定词",
               "length": 1,
               "termid": "肯定否定词_cb_是"
           },
           {
               "item": "2010年",
               "offset": 5,
               "wordtag_label": "时间类",
               "length": 5,
               "termid": "时间阶段_cb_2010年"
           },
           {
               "item": "九州出版社",
               "offset": 10,
               "wordtag_label": "组织机构类",
               "length": 5,
               "termid": "组织机构_eb_九州出版社"
           },
           {
               "item": "出版",
               "offset": 15,
               "wordtag_label": "场景事件",
               "length": 2,
               "termid": "场景事件_cb_出版"
           },
           {
               "item": "的",
               "offset": 17,
               "wordtag_label": "助词",
               "length": 1,
               "termid": "助词_cb_的"
           },
           {
               "item": "小说",
               "offset": 18,
               "wordtag_label": "作品类_概念",
               "length": 2,
               "termid": "小说_cb_小说"
           },
           {
               "item": "，",
               "offset": 20,
               "wordtag_label": "w",
               "length": 1
           },
           {
               "item": "作者",
               "offset": 21,
               "wordtag_label": "人物类_概念",
               "length": 2,
               "termid": "人物_cb_作者"
           },
           {
               "item": "是",
               "offset": 23,
               "wordtag_label": "肯定词",
               "length": 1,
               "termid": "肯定否定词_cb_是"
           },
           {
               "item": "余兼羽",
               "offset": 24,
               "wordtag_label": "人物类_实体",
               "length": 3
           },
           {
               "item": "。",
               "offset": 27,
               "wordtag_label": "w",
               "length": 1
           }
       ]
   }
   
   ```

2. 根据上述标注结果，排除掉与“作者”关系无关的类别（时间类、组织机构类、助词、标点），可以得到词类挖掘pattern：

   ```
   [W][作品类_实体][W][作者][肯定词][人物类_实体]
   ```

   其中`W`代表任意长度的通配符，使用这个词类挖掘pattern去匹配相应的文本标注结果，如匹配成功，则可从对应的词槽位中得到相应的三元组关系，下面举例这个pattern可以召回的文本：

   > 《 龙翔寰宇》是在飞库网连载的一部小说，作者是祥子。
   >
   > 《世界最动人情书精选50封》是2008年江西出版集团、江西人民出版社出版的图书，作者是刘植荣。
   >
   > 法兰西玫瑰的浪漫香颂是2009年山东美术出版社出版的图书，作者是（法）禾社德。
   >
   > ……

### 文本知识特征增强的关系抽取

关系抽取任务中，一些文本中可能不含有显式的关系，如文本“《赤裸特工》并非程小东最擅长的，但他一样做到最出色”中，只有实体“程小东“和”赤裸特工“，而没有明确提到二者存在什么关系，需要让模型得到更多的信息去完成关系推断，例如：

1. 将文本输入到知识标注模型中，得到如下结果：

   ```json
   {
       "text": "《赤裸特工》并非程小东最擅长的，但他一样做到最出色",
       "items": [
           {
               "item": "《",
               "offset": 0,
               "wordtag_label": "w",
               "length": 1
           },
           {
               "item": "赤裸特工",
               "offset": 1,
               "wordtag_label": "作品类_实体",
               "length": 4,
               "termid": "影视作品_eb_赤裸特工"
           },
           {
               "item": "》",
               "offset": 5,
               "wordtag_label": "w",
               "length": 1
           },
           {
               "item": "并非",
               "offset": 6,
               "wordtag_label": "否定词",
               "length": 2,
               "termid": "肯定否定词_cb_并不是"
           },
           {
               "item": "程小东",
               "offset": 8,
               "wordtag_label": "人物类_实体",
               "length": 3,
               "termid": "人物_eb_程小东"
           },
           {
               "item": "最",
               "offset": 11,
               "wordtag_label": "副词",
               "length": 1,
               "termid": "副词_cb_最"
           },
           {
               "item": "擅长",
               "offset": 12,
               "wordtag_label": "场景事件",
               "length": 2,
               "termid": "场景事件_cb_擅长"
           },
           {
               "item": "的",
               "offset": 14,
               "wordtag_label": "助词",
               "length": 1,
               "termid": "助词_cb_的"
           },
           {
               "item": "，",
               "offset": 15,
               "wordtag_label": "w",
               "length": 1
           },
           {
               "item": "但",
               "offset": 16,
               "wordtag_label": "连词",
               "length": 1,
               "termid": "连词_cb_但是"
           },
           {
               "item": "他",
               "offset": 17,
               "wordtag_label": "代词",
               "length": 1,
               "termid": "代词_cb_他"
           },
           {
               "item": "一样",
               "offset": 18,
               "wordtag_label": "场景事件",
               "length": 2,
               "termid": "场景事件_cb_一样"
           },
           {
               "item": "做到",
               "offset": 20,
               "wordtag_label": "场景事件",
               "length": 2,
               "termid": "场景事件_cb_做到"
           },
           {
               "item": "最",
               "offset": 22,
               "wordtag_label": "副词",
               "length": 1,
               "termid": "副词_cb_最"
           },
           {
               "item": "出色",
               "offset": 23,
               "wordtag_label": "修饰词",
               "length": 2,
               "termid": "修饰词_cb_出色"
           }
       ]
   }
   ```

   2. 根据标注结果，得到两个实体`人物_eb_程小东`和`影视作品_eb_赤裸特工`，根据termtree数据，可以查询到二者的subtype分别是“动作指导”和“动作电影”
   3. 将文本、实体和subtype共同作为模型输入，组织训练数据，则可以训练得到知识增强的关系抽取模型。



## 模型结构

模型使用ERNIE-CTM+CRF训练而成，预测时使用viterbi解码，模型结构如下：

<img src="../doc/img/image-20210427183901319.png" alt="image-20210427183901319" style="zoom: 33%;" />

## Term Linking

wordtag模型对所有的词预测到上位词类之后，会直接根据预测到的词类，映射到term体系（映射表参见代码配置），查找相应的term，查找到相应的term后，会直接link到term上，如用户需要自己定制TermTree，只需将term挂接好之后更换数据即可。

## 后期计划

1. 持续优化知识标注模型，获得更加精准的标注结果
2. 发布多粒度、多种参数规模的知识标注模型
3. 根据TermPath及TermCorpus完成细粒度term及subterm消歧

## 在论文中引用wordtag
如果您的工作成果中使用了TermTree，请增加下述引用。我们非常乐于看到TermTree对您的工作带来帮助。
```
@article{zhao2020TermTree,
	title={TermTree and Knowledge Annotation Framework for Chinese Language Understanding},
	author={Zhao, Min and Qin, Huapeng and Zhang, Guoxin and Lyu, Yajuan and Zhu, Yong},
    technical report={Baidu, Inc. TR:2020-KG-TermTree},
    year={2020}
}
```
