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
| [词性标注](#词性标注)              | `Taskflow("pos_tagging")`        | ✅        | ✅        | ✅        | ✅          | ✅          | 基于百度前沿词法分析工具LAC                            |
| [命名实体识别](#命名实体识别)      | `Taskflow("ner")`                | ✅        | ✅        | ✅        | ✅          | ✅          | 覆盖最全中文实体标签                                   |
| [依存句法分析](#依存句法分析)      | `Taskflow("dependency_parsing")` | ✅        | ✅        | ✅        |            | ✅          | 基于最大规模中文依存句法树库研发的DDParser             |
| [『解语』-知识标注](#解语知识标注) | `Taskflow("knowledge_mining")`   | ✅        | ✅        | ✅        | ✅          | ✅          | 覆盖所有中文词汇的知识标注工具                         |
| [文本纠错](#文本纠错)              | `Taskflow("text_correction")`    | ✅        | ✅        | ✅        | ✅          | ✅          | 融合拼音特征的端到端文本纠错模型ERNIE-CSC              |
| [文本相似度](#文本相似度)          | `Taskflow("text_similarity")`    | ✅        | ✅        | ✅        |            |            | 基于百度知道2200万对相似句组训练                       |
| [情感倾向分析](#情感倾向分析)      | `Taskflow("sentiment_analysis")` | ✅        | ✅        | ✅        |            | ✅          | 基于情感知识增强预训练模型SKEP达到业界SOTA             |
| [生成式问答](#生成式问答)          | `Taskflow("question_answering")` | ✅        | ✅        | ✅        |            |            | 使用最大中文开源CPM模型完成问答                        |
| [智能写诗](#智能写诗)              | `Taskflow("poetry_generation")`  | ✅        | ✅        | ✅        |            |            | 使用最大中文开源CPM模型完成写诗                        |
| [开放域对话](#开放域对话)          | `Taskflow("dialogue")`           | ✅        | ✅        | ✅        |            |            | 十亿级语料训练最强中文闲聊模型PLATO-Mini，支持多轮对话 |


## QuickStart

**环境依赖**
  - python >= 3.6
  - paddlepaddle >= 2.2.0
  - paddlenlp >= 2.2.5

![taskflow1](https://user-images.githubusercontent.com/11793384/159693816-fda35221-9751-43bb-b05c-7fc77571dd76.gif)

可进入 Jupyter Notebook 环境，在线体验 👉🏻  [进入在线运行环境](https://aistudio.baidu.com/aistudio/projectdetail/3696243)

PaddleNLP Taskflow API 支持任务持续丰富中，我们将根据开发者反馈，灵活调整功能建设优先级，可通过Issue或[问卷](https://iwenjuan.baidu.com/?code=44amg8)反馈给我们。

## 社区交流

微信扫描下方二维码加入官方交流群，与各行各业开发者充分交流，期待你的加入⬇️

<div align="center">
  <img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/release/2.2/docs/imgs/wechat.png" width="188" height="188" />
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

### 词性标注

<details><summary>&emsp;基于百度词法分析工具LAC</summary><div>

#### 支持单条和批量预测
```python
>>> from paddlenlp import Taskflow
# 单条预测
>>> tag = Taskflow("pos_tagging")
>>> tag("第十四届全运会在西安举办")
[('第十四届', 'm'), ('全运会', 'nz'), ('在', 'p'), ('西安', 'LOC'), ('举办', 'v')]

# 批量样本输入，平均速度更快
>>> tag(["第十四届全运会在西安举办", "三亚是一个美丽的城市"])
[[('第十四届', 'm'), ('全运会', 'nz'), ('在', 'p'), ('西安', 'LOC'), ('举办', 'v')], [('三亚', 'LOC'), ('是', 'v'), ('一个', 'm'), ('美丽', 'a'), ('的', 'u'), ('城市', 'n')]]
```

#### 标签集合

| 标签 | 含义     | 标签 | 含义     | 标签 | 含义     | 标签 | 含义     |
| ---- | -------- | ---- | -------- | ---- | -------- | ---- | -------- |
| n    | 普通名词 | f    | 方位名词 | s    | 处所名词 | t    | 时间     |
| nr   | 人名     | ns   | 地名     | nt   | 机构名   | nw   | 作品名   |
| nz   | 其他专名 | v    | 普通动词 | vd   | 动副词   | vn   | 名动词   |
| a    | 形容词   | ad   | 副形词   | an   | 名形词   | d    | 副词     |
| m    | 数量词   | q    | 量词     | r    | 代词     | p    | 介词     |
| c    | 连词     | u    | 助词     | xc   | 其他虚词 | w    | 标点符号 |
| PER  | 人名     | LOC  | 地名     | ORG  | 机构名   | TIME | 时间     |

#### 自定义词典

你可以通过装载自定义词典来定制化分词和词性标注结果。词典文件每一行表示一个自定义item，可以由一个单词或者多个单词组成，单词后面可以添加自定义标签，格式为`item/tag`，如果不添加自定义标签，则使用模型默认标签`n`。

词典文件`user_dict.txt`示例：

```text
赛里木湖/LAKE
高/a 山/n
海拔最高
```

装载自定义词典及输出结果示例：

```python
>>> from paddlenlp import Taskflow
>>> tag = Taskflow("pos_tagging")
>>> tag("赛里木湖是新疆海拔最高的高山湖泊")
[('赛里木湖', 'LOC'), ('是', 'v'), ('新疆', 'LOC'), ('海拔', 'n'), ('最高', 'a'), ('的', 'u'), ('高山', 'n'), ('湖泊', 'n')]
>>> my_tag = Taskflow("pos_tagging", user_dict="user_dict.txt")
>>> my_tag("赛里木湖是新疆海拔最高的高山湖泊")
[('赛里木湖', 'LAKE'), ('是', 'v'), ('新疆', 'LOC'), ('海拔最高', 'n'), ('的', 'u'), ('高', 'a'), ('山', 'n'), ('湖泊', 'n')]
```
#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `user_dict`：用户自定义词典文件，默认为None。
* `task_path`：自定义任务路径，默认为None。
</div></details>

### 命名实体识别

<details><summary>&emsp;最全中文实体标签</summary><div>

#### 支持两种模式

```python
# 精确模式（默认），基于百度解语，内置66种词性及专名类别标签
>>> from paddlenlp import Taskflow
>>> ner = Taskflow("ner")
>>> ner("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
[('《', 'w'), ('孤女', '作品类_实体'), ('》', 'w'), ('是', '肯定词'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('的', '助词'), ('小说', '作品类_概念'), ('，', 'w'), ('作者', '人物类_概念'), ('是', '肯定词'), ('余兼羽', '人物类_实体')]

>>> ner = Taskflow("ner", entity_only=True)  # 只返回实体/概念词
>>> ner("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
[('孤女', '作品类_实体'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('小说', '作品类_概念'), ('作者', '人物类_概念'), ('余兼羽', '人物类_实体')]

# 快速模式，基于百度LAC，内置24种词性和专名类别标签
>>> from paddlenlp import Taskflow
>>> ner = Taskflow("ner", mode="fast")
>>> ner("三亚是一个美丽的城市")
[('三亚', 'LOC'), ('是', 'v'), ('一个', 'm'), ('美丽', 'a'), ('的', 'u'), ('城市', 'n')]
```

#### 批量样本输入，平均速度更快
```python
>>> from paddlenlp import Taskflow
>>> ner = Taskflow("ner")
>>> ner(["热梅茶是一道以梅子为主要原料制作的茶饮", "《孤女》是2010年九州出版社出版的小说，作者是余兼羽"])
[[('热梅茶', '饮食类_饮品'), ('是', '肯定词'), ('一道', '数量词'), ('以', '介词'), ('梅子', '饮食类'), ('为', '肯定词'), ('主要原料', '物体类'), ('制作', '场景事件'), ('的', '助词'), ('茶饮', '饮食类_饮品')], [('《', 'w'), ('孤女', '作品类_实体'), ('》', 'w'), ('是', '肯定词'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('的', '助词'), ('小说', '作品类_概念'), ('，', 'w'), ('作者', '人物类_概念'), ('是', '肯定词'), ('余兼羽', '人物类_实体')]]
```

#### 实体标签说明

- 精确模式采用的标签集合

包含66种词性及专名类别标签，标签集合如下表：

<table>

<tr><th colspan='6'>WordTag标签集合
<tr><td>人物类_实体<td>物体类<td>生物类_动物<td>医学术语类<td>链接地址<td>肯定词
<tr><td>人物类_概念<td>物体类_兵器<td>品牌名<td>术语类_生物体<td>个性特征<td>否定词
<tr><td>作品类_实体<td>物体类_化学物质<td>场所类<td>疾病损伤类<td>感官特征<td>数量词
<tr><td>作品类_概念<td>其他角色类<td>场所类_交通场所<td>疾病损伤类_植物病虫害<td>场景事件<td>叹词
<tr><td>组织机构类<td>文化类<td>位置方位<td>宇宙类<td>介词<td>拟声词
<tr><td>组织机构类_企事业单位<td>文化类_语言文字<td>世界地区类<td>事件类<td>介词_方位介词<td>修饰词
<tr><td>组织机构类_医疗卫生机构<td>文化类_奖项赛事活动<td>饮食类<td>时间类<td>助词<td>外语单词
<tr><td>组织机构类_国家机关<td>文化类_制度政策协议<td>饮食类_菜品<td>时间类_特殊日<td>代词<td>英语单词
<tr><td>组织机构类_体育组织机构<td>文化类_姓氏与人名<td>饮食类_饮品<td>术语类<td>连词<td>汉语拼音
<tr><td>组织机构类_教育组织机构<td>生物类<td>药物类<td>术语类_符号指标类<td>副词<td>词汇用语
<tr><td>组织机构类_军事组织机构<td>生物类_植物<td>药物类_中药<td>信息资料<td>疑问词<td>w(标点)

</table>

- 快速模式采用的标签集合

| 标签 | 含义     | 标签 | 含义     | 标签 | 含义     | 标签 | 含义     |
| ---- | -------- | ---- | -------- | ---- | -------- | ---- | -------- |
| n    | 普通名词 | f    | 方位名词 | s    | 处所名词 | t    | 时间     |
| nr   | 人名     | ns   | 地名     | nt   | 机构名   | nw   | 作品名   |
| nz   | 其他专名 | v    | 普通动词 | vd   | 动副词   | vn   | 名动词   |
| a    | 形容词   | ad   | 副形词   | an   | 名形词   | d    | 副词     |
| m    | 数量词   | q    | 量词     | r    | 代词     | p    | 介词     |
| c    | 连词     | u    | 助词     | xc   | 其他虚词 | w    | 标点符号 |
| PER  | 人名     | LOC  | 地名     | ORG  | 机构名   | TIME | 时间     |

#### 自定义词典

你可以通过装载自定义词典来定制化命名实体识别结果。词典文件每一行表示一个自定义item，可以由一个term或者多个term组成，term后面可以添加自定义标签，格式为`item/tag`，如果不添加自定义标签，则使用模型默认标签。

词典文件`user_dict.txt`示例：

```text
长津湖/电影类_实体
收/词汇用语 尾/术语类
最 大
海外票仓
```

以"《长津湖》收尾，北美是最大海外票仓"为例，原本的输出结果为：

```text
[('《', 'w'), ('长津湖', '作品类_实体'), ('》', 'w'), ('收尾', '场景事件'), ('，', 'w'), ('北美', '世界地区类'), ('是', '肯定词'), ('最大', '修饰词'), ('海外', '场所类'), ('票仓', '词汇用语')]
```

装载自定义词典及输出结果示例：

```python
>>> from paddlenlp import Taskflow

>>> my_ner = Taskflow("ner", user_dict="user_dict.txt")
>>> my_ner("《长津湖》收尾，北美是最大海外票仓")
[('《', 'w'), ('长津湖', '电影类_实体'), ('》', 'w'), ('收', '词汇用语'), ('尾', '术语类'), ('，', 'w'), ('北美', '世界地区类'), ('是', '肯定词'), ('最', '修饰词'), ('大', '修饰词'), ('海外票仓', '场所类')]
```

#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `user_dict`：用户自定义词典文件，默认为None。
* `task_path`：自定义任务路径，默认为None。
* `entity_only`：只返回实体/概念词及其对应标签。
</div></details>


### 依存句法分析
<details><summary>&emsp;基于最大规模中文依存句法树库研发的DDParser </summary><div>

#### 支持多种形式输入

未分词输入:

```python
>>> from paddlenlp import Taskflow
>>> ddp = Taskflow("dependency_parsing")
>>> ddp("2月8日谷爱凌夺得北京冬奥会第三金")
[{'word': ['2月8日', '谷爱凌', '夺得', '北京冬奥会', '第三金'], 'head': [3, 3, 0, 5, 3], 'deprel': ['ADV', 'SBV', 'HED', 'ATT', 'VOB']}]

```

使用分词结果来输入:

```python
>>> ddp = Taskflow("dependency_parsing")
>>> ddp.from_segments([['2月8日', '谷爱凌', '夺得', '北京冬奥会', '第三金']])
[{'word': ['2月8日', '谷爱凌', '夺得', '北京冬奥会', '第三金'], 'head': [3, 3, 0, 5, 3], 'deprel': ['ADV', 'SBV', 'HED', 'ATT', 'VOB']}]
```

#### 批量样本输入，平均速度更快

```python
>>> from paddlenlp import Taskflow
>>> ddp(["2月8日谷爱凌夺得北京冬奥会第三金", "他送了一本书"])
[{'word': ['2月8日', '谷爱凌', '夺得', '北京冬奥会', '第三金'], 'head': [3, 3, 0, 5, 3], 'deprel': ['ADV', 'SBV', 'HED', 'ATT', 'VOB']}, {'word': ['他', '送', '了', '一本', '书'], 'head': [2, 0, 2, 5, 2], 'deprel': ['SBV', 'HED', 'MT', 'ATT', 'VOB']}]
```

#### 多种模型选择，满足精度、速度需求

使用ERNIE 1.0进行预测

```python
>>> ddp = Taskflow("dependency_parsing", model="ddparser-ernie-1.0")
>>> ddp("2月8日谷爱凌夺得北京冬奥会第三金")
[{'word': ['2月8日', '谷爱凌', '夺得', '北京冬奥会', '第三金'], 'head': [3, 3, 0, 5, 3], 'deprel': ['ADV', 'SBV', 'HED', 'ATT', 'VOB']}]
```

除ERNIE 1.0外，还可使用ERNIE-Gram预训练模型，其中`model=ddparser`（基于LSTM Encoder）速度最快，`model=ddparser-ernie-gram-zh`和`model=ddparser-ernie-1.0`效果更优（两者效果相当）。

#### 输出方式

输出概率值和词性标签:

```python
>>> ddp = Taskflow("dependency_parsing", prob=True, use_pos=True)
>>> ddp("2月8日谷爱凌夺得北京冬奥会第三金")
[{'word': ['2月8日', '谷爱凌', '夺得', '北京冬奥会', '第三金'], 'head': [3, 3, 0, 5, 3], 'deprel': ['ADV', 'SBV', 'HED', 'ATT', 'VOB'], 'postag': ['TIME', 'PER', 'v', 'ORG', 'n'], 'prob': [0.97, 1.0, 1.0, 0.99, 0.99]}]
```

依存关系可视化

```python
>>> from paddlenlp import Taskflow
>>> ddp = Taskflow("dependency_parsing", return_visual=True)
>>> result = ddp("2月8日谷爱凌夺得北京冬奥会第三金")[0]['visual']
>>> import cv2
>>> cv2.imwrite('test.png', result)
```

<p align="center">
 <img src="https://user-images.githubusercontent.com/11793384/159904566-40f42e19-d3ef-45e7-b798-ae7ad954fca5.png" align="middle">
<p align="center">

#### 依存句法分析标注关系集合

| Label |  关系类型  | 说明                     | 示例                           |
| :---: | :--------: | :----------------------- | :----------------------------- |
|  SBV  |  主谓关系  | 主语与谓词间的关系       | 他送了一本书(他<--送)          |
|  VOB  |  动宾关系  | 宾语与谓词间的关系       | 他送了一本书(送-->书)          |
|  POB  |  介宾关系  | 介词与宾语间的关系       | 我把书卖了（把-->书）          |
|  ADV  |  状中关系  | 状语与中心词间的关系     | 我昨天买书了（昨天<--买）      |
|  CMP  |  动补关系  | 补语与中心词间的关系     | 我都吃完了（吃-->完）          |
|  ATT  |  定中关系  | 定语与中心词间的关系     | 他送了一本书(一本<--书)        |
|   F   |  方位关系  | 方位词与中心词的关系     | 在公园里玩耍(公园-->里)        |
|  COO  |  并列关系  | 同类型词语间关系         | 叔叔阿姨(叔叔-->阿姨)          |
|  DBL  |  兼语结构  | 主谓短语做宾语的结构     | 他请我吃饭(请-->我，请-->吃饭) |
|  DOB  | 双宾语结构 | 谓语后出现两个宾语       | 他送我一本书(送-->我，送-->书) |
|  VV   |  连谓结构  | 同主语的多个谓词间关系   | 他外出吃饭(外出-->吃饭)        |
|  IC   |  子句结构  | 两个结构独立或关联的单句 | 你好，书店怎么走？(你好<--走)  |
|  MT   |  虚词成分  | 虚词与中心词间的关系     | 他送了一本书(送-->了)          |
|  HED  |  核心关系  | 指整个句子的核心         |                                |

#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `model`：选择任务使用的模型，可选有`ddparser`，`ddparser-ernie-1.0`和`ddparser-ernie-gram-zh`。
* `tree`：确保输出结果是正确的依存句法树，默认为True。
* `prob`：是否输出每个弧对应的概率值，默认为False。
* `use_pos`：是否返回词性标签，默认为False。
* `use_cuda`：是否使用GPU进行切词，默认为False。
* `return_visual`：是否返回句法树的可视化结果，默认为False。
* `task_path`：自定义任务路径，默认为None。
</div></details>

### 解语知识标注
<details><summary>&emsp;覆盖所有中文词汇的知识标注工具</summary><div>

#### 词类知识标注

```python
>>> from paddlenlp import Taskflow
>>> wordtag = Taskflow("knowledge_mining")
>>> wordtag("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
[{'text': '《孤女》是2010年九州出版社出版的小说，作者是余兼羽', 'items': [{'item': '《', 'offset': 0, 'wordtag_label': 'w', 'length': 1}, {'item': '孤女', 'offset': 1, 'wordtag_label': '作品类_实体', 'length': 2}, {'item': '》', 'offset': 3, 'wordtag_label': 'w', 'length': 1}, {'item': '是', 'offset': 4, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '2010年', 'offset': 5, 'wordtag_label': '时间类', 'length': 5, 'termid': '时间阶段_cb_2010年'}, {'item': '九州出版社', 'offset': 10, 'wordtag_label': '组织机构类', 'length': 5, 'termid': '组织机构_eb_九州出版社'}, {'item': '出版', 'offset': 15, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_出版'}, {'item': '的', 'offset': 17, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '小说', 'offset': 18, 'wordtag_label': '作品类_概念', 'length': 2, 'termid': '小说_cb_小说'}, {'item': '，', 'offset': 20, 'wordtag_label': 'w', 'length': 1}, {'item': '作者', 'offset': 21, 'wordtag_label': '人物类_概念', 'length': 2, 'termid': '人物_cb_作者'}, {'item': '是', 'offset': 23, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '余兼羽', 'offset': 24, 'wordtag_label': '人物类_实体', 'length': 3}]}]
```

**可配置参数说明：**
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `linking`：实现基于词类的linking，默认为True。
* `task_path`：自定义任务路径，默认为None。
* `user_dict`：用户自定义词典文件，默认为None。


知识挖掘-词类知识标注任务共包含66种词性及专名类别标签，标签集合如下表：

<table>

<tr><th colspan='6'>WordTag标签集合
<tr><td>人物类_实体<td>物体类<td>生物类_动物<td>医学术语类<td>链接地址<td>肯定词
<tr><td>人物类_概念<td>物体类_兵器<td>品牌名<td>术语类_生物体<td>个性特征<td>否定词
<tr><td>作品类_实体<td>物体类_化学物质<td>场所类<td>疾病损伤类<td>感官特征<td>数量词
<tr><td>作品类_概念<td>其他角色类<td>场所类_交通场所<td>疾病损伤类_植物病虫害<td>场景事件<td>叹词
<tr><td>组织机构类<td>文化类<td>位置方位<td>宇宙类<td>介词<td>拟声词
<tr><td>组织机构类_企事业单位<td>文化类_语言文字<td>世界地区类<td>事件类<td>介词_方位介词<td>修饰词
<tr><td>组织机构类_医疗卫生机构<td>文化类_奖项赛事活动<td>饮食类<td>时间类<td>助词<td>外语单词
<tr><td>组织机构类_国家机关<td>文化类_制度政策协议<td>饮食类_菜品<td>时间类_特殊日<td>代词<td>英语单词
<tr><td>组织机构类_体育组织机构<td>文化类_姓氏与人名<td>饮食类_饮品<td>术语类<td>连词<td>汉语拼音
<tr><td>组织机构类_教育组织机构<td>生物类<td>药物类<td>术语类_符号指标类<td>副词<td>词汇用语
<tr><td>组织机构类_军事组织机构<td>生物类_植物<td>药物类_中药<td>信息资料<td>疑问词<td>w(标点)

</table>


#### 名词短语标注
```python
>>> from paddlenlp import Taskflow
>>> nptag = Taskflow("knowledge_mining", model="nptag")
>>> nptag("糖醋排骨")
[{'text': '糖醋排骨', 'label': '菜品'}]

>>> nptag(["糖醋排骨", "红曲霉菌"])
[{'text': '糖醋排骨', 'label': '菜品'}, {'text': '红曲霉菌', 'label': '微生物'}]

# 使用`linking`输出粗粒度类别标签`category`，即WordTag的词汇标签。
>>> nptag = Taskflow("knowledge_mining", model="nptag", linking=True)
>>> nptag(["糖醋排骨", "红曲霉菌"])
[{'text': '糖醋排骨', 'label': '菜品', 'category': '饮食类_菜品'}, {'text': '红曲霉菌', 'label': '微生物', 'category': '生物类_微生物'}]
```
**可配置参数说明：**
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `max_seq_len`：最大序列长度，默认为64。
* `linking`：实现与WordTag类别标签的linking，默认为False。
* `task_path`：自定义任务路径，默认为None。


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
<details><summary>&emsp;基于百度知道2200万对相似句组训练SimBERT达到前沿文本相似效果</summary><div>

#### 单条输入

```python
>>> from paddlenlp import Taskflow
>>> similarity = Taskflow("text_similarity")
>>> similarity([["春天适合种什么花？", "春天适合种什么菜？"]])
[{'text1': '春天适合种什么花？', 'text2': '春天适合种什么菜？', 'similarity': 0.8340253}]
```

#### 批量样本输入，平均速度更快

```python
>>> from paddlenlp import Taskflow
>>> similarity([["光眼睛大就好看吗", "眼睛好看吗？"], ["小蝌蚪找妈妈怎么样", "小蝌蚪找妈妈是谁画的"]])
[{'text1': '光眼睛大就好看吗', 'text2': '眼睛好看吗？', 'similarity': 0.74502707}, {'text1': '小蝌蚪找妈妈怎么样', 'text2': '小蝌蚪找妈妈是谁画的', 'similarity': 0.8192149}]
```

#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `max_seq_len`：最大序列长度，默认为128。
* `task_path`：自定义任务路径，默认为None。
</div></details>

### 情感倾向分析
<details><summary>&emsp;基于情感知识增强预训练模型SKEP达到业界SOTA </summary><div>

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
```

#### 批量样本输入，平均速度更快
```python
>>> from paddlenlp import Taskflow
>>> senta(["这个产品用起来真的很流畅，我非常喜欢", "作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间"])
[{'text': '这个产品用起来真的很流畅，我非常喜欢', 'label': 'positive', 'score': 0.9938690066337585}, {'text': '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间', 'label': 'positive', 'score': 0.985750675201416}]
```

#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `model`：选择任务使用的模型，可选有`bilstm`和`skep_ernie_1.0_large_ch`。
* `task_path`：自定义任务路径，默认为None。
</div></details>

### 生成式问答
<details><summary>&emsp; 使用最大中文开源CPM模型完成问答</summary><div>

#### 支持单条、批量预测

```python
>>> from paddlenlp import Taskflow
>>> qa = Taskflow("question_answering")
# 单条输入
>>> qa("中国的国土面积有多大？")
[{'text': '中国的国土面积有多大？', 'answer': '960万平方公里。'}]
# 多条输入
>>> qa(["中国国土面积有多大？", "中国的首都在哪里？"])
[{'text': '中国国土面积有多大？', 'answer': '960万平方公里。'}, {'text': '中国的首都在哪里？', 'answer': '北京。'}]
```

#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
</div></details>

### 智能写诗
<details><summary>&emsp; 使用最大中文开源CPM模型完成写诗 </summary><div>

#### 支持单条、批量预测

```python
>>> from paddlenlp import Taskflow
>>> poetry = Taskflow("poetry_generation")
# 单条输入
>>> poetry("林密不见人")
[{'text': '林密不见人', 'answer': ',但闻人语响。'}]
# 多条输入
>>> poetry(["林密不见人", "举头邀明月"])
[{'text': '林密不见人', 'answer': ',但闻人语响。'}, {'text': '举头邀明月', 'answer': ',低头思故乡。'}]
```

#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
</div></details>

### 开放域对话
<details><summary>&emsp;十亿级语料训练最强中文闲聊模型PLATO-Mini，支持多轮对话</summary><div>

#### 非交互模式
```python
>>> from paddlenlp import Taskflow
>>> dialogue = Taskflow("dialogue")
>>> dialogue(["吃饭了吗"])
['刚吃完饭,你在干什么呢?']

>>> dialogue(["你好", "吃饭了吗"], ["你是谁？"])
['吃过了,你呢', '我是李明啊']
```

可配置参数：

* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `max_seq_len`：最大序列长度，默认为512。

#### 交互模式
```python
>>> from paddlenlp import Taskflow

>>> dialogue = Taskflow("dialogue")
# 输入`exit`可退出交互模式
>>> dialogue.interactive_mode(max_turn=3)

'''
[Human]:你好
[Bot]:你好,很高兴认识你,我想问你一下,你喜欢运动吗?
[Human]:喜欢
[Bot]:那你喜欢什么运动啊?
[Human]:篮球,你喜欢篮球吗
[Bot]:当然了,我很喜欢打篮球的
'''
```

交互模式参数：
* `max_turn`：任务能记忆的对话轮数，当max_turn为1时，模型只能记住当前对话，无法获知之前的对话内容。
  </div></details>


## PART Ⅱ &emsp; 定制化训练

<details><summary>适配任务列表</summary><div>

如果你有自己的业务数据集，可以对模型效果进一步调优，支持定制化训练的任务如下：

|                           任务名称                           |                           默认路径                           |                                                              |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|         `Taskflow("word_segmentation", mode="base")`         |             `$HOME/.paddlenlp/taskflow/lac`                  | [示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/lexical_analysis) |
|       `Taskflow("word_segmentation", mode="accurate")`       |             `$HOME/.paddlenlp/taskflow/wordtag`              | [示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/ernie-ctm) |
|       `Taskflow("pos_tagging")`                              |             `$HOME/.paddlenlp/taskflow/lac`                  | [示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/lexical_analysis) |
|                `Taskflow("ner", mode="fast")`                |             `$HOME/.paddlenlp/taskflow/lac`                  | [示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/lexical_analysis) |
|              `Taskflow("ner", mode="accurate")`              |             `$HOME/.paddlenlp/taskflow/wordtag`              | [示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/ernie-ctm) |
|     `Taskflow("text_correction", model="ernie-csc")`     |  `$HOME/.paddlenlp/taskflow/text_correction/ernie-csc`   | [示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_correction/ernie-csc) |
|      `Taskflow("dependency_parsing", model="ddparser")`      |   `$HOME/.paddlenlp/taskflow/dependency_parsing/ddparser`    | [示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/dependency_parsing/ddparser) |
| `Taskflow("dependency_parsing", model="ddparser-ernie-1.0")` | `$HOME/.paddlenlp/taskflow/dependency_parsing/ddparser-ernie-1.0` | [示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/dependency_parsing/ddparser) |
| `Taskflow("dependency_parsing", model="ddparser-ernie-gram-zh")` | `$HOME/.paddlenlp/taskflow/dependency_parsing/ddparser-ernie-gram-zh` | [示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/dependency_parsing/ddparser) |
| `Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch")` | `$HOME/.paddlenlp/taskflow/sentiment_analysis/skep_ernie_1.0_large_ch` | [示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/sentiment_analysis/skep) |
|       `Taskflow("knowledge_mining", model="wordtag")`        |             `$HOME/.paddlenlp/taskflow/wordtag`              | [示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/ernie-ctm) |
|        `Taskflow("knowledge_mining", model="nptag")`         |      `$HOME/.paddlenlp/taskflow/knowledge_mining/nptag`      | [示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/nptag) |

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
  <tr><td>词性标注<td>BiGRU+CRF<td> <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/lexical_analysis"> 训练详情 <td> 百度自建数据集，包含2200万句子，覆盖多种场景
  <tr><td rowspan="2">命名实体识别<td>精确模式：WordTag<td> <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/ernie-ctm"> 训练详情 <td> 百度自建数据集，词类体系基于TermTree构建
  <tr><td>快速模式：BiGRU+CRF <td> <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/lexical_analysis"> 训练详情 <td> 百度自建数据集，包含2200万句子，覆盖多种场景
  <tr><td>依存句法分析<td>DDParser<td> <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/dependency_parsing/ddparser"> 训练详情 <td> 百度自建数据集，DuCTB 1.0中文依存句法树库
  <tr><td rowspan="2">解语知识标注<td>词类知识标注：WordTag<td> <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/ernie-ctm"> 训练详情 <td> 百度自建数据集，词类体系基于TermTree构建
  <tr><td>名词短语标注：NPTag <td> <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/nptag"> 训练详情 <td> 百度自建数据集
  <tr><td>文本纠错<td>ERNIE-CSC<td> <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_correction/ernie-csc"> 训练详情 <td> SIGHAN简体版数据集及 <a href="https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml"> Automatic Corpus Generation生成的中文纠错数据集
  <tr><td>文本相似度<td>SimBERT<td> - <td> 收集百度知道2200万对相似句组
  <tr><td rowspan="2">情感倾向分析<td> BiLSTM <td> - <td> 百度自建数据集
  <tr><td> SKEP <td> <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/sentiment_analysis/skep"> 训练详情 <td> 百度自建数据集
  <tr><td>生成式问答<td>CPM<td> - <td> 100GB级别中文数据
  <tr><td>智能写诗<td>CPM<td> - <td> 100GB级别中文数据
  <tr><td>开放域对话<td>PLATO-Mini<td> - <td> 十亿级别中文对话数据
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
2. [ZhuiyiTechnology/simbert]( https://github.com/ZhuiyiTechnology/simbert)
3. [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://arxiv.org/abs/2012.00413)

</div></details>
