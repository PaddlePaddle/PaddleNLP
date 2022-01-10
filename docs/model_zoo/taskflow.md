# PaddleNLP Taskflow

- [PaddleNLP Taskflow](#paddlenlp-taskflow)
  - [介绍](#介绍)
    - [任务清单](#任务清单)
  - [用法](#用法)
    - [查看使用示例](#查看使用示例)
    - [中文分词](#中文分词)
    - [词性标注](#词性标注)
    - [命名实体识别](#命名实体识别)
    - [文本纠错](#文本纠错)
    - [句法分析](#句法分析)
    - [情感分析](#情感分析)
    - [文本相似度](#文本相似度)
    - [『解语』-词类知识标注](#知识挖掘-词类知识标注)
    - [『解语』-名词短语标注](#知识挖掘-名词短语标注)
    - [生成式问答](#生成式问答)
    - [智能写诗](#智能写诗)
    - [开放域对话](#开放域对话)
  - [FAQ](#FAQ)

## 介绍

`paddlenlp.Taskflow`提供开箱即用的NLP预置任务，覆盖自然语言理解与自然语言生成两大核心应用，在中文场景上提供产业级的效果与极致的预测性能。

### 任务清单

| 自然语言理解任务  | 自然语言生成任务 |
| :------------  | ---- |
| 中文分词 | 生成式问答 |
| 词性标注 | 智能写诗 |
| 命名实体识别  | 开放域对话 |
| 文本纠错 | 文本翻译(TODO) |
| 句法分析 | 自动对联(TODO) |
| 情感分析 |  |
| 文本相似度 |  |
| 『解语』-词类知识标注 |  |
| 『解语』-名词短语标注 |  |

随着版本迭代会持续开放更多的应用场景。

## 安装

### 环境依赖
- python >= 3.6
- paddlepaddle >= 2.2.0
- paddlenlp >= 2.2.0

## 用法

### 查看使用示例
```python
from paddlenlp import Taskflow

seg = Taskflow("word_segmentation")
seg.help()
>>> Examples:
        from paddlenlp import Taskflow

        seg = Taskflow("word_segmentation")
        seg("第十四届全运会在西安举办")
        '''
        ['第十四届', '全运会', '在', '西安', '举办']
        '''

        seg(["第十四届全运会在西安举办", "三亚是一个美丽的城市"])
        '''
        [['第十四届', '全运会', '在', '西安', '举办'], ['三亚', '是', '一个', '美丽', '的', '城市']]
        '''
```

### 中文分词

```python
from paddlenlp import Taskflow

seg = Taskflow("word_segmentation")
seg("第十四届全运会在西安举办")
>>> ['第十四届', '全运会', '在', '西安', '举办']

seg(["第十四届全运会在西安举办", "三亚是一个美丽的城市"])
>>> [['第十四届', '全运会', '在', '西安', '举办'], ['三亚', '是', '一个', '美丽', '的', '城市']]
```

#### 自定义词典

用户可以通过装载自定义词典来定制化分词结果。词典文件每一行表示一个自定义item，可以由一个单词或者多个单词组成。

词典文件`user_dict.txt`示例：

```text
平原上的火焰
年 末
```

以"平原上的火焰计划于年末上映"为例，原本的输出结果为：

```text
['平原', '上', '的', '火焰', '计划', '于', '年末', '上映']
```

装载自定义词典及输出结果示例：

```python
from paddlenlp import Taskflow

my_seg = Taskflow("word_segmentation", user_dict="user_dict.txt")
my_seg("平原上的火焰计划于年末上映")
>>> ['平原上的火焰', '计划', '于', '年', '末', '上映']
```

#### 自定义任务

任务的默认路径为`$HOME/.paddlenlp/taskflow/word_sementation/lac/`，默认路径下包含了执行该任务需要的所有文件。

用户也可以使用自己的数据训练自定义中文分词模型，参考[词法分析训练示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/lexical_analysis)。

有了自定义模型后可通过`task_path`指定用户自定义路径，自定义路径下的文件需要和默认路径的文件一致。

自定义路径需要有如下文件（用户自己的模型权重、标签字典）：
```text
custom_task_path/
├── model_state.pdparams
├── word.dic
├── tag.dic
└── q2b.dic
```

使用Taskflow加载自定义模型进行一键预测：

```python
from paddlenlp import Taskflow

my_seg = Taskflow("word_segmentation", task_path="./custom_task_path/")
```
#### 可配置参数说明

* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `user_dict`：用户自定义词典文件，默认为None。
* `task_path`：自定义任务路径，默认为None。

### 词性标注

```python
from paddlenlp import Taskflow

tag = Taskflow("pos_tagging")
tag("第十四届全运会在西安举办")
>>>[('第十四届', 'm'), ('全运会', 'nz'), ('在', 'p'), ('西安', 'LOC'), ('举办', 'v')]

tag(["第十四届全运会在西安举办", "三亚是一个美丽的城市"])
>>> [[('第十四届', 'm'), ('全运会', 'nz'), ('在', 'p'), ('西安', 'LOC'), ('举办', 'v')], [('三亚', 'LOC'), ('是', 'v'), ('一个', 'm'), ('美丽', 'a'), ('的', 'u'), ('城市', 'n')]]
```

- 标签集合：

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

用户可以通过装载自定义词典来定制化分词和词性标注结果。词典文件每一行表示一个自定义item，可以由一个单词或者多个单词组成，单词后面可以添加自定义标签，格式为`item/tag`，如果不添加自定义标签，则使用模型默认标签。

词典文件`user_dict.txt`示例：

```text
赛里木湖/LAKE
高/a 山/n
海拔最高
湖 泊
```

以"赛里木湖是新疆海拔最高的高山湖泊"为例，原本的输出结果为：

```text
[('赛里木湖', 'LOC'), ('是', 'v'), ('新疆', 'LOC'), ('海拔', 'n'), ('最高', 'a'), ('的', 'u'), ('高山', 'n'), ('湖泊', 'n')]
```

装载自定义词典及输出结果示例：

```python
from paddlenlp import Taskflow

my_tag = Taskflow("pos_tagging", user_dict="user_dict.txt")
my_tag("赛里木湖是新疆海拔最高的高山湖泊")
>>> [('赛里木湖', 'LAKE'), ('是', 'v'), ('新疆', 'LOC'), ('海拔最高', 'n'), ('的', 'u'), ('高', 'a'), ('山', 'n'), ('湖', 'n'), ('泊', 'n')]
```

#### 自定义任务

任务的默认路径为`$HOME/.paddlenlp/taskflow/word_sementation/lac/`，默认路径下包含了执行该任务需要的所有文件。

用户也可以使用自己的数据训练自定义中文分词模型，参考[词法分析训练示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/lexical_analysis)。

有了自定义模型后可通过`task_path`指定用户自定义路径，自定义路径下的文件需要和默认路径的文件一致。

自定义路径需要有如下文件（用户自己的模型权重、标签字典）：
```text
custom_task_path/
├── model_state.pdparams
├── word.dic
├── tag.dic
└── q2b.dic
```

使用Taskflow加载自定义模型进行一键预测：

```python
from paddlenlp import Taskflow

my_tag = Taskflow("pos_tagging", task_path="./custom_task_path/")
```
#### 可配置参数说明

* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `user_dict`：用户自定义词典文件，默认为None。
* `task_path`：自定义任务路径，默认为None。

### 命名实体识别

```python
from paddlenlp import Taskflow

ner = Taskflow("ner")
ner("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
>>> [('《', 'w'), ('孤女', '作品类_实体'), ('》', 'w'), ('是', '肯定词'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('的', '助词'), ('小说', '作品类_概念'), ('，', 'w'), ('作者', '人物类_概念'), ('是', '肯定词'), ('余兼羽', '人物类_实体')]

ner(["热梅茶是一道以梅子为主要原料制作的茶饮", "《孤女》是2010年九州出版社出版的小说，作者是余兼羽"])
>>> [[('热梅茶', '饮食类_饮品'), ('是', '肯定词'), ('一道', '数量词'), ('以', '介词'), ('梅子', '饮食类'), ('为', '肯定词'), ('主要原料', '物体类'), ('制作', '场景事件'), ('的', '助词'), ('茶饮', '饮食类_饮品')], [('《', 'w'), ('孤女', '作品类_实体'), ('》', 'w'), ('是', '肯定词'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('的', '助词'), ('小说', '作品类_概念'), ('，', 'w'), ('作者', '人物类_概念'), ('是', '肯定词'), ('余兼羽', '人物类_实体')]]
```

- 标签集合：

Taskflow提供的NER任务共包含66种词性及专名类别标签，标签集合如下表

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

#### 自定义词典

用户可以通过装载自定义词典来定制化分词和词性标注结果。词典文件每一行表示一个自定义item，可以由一个单词或者多个单词组成，单词后面可以添加自定义标签，格式为`item/tag`，如果不添加自定义标签，则使用模型默认标签。

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
from paddlenlp import Taskflow

my_ner = Taskflow("ner", user_dict="user_dict.txt")
my_ner("《长津湖》收尾，北美是最大海外票仓")
>>> [('《', 'w'), ('长津湖', '电影类_实体'), ('》', 'w'), ('收', '词汇用语'), ('尾', '术语类'), ('，', 'w'), ('北美', '世界地区类'), ('是', '肯定词'), ('最', '修饰词'), ('大', '修饰词'), ('海外票仓', '场所类')]
```

#### 自定义任务

任务的默认路径为`$HOME/.paddlenlp/taskflow/ner/wordtag/`，默认路径下包含了执行该任务需要的所有文件。

用户也可以使用自己的数据训练自定义NER模型，参考[NER-WordTag增量训练示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/ernie-ctm)。

有了自定义模型后可通过`task_path`指定用户自定义路径，自定义路径下的文件需要和默认路径的文件一致。

自定义路径需要有如下文件（用户自己的模型权重、标签文件）：
```text
custom_task_path/
├── model_state.pdparams
└── tags.txt
```

使用Taskflow加载自定义模型进行一键预测：

```python
from paddlenlp import Taskflow

my_ner = Taskflow("ner", task_path="./custom_task_path/")
```

#### 可配置参数说明

* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `user_dict`：用户自定义词典文件，默认为None。
* `task_path`：自定义任务路径，默认为None。

### 文本纠错

```python
from paddlenlp import Taskflow

corrector = Taskflow("text_correction")
corrector('遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。')
>>> [{'source': '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 'target': '遇到逆境时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 'errors': [{'position': 3, 'correction': {'竟': '境'}}]}]

corrector(['遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
                '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。'])
>>> [{'source': '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 'target': '遇到逆境时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 'errors': [{'position': 3, 'correction': {'竟': '境'}}]}, {'source': '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。', 'target': '人生就是如此，经过磨练才能让自己更加茁壮，才能使自己更加乐观。', 'errors': [{'position': 18, 'correction': {'拙': '茁'}}]}]
```

#### 自定义任务

任务的默认路径为`$HOME/.paddlenlp/taskflow/text_correction/csc-ernie-1.0/`，默认路径下包含了执行该任务需要的所有文件。

用户也可以使用自己的数据训练自定义模型，参考[文本纠错训练示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_correction/ernie-csc)。

有了自定义模型后可通过`task_path`指定用户自定义路径，自定义路径下的文件需要和默认路径的文件一致。

自定义路径需要有如下文件（用户自己的模型权重、拼音字典文件）：
```text
custom_task_path/
├── model_state.pdparams
└── pinyin_vocab.txt
```

使用Taskflow加载自定义模型进行一键预测：

```python
from paddlenlp import Taskflow

my_ner = Taskflow("text_correction", task_path="./custom_task_path/")
```

#### 可配置参数说明

* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `task_path`：自定义任务路径，默认为None。

### 句法分析

未分词输入:

```python
from paddlenlp import Taskflow

ddp = Taskflow("dependency_parsing")
ddp("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫")
>>> [{'word': ['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什球场', '击败', '俄罗斯', '球员', '梅德韦杰夫'], 'head': [2, 6, 6, 5, 6, 0, 8, 9, 6], 'deprel': ['ATT', 'ADV', 'SBV', 'MT', 'ADV', 'HED', 'ATT', 'ATT', 'VOB']}]

ddp(["9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫", "他送了一本书"])
>>> [{'word': ['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什球场', '击败', '俄罗斯', '球员', '梅德韦杰夫'], 'head': [2, 6, 6, 5, 6, 0, 8, 9, 6], 'deprel': ['ATT', 'ADV', 'SBV', 'MT', 'ADV', 'HED', 'ATT', 'ATT', 'VOB']}, {'word': ['他', '送', '了', '一本', '书'], 'head': [2, 0, 2, 5, 2], 'deprel': ['SBV', 'HED', 'MT', 'ATT', 'VOB']}]
```

输出概率值和词性标签:

```python
ddp = Taskflow("dependency_parsing", prob=True, use_pos=True)
ddp("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫")
>>> [{'word': ['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什', '球场', '击败', '俄罗斯', '球员', '梅德韦杰夫'], 'head': [2, 7, 7, 6, 6, 7, 0, 9, 10, 7], 'deprel': ['ATT', 'ADV', 'SBV', 'MT', 'ATT', 'ADV', 'HED', 'ATT', 'ATT', 'VOB'], 'postag': ['TIME', 'TIME', 'PER', 'p', 'PER', 'n', 'v', 'LOC', 'n', 'PER'], 'prob': [0.79, 0.98, 1.0, 0.49, 0.97, 0.86, 1.0, 0.85, 0.97, 0.99]}]
```

使用ddparser-ernie-1.0进行预测:

```python
ddp = Taskflow("dependency_parsing", model="ddparser-ernie-1.0")
ddp("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫")
>>> [{'word': ['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什球场', '击败', '俄罗斯', '球员', '梅德韦杰夫'], 'head': [2, 6, 6, 5, 6, 0, 8, 9, 6], 'deprel': ['ATT', 'ADV', 'SBV', 'MT', 'ADV', 'HED', 'ATT', 'ATT', 'VOB']}]
```

使用分词结果来输入:

```python
ddp = Taskflow("dependency_parsing")
ddp.from_segments([['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什球场', '击败', '俄罗斯', '球员', '梅德韦杰夫']])
>>> [{'word': ['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什球场', '击败', '俄罗斯', '球员', '梅德韦杰夫'], 'head': [2, 6, 6, 5, 6, 0, 8, 9, 6], 'deprel': ['ATT', 'ADV', 'SBV', 'MT', 'ADV', 'HED', 'ATT', 'ATT', 'VOB']}]
```

#### 自定义任务

任务的默认路径为`$HOME/.paddlenlp/taskflow/dependency_parsing/ddparser/`，默认路径下包含了执行该任务需要的所有文件。

用户也可以使用自己的数据训练自定义模型，参考[句法分析训练示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/dependency_parsing/ddparser)。

有了自定义模型后可通过`task_path`指定用户自定义路径，自定义路径下的文件需要和默认路径的文件一致。

自定义路径需要有如下文件（用户自己的模型权重、字典文件）：
```text
custom_task_path/
├── model_state.pdparams
├── rel_vocab.json
└── word_vocab.json
```

使用Taskflow加载自定义模型进行一键预测：

```python
from paddlenlp import Taskflow

my_ddp = Taskflow("dependency_parsing", task_path="./custom_task_path/")
```

#### 依存关系可视化：

```python
from paddlenlp import Taskflow

ddp = Taskflow("dependency_parsing", return_visual=True)
result = ddp("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫")[0]['visual']
import cv2
cv2.imwrite('test.png', result)
```

#### 标注关系说明：

| Label |  关系类型  | 说明                     | 示例                           |
| :---: | :--------: | :----------------------- | :----------------------------- |
|  SBV  |  主谓关系  | 主语与谓词间的关系       | 他送了一本书(他<--送)          |
|  VOB  |  动宾关系  | 宾语与谓词间的关系       | 他送了一本书(送-->书)          |
|  POB  |  介宾关系  | 介词与宾语间的关系       | 我把书卖了（把-->书）          |
|  ADV  |  状中关系  | 状语与中心词间的关系     | 我昨天买书了（昨天<--买）      |
|  CMP  |  动补关系  | 补语与中心词间的关系     | 我都吃完了（吃-->完）          |
|  ATT  |  定中关系  | 定语与中心词间的关系     | 他送了一本书(一本<--书)        |
|   F   |  方位关系  | 方位词与中心词的关系     | 在公园里玩耍(公园-->里)        |
|  COO  |  并列关系  | 同类型词语间关系        | 叔叔阿姨(叔叔-->阿姨)          |
|  DBL  |  兼语结构  | 主谓短语做宾语的结构     | 他请我吃饭(请-->我，请-->吃饭) |
|  DOB  | 双宾语结构 | 谓语后出现两个宾语       | 他送我一本书(送-->我，送-->书) |
|  VV   |  连谓结构  | 同主语的多个谓词间关系   | 他外出吃饭(外出-->吃饭)        |
|  IC   |  子句结构  | 两个结构独立或关联的单句  | 你好，书店怎么走？(你好<--走)  |
|  MT   |  虚词成分  | 虚词与中心词间的关系     | 他送了一本书(送-->了)          |
|  HED  |  核心关系  | 指整个句子的核心         |                               |

#### 可配置参数说明

* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `model`：选择任务使用的模型，可选有`ddparser`，`ddparser-ernie-1.0`和`ddparser-ernie-gram-zh`。
* `tree`：确保输出结果是正确的依存句法树，默认为True。
* `prob`：是否输出每个弧对应的概率值，默认为False。
* `use_pos`：是否返回词性标签，默认为False。
* `use_cuda`：是否使用GPU进行切词，默认为False。
* `return_visual`：是否返回句法树的可视化结果，默认为False。
* `task_path`：自定义任务路径，默认为None。

### 情感分析

```python
from paddlenlp import Taskflow

senta = Taskflow("sentiment_analysis")
senta("这个产品用起来真的很流畅，我非常喜欢")
>>> [{'text': '这个产品用起来真的很流畅，我非常喜欢', 'label': 'positive', 'score': 0.9938690066337585}]

senta(["这个产品用起来真的很流畅，我非常喜欢", "作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间"])
>>> [{'text': '这个产品用起来真的很流畅，我非常喜欢', 'label': 'positive', 'score': 0.9938690066337585}, {'text': '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间', 'label': 'positive', 'score': 0.985750675201416}]

# 使用SKEP情感分析预训练模型进行预测
senta = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch")
senta("作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。")
>>> [{'text': '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。', 'label': 'positive', 'score': 0.984320878982544}]
```

#### 自定义任务

可通过`task_path`指定用户自定义路径，自定义路径下的文件需要和默认路径的文件一致。

- 自定义bilstm模型

任务的默认路径为`$HOME/.paddlenlp/taskflow/sentiment_analysis/bilstm/`，默认路径下包含了执行该任务需要的所有文件。

通过`task_path`指定用户自定义路径，自定义路径下的文件需要和默认路径的文件一致。

自定义路径需要有如下文件（用户自己的模型权重、字典文件）：
```text
custom_task_path/
├── model_state.pdparams
└── vocab.txt
```

使用Taskflow加载自定义bilstm模型进行一键预测：

```python
from paddlenlp import Taskflow

my_senta = Taskflow("sentiment_analysis", task_path="./custom_task_path/")
```

- 自定义SKEP模型

任务的默认路径为`$HOME/.paddlenlp/taskflow/sentiment_analysis/skep_ernie_1.0_large_ch/`，默认路径下包含了执行该任务需要的所有文件。

用户也可以使用自己的数据训练自定义模型，参考[SKEP情感分类任务](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/sentiment_analysis/skep)。

通过`task_path`指定用户自定义路径，自定义路径下的文件需要和默认路径的文件一致。

自定义路径需要有如下文件（用户自己的模型权重、模型参数配置文件）：
```text
custom_task_path/
├── model_state.pdparams
└── model_config.json
```

使用Taskflow加载自定义SKEP模型进行一键预测：

```python
from paddlenlp import Taskflow

my_senta = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch", task_path="./custom_task_path/")
```

#### 可配置参数说明

* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `model`：选择任务使用的模型，可选有`bilstm`和`skep_ernie_1.0_large_ch`。
* `task_path`：自定义任务路径，默认为None。

### 文本相似度

```python
from paddlenlp import Taskflow

similarity = Taskflow("text_similarity")
similarity([["世界上什么东西最小", "世界上什么东西最小？"]])
>>> [{'text1': '世界上什么东西最小', 'text2': '世界上什么东西最小？', 'similarity': 0.992725}]

similarity([["光眼睛大就好看吗", "眼睛好看吗？"], ["小蝌蚪找妈妈怎么样", "小蝌蚪找妈妈是谁画的"]])
>>> [{'text1': '光眼睛大就好看吗', 'text2': '眼睛好看吗？', 'similarity': 0.74502707}, {'text1': '小蝌蚪找妈妈怎么样', 'text2': '小蝌蚪找妈妈是谁画的', 'similarity': 0.8192149}]
```

#### 可配置参数说明

* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `max_seq_len`：最大序列长度，默认为128。
* `task_path`：自定义任务路径，默认为None。

### 知识挖掘-词类知识标注

```python
from paddlenlp import Taskflow

wordtag = Taskflow("knowledge_mining")
wordtag("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
>>> [{'text': '《孤女》是2010年九州出版社出版的小说，作者是余兼羽', 'items': [{'item': '《', 'offset': 0, 'wordtag_label': 'w', 'length': 1}, {'item': '孤女', 'offset': 1, 'wordtag_label': '作品类_实体', 'length': 2}, {'item': '》', 'offset': 3, 'wordtag_label': 'w', 'length': 1}, {'item': '是', 'offset': 4, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '2010年', 'offset': 5, 'wordtag_label': '时间类', 'length': 5, 'termid': '时间阶段_cb_2010年'}, {'item': '九州出版社', 'offset': 10, 'wordtag_label': '组织机构类', 'length': 5, 'termid': '组织机构_eb_九州出版社'}, {'item': '出版', 'offset': 15, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_出版'}, {'item': '的', 'offset': 17, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '小说', 'offset': 18, 'wordtag_label': '作品类_概念', 'length': 2, 'termid': '小说_cb_小说'}, {'item': '，', 'offset': 20, 'wordtag_label': 'w', 'length': 1}, {'item': '作者', 'offset': 21, 'wordtag_label': '人物类_概念', 'length': 2, 'termid': '人物_cb_作者'}, {'item': '是', 'offset': 23, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '余兼羽', 'offset': 24, 'wordtag_label': '人物类_实体', 'length': 3}]}]

wordtag(["热梅茶是一道以梅子为主要原料制作的茶饮",
         "《孤女》是2010年九州出版社出版的小说，作者是余兼羽"])
>>> [{'text': '热梅茶是一道以梅子为主要原料制作的茶饮', 'items': [{'item': '热梅茶', 'offset': 0, 'wordtag_label': '饮食类_饮品', 'length': 3}, {'item': '是', 'offset': 3, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '一道', 'offset': 4, 'wordtag_label': '数量词', 'length': 2}, {'item': '以', 'offset': 6, 'wordtag_label': '介词', 'length': 1, 'termid': '介词_cb_以'}, {'item': '梅子', 'offset': 7, 'wordtag_label': '饮食类', 'length': 2, 'termid': '饮食_cb_梅'}, {'item': '为', 'offset': 9, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_为'}, {'item': '主要原料', 'offset': 10, 'wordtag_label': '物体类', 'length': 4, 'termid': '物品_cb_主要原料'}, {'item': '制作', 'offset': 14, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_制作'}, {'item': '的', 'offset': 16, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '茶饮', 'offset': 17, 'wordtag_label': '饮食类_饮品', 'length': 2, 'termid': '饮品_cb_茶饮'}]}, {'text': '《孤女》是2010年九州出版社出版的小说，作者是余兼羽', 'items': [{'item': '《', 'offset': 0, 'wordtag_label': 'w', 'length': 1}, {'item': '孤女', 'offset': 1, 'wordtag_label': '作品类_实体', 'length': 2}, {'item': '》', 'offset': 3, 'wordtag_label': 'w', 'length': 1}, {'item': '是', 'offset': 4, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '2010年', 'offset': 5, 'wordtag_label': '时间类', 'length': 5, 'termid': '时间阶段_cb_2010年'}, {'item': '九州出版社', 'offset': 10, 'wordtag_label': '组织机构类', 'length': 5, 'termid': '组织机构_eb_九州出版社'}, {'item': '出版', 'offset': 15, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_出版'}, {'item': '的', 'offset': 17, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '小说', 'offset': 18, 'wordtag_label': '作品类_概念', 'length': 2, 'termid': '小说_cb_小说'}, {'item': '，', 'offset': 20, 'wordtag_label': 'w', 'length': 1}, {'item': '作者', 'offset': 21, 'wordtag_label': '人物类_概念', 'length': 2, 'termid': '人物_cb_作者'}, {'item': '是', 'offset': 23, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '余兼羽', 'offset': 24, 'wordtag_label': '人物类_实体', 'length': 3}]}]
```

- 标签集合：

知识挖掘-词类知识标注任务共包含66种词性及专名类别标签，标签集合如下表

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

#### 自定义词典

用户可以通过装载自定义词典来定制化分词和词性标注结果。词典文件每一行表示一个自定义item，可以由一个单词或者多个单词组成，单词后面可以添加自定义标签，格式为`item/tag`，如果不添加自定义标签，则使用模型默认标签。

词典文件`user_dict.txt`示例：

```text
长津湖/电影类_实体
收/词汇用语 尾/术语类
最 大
海外票仓
```

以"《长津湖》收尾，北美是最大海外票仓"为例，原本的输出结果为：

```text
[{'text': '《长津湖》收尾，北美是最大海外票仓', 'items': [{'item': '《', 'offset': 0, 'wordtag_label': 'w', 'length': 1}, {'item': '长津湖', 'offset': 1, 'wordtag_label': '作品类_实体', 'length': 3, 'termid': '影视作品_eb_长津湖'}, {'item': '》', 'offset': 4, 'wordtag_label': 'w', 'length': 1}, {'item': '收尾', 'offset': 5, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_收尾'}, {'item': '，', 'offset': 7, 'wordtag_label': 'w', 'length': 1}, {'item': '北美', 'offset': 8, 'wordtag_label': '世界地区类', 'length': 2, 'termid': '世界地区_cb_北美'}, {'item': '是', 'offset': 10, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '最大', 'offset': 11, 'wordtag_label': '修饰词', 'length': 2, 'termid': '修饰词_cb_最大'}, {'item': '海外', 'offset': 13, 'wordtag_label': '场所类', 'length': 2, 'termid': '区域场所_cb_海外'}, {'item': '票仓', 'offset': 15, 'wordtag_label': '词汇用语', 'length': 2}]}]
```

装载自定义词典及输出结果示例：

```python
from paddlenlp import Taskflow

my_wordtag = Taskflow("knowledge_mining", user_dict="user_dict.txt")
my_wordtag("《长津湖》收尾，北美是最大海外票仓")
>>> [{'text': '《长津湖》收尾，北美是最大海外票仓', 'items': [{'item': '《', 'offset': 0, 'wordtag_label': 'w', 'length': 1}, {'item': '长津湖', 'offset': 1, 'wordtag_label': '电影类_实体', 'length': 3}, {'item': '》', 'offset': 4, 'wordtag_label': 'w', 'length': 1}, {'item': '收', 'offset': 5, 'wordtag_label': '词汇用语', 'length': 1}, {'item': '尾', 'offset': 6, 'wordtag_label': '术语类', 'length': 1, 'termid': '动物体构造_cb_动物尾巴'}, {'item': '，', 'offset': 7, 'wordtag_label': 'w', 'length': 1}, {'item': '北美', 'offset': 8, 'wordtag_label': '世界地区类', 'length': 2, 'termid': '世界地区_cb_北美'}, {'item': '是', 'offset': 10, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '最', 'offset': 11, 'wordtag_label': '修饰词', 'length': 1}, {'item': '大', 'offset': 12, 'wordtag_label': '修饰词', 'length': 1, 'termid': '修饰词_cb_大'}, {'item': '海外票仓', 'offset': 13, 'wordtag_label': '场所类', 'length': 4}]}]
```

#### 自定义任务

任务的默认路径为`$HOME/.paddlenlp/taskflow/knowledge_mining/wordtag/`，默认路径下包含了执行该任务需要的所有文件。

用户也可以使用自己的数据训练自定义模型，参考[WordTag增量训练示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/ernie-ctm)。
除了自定义模型，Taskflow还支持使用[自定义TermTree](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/termtree)来实现自定义Term-Linking。

可通过`task_path`指定用户自定义路径，自定义路径下的文件需要和默认路径的文件一致。

自定义路径包含如下文件（用户自己的模型权重、模型参数配置、标签、百科知识树文件）：
```text
custom_task_path/
├── model_state.pdparams
├── model_config.json
├── tags.txt
├── termtree_type.csv
└── termtree_data
```
**NOTE**: 因为该任务包含自定义模型与自定义TermTree两部分，若用户只想使用自己的WordTag模型而使用默认TermTree，则路径下只需要有`model_state.pdparams`、`model_config.json`和`tags.txt`即可；
若用户只使用自定义的TermTree而使用默认的WordTag模型，则路径下只需要有`termtree_type.csv`和`termtree_data`即可。

使用Taskflow加载自定义模型进行一键预测：

```python
from paddlenlp import Taskflow

my_wordtag = Taskflow("knowledge_mining", task_path="./custom_task_path/")
```

#### 可配置参数说明

* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `linking`：实现基于词类的linking，默认为True。
* `task_path`：自定义任务路径，默认为None。
* `user_dict`：用户自定义词典文件，默认为None。

### 知识挖掘-名词短语标注

```python
from paddlenlp import Taskflow

nptag = Taskflow("knowledge_mining", model="nptag")
nptag("糖醋排骨")
>>> [{'text': '糖醋排骨', 'label': '菜品'}]

nptag(["糖醋排骨", "红曲霉菌"])
>>> [{'text': '糖醋排骨', 'label': '菜品'}, {'text': '红曲霉菌', 'label': '微生物'}]

# 使用`linking`输出粗粒度类别标签`category`，即WordTag的词汇标签。
nptag = Taskflow("knowledge_mining", model="nptag", linking=True)
nptag(["糖醋排骨", "红曲霉菌"])
>>> [{'text': '糖醋排骨', 'label': '菜品', 'category': '饮食类_菜品'}, {'text': '红曲霉菌', 'label': '微生物', 'category': '生物类_微生物'}]
```

#### 自定义任务

任务的默认路径为`$HOME/.paddlenlp/taskflow/knowledge_mining/nptag/`，默认路径下包含了执行该任务需要的所有文件。

用户也可以使用自己的数据训练自定义模型，参考[名词短语标注训练示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/nptag)。

可通过`task_path`指定用户自定义路径，自定义路径下的文件需要和默认路径的文件一致。

自定义路径包含如下文件（用户自己的模型权重、模型参数配置、标签文件）：
```text
custom_task_path/
├── model_state.pdparams
├── model_config.json
└── name_category_map.json
```

使用Taskflow加载自定义模型进行一键预测：

```python
from paddlenlp import Taskflow

my_nptag = Taskflow("knowledge_mining", model="nptag", task_path="./custom_task_path/")
```

#### 可配置参数说明

* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `max_seq_len`：最大序列长度，默认为64。
* `linking`：实现与WordTag类别标签的linking，默认为False。
* `task_path`：自定义任务路径，默认为None。

### 生成式问答

```python
from paddlenlp import Taskflow

qa = Taskflow("question_answering")
qa("中国的国土面积有多大？")
>>> [{'text': '中国的国土面积有多大？', 'answer': '960万平方公里。'}]

qa(["中国国土面积有多大？", "中国的首都在哪里？"])
>>> [{'text': '中国国土面积有多大？', 'answer': '960万平方公里。'}, {'text': '中国的首都在哪里？', 'answer': '北京。'}]
```

#### 可配置参数说明

* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。

### 智能写诗

```python
from paddlenlp import Taskflow

poetry = Taskflow("poetry_generation")
poetry("林密不见人")
>>> [{'text': '林密不见人', 'answer': ',但闻人语响。'}]

poetry(["林密不见人", "举头邀明月"])
>>> [{'text': '林密不见人', 'answer': ',但闻人语响。'}, {'text': '举头邀明月', 'answer': ',低头思故乡。'}]
```

#### 可配置参数说明

* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。

### 开放域对话

非交互模式：
```python
from paddlenlp import Taskflow 

dialogue = Taskflow("dialogue")
dialogue(["吃饭了吗"])
>>> ['刚吃完饭,你在干什么呢?']

dialogue(["你好", "吃饭了吗"], ["你是谁？"])
>>> ['吃过了,你呢', '我是李明啊']
```

可配置参数：

* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `max_seq_len`：最大序列长度，默认为512。

交互模式：
```python
from paddlenlp import Taskflow

dialogue = Taskflow("dialogue")
# 输入`exit`可退出交互模式
dialogue.interactive_mode(max_turn=3)

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

## FAQ

### Q1 Taskflow如何修改任务保存路径？

**A:** Taskflow默认会将任务相关模型等文件保存到`$HOME/.paddlenlp`下，可以在任务初始化的时候通过`home_path`自定义修改保存路径。

示例：
```python
from paddlenlp import Taskflow

ner = Taskflow("ner", home_path="/workspace")
```
通过以上方式即可将ner任务相关文件保存至`/workspace`路径下。

### Q2 Taskflow如何自定义任务？

**A:** 参考具体任务中的`自定义任务`说明，用户可按照示例在特定路径配置任务所需的模型权重、字典等文件，然后通过`task_path`指定自定义任务路径以一键装载任务相关文件。自然语言生成任务暂时不支持自定义任务。