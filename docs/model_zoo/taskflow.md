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
    - [知识挖掘-词类知识标注](#知识挖掘-词类知识标注)
    - [知识挖掘-名词短语标注](#知识挖掘-名词短语标注)
    - [生成式问答](#生成式问答)
    - [智能写诗](#智能写诗)
  - [FAQ](#FAQ)

## 介绍

`paddlenlp.Taskflow`提供开箱即用的NLP预置任务，覆盖自然语言理解与自然语言生成两大核心应用，在中文场景上提供产业级的效果与极致的预测性能。

### 任务清单

| 自然语言理解任务  | 自然语言生成任务 |
| :------------  | ---- |
| 中文分词 | 生成式问答 |
| 词性标注 | 智能写诗 |
| 命名实体识别  | 文本翻译(TODO) |
| 文本纠错 | 开放域对话(TODO) |
| 句法分析 | 自动对联(TODO) |
| 情感分析 |  |
| 知识挖掘-词类知识标注 |  |
| 知识挖掘-名词短语标注 |  |

随着版本迭代会持续开放更多的应用场景。

## 安装

### 环境依赖
- python >= 3.6
- paddlepaddle >= 2.1.3
- paddlenlp >= 2.1.0

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

### 词性标注

```python
from paddlenlp import Taskflow

tag = Taskflow("pos_tagging")
tag("第十四届全运会在西安举办")
>>>[('第十四届', 'm'), ('全运会', 'nz'), ('在', 'p'), ('西安', 'LOC'), ('举办', 'v')]

tag(["第十四届全运会在西安举办", "三亚是一个美丽的城市"])
>>> [[('第十四届', 'm'), ('全运会', 'nz'), ('在', 'p'), ('西安', 'LOC'), ('举办', 'v')], [('三亚', 'LOC'), ('是', 'v'), ('一个', 'm'), ('美丽', 'a'), ('的', 'u'), ('城市', 'n')]]
```

### 命名实体识别

```python
from paddlenlp import Taskflow

ner = Taskflow("ner")
ner("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
>>> [('《', 'w'), ('孤女', '作品类_实体'), ('》', 'w'), ('是', '肯定词'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('的', '助词'), ('小说', '作品类_概念'), ('，', 'w'), ('作者', '人物类_概念'), ('是', '肯定词'), ('余兼羽', '人物类_实体')]
ner = Taskflow("ner", batch_size=2)
ner(["热梅茶是一道以梅子为主要原料制作的茶饮",
    "《孤女》是2010年九州出版社出版的小说，作者是余兼羽"])
>>> [[('热梅茶', '饮食类_饮品'), ('是', '肯定词'), ('一道', '数量词'), ('以', '介词'), ('梅子', '饮食类'), ('为', '肯定词'), ('主要原料', '物体类'), ('制作', '场景事件'), ('的', '助词'), ('茶饮', '饮食类_饮品')], [('《', 'w'), ('孤女', '作品类_实体'), ('》', 'w'), ('是', '肯定词'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('的', '助词'), ('小说', '作品类_概念'), ('，', 'w'), ('作者', '人物类_概念'), ('是', '肯定词'), ('余兼羽', '人物类_实体')]]
```

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

### 句法分析

```python
from paddlenlp import Taskflow

ddp = Taskflow("dependency_parsing")
ddp("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫")
>>> [{'word': ['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什球场', '击败', '俄罗斯', '球员', '梅德韦杰夫'], 'head': [2, 6, 6, 5, 6, 0, 8, 9, 6], 'deprel': ['ATT', 'ADV', 'SBV', 'MT', 'ADV', 'HED', 'ATT', 'ATT', 'VOB']}]

ddp(["9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫", "他送了一本书"])
>>> [{'word': ['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什球场', '击败', '俄罗斯', '球员', '梅德韦杰夫'], 'head': [2, 6, 6, 5, 6, 0, 8, 9, 6], 'deprel': ['ATT', 'ADV', 'SBV', 'MT', 'ADV', 'HED', 'ATT', 'ATT', 'VOB']}, {'word': ['他', '送', '了', '一本', '书'], 'head': [2, 0, 2, 5, 2], 'deprel': ['SBV', 'HED', 'MT', 'ATT', 'VOB']}]

# 输出概率值和词性标签
ddp = Taskflow("dependency_parsing", prob=True, use_pos=True)
ddp("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫")
>>> [{'word': ['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什', '球场', '击败', '俄罗斯', '球员', '梅德韦杰夫'], 'head': [2, 7, 7, 6, 6, 7, 0, 9, 10, 7], 'deprel': ['ATT', 'ADV', 'SBV', 'MT', 'ATT', 'ADV', 'HED', 'ATT', 'ATT', 'VOB'], 'postag': ['TIME', 'TIME', 'PER', 'p', 'PER', 'n', 'v', 'LOC', 'n', 'PER'], 'prob': [0.79, 0.98, 1.0, 0.49, 0.97, 0.86, 1.0, 0.85, 0.97, 0.99]}]

# 使用ddparser-ernie-1.0进行预测
ddp = Taskflow("dependency_parsing", model="ddparser-ernie-1.0")
ddp("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫")
>>> [{'word': ['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什球场', '击败', '俄罗斯', '球员', '梅德韦杰夫'], 'head': [2, 6, 6, 5, 6, 0, 8, 9, 6], 'deprel': ['ATT', 'ADV', 'SBV', 'MT', 'ADV', 'HED', 'ATT', 'ATT', 'VOB']}]
```

#### 依存关系可视化

句法树可视化示例：

```python
from paddlenlp import Taskflow

ddp = Taskflow("dependency_parsing", return_visual=True)
result = ddp("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫")[0]['visual']
import cv2
cv2.imwrite('test.png', result)
```

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

### 知识挖掘-词类知识标注

```python
from paddlenlp import Taskflow

wordtag = Taskflow("knowledge_mining")
wordtag("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
>>> [{'text': '《孤女》是2010年九州出版社出版的小说，作者是余兼羽', 'items': [{'item': '《', 'offset': 0, 'wordtag_label': 'w', 'length': 1}, {'item': '孤女', 'offset': 1, 'wordtag_label': '作品类_实体', 'length': 2}, {'item': '》', 'offset': 3, 'wordtag_label': 'w', 'length': 1}, {'item': '是', 'offset': 4, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '2010年', 'offset': 5, 'wordtag_label': '时间类', 'length': 5, 'termid': '时间阶段_cb_2010年'}, {'item': '九州出版社', 'offset': 10, 'wordtag_label': '组织机构类', 'length': 5, 'termid': '组织机构_eb_九州出版社'}, {'item': '出版', 'offset': 15, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_出版'}, {'item': '的', 'offset': 17, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '小说', 'offset': 18, 'wordtag_label': '作品类_概念', 'length': 2, 'termid': '小说_cb_小说'}, {'item': '，', 'offset': 20, 'wordtag_label': 'w', 'length': 1}, {'item': '作者', 'offset': 21, 'wordtag_label': '人物类_概念', 'length': 2, 'termid': '人物_cb_作者'}, {'item': '是', 'offset': 23, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '余兼羽', 'offset': 24, 'wordtag_label': '人物类_实体', 'length': 3}]}]

wordtag= Taskflow("knowledge_mining", batch_size=2)
wordtag(["热梅茶是一道以梅子为主要原料制作的茶饮",
         "《孤女》是2010年九州出版社出版的小说，作者是余兼羽"])
>>> [{'text': '热梅茶是一道以梅子为主要原料制作的茶饮', 'items': [{'item': '热梅茶', 'offset': 0, 'wordtag_label': '饮食类_饮品', 'length': 3}, {'item': '是', 'offset': 3, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '一道', 'offset': 4, 'wordtag_label': '数量词', 'length': 2}, {'item': '以', 'offset': 6, 'wordtag_label': '介词', 'length': 1, 'termid': '介词_cb_以'}, {'item': '梅子', 'offset': 7, 'wordtag_label': '饮食类', 'length': 2, 'termid': '饮食_cb_梅'}, {'item': '为', 'offset': 9, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_为'}, {'item': '主要原料', 'offset': 10, 'wordtag_label': '物体类', 'length': 4, 'termid': '物品_cb_主要原料'}, {'item': '制作', 'offset': 14, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_制作'}, {'item': '的', 'offset': 16, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '茶饮', 'offset': 17, 'wordtag_label': '饮食类_饮品', 'length': 2, 'termid': '饮品_cb_茶饮'}]}, {'text': '《孤女》是2010年九州出版社出版的小说，作者是余兼羽', 'items': [{'item': '《', 'offset': 0, 'wordtag_label': 'w', 'length': 1}, {'item': '孤女', 'offset': 1, 'wordtag_label': '作品类_实体', 'length': 2}, {'item': '》', 'offset': 3, 'wordtag_label': 'w', 'length': 1}, {'item': '是', 'offset': 4, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '2010年', 'offset': 5, 'wordtag_label': '时间类', 'length': 5, 'termid': '时间阶段_cb_2010年'}, {'item': '九州出版社', 'offset': 10, 'wordtag_label': '组织机构类', 'length': 5, 'termid': '组织机构_eb_九州出版社'}, {'item': '出版', 'offset': 15, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_出版'}, {'item': '的', 'offset': 17, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '小说', 'offset': 18, 'wordtag_label': '作品类_概念', 'length': 2, 'termid': '小说_cb_小说'}, {'item': '，', 'offset': 20, 'wordtag_label': 'w', 'length': 1}, {'item': '作者', 'offset': 21, 'wordtag_label': '人物类_概念', 'length': 2, 'termid': '人物_cb_作者'}, {'item': '是', 'offset': 23, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '余兼羽', 'offset': 24, 'wordtag_label': '人物类_实体', 'length': 3}]}]
```

### 知识挖掘-名词短语标注

```python
from paddlenlp import Taskflow

nptag = Taskflow("knowledge_mining", model="nptag")
nptag("糖醋排骨")
>>> [{'text': '糖醋排骨', 'label': '菜品'}]

nptag = Taskflow("knowledge_mining", model="nptag", batch_size=2)
nptag(["糖醋排骨", "红曲霉菌"])
>>> [{'text': '糖醋排骨', 'label': '菜品'}, {'text': '红曲霉菌', 'label': '微生物'}]

# 使用`linking`输出粗粒度类别标签`category`，即WordTag的词汇标签。
nptag = Taskflow("knowledge_mining", model="nptag", linking=True)
nptag(["糖醋排骨", "红曲霉菌"])
>>> [{'text': '糖醋排骨', 'label': '菜品', 'category': '饮食类_菜品'}, {'text': '红曲霉菌', 'label': '微生物', 'category': '生物类_微生物'}]
```

### 生成式问答

```python
from paddlenlp import Taskflow

qa = Taskflow("question_answering")
qa("中国的国土面积有多大？")
>>> [{'text': '中国的国土面积有多大？', 'answer': '960万平方公里。'}]

qa(["中国国土面积有多大？", "中国的首都在哪里？"])
>>> [{'text': '中国国土面积有多大？', 'answer': '960万平方公里。'}, {'text': '中国的首都在哪里？', 'answer': '北京。'}]
```

### 智能写诗

```python
from paddlenlp import Taskflow

poetry = Taskflow("poetry_generation")
poetry("林密不见人")
>>> [{'text': '林密不见人', 'answer': ',但闻人语响。'}]

poetry(["林密不见人", "举头邀明月"])
>>> [{'text': '林密不见人', 'answer': ',但闻人语响。'}, {'text': '举头邀明月', 'answer': ',低头思故乡。'}]
```

## FAQ

### Q1 Taskflow如何修改任务保存路径？

**A:** Taskflow默认会将任务相关模型等文件保存到`$HOME/.paddlenlp`下，可以在任务初始化的时候通过`home_path`自定义修改保存路径。

示例：
```python
from paddlenlp import Taskflow

ner = Taskflow("ner", home_path="/workspace")
```
通过以上方式即可将ner任务相关文件保存至`/workspace`路径下。