# PaddleNLP Taskflow

- [介绍](#介绍)
  * [任务清单](#任务清单)
- [用法](#用法)
  * [中文分词](#中文分词)
  * [词性标注](#词性标注)
  * [命名实体识别](#命名实体识别)
  * [文本纠错](#文本纠错)
  * [句法分析](#句法分析)
  * [情感分类](#情感分类)
  * [生成式问答](#生成式问答)
  * [智能写诗](#智能写诗)

## 介绍

`paddlenlp.Taskflow`是功能强大的自然语言处理库，旨在提供开箱即用的NLP预置任务，覆盖自然语言理解与自然语言生成两大核心应用，在中文场景上提供工业级的效果与极致的预测性能。

### 支持任务清单

| 自然语言理解任务  | 自然语言生成任务 |
| :------------  | ---- |
| 中文分词 | 生成式问答 |
| 词性标注 | 智能写诗 |
| 命名实体识别  | 文本翻译(TODO) |
| 文本纠错 | 开放域对话(TODO) |
| 句法分析 | 自动对联(TODO) |
| 情感分类 |  |

随着版本迭代后续会持续开放更多的应用场景。

## 用法

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

pos_tagging = Taskflow("pos_tagging")
pos_tagging("第十四届全运会在西安举办")
>>>[('第十四届', 'm'), ('全运会', 'nz'), ('在', 'p'), ('西安', 'LOC'), ('举办', 'v')]

pos_tagging(["第十四届全运会在西安举办", "三亚是一个美丽的城市"])

>>> [[('第十四届', 'm'), ('全运会', 'nz'), ('在', 'p'), ('西安', 'LOC'), ('举办', 'v')], [('三亚', 'LOC'), ('是', 'v'), ('一个', 'm'), ('美丽', 'a'), ('的', 'u'), ('城市', 'n')]]
```

### 命名实体识别

```python
from paddlenlp import Taskflow 

ner = Taskflow("ner")
ner("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
>>> [{'text': '《孤女》是2010年九州出版社出版的小说，作者是余兼羽', 'items': [{'item': '《', 'offset': 0, 'wordtag_label': 'w', 'length': 1}, {'item': '孤女', 'offset': 1, 'wordtag_label': '作品类_实体', 'length': 2}, {'item': '》', 'offset': 3, 'wordtag_label': 'w', 'length': 1}, {'item': '是', 'offset': 4, 'wordtag_label': '肯定词', 'length': 1}, {'item': '2010年', 'offset': 5, 'wordtag_label': '时间类', 'length': 5}, {'item': '九州出版社', 'offset': 10, 'wordtag_label': '组织机构类', 'length': 5}, {'item': '出版', 'offset': 15, 'wordtag_label': '场景事件', 'length': 2}, {'item': '的', 'offset': 17, 'wordtag_label': '助词', 'length': 1}, {'item': '小说', 'offset': 18, 'wordtag_label': '作品类_概念', 'length': 2}, {'item': '，', 'offset': 20, 'wordtag_label': 'w', 'length': 1}, {'item': '作者', 'offset': 21, 'wordtag_label': '人物类_概念', 'length': 2}, {'item': '是', 'offset': 23, 'wordtag_label': '肯定词', 'length': 1}, {'item': '余兼羽', 'offset': 24, 'wordtag_label': '人物类_实体', 'length': 3}]}]

ner = Taskflow("ner", batch_size=2)
ner(["热梅茶是一道以梅子为主要原料制作的茶饮",
    "《孤女》是2010年九州出版社出版的小说，作者是余兼羽",
    "中山中环广场，位于广东省中山市东区，地址是东区兴政路1号",
    "宫之王是一款打发休闲时光的迷宫游戏"])
>>> [{'text': '热梅茶是一道以梅子为主要原料制作的茶饮', 'items': [{'item': '热梅茶', 'offset': 0, 'wordtag_label': '饮食类_饮品', 'length': 3}, {'item': '是', 'offset': 3, 'wordtag_label': '肯定词', 'length': 1}, {'item': '一道', 'offset': 4, 'wordtag_label': '数量词', 'length': 2}, {'item': '以', 'offset': 6, 'wordtag_label': '介词', 'length': 1}, {'item': '梅子', 'offset': 7, 'wordtag_label': '饮食类', 'length': 2}, {'item': '为', 'offset': 9, 'wordtag_label': '肯定词', 'length': 1}, {'item': '主要原料', 'offset': 10, 'wordtag_label': '物体类', 'length': 4}, {'item': '制作', 'offset': 14, 'wordtag_label': '场景事件', 'length': 2}, {'item': '的', 'offset': 16, 'wordtag_label': '助词', 'length': 1}, {'item': '茶饮', 'offset': 17, 'wordtag_label': '饮食类_饮品', 'length': 2}]}, {'text': '《孤女》是2010年九州出版社出版的小说，作者是余兼羽', 'items': [{'item': '《', 'offset': 0, 'wordtag_label': 'w', 'length': 1}, {'item': '孤女', 'offset': 1, 'wordtag_label': '作品类_实体', 'length': 2}, {'item': '》', 'offset': 3, 'wordtag_label': 'w', 'length': 1}, {'item': '是', 'offset': 4, 'wordtag_label': '肯定词', 'length': 1}, {'item': '2010年', 'offset': 5, 'wordtag_label': '时间类', 'length': 5}, {'item': '九州出版社', 'offset': 10, 'wordtag_label': '组织机构类', 'length': 5}, {'item': '出版', 'offset': 15, 'wordtag_label': '场景事件', 'length': 2}, {'item': '的', 'offset': 17, 'wordtag_label': '助词', 'length': 1}, {'item': '小说', 'offset': 18, 'wordtag_label': '作品类_概念', 'length': 2}, {'item': '，', 'offset': 20, 'wordtag_label': 'w', 'length': 1}, {'item': '作者', 'offset': 21, 'wordtag_label': '人物类_概念', 'length': 2}, {'item': '是', 'offset': 23, 'wordtag_label': '肯定词', 'length': 1}, {'item': '余兼羽', 'offset': 24, 'wordtag_label': '人物类_实体', 'length': 3}]}, {'text': '中山中环广场，位于广东省中山市东区，地址是东区兴政路1号', 'items': [{'item': '中山中环广场', 'offset': 0, 'wordtag_label': '场所类', 'length': 6}, {'item': '，', 'offset': 6, 'wordtag_label': 'w', 'length': 1}, {'item': '位于', 'offset': 7, 'wordtag_label': '场景事件', 'length': 2}, {'item': '广东省', 'offset': 9, 'wordtag_label': '世界地区类', 'length': 3}, {'item': '中山市东', 'offset': 12, 'wordtag_label': '世界地区类', 'length': 4}, {'item': '区', 'offset': 16, 'wordtag_label': '词汇用语', 'length': 1}, {'item': '，', 'offset': 17, 'wordtag_label': 'w', 'length': 1}, {'item': '地址', 'offset': 18, 'wordtag_label': '场所类', 'length': 2}, {'item': '是', 'offset': 20, 'wordtag_label': '肯定词', 'length': 1}, {'item': '东区', 'offset': 21, 'wordtag_label': '位置方位', 'length': 2}, {'item': '兴政路1号', 'offset': 23, 'wordtag_label': '世界地区类', 'length': 5}]}, {'text': '宫之王是一款打发休闲时光的迷宫游戏', 'items': [{'item': '宫之王', 'offset': 0, 'wordtag_label': '人物类_实体', 'length': 3}, {'item': '是', 'offset': 3, 'wordtag_label': '肯定词', 'length': 1}, {'item': '一款', 'offset': 4, 'wordtag_label': '数量词', 'length': 2}, {'item': '打发', 'offset': 6, 'wordtag_label': '场景事件', 'length': 2}, {'item': '休闲', 'offset': 8, 'wordtag_label': '场景事件', 'length': 2}, {'item': '时光', 'offset': 10, 'wordtag_label': '时间类', 'length': 2}, {'item': '的', 'offset': 12, 'wordtag_label': '助词', 'length': 1}, {'item': '迷宫游戏', 'offset': 13, 'wordtag_label': '作品类_概念', 'length': 4}]}]
```

### 文本纠错

```python
from paddlenlp import Taskflow

text_correction = Taskflow("text_correction")
text_correction('遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。')
>>> [{'source': '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
    'target': '遇到逆境时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
    'errors': [{'position': 3, 'correction': {'竟': '境'}}]}
]

text_correction(['遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
                '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。'])
>>> [{'source': '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 
    'target': '遇到逆境时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 
    'errors': [{'position': 3, 'correction': {'竟': '境'}}]}, 
{'source': '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。', 
    'target': '人生就是如此，经过磨练才能让自己更加茁壮，才能使自己更加乐观。', 
    'errors': [{'position': 18, 'correction': {'拙': '茁'}}]}
]
```

### 句法分析

```python
from paddlenlp import Taskflow 

ddp = Taskflow("dependency_parsing")
ddp("百度是一家高科技公司")
>>> [{'word': ['百度', '是', '一家', '高科技', '公司'], 'head': ['2', '0', '5', '5', '2'], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']}]

ddp(["百度是一家高科技公司", "他送了一本书"])
>>> [{'word': ['百度', '是', '一家', '高科技', '公司'], 'head': ['2', '0', '5', '5', '2'], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']}, {'word': ['他', '送', '了', '一本', '书'], 'head': ['2', '0', '2', '5', '2'], 'deprel': ['SBV', 'HED', 'MT', 'ATT', 'VOB']}]

# 输出概率值和词性标签
ddp = Taskflow("dependency_parsing", prob=True, use_pos=True)
ddp("百度是一家高科技公司")
>>> [{'word': ['百度', '是', '一家', '高科技', '公司'], 'postag': ['ORG', 'v', 'm', 'n', 'n'], 'head': ['2', '0', '5', '5', '2'], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB'], 'prob': [1.0, 1.0, 1.0, 1.0, 1.0]}]

# 使用ddparser-ernie-1.0进行预测
ddp = Taskflow("dependency_parsing",model="ddparser-ernie-1.0")
ddp("百度是一家高科技公司")
>>> [{'word': ['百度', '是', '一家', '高科技', '公司'], 'head': ['2', '0', '5', '5', '2'], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']}]
```

### 情感分类

```python
from paddlenlp import Taskflow 

senta = Taskflow("sentiment_analysis")
senta("怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片")
>>> [{'text': '怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片', 'label': 'negative'}]

senta(["怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片", 
        "作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间"])
>>> [{'text': '怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片', 'label': 'negative'}, 
{'text': '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间', 'label': 'positive'}
]

# 使用SKEP情感分析预训练模型进行预测
senta = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch")
senta("作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。")
>>> [{'text': '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。', 'label': 'positive'}]
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