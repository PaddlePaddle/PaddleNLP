# 解语：TermTree（百科知识树）
TermTree（百科知识树）是一个描述所有中文词汇（包括概念、实体/专名、领域术语、语法词等，统一称之为 Term）的树状知识库，完整的 TermTree 由两部分构成：

> I. TermType 词类体系：覆盖所有中文词汇词类的树状知识体系，是对中文词汇集合的一种全划分层次表示；
>
> II. Term 关系和属性值：描述具体 Term 之间关系和 Term 属性值网状图谱，用于整合各应用知识图谱；

本次发布的 TermTreeV1.0试用版是 TermTree 的一个常用子集，包括两部分内容：

> A.  简化版的 TermType 词类体系，由160+ termtype（三层结构）和 7000+ subtype 构成。
>
> B.  约100w 的 term 集（挂接在 TermType 词类体系下），包括大多数常用概念（src=cb，基础概念库，termtype 准确率为98%）和一部分高频百科实体（src=eb，基础实体库，termtype 准确率为95%）。
>
> 开源版不包括 Term 关系和属性值，但给出了实体的百科词条链接，应用方可以利用百科链接整合其他知识图谱使用。

我们提供了 TermTreeV1.0试用版的下载链接供大家使用，[下载链接](https://kg-concept.bj.bcebos.com/TermTree/TermTree.V1.0.tar.gz) 。

**注：** 与其他常见应用知识图谱不同，TermTree 的核心是概念词，而非专名实体词。因为，在中文文本中，概念词的含义是相对稳定的，而专名实体词随应用变化（例如，不同电商有不同的商品实体集，不同的小说站有不同的小说实体集），因此，TermTree 通过 “提供常用概念集 + 可插拔的应用实体集/应用知识图谱” 来达到支持不同的应用适配。

## 自定义 TermTree

`termtree.py`文件中的 TermTree 类支持 TermTree 的加载、增加、保存操作，因为涉及到数据结构整体性和一致性，暂不支持删除和修改操作。下面提供了离线维护自定义 TermTree 的代码示例

### 文件准备

首先下载已有的 TermTreeV1.0
```shell
wget https://kg-concept.bj.bcebos.com/TermTree/TermTree.V1.0.tar.gz && tar -zxvf TermTree.V1.0.tar.gz
```

### TermTree 维护与修改

加载 TermTreeV1.0，增加新的 term
```python
from termtree import TermTree

# 加载百科知识树
termtree = TermTree.from_dir("termtree_type.csv", "TermTree.V1.0")

# 增加新term: 平原上的火焰
termtree.add_term(term="平原上的火焰",
                  base="eb",
                  term_type="影视作品")

# 保存修改, 执行后将在当前路径生成文件`termtree_data`，即新的自定义TermTree
termtree.save("./")
```

#### API 说明

- ```python
  def add_term()
  ```

- **参数**
 - term (str): 待增加的 term 名称。
 - base (str): term 属于概念词（cb）还是实体词（eb）。
 - term_type (str): term 的主类别。
 - sub_type (Optional[List[str]], optional): term 的辅助类别或细分类别，非必选。
 - sub_terms (Optional[List[str]], optional): 用于描述同类同名的 term 集，非必选。
 - alias (Optional[List[str]], optional): term 的常用别名，非必选。
 - alias_ext (Optional[List[str]], optional): term 的常用扩展别名，非必选。
 - data (Optional[Dict[str, Any]], optional): 以 dict 形式构造该 term 节点，非必选。


### 自定义 Term-Linking

Taskflow 支持使用自定义 TermTree 实现自定义 Term-Linking，该示例中"平原上的火焰"的 Term-Linking 如下:
作品类_实体(wordtag_label) -> 影视作品_eb_平原上的火焰(term_id)

通过`task_path`定义用户自定义路径，文件组成：
```text
custom_task_path/
├── termtree_type.csv
└── termtree_data
```

使用 Taskflow 加载自定义 TermTree 来进行预测：

```python
from paddlenlp import Taskflow

wordtag = Taskflow("knowledge_mining", task_path="./custom_task_path/")

wordtag("《平原上的火焰》是今年新上映的电影")
# [{'text': '《平原上的火焰》是今年新上映的电影', 'items': [{'item': '《', 'offset': 0, 'wordtag_label': 'w', 'length': 1}, {'item': '平原上的火焰', 'offset': 1, 'wordtag_label': '作品类_实体', 'length': 6, 'termid': '影视作品_eb_平原上的火焰'}, {'item': '》', 'offset': 7, 'wordtag_label': 'w', 'length': 1}, {'item': '是', 'offset': 8, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '今年', 'offset': 9, 'wordtag_label': '时间类', 'length': 2, 'termid': '时间阶段_cb_今年'}, {'item': '新', 'offset': 11, 'wordtag_label': '修饰词', 'length': 1, 'termid': '修饰词_cb_新'}, {'item': '上映', 'offset': 12, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_上映'}, {'item': '的', 'offset': 14, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '电影', 'offset': 15, 'wordtag_label': '作品类_概念', 'length': 2, 'termid': '影视作品_cb_电影'}]}]
```

## 常见问题

**常见问题1：为什么 TermTree 采用树状结构（Tree），而不是网状结构（Net/Graph）？**

- 树结构是对知识空间的全划分，网状结构是对相关关系的描述和提炼。树结构更方便做到对词类体系的全面描述。

- 树结构适合概念层次的泛化推理，网状结构适合相关性的泛化推理。树结构的知识对统计相关知识有很好的互补作用，在应用中能够更好地弥补统计模型的不足。
- 两者可以结合表示和使用：Term 集合整体以树结构组织（TermType 词类体系），Term 间的关系用网状结构描述（Term 关系和属性值）。可以将 TermTree 视为中文词汇的层次描述框架，应用知识图谱可以基于 TermType 词类体系方便地整合到 TermTree。

**常见问题2：为什么 TermTree 叫做百科知识树？是否只能用于描述百科知识？**

- 一方面，Term 可以泛指任意概念、实体/专名、领域术语、语法词等，用“百科”是为了表达 Term 的多样性，而不是限定 Term 的来源，Term 可以来自任意中文文本；
- 另一方面，各类别的词汇都可以在百科词条中找到样例，用“百科”也是为了表示对所有中文词汇词类的描述能力。

**常见问题3：中文词汇词类描述体系有很多，为什么采用这个体系？**

- TermTree 的词类体系是在大规模工业应用实践（如百科文本解析挖掘、query 理解）中打磨出来的中文词类体系，在理论上可能不是一个完备体系，但很适合通用领域中文解析挖掘任务。


## TermTree 字段说明

| 字段         | 说明                                                         | 备注                                                         |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| id           | 【必有】唯一标识符                                           | 可基于 termid 生成                                             |
| term         | 【必有】term 的名字                                           |                                                              |
| termid       | 【必有】term 的 id（唯一），构造方式为 termtype_src_term        | 采用显式构造 id 的方式，便于应用数据扩展和整合                 |
| src          | 【必有】term 的来源库，当前包括两个基础库 cb 和 eb。其中 cb 为基础概念库（concept base,收录常用词汇用语，可作为各类应用的基础集），eb 为基础实体库（entity base, 收录常见命名实体，可根据应用需求扩展） | cb、eb 的划分标准不同应用不一样，可根据需求调整；应用方也可以构造自己的应用库，与 cb、eb 整合使用。 |
| termtype     | 【必有】term 的主类别，详细描述参见 [termtree\_type](./termtree_type.csv) | 多上位的 term 会选择其中一个作为 termtype，其他上位作为 subtype，方便应用筛选 |
| subtype      | 【非必须】term 的辅助类别或细分类别                           | 如果应用特别关注某个 subtype，也可以将其升级为 termtype 使用（需要相应更新 termid 和 id） |
| subterms     | 【非必须】用于描述同类同名的 term 集，若“termtype+src”下 term 只对应一个实例，则 subterms 为空；若“termtype+src”下 term 对应多个实例，则 subterms 记录这些实例，其字段与 term 相同 | 不需要区分 subterm 的两种常见场景：1. 应用只需词类特征；2. 上下文信息不足，无法区分具体实例 |
| subterms_num | 【非必须】subterms 中的 subterm 数量                            | 如果没有 subterm，则值为0                                     |
| alias        | 【非必须】term 的常用别名                                     | 通常为歧义小的别名                                           |
| alias\_ext   | 【非必须】term 的常用扩展别名，经常是 term 或 alias 的一个子片段，单独出现有其他含义，结合上下文可识别为别名。 | 通常为歧义大的别名，便于应用筛选使用。e.g., 四维彩超的 alias_ext“四维” |
| links        | 【非必须】该 term 对应的其他 term 的 id，可以是本知识库中的 id，也可以是其他知识库如百度百科 id | 如果是本知识库中的 id，则表示两者可以指代同一实体             |

## 数据示例
```json
// 示例1：无subterms的term
{
    "id": "c472a6fe74eb2008c4e5b958a047eb5c",
    "termid": "植物_cb_苹果",
    "term": "苹果",
    "src": "cb",
    "termtype": "植物",
    "subtype": [],
    "subterms": [],
    "subterms_num": 0,
    "alias": [
        "苹果树"
    ],
    "alias_ext": [],
    "links": [
        {
            "bdbkUrl": [
                "http://baike.baidu.com/item/%E8%8B%B9%E6%9E%9C/14822460"
            ]
        }
    ]
}

// 示例2：有subterms的term
{
    "id": "824716062a4d74efc0897d676700a24e",
    "termid": "影视作品_eb_苹果",
    "term": "苹果",
    "src": "eb",
    "termtype": "影视作品",
    "subtype": [],
    "subterms": [
        {
            "id": "9bb5b38dc50233b1ccd28d1c33c37605",
            "subtype": [
                "影视作品_cb_电影",
                "影视动漫作品_cb_剧情片"
            ],
            "alias": [],
            "alias_ext": [],
            "links": [
                {
                    "bdbkUrl": [
                        "http://baike.baidu.com/item/%E8%8B%B9%E6%9E%9C/6011191"
                    ]
                }
            ]
        },
        {
            "id": "688dc07cc98f02cbd4d21e2700290590",
            "subtype": [
                "影视作品_cb_韩国电影"
            ],
            "alias": [],
            "alias_ext": [],
            "links": [
                {
                    "bdbkUrl": [
                        "http://baike.baidu.com/item/%E8%8B%B9%E6%9E%9C/6011208"
                    ]
                }
            ]
        },
        {
            "id": "bbf4abe6ac412b181eac383333ca9fef",
            "subtype": [
                "影视作品_cb_剧情电影"
            ],
            "alias": [],
            "alias_ext": [],
            "links": [
                {
                    "bdbkUrl": [
                        "http://baike.baidu.com/item/%E8%8B%B9%E6%9E%9C/6011176"
                    ]
                }
            ]
        }
    ],
    "subterms_num": 3,
    "alias": [],
    "alias_ext": [],
    "links": []
}
```

## TermTree 特点

 1. 将所有中文词汇放在一个统一类别体系下表示，包括**概念、实体/专名、领域术语、语法词**。
- 解决传统标注技术下（e.g., 词性标注、命名实体识别），概念、实体、词性特征难以统一计算的问题。

 2. 为中文精准解析挖掘服务的词汇类别体系，以全面覆盖**百科词条、搜索 query、新闻资讯**中出现的中文词汇为目标，支持通用场景文本理解。
 - 应用可以通过指定词表的 TermType，方便地整合到 TermTree 中，定制应用特化版。

 3. 尽可能收录常用概念词，并区分常用概念词（src=cb）和专名实体词（src=eb），以解决专名实体与概念在计算中容易混淆的问题。为此，特别补充收录了很多百科中缺少的概念词。
 - 例：“琴房（歌曲类实体）” VS. “琴房（区域场所类概念）”
 - 例：“甩掉（歌曲类实体）” VS. “甩掉（场景事件类概念）”

 4. 将同类同名实体拆分为 term 和 subterm 两层（参见数据示例），term 作为给定 termtype 下所有同名实体的表示，subterm 作为同类同名实体集中每一个具体实体的表示：
 - 一方面解决文本中信息不足无法区分具体实体时的标注问题；
 - 一方面减少同名词汇的消歧计算代价（只需要计算同类下的同名实体，有效解决概念词和实体词识别混淆的问题）

 5. 为重要的概念/实体构建完整上位归类路径（**注：** TermTreeV1.0试用版暂不包括），用于细粒度特征计算和知识推断，参见以下示例

    | term | 类别| src| 上位归类路径示例 |
    |---|---|---|---|
    |苹果 | 植物类|cb|苹果 → 苹果属 → 蔷薇科 → 蔷薇目 → 双子叶植物纲 → 被子植物门 → 种子植物 → 植物界 → 真核生物域 → 生物|
    | 黄香蕉苹果| 饮食类|cb|黄香蕉苹果 →苹果 →水果 → 蔬果和菌藻类 →食材 →食物 →饮食|
    |甲型流感 | 疾病类|cb|甲型流感 → 流行性感冒 → 感冒 → 呼吸道感染 → 呼吸系统疾病 → 疾病损伤 → 生物疾病|
    |甲型流感病毒| 微生物类|cb|甲型流感病毒 → 流行性感冒病毒 → 正粘病毒科 → RNA 病毒 → 生物病毒 → 病原微生物 → 微生物 → 生物|
    |琴房| 区域场所类|cb|琴房 → 音乐室 → 活动室 →活动场所 →区域场所|
    |琴房| 音乐类|eb|琴房 → 歌曲 →音乐作品 →艺术作品 →作品 → 作品与出版物|
    |认同感 | 生活用语类|cb|认同感 →正面感受 → 感受 → 知觉感受 → 个体描述  → 生活用语|
    | 认同感| 图书类|eb|认同感 →书籍 →图书 →书刊 →出版物 → 作品与出版物|
    |佛罗伦萨足球俱乐部| 体育组织机构|eb|佛罗伦萨足球俱乐部 →意大利足球联赛球队→职业足球俱乐部→足球俱乐部 →足球队 →球队 →运动队 →体育组织机构 →组织机构|
    |佛罗伦萨市 | 世界地区类|cb|佛罗伦萨市 →托斯卡纳大区 →意大利 →南欧 →欧洲 →地球区域 →世界地区|
    |言情小说 | 小说类|cb|言情小说 →情感小说 →小说 →文学作品 →作品 →作品与出版物|
    | 言情小说| 音乐类|eb|言情小说 → 歌曲 →音乐作品 →艺术作品 →作品 → 作品与出版物|
> **注：** TermType 词类体系可视为所有上位归类路径的集合。

## TermTree 应用方式

1. 直接作为词表使用，利用 termtype 和 subtype 筛选应用所需的词表（停用词表、黑白名单、概念扩展词表等）。
2. 结合中文文本知识标注工具（WordTag 等）使用，用于文本词类特征生成、挖掘/解析 pattern 生成、样本构建和优化等等，参见"[解语的应用场景](../)"。
3. 整合应用知识图谱，为应用知识图谱提供通用词汇知识补充。

## TermTree 后续规划

1. 数据覆盖扩展到全量百度百科词条，提升 TermType 归类准确率，便于应用方筛选构建应用适配的 TermTree；
2. 建立知识共建社区，支持用户提交自己的 term 词表，生成定制版 TermTree。


## 在论文中引用 TermTree
如果您的工作成果中使用了 TermTree，请增加下述引用。我们非常乐于看到 TermTree 对您的工作带来帮助。
```
@article{zhao2020TermTree,
    title={TermTree and Knowledge Annotation Framework for Chinese Language Understanding},
    author={Zhao, Min and Qin, Huapeng and Zhang, Guoxin and Lyu, Yajuan and Zhu, Yong},
    technical report={Baidu, Inc. TR:2020-KG-TermTree},
    year={2020}
}
```

## 问题与反馈

百科知识树在持续扩充优化中，如果您有任何建议或发现数据问题，欢迎提交 issue 到 Github。
