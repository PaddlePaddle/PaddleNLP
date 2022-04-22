# CCKS 2022 通用信息抽取基线

信息抽取任务旨在根据特定的抽取需求从非结构化文本中自动抽取结构化信息。
其中，特定的抽取需求是指抽取任务中的抽取框架，主要由抽取类别（人物名称、公司名称、企业上市事件）及目标结构（实体、关系、事件等）组成。
本任务为中文信息抽取任务，即按照特定的抽取框架S，从给定的一组自由文本X中抽取出所有符合抽取需求的信息结构Y（实体、关系、事件记录等）。
对于同一输入文本，不同的抽取框架会抽取不同的信息结构。

本例中包含四类抽取任务：实体抽取、关系抽取、事件抽取和情感抽取。
以“In 1997, Steve was excited to become the CEO of Apple.”为例，各个任务的目标结构为:

- 实体：Steve - 人物实体、Apple - 组织机构实体
- 关系：(Steve, Work For Apple)
- 事件：{类别: 就职事件, 触发词: become, 论元: [[雇主, Apple], [雇员, Steve]]}
- 情感：(exicted, become the CEO of Apple, Positive)

该示例展示了如何使用 PaddleNLP 快速构建 [CCKS 2022 通用信息抽取比赛](https://aistudio.baidu.com/aistudio/competition/detail/161/0/task-definition)基线，构建单个模型同时对上述四个任务进行抽取。

## 环境安装

``` bash
pip install -r requirements.txt
```

``` bash
pip install -r requirements.txt -t /home/aistudio/external-libraries
```

## 目录结构
``` text
$ tree -L 2
.
├── config              # 配置文件文件夹
│   ├── data_conf       # 不同任务数据集配置文件
│   ├── multitask       # 多任务配置文件
│   └── offset_map      # 生成的文本到边界的回标策略配置文件
├── inference.py        # 推理测试脚本，验证 Checkpoint 性能
├── output              # 训练完成模型的默认位置
├── README_CN.md        # 中文说明文件
├── README.md           # 英文说明文件
├── requirements.txt    # Python 依赖包文件
├── run_seq2seq_paddle.bash             # 单任务运行脚本
├── run_seq2seq_paddle_multitask.bash   # 多任务运行脚本
├── run_seq2seq_paddle_multitask.py     # 多任务 Python 入口
├── run_seq2seq_paddle.py               # 单任务 Python 入口
├── scripts             # 工具脚本
└── uie
    ├── extraction      # 信息抽取代码: 评分器等
    ├── sel2record      # 生成结构SEL到结构化抽取记录的转换器
    └── seq2seq_paddle  # 序列到序列代码: 数据收集器、切词器
```

## 通用信息抽取基线

### 快速基线第一步：数据预处理并加载

从比赛官网下载数据集，解压存放于 data/DuUIE 目录下，在原始数据中添加 Spot 和 Asoc 标注。

``` bash
python scripts/convert_to_spot_asoc.py
```

处理之后的数据将自动生成在同样放在 data/DuUIE_pre 下，每个实例中进一步添加了 `spot`, `asoc` 和 `spot_asoc` 三个字段。
- 在多任务样例中，spot/asoc 为每个任务中所有的 Spot/Asoc 类型，用于生成对应的 SSI。
- 在单任务样例中，spot/asoc 为每个实例中对应的 Spot/Asoc 类型，便于单任务内进行拒识噪声采样，单任务 SSI 参考 Schema 文件定义生成。

### 快速基线第二步：多任务模型训练并预测
#### 模型结构

本例中模型为统一结构生成模型，将多种不同抽取任务形式统一成通用的结构表示，并通过序列到结构生成模型进行统一建模。
具体而言，使用结构化抽取语言对不同的任务进行统一表示，使用结构化模式前缀来区分不同的抽取任务。

#### 结构化抽取语言
结构化抽取语言用于将不同的目标结构进行统一结构表示。结构化抽取语言的形式如下：
```
(
  (Spot Name: Info Span
    (Asso Name: Info Span)
    (Asso Name: Info Span)
  )
)
```
其中，
- Spot Name：信息点类别，如实体类型；
- Asso Name: 信息点关联列别，如关系类型、事件论元类型；
- Info Span：信息点所对应的文本片段。

以抽取`2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！`中信息结构为例：

- 该句中的国籍关系 SEL 表达式为：
```
(
  (人物: 谷爱凌
    (国家: 中国)
  )
)
```
- 该句中的夺冠事件 SEL 表达式为：
```
(
  (夺冠事件: 金牌
    (夺冠时间: 2月8号上午)
    (冠军: 谷爱凌)
    (夺冠赛事: 北京冬奥会自由式滑雪女子大跳台决赛)
  )
)
```

生成SEL表达式后，我们通过解析器将表达式解析成对应的抽取结果。

#### 结构化模式前缀
结构化模式前缀是在待抽取的文本前拼接上相应的结构化前缀，用于区分不同的抽取任务。
不同任务的的形式是：
- 实体抽取：[spot] 实体类别 [text]
- 关系抽取：[spot] 实体类别 [asso] 关系类别 [text]
- 事件抽取：[spot] 事件类别 [asso] 论元类别 [text]
- 情感抽取：[spot] 评价维度 [asso] 观点类别 [text]

基线采用的预训练模型为字符级别的中文生成模型 UIE-char-small，首先使用 100G 中文数据进行 Span Corruption 预训练，然后使用远距离监督产生的文本-结构数据进行结构生成预训练。

#### 多任务配置


本例中采用 Yaml 配置文件来配置不同任务的数据来源和验证方式
```
T1:
  name: duuie_company_info_relation                         # 任务名称
  task: record                                              # 结构化生成类型
  path: data/text2spotasoc/duuie/company_info_relation      # 数据文件夹位置
  decoding_format: spotasoc                                 # 目标统一编码格式
  sel2record: config/offset_map/longer_first_offset_zh.yaml # 生成文本到边界的回标策略配置
  eval_match_mode: set                                      # 模型匹配评价方式
  metrics:                                                  # 模型评价指标
    - string-rel-strict-F1

T2: ...
```

多任务的脚本如下：
```
bash run_seq2seq_paddle_multitask.bash
```

训练完成后，将生成对应的文件夹 `duuie_multi_task_b24_lr3e-4`

### 快速基线第三步：后处理提交结果
该步骤将按照实例的`id`将不同任务的预测进行合并，生成提交数据 `submit.txt`。

``` bash
python scripts/merge_to_submit.py --data data/duuie --model output/duuie_multi_task_b32_lr3e-4 --submit submit.txt
```

## Reference
- [Unified Structure Generation for Universal Information Extraction](https://arxiv.org/pdf/2203.12277.pdf)
- [DuIE: A Large-scale Chinese Dataset for Information Extraction](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction/DuIE)
- [DuEE: A Large-Scale Dataset for Chinese Event Extraction in Real-World Scenarios](https://link.springer.com/chapter/10.1007/978-3-030-60457-8_44)
- [CASA: Conversational Aspect Sentiment Analysis for Dialogue Understanding](https://jair.org/index.php/jair/article/view/12802)
