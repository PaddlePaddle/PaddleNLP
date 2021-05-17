# Transformers模型预训练数据准备

为了满足各垂类领域（金融、医疗等）继续预训练Tranformers模型，本教程给出如何在垂类领域构建预训练数据，继续预训练模型（continual pretraining），支持BERT、BERT-WWM、ERNIE等预训练模型在垂类数据上继续预训练。

为了完成继续预训练，需要完成以下几个步骤：

1. 数据格式化
2. Mask Language Model （MLM）数据构建
3. 开始预训练


## 代码结构说明

以下是本项目主要代码结构及说明：

```text
pretraining_data_prepare/
├── create_pretraining_data.py # MLM数据构建
├── main.py # 程序入口文件
├── README.md # 文档说明
├── requirements.txt # 第三方依赖
├── text_formatting # 数据格式化
│   ├── bookcorpus.py # bookcorpus数据格式化脚本
│   └── wikicorpus.py # 中英文维基百科数据格式化脚本
└── text_sharding.py # 数据切片
```

## 环境依赖

本教程除了需要安装PaddleNLP库，还需以下依赖

```text
datasets
opencc-python-reimplemented
nltk
LAC
tqdm
```

通过`pip install -r requirements.txt`即可完成依赖安装。

## 快速开始

### 数据格式化

首先，我们需要将原始数据进行格式化，整理成一行一篇文章的形式，并且文章之间以空行进行分隔。

如以下示例展示了两篇文章：

```text
班距又称班次，是指巴士、铁路、渡轮等公共交通系统计算服务时间的一种方式。每一班次相隔的时间，可以分钟、小时为单位。 由于巴士受路面因素影响较多，故为了确保班次>准确，部份路线会设立定时点，巴士停站后需等待到某个特定时间才开出。同样的，巴士也有可能晚点，导致乘客不便。 著名公共交通规划师杰里特-沃克（Jarrett Walker）说过：“短的班距就是自由。”一条公交线路的班距如果在10分钟以下，晚点的影响也不会特别大。  

标准摩尔熵（）指在标准状况（298.15 K，105Pa）下，1摩尔纯物质的规定熵，通常用符号"S°"表示，单位是J/(mol·K)（或作J·mol−1·K−1，读作「焦耳每千克开尔文」）。与标准
摩尔生成焓不同，标准莫耳熵是绝对的。 计算. 热力学第三定律表明，纯物质在绝对零度下形成的完整晶体的熵为0，据此可以算出物质的绝对熵。 假想纯固体从0 K（绝对零度）开始加热到"T" K，绝对零度熵为"S""0" = 0，则"T" K时物质的绝对熵"S""T"与各温度下的"CP"有以下关系： 也就是说，从0 K加热到298.15 K（25℃）而其间不发生相变的固体，>标准摩尔熵"S°"可由下式求出： 但如果存在同质异像间的相变，还需要加上相变的熵变formula_3。 0 K至298.15 K之间存在相变，例如发生熔化的情况下，需要加上熔化熵formula_4： 而熔化后还发生汽化的情况下，除熔化熵外，还需要加上汽化熵formula_6： 常见物质的标准摩尔熵. 对于水溶液中的阴阳离子，由于通常是将阳离子和阴离子作为整体进行
测定的，所以单独的离子的标准摩尔熵是以氢离子的标准摩尔熵为0，在无限稀释状态的假想的1 mol kg−1的理想溶液计算的。
```

本教程中提供了中英文维基百科数据的格式化方法如**text_formatting/wikicorpus.py**脚本。如需在垂类数据上预训练，则需自定义数据格式化方法。


### Mask Language Model （MLM）数据构建

**create_pretraining_data.py**脚本包含了Basic Masking （BERT所采用的数据Masking方式）和 Whole Word Masking (WWM)两种数据Masking方法。

| Masking Strategy | Chinese Sentence |
| ---------------- | ---------------- |
| original sentence	| 2018年，谷歌提出BERT模型，使用语言模型来预测下一个词的概率。|
| + Chinese Word Segmentation | 2018年 ， 谷歌 提出 BERT 模型 ， 使用 语言 模型 来 预测 下 一个 词 的 概率 。|
| + BERT Tokenizer  | 2018 年 ，谷 歌 提 出 BERT 模 型 ， 使 用 语 言 模 型 来 预 测 下 一 个 词 的 概 率 。|
| + Basic Masking | 2018 年 ，谷 [MASK] 提 出 BERT 模 型 ，使 用 语 言 [MASK] 型 来 [MASK] 测 下 一 个 词 的 概 率 。 |
| + Whole Word Masking | 2018 年 ， [MASK] [MASK] 提 出 BERT 模 型 ， 使 用 语 言 [MASK] [MASK] 来 [MASK] [MASK] 下 一 个 词 的 概 率 。 |


通过以下命令即可完成，Transformer模型的预训练数据构建

```shell
python main.py --skip_formatting True \
    --formatted_file text.txt \
    --output_dir pretraining_data/ \
    --model_name ernie-1.0 \
    --n_train_shards 256 \
    --n_test_shards 1 \
    --fraction_test_set 0.1 \
    --max_seq_length 128 \
    --max_predictions_per_seq 20 \
    --masked_lm_prob 0.15  
```

以上参数表示：

* `skip_formatting`：是否跳过数据格式化阶段。若`skip_formatting=True`表示您已完成垂类数据格式化；若`skip_formatting=False`表示按照本教程格式化中英文维基百科数据。
* `formatted_file`：已完成数据格式化的文件地址。
    **NOTE：** 如果`skip_formatting=False`，则无需指定`formatted_file`。程序将会自动下载维基百科和BookCorpus数据作为预训练数据。
* `output_dir`：预训练数据构建存放目录。
* `model_name`：具体预训练模型名称，程序将会按照这种预训练模型完成预训练数据构造。可选'bert-base-uncased', 'bert-base-chinese', 'bert-wwm-chinese','ernie-1.0'。
* `n_train_shards`: 训练集数据分片数量。为了可以利用上分布式加速训练，我们需要将预训练数据切片。
* `n_test_shards`: 测试集数据分片数量。
* `fraction_test_set`: 训练集和测试集划分比例。
* `max_seq_length`: 文本序列最大长度。
* `max_predictions_per_seq`: 文本序列中预测Token的最大数量。
* `masked_lm_prob`: 文本序列被mask概率大小。

### 预训练开始

按照以上方法构建完毕预训练数据之后，我们可以按照相应的预训练模型开始垂类预训练。

如[BERT预训练](../bert#执行pre-training)完成继续预训练。
