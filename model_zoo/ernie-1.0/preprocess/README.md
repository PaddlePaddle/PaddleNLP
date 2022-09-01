# PaddleNLP 预训练数据流程

本示例致力于打造基于PaddleNLP预训练模型的最佳实践。预训练全部流程的整体详细介绍文档，请参考[ERNIE 中文预训练介绍](../pretraining_introduction.md)。本文档主要介绍预训练数据流程。


我们将预训练数据过程划分为以下部分

- 原始数据转换，原始文本转换为jsonl的json字符串格式。
- 数据ID化，断句、分词、tokenize转化为token id格式。
- 训练index文件生成，生成train、valid、test的每个样本索引。
- token动态mask(可选)，python 层实时mask文本。

本目录下主要包含一下文件：
```
├── create_pretraining_data.py
├── dataset_utils.py
├── ernie_dataset.py
├── helpers.cpp
├── Makefile
├── README.md
└── trans_to_json.py

```
其中，`trans_to_json.py`是原始数据转化的脚本，将数据转化为json串格式。
`create_pretraining_data.py`将jsonl文本，断句、分词后，tokenizer转化为token id。
`dataset_utils.py`中包含了index生成、动态mask的实现。
`ernie_dataset.py`通过调用`dataset_utils.py`的一些函数，产生ernie的输入dataset。


### 环境依赖

 - tqdm
 - numpy
 - pybind11
 - tool_helpers
 - lac (可选)
 - zstandard (可选)

安装命令`pip install tqdm numpy pybind11 tool_helpers lac zstandard`。另，部分功能需要`g++>=4.8`编译支持


## 训练全流程数据Pipeline

飞桨是自主研发、功能完备、开源开放的产业级深度学习平台，集深度学习核心训练和推理框架、基础模型库、端到端开发套件和丰富的工具组件于一体

|步骤|阶段&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|数据格式| 样例|
|-|-|-|-|
| 0️⃣初始状态 | -|原始数据： <br/> **每个doc之间用空行间隔开** <br/> - 中文，默认每句换行符，作为句子结束。<br/> - 英文，默认使用nltk判断句子结束  | ```飞桨是功能完备、开源开放的产业级深度学习平台。``` <br/> ```飞桨拥有核心训练和推理框架、基础模型库。``` <br/><br/> ```PaddleNLP是自然语言处理领域的优秀工具。```  |
|1️⃣原始数据转换<br/>`trans_to_json.py`|预处理 <br>输入：0️⃣初始状态 <br>输出：jsonl|jsonl格式：每个doc对应一行json字符串| ```{"text": "飞桨是功能完备、开源开放的产业级深度学习平台。飞桨拥有..."}```<br/>```{"text": "PaddleNLP是自然语言..."}```
|❇️(**可选**)数据中文分词<br/>`words_segmentation.py`|语料分词：中文WWM <br>输入：jsonl  <br> 输出：0️⃣初始状态| 将jsonl格式的数据，恢复成分词后的原始格式数据 <br> | ```飞桨 是 功能 完备、开源 开放的 产业级 深度学习 平台。``` <br/> ```飞桨 拥有 核心 训练和推理 框架、基础 模型库。``` <br/><br/> ```PaddleNLP 是 自然语言处理领域 的 优秀工具。```
|2️⃣数据ID化<br/>`create_pretrain_data.py`|预处理| npy格式：数据id化后的token id <br/>npz格式：数据句子、文章位置索引 | -
|3️⃣训练index文件生成|训练启动|npy格式：<br/> 根据训练步数max_steps生成<br/>train、valid、test的每个样本索引文件| -
|4️⃣token动态mask（可选）| Dataset取数据 | 无 |-


注意：
- **❇️(**可选**)数据中文分词** 是中文预训练做 WWM 的可选步骤
  - 当你的数据比较少时，分词耗时较少，不需要词步骤。直接在`create_pretrain_data.py`步骤中分词即可。
  - 目的是为了提前分词，加快后续数据ID转化步骤。
  - 如果这里输入的是 jsonl格式文件，最好为多文件，`trans_to_json.py` 时候开启`no-merge`选项。
  - 当你的数据集比较大，或者需要尝试多次转换数据的时候，提前分词可以避免`create_pretrain_data.py`时每次都运行一次分词程序。
- 转换后，需要重新 进行步骤 1️⃣`原始数据转换 trans_to_json.py`，最后2️⃣`数据ID化`步骤设置`--cn_splited=True`参数。
- 2️⃣`数据ID化`也可以在转化ID的同时，一起实现分词。不需要❇️`数据中文分词`步骤。


## 数据教程汇总

针对目前开源的数据集，PaddleNLP提供了详细的数据教程，点击对应数据集的链接，即可开始进行数据制作：

| 名称 | 文本类型 | 纯文本大小 | 适配模型
|-|-|-|-|
| [CLUECorpusSmall](./docs/CLUECorpusSmall.md)| 中文 | 14GB | ERNIE
| [OpenWebText2](./docs/OpenWebText2.md) | 英文 | 70GB | GPT
| [WuDaoCorpus2.0 Base](./docs/WuDaoCorpusBase.md)| 中文 |  200GB | ERNIE
| [CLUECorpus2020](./docs/CLUECorpus2020.md)| 中文 | 200GB | ERNIE

## ERNIE预训练详细准备

下面以ERNIE预训练为例，简要介绍一下预训练的全流程。

### 原始数据
首先下载样例数据：
```
mkdir data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/data_tools/baike.txt
cd ..
```

### 原始数据转换 jsonl 格式
使用`trans_to_json.py`转化为json串格式，下面是脚本的使用说明
```
optional arguments:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        Path to you raw files. Folder or file path.
                        必须设置，可以是文件夹或者单个文件。文件夹中的目录默认最多搜索两层子目录。
  --output_path OUTPUT_PATH
                        Path to save the output json files.
                        必须设置，输出文件的名字。
  --json_key JSON_KEY   The content key of json file.
                        建议不修改，默认的key是text
  --doc_spliter DOC_SPLITER
                        Spliter between documents. We will strip the line, if you use blank line to split doc, leave it blank.
                        根据实际情况修改，默认空行作为文章换行符。
  --min_doc_length MIN_DOC_LENGTH
                        Minimal char of a documment.
                        可选。过滤掉长度多短的文章，默认值10
  --workers WORKERS     Number of worker processes to launch
                        可选。多进程转化文件，适用于 input_path 中包含的文件数据较多的情况。每个文件，分配给不同worker处理
  --log_interval LOG_INTERVAL
                        Interval between progress updates.
                        可选。此处的interval是值处理完文件个数的间隔。
  --no-merge            Don't merge the file.
                        可选。默认不开启这个选项，默认每个文件转换的jsonl文本，会拼接成到同一个文件。
  --no-shuffle          Don't shuffle the file.
                        可选。默认不开启这个选项，默认对处理完进行shuffle。
```
根据说明，我们使用下面简单命令，可以得到`baike_sample.jsonl`文件。此处，我们对文章所有doc进行了shuffle。
```shell
python trans_to_json.py  --input_path ./data --output_path baike_sample

#查看数据
head -1 baike_sample.jsonl
{"text": "中国效仿西方发展工业的过程，于中华民国国民政府成立后至中日战争开战前夕已顺畅发展，尽管其间受到内外因素的多重干扰。尔后直至中日战争和国共战争的结束，
中国始有较为长期的和平发展时期。\n1980年代以来，邓小平政府宣布改革开放，开始实行社会主义市场经济并推行经济体制改革。中国大陆近年至2010年，GDP超过72000亿美元，
已经成为美国之后的世界第二经济大国，普遍认为中国是世界上发展速度最快的经济体，但是人均国民生产总值仍位于世界中等水平（第89位），并逐渐受到资源限制和贫富差距加
大的制约。中华人民共和国省份中，广东为GDP最高的第一强省，浙江为人均收入最高的第一富省。中国大陆、香港、澳门、台湾之间的经济联系在全球化的过程中日益紧密。\n"}
```

### 数据ID化
本部分，我们使用 `create_pretraining_data.py` 脚本将前面得到的 `baike_sample.jsonl` 进行tokenize id化处理。
```
optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        What model to use.
                        必须设置，如：ernie-1.0-base-zh, 可以参考已有的模型名称 https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer
  --tokenizer_name {ErnieTokenizer,BertTokenizer,GPTTokenizer,GPTChineseTokenizer}
                        What type of tokenizer to use.
                        模型对应的tokenizer, 目前暂时只支持 ERNIE，BERT，GPT
data input/output:
  --input_path INPUT_PATH
                        Path to input JSON files.
                        必须设置，输入文件jsonl的目录
  --output_prefix OUTPUT_PREFIX
                        Output prefix to store output file.
                        必须设置，输出文件的名称。
                        假设名称为XXX，则会输出 XXX_ids.npy, XXX_idx.npz 两个文件。
                        npy文件，数据id化后的token ids; npz文件，数据句子、文章位置索引。
  --data_format {JSON}  Only support json format for now. One document per line.
                        不需要设置。目前默认处理jsonl数据格式
  --json_key JSON_KEY   For JSON format. Space separate listed of keys to extract from json
                        文本串json的key值。同前面trans_to_json.py的json_key，默认text为key
  --split_sentences     Split documents into sentences.
                        是否需要将文章划分成句子。一般而言，GPT不需要，BERT/ERNIE模型需要

chinese words:
  --chinese             Is corpus need words segmentation step for chinese words.
                        中文情形必须设置。处理的文本类型是否是中文。
  --cn_whole_word_segment
                        Is corpus need words segmentation step for chinese words WWM.
                        可选。是否需要WWM策略。一般而言，BERT/ERNIE模型需要，GPT不需要。
  --cn_seg_func {lac,seg,jieba}
                        Words segment function for chinese words.
                        默认jieba，jieba速度较快，lac模型更准确，计算量高。
  --cn_splited          Is chinese corpus is splited in to words.
                        分词后的文本，可选。设置此选项则，cn_seg_func不起作用。
                        例如分词后文本串 "中国 效仿 西方 发展 工业 的过 程"
  --cn_split_dimer CN_SPLIT_DIMER
                        Split dimer between chinese words.
                        配合cn_splited使用，默认空格表示分词间隔。

common config:
  --append_eos          Append an <eos> token to the end of a document.
                        gpt模型专用，gpt设置此选项，表示doc结束。
  --log_interval LOG_INTERVAL
                        Interval between progress updates
                        打印日志间隔，interval表示处理 文本行数/doc数的 间隔。
  --workers WORKERS     Number of worker processes to launch
                        处理文本id化的进程个数。
```
通过下面脚本转化，我们可以得到处理好的预训练数据，token ids:`baike_sample_ids.npy`, 文章索引信息`baike_sample_idx.npz`.
```
python -u  create_pretraining_data.py \
    --model_name ernie-1.0-base-zh \
    --tokenizer_name ErnieTokenizer \
    --input_path baike_sample.jsonl \
    --split_sentences\
    --chinese \
    --cn_whole_word_segment \
    --output_prefix baike_sample  \
    --workers 1 \
    --log_interval 5
```
1. 如果您使用已经分好词的语料，可以设置 --cn_splited 为 True，同时指定--cn_split_dimer如空格。
2. 使用自定义词表的话，请指定model_name为词表所在的文件夹地址。


### ERNIE 预训练开始
得到了处理好的训练数据，就可以开始ERNIE模型的预训练了。ERNIE预训练的代码在`model_zoo/ernie-1.0`。
简单将预处理好的数据，拷贝到data目录，即可开始ERNIE模型预训练。
```
mkdir data
mv ./preprocess/baike_sample* ./data
sh run_static.sh
# 建议修改 run_static.sh 中的配置，将max_steps设置小一些。
```
代码说明：

- ernie预训练使用的 dataset 代码文件在 `./data_tools/ernie_dataset.py`
- 数据集index生成，动态mask相关代码实现在`./data_tools/dataset_utils.py`

用户可以根据自己的需求，灵活修改mask方式。具体可以参考`dataset_utils.py`中`create_masked_lm_predictions`函数。
可以自定义的选项有do_whole_word_mask, favor_longer_ngram, do_permutation, geometric_dist等，
可以参考[Megatron](https://github.com/NVIDIA/Megatron-LM)使用这些lm_mask策略。

### FAQ

#### C++代码编译失败怎么办？
- 请先检查pybind11包是否安装，g++、make工具是否正常。
- 编译失败可能是本文件夹下的Makefile命令出现了一些问题。可以将Makefile中的python3、python3-config设置成完全的路径，如/usr/bin/python3.7。

## 参考内容

注: 大部分数据流程，参考自[Megatron](https://github.com/NVIDIA/Megatron-LM)，特此表达感谢。
