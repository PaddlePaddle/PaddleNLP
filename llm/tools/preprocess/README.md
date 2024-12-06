# PaddleNLP 预训练数据流程

本示例致力于打造基于 PaddleNLP 预训练模型的最佳实践。


我们将预训练数据过程划分为以下部分

- 原始数据转换，原始文本转换为 jsonl 的 json 字符串格式。
- 数据 ID 化，断句、分词、tokenize 转化为 token id 格式。
- 训练 index 文件生成，生成 train、valid、test 的每个样本索引。
- token 动态 mask(可选)，python 层实时 mask 文本。

本目录下主要包含一下文件：
```
├── create_pretraining_data.py
├── merge.py
├── trans_to_json.py
├── words_segmentation.py
└── README.md
```

### 环境依赖

 - tqdm
 - numpy
 - pybind11
 - fast_dataindex
 - lac (可选)
 - zstandard (可选)

安装命令`pip install tqdm numpy pybind11 fast_dataindex lac zstandard`。另，部分功能需要`g++>=4.8`编译支持


## 训练全流程数据 Pipeline

飞桨是自主研发、功能完备、开源开放的产业级深度学习平台，集深度学习核心训练和推理框架、基础模型库、端到端开发套件和丰富的工具组件于一体

| 步骤                                                | 阶段&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | 数据格式                                                                                                                        | 样例                                                                                                                                                                               |
|-----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0️⃣初始状态                                           | -                                                                                                                                  | 原始数据： <br/> **每个 doc 之间用空行间隔开** <br/> - 中文，默认每句换行符，作为句子结束。<br/> - 英文，默认使用 nltk 判断句子结束 | ```飞桨是功能完备、开源开放的产业级深度学习平台。``` <br/> ```飞桨拥有核心训练和推理框架、基础模型库。``` <br/><br/> ```PaddleNLP是自然语言处理领域的优秀工具。```                 |
| 1️⃣原始数据转换<br/>`trans_to_json.py`                | 预处理 <br>输入：0️⃣初始状态 <br>输出：jsonl                                                                                         | jsonl 格式：每个 doc 对应一行 json 字符串                                                                                            | ```{"text": "飞桨是功能完备、开源开放的产业级深度学习平台。飞桨拥有..."}```<br/>```{"text": "PaddleNLP是自然语言..."}```                                                           |
| ❇️(**可选**)数据中文分词<br/>`words_segmentation.py` | 语料分词：中文 WWM <br>输入：jsonl  <br> 输出：0️⃣初始状态                                                                            | 将 jsonl 格式的数据，恢复成分词后的原始格式数据 <br>                                                                              | ```飞桨 是 功能 完备、开源 开放的 产业级 深度学习 平台。``` <br/> ```飞桨 拥有 核心 训练和推理 框架、基础 模型库。``` <br/><br/> ```PaddleNLP 是 自然语言处理领域 的 优秀工具。``` |
| 2️⃣数据 ID 化<br/>`create_pretrain_data.py`             | 预处理                                                                                                                             | bin 格式：数据 id 化后的 token id <br/>idx 格式：数据句子、文章位置索引                                                              | -                                                                                                                                                                                  |
| 3️⃣训练 index 文件生成                                  | 训练启动                                                                                                                           | npy 格式：<br/> 根据训练步数 max_steps 生成<br/>train、valid、test 的每个样本索引文件                                               | -                                                                                                                                                                                  |
| 4️⃣token 动态 mask（可选）                              | Dataset 取数据                                                                                                                      | 无                                                                                                                              | -                                                                                                                                                                                  |


注意：
- **❇️(**可选**)数据中文分词** 是中文预训练做 WWM 的可选步骤
  - 当你的数据比较少时，分词耗时较少，不需要分词步骤。直接在`create_pretrain_data.py`步骤中分词即可。
  - 目的是为了提前分词，加快后续数据 ID 转化步骤。
  - 如果这里输入的是 jsonl 格式文件，最好为多文件，`trans_to_json.py` 时候开启`no-merge`选项。
  - 当你的数据集比较大，或者需要尝试多次转换数据的时候，提前分词可以避免`create_pretrain_data.py`时每次都运行一次分词程序。
- 转换后，需要重新进行步骤 1️⃣`原始数据转换 trans_to_json.py`，最后2️⃣`数据 ID 化`步骤设置`--cn_splited=True`参数。
- 2️⃣`数据 ID 化`也可以在转化 ID 的同时，一起实现分词。不需要❇️`数据中文分词`步骤。


## 数据教程汇总

针对目前开源的数据集，PaddleNLP 提供了详细的数据教程，点击对应数据集的链接，即可开始进行数据制作：

| 名称                                             | 文本类型 | 纯文本大小 | 适配模型 |
|--------------------------------------------------|----------|------------|----------|
| [CLUECorpusSmall](./docs/CLUECorpusSmall.md)     | 中文     | 14GB       | Llama    |
| [OpenWebText2](./docs/OpenWebText2.md)           | 英文     | 70GB       | Llama    |
| [WuDaoCorpus2.0 Base](./docs/WuDaoCorpusBase.md) | 中文     | 200GB      | Llama    |
| [CLUECorpus2020](./docs/CLUECorpus2020.md)       | 中文     | 200GB      | Llama    |

## 预训练详细准备

下面以 ziya-llama-13b-v1预训练为例，简要介绍一下预训练的全流程。

### 原始数据
首先下载样例数据：
```
mkdir data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/data_tools/baike.txt
cd ..
```

### 原始数据转换 jsonl 格式
使用`trans_to_json.py`转化为 json 串格式，下面是脚本的使用说明
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
根据说明，我们使用下面简单命令，可以得到`baike_sample.jsonl`文件。此处，我们对文章所有 doc 进行了 shuffle。
```shell
python trans_to_json.py  --input_path ./data --output_path baike_sample
```

```shell
#查看数据
head -1 baike_sample.jsonl
{"text": "中国效仿西方发展工业的过程，于中华民国国民政府成立后至中日战争开战前夕已顺畅发展，尽管其间受到内外因素的多重干扰。尔后直至中日战争和国共战争的结束，
中国始有较为长期的和平发展时期。\n1980年代以来，邓小平政府宣布改革开放，开始实行社会主义市场经济并推行经济体制改革。中国大陆近年至2010年，GDP超过72000亿美元，
已经成为美国之后的世界第二经济大国，普遍认为中国是世界上发展速度最快的经济体，但是人均国民生产总值仍位于世界中等水平（第89位），并逐渐受到资源限制和贫富差距加
大的制约。中华人民共和国省份中，广东为GDP最高的第一强省，浙江为人均收入最高的第一富省。中国大陆、香港、澳门、台湾之间的经济联系在全球化的过程中日益紧密。\n"}
```

### 数据 ID 化
本部分，我们使用 `create_pretraining_data.py` 脚本将前面得到的 `baike_sample.jsonl` 进行 tokenize id 化处理。
```
optional arguments:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        What model to use.
                        必须设置，如：idea-ccnl/ziya-llama-13b-v1, 可以参考已有的模型名称 https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm
data input/output:
  --input_path INPUT_PATH
                        Path to input JSON files.
                        必须设置，输入文件jsonl的目录
  --output_prefix OUTPUT_PREFIX
                        Output prefix to store output file.
                        必须设置，输出文件的名称。
                        假设名称为XXX，则会输出 XXX.bin, XXX.idx 两个文件。
                        bin文件，数据id化后的token ids; idx文件，数据句子、文章位置索引。
  --data_format {JSON}  Only support json format for now. One document per line.
                        不需要设置。目前默认处理jsonl数据格式
  --json_key JSON_KEY   For JSON format. Space separate listed of keys to extract from json
                        文本串json的key值。同前面trans_to_json.py的json_key，默认text为key
  --split_sentences     Split documents into sentences.
                        是否需要将文章划分成句子。一般而言，GPT不需要，BERT/ERNIE模型需要
  --data_impl {mmap,lazy}
                        Convert the json into mmap/lazy format.
                        处理后的数据格式，可选“mmap”或“lazy”，其中“mmap”格式在读入数据时会建立内存映射，“lazy”格式在读入数据时直接从文件读取。

chinese words:
  --chinese             Is corpus need words segmentation step for chinese words.
                        若设置了split_sentences，并处理中文则需要设置。
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
                        gpt类模型专用，gpt设置此选项，表示doc结束。针对tokenier中不包含eos_token情况，输出提示warning并且不添加<eos>。
  --log_interval LOG_INTERVAL
                        Interval between progress updates
                        打印日志间隔，interval表示处理 文本行数/doc数的 间隔。
  --workers WORKERS     Number of worker processes to launch
                        处理文本id化的进程个数。
  --max_repeated_len    Max length of repeated chars to keep
                        最大保留重复的字符个数。
```
通过下面脚本转化，我们可以得到处理好的预训练数据，token ids:`baike_sample.bin`, 文章索引信息`baike_sample.idx`.

* 针对 llama 模型
```shell
python -u  create_pretraining_data.py \
    --model_name_or_path "idea-ccnl/ziya-llama-13b-v1" \
    --input_path "baike_sample.jsonl" \
    --output_prefix "baike_sample"  \
    --data_format "JSON" \
    --json_key "text" \
    --data_impl "mmap" \
    --append_eos \
    --log_interval 5 \
    --workers 40

```

* 针对 ernie 模型
```shell
python -u  create_pretraining_data.py \
    --model_name_or_path "ernie-3.0-base-zh" \
    --input_path "baike_sample.jsonl" \
    --output_prefix "baike_sample"  \
    --data_format "JSON" \
    --json_key "text" \
    --split_sentences \
    --data_impl "mmap" \
    --chinese \
    --cn_whole_word_segment \
    --cn_seg_func "jieba" \
    --log_interval 5 \
    --workers 40
```
1. 如果您使用已经分好词的语料，可以设置 --cn_splited 为 True，同时指定--cn_split_dimer 如空格。
2. 使用自定义词表的话，请指定 model_name 为词表所在的文件夹地址。

若需要预处理的文件过大，该脚本所耗费的时间可能会很长。此时可以考虑将 jsonl 文件拆分为多个小文件，并行使用 create_pretraining_data.py 进行处理，得到多个.bin & .idx 文件。
之后使用如下 merge 脚本合并多个小的.bin & .idx 文件。
```
python merge.py \
    --input /root/data \
    --output-prefix /root/data/merged \
    --data_impl mmap
```
使用说明：
```
arguments:
  --input INPUT_PATH
                        Path to the folder where the files to be merged.
                        待合并的文件所在文件夹，文件夹内各个小文件需按merge的顺序排列，如1.bin / 1.idx，2.bin / 2.idx...
  --output_prefix OUTPUT_PREFIX
                        Output prefix to store output file.
                        合并后输出文件的名称，假设名称为XXX，则会输出 XXX.bin, XXX.idx 两个文件。
  --data_impl {mmap,lazy}
                        Convert the json into mmap/lazy format.
                        merge前后的数据格式，可选“mmap”或“lazy，各个待merge的文件需格式一致。”。
```

### 预训练开始
得到了处理好的训练数据，就可以开始模型的预训练了。简单将预处理好的数据，拷贝到 data 目录，即可开始预训练。
```shell
mkdir data
mv ./preprocess/baike_sample* ./data
```

* llama 预训练请参考[预训练](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm)。
* ernie 预训练请参考[预训练](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/slm/model_zoo/ernie-1.0/pretraining_introduction.md)。


代码说明：
- 动态 mask 相关代码实现在`./data_tools/dataset_utils.py`
  用户可以根据自己的需求，灵活修改 mask 方式。具体可以参考`dataset_utils.py`中`create_masked_lm_predictions`函数。
  可以自定义的选项有 do_whole_word_mask, favor_longer_ngram, do_permutation, geometric_dist 等，
  可以参考[Megatron](https://github.com/NVIDIA/Megatron-LM)使用这些 lm_mask 策略。

## 参考内容

注: 大部分数据流程，参考自[Megatron](https://github.com/NVIDIA/Megatron-LM)，特此表达感谢。
