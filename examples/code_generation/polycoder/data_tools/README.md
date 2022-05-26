# PaddleNLP 预训练数据流程

本示例致力于打造基于PaddleNLP预训练模型的最佳实践。

我们将预训练数据过程划分为以下部分

- 原始数据转换，原始文本转换为jsonl的json字符串格式。
- 数据ID化，断句、分词、tokenize转化为token id格式。
- 训练index文件生成，生成train、valid、test的每个样本索引。
- token动态mask(可选)，python 层实时mask文本。

本目录下主要包含一下文件：
```
../data_tools
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
 - lac (可选)
 - zstandard (可选)

安装命令`pip install tqdm numpy pybind11 lac zstandard`。另，部分功能需要`g++>=4.8`编译支持


## 训练全流程数据Pipeline

|步骤|阶段|数据格式| 样例|
|-|-|-|-|
| - |-|原始数据： <br/> 每个doc之间用空行间隔开 <br/> - 中文，默认每句换行符，作为句子结束。<br/> - 英文，默认使用nltk判断句子结束  | ```百度，是一家中国互联网公司。``` <br/> ```百度为用户提供搜索服务。``` <br/><br/> ```PaddleNLP是自然语言处理领域的优秀工具。```  |
|原始数据转换<br/>`trans_to_json.py`|预处理|jsonl格式：每个doc对应一行json字符串| ```{"text": "百度是一家中国互联网公司。百度为..."}```<br/>```{"text": "PaddleNLP是自然语言..."}```
|数据ID化<br/>`create_pretrain_data.py`|预处理| npy格式：数据id化后的token id <br/>npz格式：数据句子、文章位置索引 | -
|训练index文件生成|训练启动|npy格式：<br/> 根据训练步数max_steps生成<br/>train、valid、test的每个样本索引文件| -
|token动态mask（可选）| Dataset取数据 | 无 |-


## 参数说明

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
根据说明，我们使用下面简单命令，可以得到`data_sample.jsonl`文件。此处，我们对文章所有doc进行了shuffle。
```shell
python trans_to_json.py  --input_path ./data --output_path data_sample
```

### 数据ID化
本部分，我们使用 `create_pretraining_data.py` 脚本将前面得到的 `data_sample.jsonl` 进行tokenize id化处理。
```
optional arguments:
  -h, --help            show this help message and exit
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

common config:
  --append_eos          Append an <eos> token to the end of a document.
                        gpt模型专用，gpt设置此选项，表示doc结束。
  --log_interval LOG_INTERVAL
                        Interval between progress updates
                        打印日志间隔，interval表示处理 文本行数/doc数的 间隔。
  --workers WORKERS     Number of worker processes to launch
                        处理文本id化的进程个数。
```
通过下面脚本转化，我们可以得到处理好的预训练数据，token ids:`data_sample_ids.npy`, 文章索引信息`data_sample_idx.npz`.
```
python -u create_pretraining_data.py \
    --data_format JSON \
    --input_path data_sample.jsonl \
    --output_prefix data_sample  \
    --workers 1 \
    --log_interval 10000
```

### FAQ

#### C++代码编译失败怎么办？
- 请先检查pybind11包是否安装，g++、make工具是否正常。
- 编译失败可能是本文件夹下的Makefile命令出现了一些问题。可以将Makefile中的python3、python3-config设置成完全的路径，如/usr/bin/python3.7。

## 参考内容

注: 大部分数据流程，参考自[Megatron](https://github.com/NVIDIA/Megatron-LM)，特此表达感谢。
