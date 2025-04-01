# 一、Pretrain 数据集

## 1. 内置数据集

名称|文本类型|纯文本的大小|适配模型|制作时间|出处|下载链接 bin|下载链接 idx|
|-|-|-|-|-|-|-|-|
OpenWebText2|英文|70GB|`meta-llama/Llama-2-7b`<br> `meta-llama/Llama-2-7b-chat`<br> `meta-llama/Llama-2-13b`<br> `meta-llama/Llama-2-13b-chat` <br>`facebook/llama-7b`<br> `facebook/llama-13b`<br>| 42min |  [链接](https://skylion007.github.io/OpenWebTextCorpus/) |[*bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/openwebtext2/llama/mmap/llama_mmap.bin) | [*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/openwebtext2/llama/mmap/llama_mmap.idx) |
|OpenWebText2|英文|70GB|`gpt2-en`|37min|[链接](https://skylion007.github.io/OpenWebTextCorpus/)|[*bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/openwebtext2/gpt/mmap/gpt2-en-mmap.bin)|[*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/openwebtext2/gpt/mmap/gpt2-en-mmap.idx)|
CLUECorpusSmall|中文|14GB|`idea-ccnl/ziya-llama-13b-v1`|15min|[链接](https://github.com/CLUEbenchmark/CLUECorpus2020)|[*bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/ziya/mmap/ziya_mmap.bin)|[*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/ziya/mmap/ziya_mmap.idx)|
-|中文|14GB|`baichuan-inc/Baichuan-7B`|12min||[* bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/baichuan/mmap/baichuan_mmap.bin)|[*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/baichuan/mmap/baichuan_mmap.idx)|
-|中文|14GB|`linly-ai/chinese-llama-2-7b` <br>`linly-ai/chinese-llama-2-13b`|19min||[*bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/linly/mmap/linly_mmap.bin)|[*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/linly/mmap/linly_mmap.idx)|
-|中文|14GB|`baichuan-inc/Baichuan-13B-Base` <br>`baichuan-inc/Baichuan-13B-Chat`|14min || [*bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/baichuan13b/mmap/baichuan13b_mmap.bin)|[*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/baichuan13b/mmap/baichuan13b_mmap.idx)|
-|中文|14GB|`baichuan-inc/Baichuan2-7B-Base`<br> `baichuan-inc/Baichuan2-7B-Chat`<br> `baichuan-inc/Baichuan2-13B-Base`<br> `baichuan-inc/Baichuan2-13B-Chat` |13min||[*bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/baichuan2/mmap/baichuan2_mmap.bin)|[*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/baichuan2/mmap/baichuan2_mmap.idx)|
-|中文|14GB|`meta-llama/Llama-2-7b`<br> `meta-llama/Llama-2-7b-chat`<br> `meta-llama/Llama-2-13b`<br> `meta-llama/Llama-2-13b-chat`<br> `facebook/llama-7b` <br> `facebook/llama-13b`<br> `FlagAlpha/Llama2-Chinese-7b-Chat`<br> `FlagAlpha/Llama2-Chinese-13b-Chat` |20min|| [*bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/llama/mmap/llama_mmap.bin)|[*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/llama/mmap/llama_mmap.idx)|
WuDaoCorpus2.0 Base|中文|200GB|`idea-ccnl/ziya-llama-13b-v1`|3h 35min| [链接](https://data.baai.ac.cn/details/WuDaoCorporaText)|[*bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/wudao/ziya/mmap/ziya_mmap.bin)|[*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/wudao/ziya/mmap/ziya_mmap.idx)|
WuDaoCorpus2.0 Base|中文|200GB|`baichuan-inc/Baichuan-7B`|2h 52min||[*bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/wudao/baichuan/mmap/baichuan_mmap.bin)|[*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/wudao/baichuan/mmap/baichuan_mmap.idx)|

下载 bin 和 idx 放在同一个目录下，预训练脚本指定 input_dir 即可.

若需要自行制作数据集，整体制作流程如2.1所示，详细步骤如以下2.2所示。

##  2. 自定义数据集

### 2.1 数据创建流程

|步骤|阶段&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|数据格式| 样例|
|-|-|-|-|
| 0️⃣初始状态 | -|原始数据： <br/> **每个 doc 之间用空行间隔开** <br/> - 中文，默认每句换行符，作为句子结束。<br/> - 英文，默认使用 nltk 判断句子结束  | ```飞桨是功能完备、开源开放的产业级深度学习平台。``` <br/> ```飞桨拥有核心训练和推理框架、基础模型库。``` <br/><br/> ```PaddleNLP是自然语言处理领域的优秀工具。```  |
|1️⃣原始数据转换<br/>`trans_to_json.py`|预处理 <br>输入：0️⃣初始状态 <br>输出：jsonl|jsonl 格式：每个 doc 对应一行 json 字符串| ```{"text": "飞桨是功能完备、开源开放的产业级深度学习平台。飞桨拥有..."}```<br/>```{"text": "PaddleNLP是自然语言..."}```
|2️⃣数据 ID 化<br/>`create_pretrain_data.py`|预处理| bin 格式：数据 id 化后的 token id <br/>idx 格式：数据句子、文章位置索引 | -

### 2.2 详细准备
下面以 ziya-llama-13b-v1模型为例，简要介绍数据制备的全流程。

**2.2.1 原始数据**

首先下载样例数据：
```
mkdir data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/data_tools/baike.txt
cd ..
```

**2.2.2 原始数据转换 jsonl 格式**

使用 trans_to_json.py 转化为 json 串格式，下面是脚本的使用说明
```bash
optional arguments:
  -h, --help
  --input_path INPUT_PATH
                        "必须设置，可以是文件夹或者单个文件。文件夹中的目录默认最多搜索两层子目录。"
  --output_path OUTPUT_PATH
                        "必须设置，输出文件的名字。"
  --json_key JSON_KEY
                        "建议不修改，默认的key是text"
  --doc_spliter DOC_SPLITER
                        "文章换行符，可以根据实际情况修改，默认空行作为文章换行符。"
  --min_doc_length MIN_DOC_LENGTH
                        "可选。过滤掉长度多短的文章，默认值10"
  --workers WORKERS
                        "可选。多进程转化文件，适用于 input_path 中包含的文件数据较多的情况。"
                        "每个文件，分配给不同worker处理"
  --log_interval LOG_INTERVAL
                        "可选。此处的interval是值处理完文件个数的间隔。"
  --no-merge
                        "可选。默认不开启这个选项，默认每个文件转换的jsonl文本，会拼接成到同一个文件。"
  --no-shuffle
                        "可选。默认不开启这个选项，默认对处理完进行shuffle。"
```
根据说明，我们使用下面简单命令，可以得到 baike_sample.jsonl 文件。此处，我们对文章所有 doc 进行了 shuffle。
```
python trans_to_json.py  --input_path ./data --output_path baike_sample
```
查看数据:
```
head -1 baike_sample.jsonl
{"text": "中国效仿西方发展工业的过程，于中华民国国民政府成立后至中日战争开战前夕已顺畅发展，尽管其间受到内外因素的多重干扰。尔后直至中日战争和国共战争的结束，
中国始有较为长期的和平发展时期。\n1980年代以来，邓小平政府宣布改革开放，开始实行社会主义市场经济并推行经济体制改革。中国大陆近年至2010年，GDP超过72000亿美元，
已经成为美国之后的世界第二经济大国，普遍认为中国是世界上发展速度最快的经济体，但是人均国民生产总值仍位于世界中等水平（第89位），并逐渐受到资源限制和贫富差距加
大的制约。中华人民共和国省份中，广东为GDP最高的第一强省，浙江为人均收入最高的第一富省。中国大陆、香港、澳门、台湾之间的经济联系在全球化的过程中日益紧密。\n"}
```

**2.2.3 数据 ID 化**

在这一部分，我们使用 `create_pretraining_data.py` 脚本将前面得到的 `baike_sample.jsonl` 进行 tokenize id 化处理。模型可以参考已有的列表。
```bash
optional arguments:
  --model_name MODEL_NAME.
                        "必须设置，如：idea-ccnl/ziya-llama-13b-v1"
  --tokenizer_name {LlamaTokenizer}
                        "模型对应的tokenizer, Llama模型需使用LlamaTokenizer"
data input/output:
  --input_path INPUT_PATH
                        "必须设置，输入文件jsonl的目录"
  --output_prefix OUTPUT_PREFIX
                        "必须设置，输出文件的名称。"
                        "假设名称为XXX，则会输出 XXX.bin, XXX.idx 两个文件。"
                        "其中bin文件：数据id化后的token ids; idx文件：数据句子、文章位置索引。"
  --data_format {JSON}
                        "不需要设置。目前默认处理jsonl数据格式"
  --json_key JSON_KEY
                        "文本串json的key值。同前面trans_to_json.py的json_key，默认text为key"
  --split_sentences
                        "是否需要将文章划分成句子。一般而言，GPT不需要。"
  --data_impl
                        "处理后的数据格式，可选“mmap”或“lazy”，"
                        "其中“mmap”在训练时读入数据会建立内存映射，“lazy”在读入数据时直接从文件读取。"

chinese words:
  --chinese
                        "若设置了split_sentences，并处理中文则需要设置。"
  --cn_whole_word_segment
                        "可选。是否需要WWM策略。一般而言，GPT类模型不需要。"
  --cn_seg_func {lac,seg,jieba}
                        "默认jieba，jieba速度较快，lac模型更准确，计算量高。"
  --cn_splited
                        "可选。对分词后的文本，设置此选项则，cn_seg_func不起作用。"
                        "例如分词后文本串: 中国 效仿 西方 发展 工业 的过 程"
  --cn_split_dimer CN_SPLIT_DIMER
                        "配合cn_splited使用，默认空格表示分词间隔。"

common config:
  --append_eos
                        "gpt模型专用，gpt设置此选项，表示doc结束。"
  --log_interval LOG_INTERVAL

                        "打印日志间隔，interval表示处理 文本行数/doc数的 间隔。"
  --workers WORKERS
                        "处理文本id化的进程个数。"
```
我们可以通过以下训练脚本得到处理好的预训练数据：
```bash
python -u  create_pretraining_data.py \
    --model_name "idea-ccnl/ziya-llama-13b-v1" \
    --tokenizer_name "LlamaTokenizer" \
    --data_format "JSON" \
    --input_path "/home/data/baike_sample.jsonl" \
    --append_eos \
    --output_prefix "/home/data/baike_sample"  \
    --workers 1 \
    --log_interval 5 \
    --data_impl "mmap"
```
1. 如果您使用已经分好词的语料，可以设置 `--cn_splitd` 为 True，同时指定`--cn_split_dimer`如空格。
2. 使用自定义词表的话，请指定 model_name 为词表所在的文件夹地址。

经过以上处理，在 “/home/data/” 文件夹下可以得到预处理后的训练数据 baike_sample.bin, 与文章索引信息：`baike_sample.idx`.

**2.2.4（可选） 合并数据集**

若需要预处理的文件过大，该脚本所耗费的时间可能会很长。此时可以考虑将 jsonl 文件拆分为多个小文件，并行使用 create_pretraining_data.py 进行处理，得到多个.bin & .idx 文件，之后使用如下 merge 脚本合并多个小的.bin & .idx 文件。
使用 merge 脚本将两份500g 文件合并为1T 的时间约1h。
```bash
python merge.py \
    --input "/home/data/" \
    --output-prefix "/home/data/merged" \
    --data_impl mmap
```
使用说明：
```bash
arguments:
  --input INPUT_PATH
                        "待合并的文件所在文件夹，文件夹内各个小文件按需要merge的顺序命名"
                        "如1.bin / 1.idx，2.bin / 2.idx..."
  --output_prefix OUTPUT_PREFIX
                        "合并后输出文件的名称，假设名称为XXX，则会输出 XXX.bin, XXX.idx 两个文件"。
  --data_impl {mmap,lazy}
                        "merge前后的数据格式，可选“mmap”或“lazy”，各个待merge的文件需格式一致。"
```
经过以上 merge 脚本处理，“/home/data”目录下可以得到由“/home/data/”下的小文件合并而成的 merged.bin 和 merged.idx 文件。

**注意：单个数据集不宜过大，容易出现 int32越界，建议单个文件 docs 数目不超过5亿。**




## 常用数据集制作

[CLUECorpus2020 语料制作](./tools/preprocess/docs/CLUECorpus2020.md)

[CLUECorpusSmall 语料制作](./tools/preprocess/docs/CLUECorpusSmall.md)

[OpenWebText2 语料制作](./tools/preprocess/docs/OpenWebText2.md)

[WuDaoCorpus2.0 Base 语料](./tools/preprocess/docs/WuDaoCorpusBase.md)
