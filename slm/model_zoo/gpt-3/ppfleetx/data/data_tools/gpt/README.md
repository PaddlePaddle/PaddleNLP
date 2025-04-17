## GPT 模型预训练数据准备流程(中文数据处理正在支持中)

我们将预训练数据过程划分为以下2个部分：

1. 原始数据转换，原始文本转换为jsonl的json字符串格式。
2. 数据ID化，断句、分词、tokenize转化为token id格式。

本目录下主要包含以下文件：
```
├── preprocess_data.py # 将jsonl文本，断句、分词后，tokenizer转化为token id。
├── README.md # 预训练数据准备流程教程
└── raw_trans_to_json.py # 原始文本数据转化的脚本，将数据转化为json串格式。
```

## 目录切换
```
# 如果您还未下载 PaddleFleetX 套件，请先 clone 套件
# git clone https://github.com/PaddlePaddle/PaddleFleetX.git
cd PaddleNLP/model_zoo/gpt-3

# 以下所有命令都在 PaddleFleetX 根目录中执行
```

## 环境依赖

 - paddlepaddle-gpu>=2.3.0
 - python==3.7
 - tqdm==4.54.1
 - numpy==1.20.1
 - pybind11==2.10.0

安装命令`pip install -r requirements.txt`。


## 训练全流程数据 Pipeline

|步骤|阶段|数据格式| 样例|
|-|-|-|-|
| 原始数据清洗 | 原始数据准备|原始数据： <br/> 每个doc之间用空行间隔开 <br/> - 中文，默认每句换行符，作为句子结束。<br/> - 英文，默认使用nltk判断句子结束。doc是又一段或多端文字组成，每段文字由一句或多句话文字组成。  | ```飞桨是功能完备、开源开放的产业级深度学习平台。``` <br/> ```飞桨拥有核心训练和推理框架、基础模型库。``` <br/><br/> ```PaddleNLP是自然语言处理领域的优秀工具。```  |
|原始数据转换<br/>`raw_trans_to_json.py`|预处理|jsonl格式：每个doc对应一行json字符串| ```{"text": "飞桨是功能完备、开源开放的产业级深度学习平台。飞桨拥有..."}```<br/>```{"text": "PaddleNLP是自然语言..."}```
|数据ID化<br/>`preprocess_data.py`|预处理| npy格式：数据id化后的token id <br/>npz格式：数据句子、文章位置索引 | -


## 全流程示例

下面以 GPT 预训练为例，简要介绍一下预训练数据处理的全流程。

### 原始数据
首先下载样例数据：
```
mkdir -p dataset/wikitext_103_en
wget -O dataset/wikitext_103_en/wikitext-103-en.txt http://fleet.bj.bcebos.com/datasets/gpt/wikitext-103-en.txt
```
### 原始数据转换 jsonl 格式
使用`raw_trans_to_json.py`转化为json串格式，下面是脚本的使用说明
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
根据说明，我们使用下面简单命令，可以得到`wikitext_103_en.jsonl`文件。此处，我们对所有doc进行了shuffle。
```shell
python ppfleetx/data/data_tools/gpt/raw_trans_to_json.py  --input_path ./dataset/wikitext_103_en --output_path ./dataset/wikitext_103_en/wikitext_103_en

# output of terminal
# Time to startup: 0.0075109004974365234
# Processed 1 files (0.12870440603278582 files/s, 64.80481421466284 MB/s).
# Merging files into wikitext_103_en.jsonl
# File save in wikitext_103_en.jsonl
# Shuffling the jsonl file...
# File shuffled!!!

# 查看数据。因为对数据有 shuffle，下面的内容可能会不一样。
tail -1 ./dataset/wikitext_103_en/wikitext_103_en.jsonl
{"text": "The album was released in June 1973 . Although it received good reviews , it did not sell well , except in Austin , where it sold more copies than earlier records by Nelson did nationwide . The recording led Nelson to a new style ; he later stated regarding his new musical identity that Shotgun Willie had \" cleared his throat . \" It became his breakthrough record , and one of the first of the outlaw movement , music created without the influence of the conservative Nashville Sound . The album — the first to feature Nelson with long hair and a beard on the cover — gained him the interest of younger audiences . It peaked at number 41 on Billboard 's album chart and the songs \" Shotgun Willie \" and \" Stay All Night ( Stay A Little Longer ) \" peaked at number 60 and 22 on Billboard Hot 100 respectively .\nRolling Stone wrote : \" With this flawless album , Willie Nelson finally demonstrates why he has for so long been regarded as a Country & Western singer @-@ songwriter 's singer @-@ songwriter ... At the age of 39 , Nelson finally seems destined for the stardom he deserves \" . Robert Christgau wrote : \" This attempt to turn Nelson into a star runs into trouble when it induces him to outshout Memphis horns or Western swing . \"\nBillboard wrote : \" This is Willie Nelson at his narrative best . He writes and sings with the love and the hurt and the down @-@ to @-@ earth things he feels , and he has a few peers . \" Texas Monthly praised Nelson and Wexler regarding the change in musical style : \" They 've switched his arrangements from Ray Price to Ray Charles — the result : a revitalized music . He 's the same old Willie , but veteran producer Jerry Wexler finally captured on wax the energy Nelson projects in person \" . School Library Journal wrote : \" Willie Nelson differs ( from ) rock artists framing their music with a country & western facade — in that he appears a honky @-@ tonk stardust cowboy to the core . This album abounds in unabashed sentimentalism , nasal singing , lyrics preoccupied with booze , religion , and love gone bad , and stereotyped Nashville instrumentation ( twangy steel guitars , fiddles , and a clean rhythm section characterized by the minimal use of bass drum and cymbals , both of which gain heavy mileage with rock performers ) .\nStephen Thomas Erlewine wrote in his review for Allmusic : \" Willie Nelson offered his finest record to date for his debut – possibly his finest album ever . Shotgun Willie encapsulates Willie 's world view and music , finding him at a peak as a composer , interpreter , and performer . This is laid @-@ back , deceptively complex music , equal parts country , rock attitude , jazz musicianship , and troubadour storytelling \" .\n"}
```

### 数据ID化
我们使用 `preprocess_data.py` 脚本将前面得到的 `wikitext_103_en.jsonl` 进行tokenize id化处理。
```
optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        What model to use.
                        必须设置，如：gpt2
  --tokenizer_name {ErnieTokenizer,BertTokenizer,GPTTokenizer,GPTChineseTokenizer}
                        What type of tokenizer to use.
                        模型对应的tokenizer, 目前暂时只支持 Ernie，Bert，GPT
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
                        是否需要将文章划分成句子。一般而言，GPT不需要，Bert/Ernie模型需要

chinese words:
  --chinese             Is corpus need words segmentation step for chinese words.
                        中文情形必须设置。处理的文本类型是否是中文。
  --cn_whole_word_segment
                        Is corpus need words segmentation step for chinese words WWM.
                        可选。是否需要WWM策略。一般而言，Bert/Ernie模型需要，GPT不需要。
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
通过下面脚本转化，我们可以得到处理好的预训练数据，token ids:`wikitext_103_en.npy`, 文章索引信息`wikitext_103_en.npz`.
在使用 `GPTTokenizer` 时需要用到 `gpt2-vocab.json` 与 `gpt2-merges.txt`，如果没有下载缓存过这两个文件，脚本会自动下载并缓存。当遇到网络问题时，可以自行下载并将这两个文件放置在 `~/.cache/ppfleetx/` 目录下。
```
python ppfleetx/data/data_tools/gpt/preprocess_data.py \
    --model_name gpt2 \
    --tokenizer_name GPTTokenizer \
    --data_format JSON \
    --input_path ./dataset/wikitext_103_en/wikitext_103_en.jsonl \
    --append_eos \
    --output_prefix ./dataset/wikitext_103_en/wikitext_103_en  \
    --workers 40 \
    --log_interval 1000

# 处理完后 terminal 输出
# Processed 267000 documents (9843.34 docs/s, 18.4880 MB/s).
# Processed 268000 documents (9869.46 docs/s, 18.5351 MB/s).
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:27<00:00, 27.17s/it]
# Saving tokens to files...
# Total sentences num: 268492
# Total documents num: 268492
# Total tokens num: 114130026
# Average tokens per sentence: 425.08
# Average tokens per document: 425.08
```

## 参考内容

注: 大部分数据流程，参考自[Megatron](https://github.com/NVIDIA/Megatron-LM)和[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)，特此表达感谢。
