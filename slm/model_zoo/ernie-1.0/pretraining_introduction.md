# ERNIE 中文预训练介绍

ERNIE 是百度提出的大规模预训练模型，曾在中文场景下取得了 SOTA 效果。
PaddleNLP 致力于预训练开源工作，使用开源中文语料 CLUE、WuDao 总共400GB，发布大规模开源语料预训练全流程。从零开始，轻松构建预训练模型。

本项目，从数据下载，词表制作，数据转化，模型训练，所有流程，完全开源开放，可复现。
并训练发布开源最优的模型参数。

接下来将从下面几个方面，详细介绍整个数据制作全流程，从零开始，构建一个预训练模型。

* [1. 数据准备](#数据准备)
    * [1.1 大规模中文数据](#大规模中文数据)
    * [1.2 高精准中文分词](#高精准中文分词)
    * [1.3 快速 Token ID 转化](#快速 TokenID 转化)
* [2. 全字符中文词表制作](#中文词表制作)
    - [2.1 分析准备](#分析准备)
    - [2.2 文本字符统计](#文本字符统计)
    - [2.3 英文字符词表](#英文字符词表)
    - [2.4 合并词表](#合并词表)
* [3. 开始训练](#开始训练)
    - [3.1 训练脚本](#训练脚本)
    - [3.2 训练网络配置](#networks)
    - [3.3 训练速度配置](#speed)
    - [3.4 训练数据流配置](#data_pipe)
    - [3.5 观察评估](#观察评估)
- [4. 训练效果](#release_models)
    - [4.1 ERNIE 1.0-Base-zh-cw 模型](#ernie-1.0-base-zh-cw)
    - [4.2 ERNIE 1.0-Large-zh-cw 模型](#ernie-1.0-large-zh-cw)
* [5. 参考](#references)

全部流程介绍图如下：

<p align="center">
  <img src="https://user-images.githubusercontent.com/16911935/187170152-0778a6c1-6510-4c01-84d0-8e0ea3c05231.png" align="middle"  width="500" />
</p>


**环境依赖**

- fast_dataindex
- visualdl
- pybind11
- lac (可选)

安装命令 `pip install fast_dataindex visualdl pybind11 lac`

<a name="数据准备"> </a>

## 1. 数据准备

数据流是预训练的非常重要的，[预处理文档](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/tools/preprocess)提供了整体的数据变动的流程示意，用户可以查看数据制作的细节文档。


<a name="大规模中文数据"> </a>

### 1.1 大规模中文数据

模型的根本是数据，大数据才能有望获得更好的训练效果。我们希望语料有如下特点:
- **大规模**：目前像 ERNIE-3.0，GPT-3，CPM 等模型，动辄数 T 的文本语料。而目前开源的一些中文模型，确是基于15G 左右的 CLUECorpus 语料训练，大大限制了模型的效果，
- **开源开放**：为了让用户也可以比较容易复现整体的数据流程，采用的数据希望是**开源**的，人人可以获取的。

综上，我们选用的预料为 CLUECorpus2020 语料 200G， WuDaoCorpus2.0 Base 语料 200G。

**CLUECorpus2020 语料**

CLUECorpus2020 是通过 Common Crawl 中文部分语料清洗得到。开源部分提供了约200G 左右的语料文本，详细介绍见[官网](https://github.com/CLUEbenchmark/CLUECorpus2020#%E6%95%B0%E6%8D%AE%E4%B8%8B%E8%BD%BD)，用户可以通过邮件申请下载。

**WuDaoCorpus2.0 Base 语料**

WuDaoCorpora 是悟道爬取的中文大规模语料。整体数量为3TB，目前开源的部分为 WuDaoCorpus2.0 bases 数据集，大小为200GB。
用户微信登录[官网](https://resource.wudaoai.cn/home)，即可直接下载数据。下载好的压缩数据约 64GB。


为了方便用户测试，我们提供了少量 part 的 WuDao 数据供大家使用，（如有侵权，请联系我们删除）
```
wget https://bj.bcebos.com/paddlenlp/models/transformers/data_tools/WuDaoCorpus2.0_base_200G_sample.tar.gz
tar -xvf WuDaoCorpus2.0_base_200G_sample.tar.gz
```
用户可以用这份数据跑完后续全程。数据量约为2GB。


<a name="高精准中文分词"> </a>

### 1.2 高精准中文分词

ERNIE 使用知识嵌入的方式进行预训练。文本中的知识，比如 文本的中的人名、地名、成语、短语等都是知识。如何把这知识训练融合到模型中呢？ERNIE 给出的方案是对这些知识短语一起 MASK，然后预测，也就是 Whole Words MASK。

在我们数据处理层面，如何尽可能精确的从原始文本中提取知识，直接关系预训练模型的效果。我们对目前 PaddleNLP 常用的分词方式的有`jieba`，`lac`，`seg`进行分析。`jieba`采用 HMM 隐马尔可模型，`lac`是 LSTM 模型。

效果、速度对比表格如下，假设 CPU 使用40线程，GPU 使用16卡，处理200G 文本：

| 切词方式 | 效果 | 速度 | 预估耗时
|-|-|-|-|
| jieba | 一般 | 607 KB/s |  2.5 h |
| lac   | 好 | 106 KB/s | 13.9 h
| wordtag (弃用)| 最好 | 0.94 KB/s | 159 D (GPU)|

综合考虑分词的效果与速度，我们选择百度的 LAC（seg）作为我们的文本分词工具。


本文档以 WuDao 数据为例，对数据进行分词：


```shell
python $PADDLENLP_PATH/llm/tools/preprocess/words_segmentation.py \
    --input_path "./WuDaoCorpus2.0_base_200G" \
    --output_path "./wudao_lac_cut" \
    --data_format "wudao" \
    --cn_seg_func "seg" \
    --workers 48
```

注：预训练需要实现 SOP( Sentence Order Predict) 任务，在分词的同时，我们使用 简单规则 进行了文本断句。如果语料只有一句话，建议去除 SOP loss，训练时设置 `binary_head=False`。

文本转化完成后。我们使用 `$PADDLENLP_PATH/llm/tools/preprocess/trans_to_json.py`重新转换为 jsonl 格式（分词完毕）。
```shell
python $PADDLENLP_PATH/llm/tools/preprocess/trans_to_json.py  \
    --input_path "./wudao_lac_cut" \
    --output_path "wudao_corpus_200g_sample.jsonl" \
    --workers 40 \
    --no-shuffle
```
使用 WuDaoCorpus2.0_base_200G_sample.tar.gz 数据可以得到 jsonl 文本为:
```
wget https://bj.bcebos.com/paddlenlp/models/transformers/data_tools/wudao_corpus_200g_sample.jsonl
```
用户可以下载处理好的数据，进行 tokenizer 转换。


<a name="快速 TokenID 转化"> </a>

## 1.3 快速 Token ID 转化

预料、词表准备妥当后，我们可以开始进行最后的数据 ID 转化。

- 高效的 Multiprocessing 多进程实现
- 使用内存 BytesIO 存储 ID 数据

由于转换的逻辑复杂，需要定义`class Converter`对象来进行转化处理。如果每次处理新的文本，都实例化一次 class 对象，速度瓶颈会在处理函数的实例化。
我们使用了提前 multiprocessing.Pool 的`initializer`，对处理函数进行提前实例化，提高处理效率。

处理后的 token id 数量巨大，可以达到数百 Billion，如果使用普通的数据结构，如 python 的 list 保存，会出现存储瓶颈，不仅占用空间大，list 对象还需要重新分配内存空间。这里我们采用了 BytesIO 的方式，类似写入内存文件的方式，速度快，可以非常方便转化为 numpy 文件保存。

使用 Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz CPU 测试，40线程，处理速度 8+MB/s，约7个小时左右，即可完成 200GB 文本转化为 ID.

```shell
python -u  $PADDLENLP_PATH/llm/tools/preprocess/create_pretraining_data.py \
    --model_name "ernie-3.0-base-zh" \
    --tokenizer_name "ErnieTokenizer" \
    --input_path "wudao_corpus_200g.jsonl" \
    --output_prefix "wudao_corpus_200g" \
    --split_sentences\
    --data_impl "mmap" \
    --chinese \
    --cn_splited \
    --cn_whole_word_segment \
    --workers 48 \
    --log_interval 1000
```

此处需要指定词表文件进行 ID 转化，用户可以使用 paddlenlp 内置的部分词表如`ernie-1.0-base-zh,ernie-3.0-base-zh`，设置`model_name`参数为对应参数名即可。
也可以根据自己的需求，重新开始制作词表，然后`model_name`传入词表所在的文件夹目录即可。词表制作，请参考下一章节[全字符中文词表制作](#全字符中文词表制作)。

转化后的数据如下，使用这份数据，即可开始 ERNIE 预训练：
```
wudao_corpus_200g.bin
wudao_corpus_200g.idx
```
同样，对于 WuDaoCorpus2.0_base_200G_sample.tar.gz 数据，使用`ernie-3.0-bash-zh`的 tokenizer，可以得到数据。
```
mkdir data && cd data
wget https://paddlenlp.bj.bcebos.com/paddlenlp/models/transformers/data_tools/wudao_corpus_200g_sample_ernie-3.0-base-zh.bin
wget https://paddlenlp.bj.bcebos.com/paddlenlp/models/transformers/data_tools/wudao_corpus_200g_sample_ernie-3.0-base-zh.idx
```

<a name="中文词表制作"> </a>

### 2. 全字符中文词表制作

之前的 数据 id 化中，使用了已有的词表进行转化，当没有词表时，需要从头开始进行词表制作。如果你没有制作新词表的需求，请跳过此部分，直接阅读 [第三节，开始训练](#开始训练)。

那制作 ERNIE 的词表有什么特点需要注意呢？常见的方法是使用 sentencepiece 切词，使用 BPE 去找通用的子词串。但是，ERNIE 之类的中文模型，是属于字模型，不会出现连续汉字作为子词 如`##中国`。一般是通过 BasicTokenizer，给所有中文汉字之间，添加空格，然后再去切分 子词 subword，这样每个汉字就都是独立的。
```
china -> ch #ina
我爱china -> 我 爱 china -> 我 爱 ch #ina
```

这里提供了 ERNIE 模型词表制作的两种方案：

- 第一种，词表组合方案
    1. 统计字符
    2. 制作英文词表
    3. 合并词表

- 第二种，预处理后直接生成，方案
    1. 文本预处理（中文加空格，文本 normalize）
    2. 使用 sentencepeice 制作词表

第二种方案需要对文本先使用`BasicTokenizer`切分一遍语料。
第一种方案，自定义程度高，但存在一些局限性。本项目采用了第一种方案，详细介绍如下：

### 2.1 分析准备
词表大小： 这里我们考虑的因素主要有两个
- 已有模型对照：
    - ERNIE 3.0系列模型的词表，词表大小为 40000 左右。
- 预训练数据存储占用：
    - 文本 token id 化后，希望使用 uint16表示，此时表示的最大字符为65536。
    - 同时考虑到 ERNIE 虽然是字模型，我们的仍然需要 `##中` 之类的中文字符表示分词信息。假设使用中文全字符20902(0x4E00-0x9FA5)个字符，那么剩余 vocab 大小不能超过 44634。

综上，本项目决定采用 40000 左右的 vocab 容量。
其中：
- 中文全字符 `20902`
- 英文字符 `17000`
- 其他字符约 `2000` 左右


### 2.2 文本字符统计
首先第一步是对文本字符进行统计。字符统计的目的主要是添加常用的中文字符、特殊字符。

由于语料文本过大，我们随机选取 10G 左右的原始文本进行了字符统计。
```
python ./vocab/gen_char.py path_to_corpus.txt
```
可以在本地文件夹得到`char_dict.pickle`字符频率文件。同时我们也提供了自己统计的词频文件，方便用户复现：
```
wget https://bj.bcebos.com/paddlenlp/models/transformers/data_tools/char_dict.pickle
```

### 2.3 英文字符词表
基于字符的词频统计，使得英文字符也切割为字母，为此我们需要添加英文词表。
英文部分，我们使用了 [WikiText](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip)  数据集，来构造词表。
下载解压数据，使用 BPE 切词
```
wget  https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip
python ./vocab/gen_vocab.py ./wikitext-103-raw/wiki.train.raw
```
即可产生英文部分的词表。这里我们也提供了处理好的 vocab 方便用户验证。
```
wget https://bj.bcebos.com/paddlenlp/models/transformers/data_tools/eng.vocab
```


### 2.4 合并词表

目前我们得到了字符统计表，和英文字符词表。下一步，我们将词表进行合并。

将`char_dict.pickle`，`eng.vocab`放置到当前目录，使用下面命令
```
python ./vocab/merge_vocab.py
```
即可在 当前 目录生成 vocab.txt 得到最终词表。

此阶段需要注意的一些问题是：
1. 对于一些日文、谚文文字字符，需要进行 normalize
2. 添加 special_tokens

### 2.5 问题遗留
本项目采用的第一种方式，即拼接产出的词表，对连续非中、英文字符文本，会出现 UNK 的情况。
如 issue: [#2927](https://github.com/PaddlePaddle/PaddleNLP/issues/2927)、 [#2585](https://github.com/PaddlePaddle/PaddleNLP/issues/2585)。本项目做了两点改进:

1. 对 Symbol 字符默认添加空格，变成独立字符
2. 对 日文、谚文 在合并词表阶段默认添加 ## 字符。

虽然有上述两点修复，任然无法避免 [#2927](https://github.com/PaddlePaddle/PaddleNLP/issues/2927) 现象。
彻底解决的话，建议使用第二种方式制作 vocab 文件。

### 2.6 方案二：预处理后直接生成
此方案没有被采用，这里也简单说明一下具体的方案：
1. 对语料使用 BasicTokenizer 转换
```python
from paddlenlp.transformers import
tokenizer = BasicTokenizer()
basic_toknizer = lambda x: " ".join(tokenizer.tokenize(x))
# 对语料使用 basic_toknizer 转换
# 并存储为新的语料 afer_basic_toknizer_corpus.txt
```
2. 处理转换后的语料
```shell
python ./vocab/gen_vocab.py afer_basic_toknizer_corpus.txt
```
对处理好的 vocab 文件手动替换一些`<pad> -> [PAD]`之类的 special_tokens，即可产出词表。


## 3. 开始训练

使用开源中文语料 CLUE、WuDao 总共400GB，提供上面提供的大规模语料数据集制作教程。接下来，看是模型训练。

<p align="center">
  <img src="https://user-images.githubusercontent.com/16911935/187134299-72628dce-cc04-49d7-89ef-078fad487724.png" align="middle"  width="500" />
</p>

### 3.1 训练脚本

训练脚本如下。环境配置和路径配置，不是必要的，如果用户只想简单训练，可以直接跳到[继续训练](#继续训练)部分，直接训练。

<b>环境配置</b>
- PYTHONPATH 设置为当前目录（适合 paddlenlp develop 运行）
- 设置了一些 FLAGS，包括增强报错，动态图 Flag，提高矩阵乘法精度。
- 多机情况下，可以设置`NCCL_SOCKET_IFNAME`指明 NCCL 使用的通信网口。

<details>
<summary>环境配置脚本</summary>

```shell
set -x

# cd PaddleNLP/model_zoo/ernie-1.0
export PYTHONPATH=$PYTHONPATH:../../

export FLAGS_call_stack_level=2
# export NCCL_SOCKET_IFNAME=xgbe0
export FLAGS_gemm_use_half_precision_compute_type=False
export FLAGS_enable_eager_mode=1
unset CUDA_VISIBLE_DEVICES
```
</details>

<b>路径配置</b>

- 主要配置输入输出目录
- 这里的`vocab_dir`如果没有使用自定义词表的话，请设置为内置的 tokenizer，如`ernie-1.0-base-zh,ernie-3.0-base-zh`等。
- 这里的 `data_dir` 设置多份数据集，用户不使用多份数据集的话，直接`data_dir="./data"`即可。

<details>
<summary>路径配置</summary>

```shell
trainer_id=${PADDLE_TRAINER_ID:-"0"}
task_name="0809-ernie-1.0-base-cw-dp16-gb1024"

base_nfs="/path/to/your/nfs/mount/point"
base_dir="${base_nfs}/ernie-cw/output/${task_name}"
data_dir="5.0 ${base_nfs}/clue_oscar/clue_corpus_oscar_0630 7.0 ${base_nfs}/clue_train/clue_corpus_train_0629 12.0 ${base_nfs}/wudao_200g/wudao_200g_0703"
vocab_dir="${base_nfs}/"
```
</details>

**启动训练**：这里启动的是单机8卡任务，整体全局的 batch_size 512 (64*8)。如果指定 ips 参数，进行多机运行，如 `python3 -u  -m paddle.distributed.launch  --gpus "0,1,2,3,4,5,6,7" --ips 192.168.1.101,192.168.1.101 `

```shell
python3 -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "${base_dir}/log_${trainer_id}" \
    run_pretrain.py \
    --model_type "ernie" \
    --model_name_or_path "ernie-3.0-base-zh" \
    --tokenizer_name_or_path "${vocab_dir}" \
    --input_dir "${data_dir}" \
    --output_dir "${base_dir}" \
    --split 949,50,1 \
    --max_seq_len 512 \
    --binary_head true \
    --micro_batch_size 64 \
    --use_amp true \
    --fp16_opt_level "O1" \
    --use_recompute false \
    --max_lr 0.0001 \
    --min_lr 0.00001 \
    --max_steps 4000000 \
    --save_steps 100000 \
    --checkpoint_steps 5000 \
    --decay_steps 3900000 \
    --weight_decay 0.01 \
    --warmup_rate 0.01 \
    --grad_clip 1.0 \
    --logging_freq 20 \
    --num_workers 3 \
    --eval_freq 1000 \
    --device "gpu"\
    --share_folder true \
    --hidden_dropout_prob 0.1 \
    --attention_probs_dropout_prob 0.1 \
    --seed 1234 \
```


其中参数释义如下：
- `model_name_or_path` 要训练的模型或者之前训练的 checkpoint。
- `tokenizer_name_or_path` 模型词表文件所在的文件夹(对于 ernie，词表文件名一般命名为 vocab.txt)，或者 PaddleNLP 内置 tokenizer 的名字。
- `continue_training` 默认 false，模型从随机初始化，开始训练。如果为 True，从已有的预训练权重加载，开始训练。如果为 True， 训练初始 loss 为2.x 是正常 loss，如果未 False，随机初始化，初始 loss 一般为10+。
- `input_dir` 指定输入文件，可以使用目录，指定目录时将包括目录中的所有文件。
- `output_dir` 指定输出文件。
- `split` 划分数据集为 train、valid、test 的比例。整个数据集会按照这个比例划分数据。默认`split=949,50,1`, 使用1/1000的数据为 test，当样本数太少时，增大测试的样本数目。
- `max_seq_len` 输入文本序列的长度，默认值`512`。
- `binary_head` 是否使用 SOP(Sentences Order Predicet) loss，默认为 True，使用此 loss。如果用户句子语料很短，无法组合成句子对，请设置此参数为`false`。
- `micro_batch_size` 单卡 batch size 大小，比如此处单卡 bs=64, 采用8卡训练`global_batch_size=64*8=512`。
- `use_amp` 开启混合精度策略。
- `fp16_opt_level` 混合精度策略，支持 O1 自动混合精度，O2 pure fp16精度训练。
- `max_lr` 训练学习率。
- `min_lr` 学习率衰减到最小值后，学习率将一直保持为`min_lr`。
- `max_steps` 最大训练步数。训练不支持通过`epoch`控制，第一次制造数据 index 时候，日志会显示数据会被计算的 epoch 数，请注意查看。
- `save_steps` 保存模型间隔。默认保存地址格式为`output_dir/model_50000`(5w 步时的权重)。
- `checkpoint_steps` 模型 checkpoint 间隔，用于模型断点重启训练。默认地址为`output_dir/model_last`.
- `weight_decay` 权重衰减参数。
- `warmup_rate` 学习率 warmup 参数。
- `grad_clip` 梯度裁剪范围。
- `logging_freq` 日志输出间隔。
- `num_workers` DataLoader 采样进程，当数据输入为瓶颈时，可尝试提高采样进程数目。
- `eval_freq` 模型评估间隔。
- `device` 训练设备，默认为 GPU。
- `share_folder` 多机训练时，如果多机`input_dir`为挂载的同一个 nfs 网络位置，可以开启次选项，多机共享同一份数据。（每次运行，会制作训练的 index 数据，如果为挂载的统一 nfs 位置，则一台机器制作数据即可，否则每台机器都需要制作）

<b>继续训练</b>
<a name="继续训练"> </a>

很多同学的需求，是从已有的预训练参数开始，继续训练过程，这里我们使用前面教程提供的`WuDaoCorpus2.0_base_200G_sample.tar.gz`样本数据，在`ernie-3.0-base-zh`权重上继续训练。脚本如下：

<details>
<summary><b>展开脚本</b></summary>

```
python3 -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/ernie_continue_training/logs" \
    run_pretrain.py \
    --model_type "ernie" \
    --model_name_or_path "ernie-3.0-base-zh" \
    --tokenizer_name_or_path  "ernie-3.0-base-zh" \
    --continue_training true \
    --input_dir ./data \
    --output_dir output/ernie_continue_training/ \
    --split 949,50,1 \
    --max_seq_len 512 \
    --binary_head true \
    --micro_batch_size 64 \
    --use_amp true \
    --fp16_opt_level "O1" \
    --use_recompute false \
    --max_lr 0.0001 \
    --min_lr 0.00001 \
    --max_steps 500000 \
    --save_steps 100000 \
    --checkpoint_steps 5000 \
    --decay_steps 490000 \
    --weight_decay 0.01 \
    --warmup_rate 0.01 \
    --grad_clip 1.0 \
    --logging_freq 1 \
    --num_workers 3 \
    --eval_freq 1000 \
    --device "gpu"\
    --scale_loss 1024\
    --seed 1234 \
```
</details>


<a name="networks"> </a>

### 3.2 训练网络配置

本小节

- SOP Loss
    - SOP (Sentence Order Predict) 损失，是 模型训练的常用损失。将文本中的句子顺序分为两段打乱，最后判断文本是否被打乱。下图是数据组织形式的展示：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/16911935/187140981-924fd21c-fb67-4ba8-a421-490fd293175c.png" align="middle"  width="600" />
    </p>

    - *<u>使用方法</u>*: 此开关由 `binary_head` 选项开启，`binary_head=True`添加 sop loss， `binary_head=False` 关闭 sop loss。
    - **注意：如果你使用的语料文本中，只有一句话，无法分为多个句子段落，请设置 `binary_head=False`。否则，不符合要求的数据默认被删去，导致可训练的数据过小。**
- MASK
    -  MLM (Mask Language Model) 是通过随机将文本中的部分 token，随机替换为`[MASK]` token，最后预测出真实的 token 值。ERNIE 默认采用了 Whole Word MASK 方式，选定一些词语进行 MASK。
    - *<u>使用方法</u>*: 用户可以设置 `masked_lm_prob` 控制 mask 的 token 占文本总 token 长度的比例。默认`masked_lm_prob=0.15` 随机 mask 15% 的 token 数目。
    - 设置`short_seq_prob`， 控制长度小于 max_seq_length 的样本比例，默认值`short_seq_prob=0.1`。制作数据时候，会有相应比例的数据 最大长度会设置为 一个小于 max_seq_length 的随机值。
- Ngram MASK
    - 项目还支持了 n-gram mask 策略，如下图所示，在 WWM 进行词语级别 MASK 的基础上（如此处 mask 掉的`[模型]`词组），n-gram 可以 MASK 掉连续 n 个词组。下面例子中，连续 mask 了2个词组，`【[语言][模型]】`同时进行了 mask。
    <p align="center">
    <img src="https://user-images.githubusercontent.com/16911935/187145669-7c55386d-f57a-4589-9e6d-e4a36b93e24c.png" align="middle"  width="600" />
    </p>

    - *<u>使用方法</u>*: 用户通过`max_ngrams`设置最大的`ngram`长度。默认`max_ngrams=3`。
    - 注：
        - ernie 预训练使用的 dataset 代码文件在 `./data_tools/ernie_dataset.py`
        - 数据集 index 生成，动态 mask 相关代码实现在`./data_tools/dataset_utils.py`

        - 用户可以根据自己的需求，灵活修改 mask 方式。具体可以参考`dataset_utils.py`中`create_masked_lm_predictions`函数。可以自定义的选项有 do_whole_word_mask, favor_longer_ngram, do_permutation, geometric_dist 等，可以参考[Megatron](https://github.com/NVIDIA/Megatron-LM)使用这些 lm_mask 策略。

- Dropout
    - Dropout 是常用的防止过拟合策略。对于大规模数据集训练，如`ernie-3.0`系列4T 文本语料，可以设置 `dropout=0`，不考虑过拟合。实际`ernie-3.0-base-zh`训练中，没有开启 Dropout。
    - *<u>使用方法</u>*: 用户可以设置 `hidden_dropout_prob`，`attention_probs_dropout_prob`。默认值为 `0.1`。


<a name="speed"> </a>

### 3.3 训练速度配置

**训练速度方面**，我们支持了如下策略，加
速计算过程，减小显存占用，扩大 batch_size：

- **多卡多机**训练：
    - 基于飞桨 Fleet 分布式 API，用户可以十分方便的通过数据并行的方法，将训练扩展到多机多卡。
    - *<u>使用方法</u>*：
        - 单机八卡
        ```shell
        python3 -u  -m paddle.distributed.launch \
            --gpus "0,1,2,3,4,5,6,7" \
            run_pretrain.py
        ```
        - 多机，假设机器 ip 为 `192.168.1.101,192.168.1.102` **注**：多台机器启动的 ips 参数需要顺序一致。
        ```shell
        python3 -u  -m paddle.distributed.launch \
            --gpus "0,1,2,3,4,5,6,7" \
            --ips 192.168.1.101,192.168.1.102 \
            run_pretrain.py
        ```
- **混合精度**训练：
    - 部分算子使用 FP16计算 kernel，加速计算过程。支持 AMP 混合精度 O1，和 Pure FP16全 FP 训练策略 O2。
    - 如下图所示，使用 AMP O1时，一些参数自动从 fp32 cast 为 FP16类型计算。使用`O2` pure fp16时，模型参数为 fp16。
    - *<u>使用方法</u>*: 设置`use_amp=True`开启混合精度训练。设置`fp16_opt_level=O1`，切换 pure_fp16请设置为`O2`。
    <p align="center">
    <img src="https://user-images.githubusercontent.com/16911935/187338824-8b522935-4d6e-48d4-a5f6-55695ed3b182.png" align="middle" width=600 />
    </p>
- **梯度累积**训练：
    - 用户可以指定梯度累积的步数，在梯度累积的 step 中。
    - 减少多卡之间梯度的通信，减少更新的次数，扩大训练的 batch_size.
    - <u>*使用方法*</u>：用户设置 `gobal_batch_size`为 `micro_batch_size*卡数`的倍数，即可开启梯度累积。如：单卡 bs=16，8卡，此时如果设置`gobal_batch_size=512`，则梯度累积次数为`gobal_batch_size/bs/card_num=512/16/8=4`。
- **重计算**训练：
    - 通过重新计算前向的方式，减少前向网络中间变量的存储，可以显著减少显存占用。理论上，该方式以时间换空间，但在 batch size 显著扩大的情况下，速度下降幅度较小。
    - 如图所示：原来训练过程中，中间变量需要常驻显存，等待反向计算。使用重计算之后，修改成了反向需要时，再重新计算一遍前向过程，生成中间变量。避免常驻显存，减小显存占用。
    - <u>*使用方法*</u>：用户设置`use_recompute=True`即可使用。注意使用时，可同时扩大`micro_batch_size`参数。
    <p align="center">
    <img src="https://user-images.githubusercontent.com/16911935/187176881-06103714-3061-42ab-8322-0b63422e7087.png" align="middle"  width="600" />
    </p>


<a name="data_pipe"> </a>

### 3.4 训练数据流配置
**训练数据流方面**，我们针对训练数据流扩展、混合、重启等方面做了针对性优化提升

数据流
- **多机扩展**
    - 用户可以将数据放置到 NFS 服务器上，多机同时挂载数据即可。
    - 解析：当用户需要在多台机器之间，一起多机训练，或者切换到空闲的机器上训练时。由于数据集很大(数百 GB)，迁移不方便。训练数据与计算资源分离，是非常适合的策略。
    - <u>*使用方法*</u>：参考[NFS 服务搭建教程](https://blog.csdn.net/eijiyey/article/details/123184529)，用户将制作好的数据，放到 NFS 机器，然后挂载到有训练资源的其他机器训练即可。
    <p align="center">
    <img src="https://user-images.githubusercontent.com/16911935/187355897-478e7aeb-560f-4ea7-a29c-4bea9d8a7712.png" align="middle"  width="500" />
    </p>

- **多数据混合**
    -  <u>*简介*</u>：训练数据集支持多个文件，即插即用，可设置不同数据集占比权重。上面的多机训练的架构，混合使用了四份数据集。
    - <u>*使用方法*</u>：传入参数即可`input_dir="1.0  dateset_a/prefix  2.0 dataset_b/prefix"`
    - **注意**：如果文件夹中只有一份数据如`data/wudao_200g_0703_ids.npy data/wudao_200g_0703_idx.npz`，可以直接设置`input_dir=./data`为输入目录即可。如果需要设定多份数据集，必须写上数据集前缀，如`input_dir="1.0 data/wudao_200g_0703 1.0 data2/clue_corpus_train_0629"`。写前缀即可，不要加上后面类似`_ids.npy _idx.npz`的尾缀。
- **稳定可复现**
    - <u>*简介*</u>：MLM 任务具有一定随机性，需要随机 mask 数据。本数据流通过固定每一个 step 数据的随机种子，实验数据流稳定可复现。
    - <u>*使用方法*</u>： 传入`seed`参数即可，修改参数后会重新生成 index 数据，打乱数据顺序。
- **快加载**
    -  <u>*简介*</u>：数据文件使用 mmap 读取，避免直接将数据加载到内存，加载数百 GB 文件几乎不耗时。
- **断点重启**
    -  <u>*简介*</u>：用户可以单独设置，`checkpoint_steps` 参数可设置较小，重启训练默认加载最新 checkpoint。
    - 断点数据自动恢复，学习率等参数也自动恢复。
    - **注意：** 此`checkpoint_steps`参数仅保留最后一个`checkpoint`到`model_last`文件夹，默认每次覆盖。用户需要永久保存参数，请设置`save_steps`。建议可以设置`checkpoint_steps`为需要间隔训练半小时、一小时左右的时间，一旦环境故障，可以获取到最新的`checkpoint`。


### 3.4 观察评估

- **训练过程观察**：VisualDL 可视化日志记录
    - 日志展示为全局 loss，波动小。
    - 记录混合精度，loss_scaling 等信息，方便用户 debug。
    - 对模型结构，配置参数，paddle 版本信息进行记录，方便复现环境

<p align="center">
<img src="https://user-images.githubusercontent.com/16911935/187404575-52d53892-4272-4c9d-b29d-064352628951.png" align="middle"  width="900" />
</p>


- **下游任务评估**：CLUE Benchmark 搜索评估参数效果
    - 使用[批量启动-grid-search](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/examples/benchmark/clue#%E6%89%B9%E9%87%8F%E5%90%AF%E5%8A%A8-grid-search)，可以进行批量搜索任务
    - 注意，这里使用的是训练中的 checkpoint 进行评估，可以直接试着 评估待评估的参数为，所在的路径地址，即如 `python grid_seach.py output/ernie-base-outdir/model_100000` 之类的 checkpoint 地址。


<a name="release_models"></a>
## 4. 训练效果

**训练效果方面**，我们 release 了 base、large 两个模型。均取得了较好的预训练效果。

<a name="ernie-1.0-base-zh-cw"></a>

### 4.1 ERNIE 1.0-Base-zh-cw 模型

使用 CLUE，WuDao 共计400GB 的语料，batch_size 1024, 训练 400w step，即可训练得到`ernie-3.0-base-zh`类似的模型效果。相关模型参数，开源为`ernie-1.0-base-zh-cw`，用户加载即可使用。使用 CLUE benchmark 对最优超参数进行 GradSearch 搜索：

Model&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Arch | CLUE AVG |  AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUE WSC2020 | CSL | CMRC | CHID | C3
-- | -- | -- | -- | -- | -- | -- |  -- | -- | -- | -- | -- |  -- |
 Metrics |   |   | Acc | Acc | Acc | Acc | Acc | Acc | Acc | Exact/F1| Acc| Acc
ERNIE 1.0-Base-zh-cw | 12L768H | <b>76.47</b> | 76.04 |    57.86 |    59.91 |   <b>83.41</b> | 79.58 |    89.91 |    83.42 |  72.88/90.78 |    <b>84.68</b> |    76.98 |
ERNIE 2.0-Base-zh | 12L768H | 74.32  | 75.65 |  58.25 | 61.64 |  82.62 |  78.71 |    81.91 |  82.33 | 66.08/87.46    | 82.78    | 73.19
ERNIE 1.0-Base-zh | 12L768H | 74.17 | 74.84 |    58.91 |    62.25 |    81.68 |    76.58 |    85.20 |    82.77 | 67.32/87.83 | 82.47 | 69.68


<a name="ernie-1.0-large-zh-cw"> </a>

### 4.2 ERNIE 1.0-Large-zh-cw 模型

除了 base 模型外，我们还训练了 large 模型。命名为`ernie-1.0-large-zh-cw`。使用开源语料，batch_size 512, 训练 400w step，训练去除 SOP 任务，只保留 MLM 损失，使用 CLUE benchmark 对最优超参数进行 GradSearch 搜索：

Model&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  | Arch | CLUE AVG |  AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUE WSC2020 | CSL | CMRC | CHID | C3
-- | -- | -- | -- | -- | -- | -- |  -- | -- | -- | -- | -- |  -- |
Metrics |   |   | Acc | Acc | Acc | Acc | Acc | Acc | Acc | Exact/F1 | Acc| Acc
ERNIE 1.0-Large-zh-cw| 24L1024H | <b>79.03</b> | 75.97 |    59.65 |    62.91 |    85.09 |    81.73| 93.09 |    84.53 | 74.22/91.88 | 88.57 | 84.54
ERNIE 3.0-Xbase-zh| 20L1024H | 78.39 | 76.16 | 59.55 | 61.87 | 84.40 |  81.73 | 88.82 | 83.60 |    75.99/93.00 | 86.78 | 84.98
RoBERTa-wwm-ext-large | 24L1024H | 76.61 |    76.00 |    59.33 |    62.02 |    83.88 |    78.81 |    90.79 |    83.67 |    70.58/89.82 |    85.72 |    75.26

<a name="references"> </a>

## 5. 参考文献

感谢 CLUE，WuDao 提供的开源文本语料，主要数据流部分参考自[Megatron](https://github.com/NVIDIA/Megatron-LM)，参考资料：
- Xu, L., Zhang, X. and Dong, Q., 2020. CLUECorpus2020: A large-scale Chinese corpus for pre-training language model. arXiv preprint arXiv:2003.01355.
- Yuan, S., Zhao, H., Du, Z., Ding, M., Liu, X., Cen, Y., Zou, X., Yang, Z. and Tang, J., 2021. Wudaocorpora: A super large-scale chinese corpora for pre-training language models. AI Open, 2, pp.65-68.
- https://github.com/CLUEbenchmark/CLUECorpus2020
- https://resource.wudaoai.cn
- https://github.com/NVIDIA/Megatron-LM
