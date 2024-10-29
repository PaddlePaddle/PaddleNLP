# ERNIE: Enhanced Representation through kNowledge IntEgration

**目录**
- [1. 模型简介](#模型简介)
    - [1.1 目录结构](#目录结构)
    - [1.1 环境依赖](#环境依赖)
- [2. 中文预训练](#中文预训练)
    - [2.1 小规模语料预训练: 14GB - CLUECorpusSmall](#CLUECorpusSmall)
    - [2.2 大规模语料预训练: 400GB - CLUE & WuDao](#ERNIE-CW)
    - [2.3 预训练模型贡献](#预训练模型贡献)
- [3. 下游任务微调](#下游任务微调)
  - [3.1 序列分类](#序列分类)
  - [3.2 Token 分类](#序列分类)
  - [3.3 阅读理解](#阅读理解)
- [4. 预测部署](#预测部署)
- [5. 参考文献](#参考文献)



<a name="模型简介"></a>

## 1. 模型简介

ERNIE 是百度开创性提出的基于知识增强的持续学习语义理解框架，它将大数据预训练与多源丰富知识相结合，通过持续学习技术，不断吸收海量文本数据中词汇、结构、语义等方面的知识，实现模型效果不断进化。

ERNIE 在情感分析、文本匹配、自然语言推理、词法分析、阅读理解、智能问答等16个公开数据集上全面显著超越世界领先技术，在国际权威的通用语言理解评估基准 GLUE 上，得分首次突破90分，获得全球第一。
相关创新成果也被国际顶级学术会议 AAAI、IJCAI 收录。
同时，ERNIE 在工业界得到了大规模应用，如搜索引擎、新闻推荐、广告系统、语音交互、智能客服等。

ERNIE 通过建模海量数据中的词、实体及实体关系，学习真实世界的语义知识。相较于 BERT 学习原始语言信号，ERNIE 直接对先验语义知识单元进行建模，增强了模型语义表示能力。

这里我们举个例子：
```
Learnt by BERT ：哈 [mask] 滨是 [mask] 龙江的省会，[mask] 际冰 [mask] 文化名城。
Learnt by ERNIE：[mask] [mask] [mask] 是黑龙江的省会，国际 [mask] [mask] 文化名城。
```
在 BERT 模型中，我们通过『哈』与『滨』的局部共现，即可判断出『尔』字，模型没有学习与『哈尔滨』相关的任何知识。而 ERNIE 通过学习词与实体的表达，使模型能够建模出『哈尔滨』与『黑龙江』的关系，学到『哈尔滨』是 『黑龙江』的省会以及『哈尔滨』是个冰雪城市。

<a name="项目特色"></a>

**项目特色**
- **中文预训练**
    - 提供了完整中文预训练流程，从词表构造、数据处理、任务训练，到下游任务。
    - 提供中文 Whole Word Mask，支持文本动态 Mask。
- **数据流程**，
    - 数据预处理流程高效，40分钟即可完成14G ERNIE 数据制作。
    - 数据稳定可复现，多数据集即插即用。
- **分布式训练**，
    - 支持多机多卡，支持混合精度、重计算、梯度累积等功能。

<a name="目录结构"></a>

### 1.1 目录结构

整体的目录结构如下：

```shell
./
├── args.py     训练配置参数文件
├── converter   静态图参数转换为动态图的脚本
│   └── params_static_to_dygraph.py
├── finetune    下游任务finetune脚本
│   ├── config.yml                 训练参数配置文件
│   ├── question_answering.py      阅读理解任务预处理代码
│   ├── sequence_classification.py 序列分类任务预处理代码
│   ├── token_classification.py    TOKEN分类任务预处理代码
│   ├── README.md       说明文档
│   ├── run_ner.py      命名实体识别任务运行脚本
│   ├── run_qa.py       阅读理解任务运行脚本
│   ├── run_seq_cls.py  序列分类任务运行脚本
│   └── utils.py
├── README.md  说明文档
├── pretraining_introduction.md 中文预训练详细介绍文档
├── preprocess
│   ├── docs                部分数据制作文档，包括CLUECorpusSmall，WuDaoCorpusBase
│   ├─ xxx.py              文件处理的python脚本
│   └──README.md            PaddleNLP 预训练数据流程
├── vocab                   全中文字符词表制作教程
├── run_gb512_s1m.sh        训练启动shell脚本，batch size 512. max steps 100w
├── run_gb512_s1m_static.sh
├── run_gb512_s1m_trainer.sh
├── run_pretrain.py         训练启动python脚本
└── run_pretrain_trainer.py
```

<a name="环境依赖"></a>

### 1.2 环境依赖

- fast_dataindex
- visualdl
- pybind11

安装命令 `pip install visualdl pybind11 fast_dataindex`

<a name="中文预训练"></a>

## 2. 中文预训练

ERNIE 预训练采用的是 MLM（Mask Language Model）的训练方式，采用 WWM（Whole Word Mask）方式，对于完整语义单元的 Token，会同时进行 Mask。整体的训练损失 loss 是 mlm_loss + sop_loss。

ERNIE 中文预训练更详细的介绍文档请可以参见[ERNIE 中文预训练介绍](./pretraining_introduction.md)。


本样例为用户提供了高效的训练流程，
- **支持动态文本 mask**： 用户可以根据自己的需求，灵活修改 mask 方式。具体可以参考修改`data_tools/dataset_utils.py`中`create_masked_lm_predictions`函数。
- **支持自动断点训练重启恢复**。 用户可以设置`checkpoint_steps`，间隔`checkpoint_steps`数，即保留最新的 checkpoint 到`model_last`文件夹。重启训练时，程序默认从最新 checkpoint 重启训练，学习率、数据集都可以恢复到 checkpoint 时候的状态。


<a name="CLUECorpusSmall"></a>

### 2.1 小规模语料预训练: 14GB - CLUECorpusSmall
下面是使用 CLUECorpusSmall 14G 文本进行预训练的流程：

<details>
<summary><b>CLUECorpusSmall 数据准备</b></summary>

#### 数据准备
数据下载部分请参考[preprocess](../../../llm/tools/preprocess)目录，根据文档中`CLUECorpusSmall 数据集处理教程`，下载数据。下载好后:

解压文件
```shell
unzip comment2019zh_corpus.zip -d  clue_corpus_small_14g/comment2019zh_corpus
unzip news2016zh_corpus.zip    -d  clue_corpus_small_14g/news2016zh_corpus
unzip webText2019zh_corpus.zip -d  clue_corpus_small_14g/webText2019zh_corpus
unzip wiki2019zh_corpus.zip    -d  clue_corpus_small_14g/wiki2019zh_corpus
```
将 txt 文件转换为 jsonl 格式
```
python preprocess/trans_to_json.py  --input_path ./clue_corpus_small_14g --output_path clue_corpus_small_14g.jsonl
```
现在我们得到了 jsonl 格式的数据集，下面是针对训练任务的数据集应用，此处以 ernie 为例。
```
python -u  preprocess/create_pretraining_data.py \
    --model_name ernie-1.0-base-zh \
    --tokenizer_name ErnieTokenizer \
    --input_path clue_corpus_small_14g.jsonl \
    --split_sentences \
    --data_impl mmap \
    --chinese \
    --cn_whole_word_segment \
    --cn_seg_func jieba \
    --output_prefix clue_corpus_small_14g_20220104 \
    --workers 48 \
    --log_interval 10000
```
数据共有文档`15702702`条左右，由于分词比较耗时，大概一小时左右可以完成。在当前目录下产出训练所需数据。
```
clue_corpus_small_14g_20220104.bin
clue_corpus_small_14g_20220104.idx
```

</details>


<details>
<summary><b>CLUECorpusSmall 开始训练</b></summary>


####  开始训练

将制作好的数据`clue_corpus_small_14g_20220104.bin,clue_corpus_small_14g_20220104.idx`移动到 input_dir 中，即可开始训练。
这里以8卡 GPU 训练为例任务脚本为例：
```
python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/ernie-1.0-dp8-gb512/log" \
    run_pretrain.py \
    --model_type "ernie" \
    --model_name_or_path "ernie-1.0-base-zh" \
    --tokenizer_name_or_path "ernie-1.0-base-zh" \
    --input_dir "./data" \
    --data_impl "mmap" \
    --output_dir "output/ernie-1.0-dp8-gb512" \
    --split 949,50,1 \
    --max_seq_len 512 \
    --micro_batch_size 64 \
    --use_amp true \
    --fp16_opt_level O2 \
    --max_lr 0.0001 \
    --min_lr 0.00001 \
    --max_steps 1000000 \
    --save_steps 50000 \
    --checkpoint_steps 5000 \
    --decay_steps 990000 \
    --weight_decay 0.01 \
    --warmup_rate 0.01 \
    --grad_clip 1.0 \
    --logging_freq 20 \
    --num_workers 2 \
    --eval_freq 1000 \
    --device "gpu" \
    --share_folder false \
```

使用8卡 MLU 训练示例：
```
python -u  -m paddle.distributed.launch \
    --mlus "0,1,2,3,4,5,6,7" \
    --log_dir "output/ernie-1.0-dp8-gb512/log" \
    run_pretrain.py \
    --model_type "ernie" \
    --model_name_or_path "ernie-1.0-base-zh" \
    --tokenizer_name_or_path "ernie-1.0-base-zh" \
    --input_dir "./data" \
    --data_impl "mmap" \
    --output_dir "output/ernie-1.0-dp8-gb512" \
    --split 949,50,1 \
    --max_seq_len 512 \
    --micro_batch_size 64 \
    --use_amp true \
    --fp16_opt_level O2 \
    --max_lr 0.0001 \
    --min_lr 0.00001 \
    --max_steps 1000000 \
    --save_steps 50000 \
    --checkpoint_steps 5000 \
    --decay_steps 990000 \
    --weight_decay 0.01 \
    --warmup_rate 0.01 \
    --grad_clip 1.0 \
    --logging_freq 20 \
    --num_workers 2 \
    --eval_freq 1000 \
    --device "mlu" \
    --share_folder false \
```

其中参数释义如下：
- `model_name_or_path` 要训练的模型或者之前训练的 checkpoint。
- `tokenizer_name_or_path` 模型词表文件所在的文件夹，或者 PaddleNLP 内置 tokenizer 的名字。
- `continue_training` 默认 false，模型从随机初始化，开始训练。如果为 True，从已有的预训练权重加载，开始训练。如果为 True， 训练初始 loss 为2.x 是正常 loss，如果未 False，随机初始化，初始 loss 一般为10+。
- `input_dir` 指定输入文件，可以使用目录，指定目录时将包括目录中的所有文件。
- `data_impl` 指定输入文件数据制作类型，默认为`mmap`，可指定`mmap`或`lazy`，`mmap`格式在读入数据时会建立内存映射，`lazy`格式在读入数据时直接从文件读取。
- `output_dir` 指定输出文件。
- `split` 划分数据集为 train、valid、test 的比例。整个数据集会按照这个比例划分数据。默认1/1000的数据为 test，当样本数太少时，请修改此比例。
- `max_seq_len` 输入文本序列的长度。
- `micro_batch_size` 单卡 batch size 大小，比如此处单卡 bs=64, 采用8卡训练`global_batch_size=64*8=512`。
- `use_amp` 开启混合精度策略。
- `fp16_opt_level` 混合精度策略，支持 O1 自动混合精度，O2 pure fp16精度训练。
- `max_lr` 训练学习率。
- `min_lr` 学习率衰减的最小值。
- `max_steps` 最大训练步数。
- `save_steps` 保存模型间隔。默认保存地址格式为`output_dir/model_50000`(5w 步时的权重)。
- `checkpoint_steps` 模型 checkpoint 间隔，用于模型断点重启训练。默认地址为`output_dir/model_last`.
- `weight_decay` 权重衰减参数。
- `warmup_rate` 学习率 warmup 参数。
- `grad_clip` 梯度裁剪范围。
- `logging_freq` 日志输出间隔。
- `num_workers` DataLoader 采样进程，当数据输入为瓶颈时，可尝试提高采样进程数目。
- `eval_freq` 模型评估间隔。
- `device` 训练设备，默认为 GPU。
- `share_folder` 多机训练时，如果多机 input_dir 为挂载的同一个 nfs 网络位置，可以开启次选项，多机共享同一份数据。


注：
- 训练支持断点重启，直接启动即可，程序会找到最新的 checkpoint(`output_dir/model_last`)，开始重启训练。请确保重启的训练配置与之前相同。
- visualdl 的日志在 `./output/ernie-1.0-dp8-gb512/train_log/xxx` 中。
</details>



<details>
<summary><b>CLUECorpusSmall 数据集训练效果</b></summary>

#### CLUECorpusSmall 数据集训练效果

使用创建好的训练 clue_corpus_small_14g 数据集。使用本训练脚本, batch_size=512, max_steps=100w，[详细训练日志](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=3fddf650db14b9319f9dc3a91dfe4ac6)

最终训练 loss 结果：

<img width="400" alt="image" src="https://user-images.githubusercontent.com/16911935/167784987-3e51a2ae-df3d-4be6-bacf-0a20e9c272b7.png">

<img width="400" alt="image" src="https://user-images.githubusercontent.com/16911935/167785241-0da271ec-0cd9-446d-a425-64022098a271.png">

|Loss | Train | Validation |
|-|-|-|
|loss |2.59 | 2.48 |
|lm_loss|2.48 | 2.38 |
|sop_loss|0.11 | 0.10 |

训练集 lm_loss=2.48 左右, 验证集 lm_loss=2.38 左右。

使用训练好的模型参数，在下游任务重进行 finetune。这里报告部分数据集上的 finetune 结果：

CLUE 评估结果：

Model | Arch | CLUE AVG |  AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUEWSC2020 | CSL
-- | -- | -- | -- | -- | -- | -- |  -- | -- | --
Metrics |   |   |   | Acc | Acc | Acc | Acc | Acc | Acc | Acc
ERNIE-1.0 Base | 12L768H | 73.78 |  74.95 | 58.73 | 61.37 | 81.77 | 75.46 | 81.25 | 82.93
ERINE-1.0-cluecorpussmall | 12L768H | 73.24(-0.54) | 74.26 | 57.24 | 60.79 | 81.15 | 76.64 | 81.25 | 81.33

注:
- `ERNIE-1.0 Base`官方预训练参数，采用的训练配置是 batch_size=1024、steps=100w，
- `ERINE-1.0-cluecorpussmall`复现版本，采用的是 batch_size=512、steps=100w。

</details>

<a name="ERNIE-CW"></a>

### 2.2 大规模语料预训练: 400GB - CLUE & WuDao

PaddleNLP 致力于预训练开源工作，使用开源中文语料 CLUE、WuDao 总共400GB，提供大规模语料训练教程，让用户可以从零开始构建，基于大规模语料，训练预训练模型。

[ERNIE 中文预训练介绍](./pretraining_introduction.md)，从数据下载，词表制作，数据转化，模型训练，所有流程，完全开源开放，可复现。
并训练发布开源最优的模型参数。

#### 数据准备

数据下载，数据转化部分，请参见[数据预处理文档](../../../llm/tools/preprocess/README.md)，
- [CLUECorpus2020数据处理](../../../llm/tools/preprocess/docs/CLUECorpus2020.md)
- [WuDaoCorpusBase 数据处理](../../../llm/tools/preprocess/docs/WuDaoCorpusBase.md)

如果需要定制化词表，词表制作部分请参考[词表制作](./vocab/README.md)。


#### 训练脚本

训练脚本如下

**环境配置**

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

**路径配置**

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

**启动训练**：

对于`ernie-3.0-base-zh`我们提供了悟道的一个小规模样本的数据：
```
mkdir data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/data_tools/wudao_200g_sample_ernie-3.0-base-zh_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/data_tools/wudao_200g_sample_ernie-3.0-base-zh_idx.npz
cd -
```
同时我们也提供了 `ernie-1.0-base-zh` 的悟道一个小规模样本的数据：
```
https://paddlenlp.bj.bcebos.com/models/transformers/data_tools/wudao_200g_sample_ernie-1.0-base-zh_ids.npy
https://paddlenlp.bj.bcebos.com/models/transformers/data_tools/wudao_200g_sample_ernie-1.0-base-zh_idx.npz
```

可以指定`tokenizer_name_or_path=ernie-3.0-bash-zh`,`input_dir=./data` 用下面的脚本训练。

这里启动的是单机8卡任务，整体全局的 batch_size 512 (64*8)。如果指定 ips 参数，进行多机运行，如 `python3 -u  -m paddle.distributed.launch  --gpus "0,1,2,3,4,5,6,7" --ips 192.168.1.101,192.168.1.101 `
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


<p align="center">
  <img src="https://user-images.githubusercontent.com/16911935/187134299-72628dce-cc04-49d7-89ef-078fad487724.png" align="middle"  width="500" />
</p>

接下来我们主要介绍训练流程部分的特性的简单介绍：详细参数配置介绍请参见[ERNIE 中文预训练介绍](./pretraining_introduction.md)。

- **训练网络配置方面：**

    本小节主要针对，任务的损失函数、MASK 参数等配置进行了简单介绍。
    - SOP Loss
        - SOP (Sentence Order Predict) 损失，是 模型训练的常用损失。将文本中的句子顺序分为两段打乱，最后判断文本是否被打乱。可以通过设置`binary_head`开启或者关闭。
    - MASK
        -  MLM (Mask Language Model) 是通过随机将文本中的部分 token，随机替换为`[MASK]` token，最后预测出真实的 token 值。ERNIE 默认采用了 Whole Word MASK 方式，选定一些词语进行 MASK。
        - *<u>使用方法</u>*: 用户可以设置 `masked_lm_prob` 控制 mask 的 token 占文本总 token 长度的比例。默认`masked_lm_prob=0.15` 随机 mask 15% 的 token 数目。
    - Ngram MASK
        - 项目还支持了 n-gram mask 策略，如下图所示，在 WWM 进行词语级别 MASK 的基础上（如此处 mask 掉的`[模型]`词组），n-gram 可以 MASK 掉连续 n 个词组。下面例子中，连续 mask 了2个词组，`【[语言][模型]】`同时进行了 mask。
        <p align="center">
        <img src="https://user-images.githubusercontent.com/16911935/187145669-7c55386d-f57a-4589-9e6d-e4a36b93e24c.png" align="middle"  width="600" />
        </p>

        - *<u>使用方法</u>*: 用户通过`max_ngrams`设置最大的`ngram`长度。默认`max_ngrams=3`。

    - Dropout
        - Dropout 是常用的防止过拟合策略。对于大规模数据集训练，如`ernie-3.0`系列4T 文本语料，可以设置 `dropout=0`，不考虑过拟合。实际`ernie-3.0-base-zh`训练中，没有开启 Dropout。

详细参数配置介绍请参见[ERNIE 中文预训练介绍](./pretraining_introduction.md)。


- **训练速度方面**

    我们支持了如下策略，加速计算过程，减小显存占用，扩大 batch_size：

    - **多卡多机训练**：
        - 基于飞桨 Fleet 分布式 API，用户可以十分方便的通过数据并行的方法，将训练扩展到多机多卡。
    - **混合精度训练**：
        - 部分算子使用 FP16计算 kernel，加速计算过程。支持 AMP 混合精度 O1，和 Pure FP16全 FP 训练策略 O2。
    - **梯度累积训练**：
        - 用户可以指定梯度累积的步数，在梯度累积的 step 中，减少多卡之间梯度的通信，减少更新的次数，可以扩大训练的 batch_size.
    - **重计算训练**：
        -  通过重新计算前向的方式，减少前向网络中间变量的存储，可以显著减少显存占用，

详细参数配置介绍请参见[ERNIE 中文预训练介绍](./pretraining_introduction.md)。


- **训练数据流方面**

    我们针对训练数据流扩展、混合、重启等方面做了针对性优化提升
    <p align="center">
    <img src="https://user-images.githubusercontent.com/16911935/187355897-478e7aeb-560f-4ea7-a29c-4bea9d8a7712.png" align="middle"  width="500" />
    </p>

    - **多机扩展**
        - 用户可以将数据放置到 NFS 服务器上，多机同时挂载数据即可。训练数据与计算资源分离。
    - **多数据混合**
        - 训练数据集支持多个文件，即插即用，设置权重，传入参数即可`input_dir="1.0  dateset_a/prefix  2.0 dataset_b/prefix"`
    - **稳定可复现**
        - MLM 任务具有一定随机性，需要随机 mask 数据。本数据流通过固定每一个 step 数据的随机种子，实验数据流稳定可复现。
    - **快加载**
        - 数据文件使用 mmap 读取，加载数百 GB 文件几乎不耗时。
    - **断点重启**
        - 用户可以单独设置，checkpoints steps 参数可设置较小，重启训练默认加载最新 checkpoint。
        - 断点数据自动恢复，学习率等参数也自动恢复。

详细参数配置介绍请参见[ERNIE 中文预训练介绍](./pretraining_introduction.md)。

- **观察评估方面**

    - **可视化日志记录**
        - 日志展示为全局 loss，波动小。
        - 记录混合精度，loss_scaling 等信息，方便用户 debug。
        - 对模型结构，配置参数，paddle 版本信息进行记录，方便复现环境
    - **下游任务评估**：CLUE Benchmark 搜索评估参数效果
        - 使用[批量启动-grid-search](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/examples/benchmark/clue#%E6%89%B9%E9%87%8F%E5%90%AF%E5%8A%A8-grid-search)，可以进行批量搜索任务
        - 注意，这里使用的是训练中的 checkpoint 进行评估，可以直接试着 评估待评估的参数为，所在的路径地址，即如 `python grid_seach.py output/ernie-base-outdir/model_100000` 之类的 checkpoint 地址。

详细介绍请参见[ERNIE 中文预训练介绍](./pretraining_introduction.md)。


- **训练效果方面**

    我们 release 了 base、large 两个模型。均取得了较好的预训练效果。

    - **ERNIE 1.0-Base-zh-cw** 模型：
        - 使用 CLUE，WuDao 共计400GB 的语料，batch_size 1024, 训练 400w step，即可训练得到`ernie-3.0-base-zh`类似的模型效果。相关模型参数，开源为`ernie-1.0-base-zh-cw`，用户加载即可使用。使用 CLUE benchmark 对最优超参数进行 GradSearch 搜索：

Model&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Arch | CLUE AVG |  AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUE WSC2020 | CSL | CMRC | CHID | C3
-- | -- | -- | -- | -- | -- | -- |  -- | -- | -- | -- | -- |  -- |
 Metrics |   |   | Acc | Acc | Acc | Acc | Acc | Acc | Acc | Exact/F1| Acc| Acc | Acc
ERNIE 1.0-Base-zh-cw | 12L768H | <b>76.47</b> | 76.07 |    57.86 |    59.91 |    83.41 | 79.91 |    89.91 |   <b>83.42</b> |  72.88/90.78 |    <b>84.68</b> |    76.98 |
ERNIE 2.0-Base-zh | 12L768H | 74.95  | 76.25 |    58.53 |    61.72 |    83.07 |    78.81 |    84.21 |    82.77 | 68.22/88.71    | 82.78    | 73.19
ERNIE 1.0-Base-zh | 12L768H | 74.17 | 74.84 |    58.91 |    62.25 |    81.68 |    76.58 |    85.20 |    82.77 | 67.32/87.83 | 82.47 | 69.68
-
    - **ERNIE 1.0-Large-zh-cw** 模型：

        - 除了 base 模型外，我们还训练了放出了 large 模型。此模型参数采用的是词表与 ernie-1.0相同，因此命名为`ernie-1.0-large-zh-cw`。使用开源语料，batch_size 512, 训练 400w step，训练去除 SOP 任务，只保留 MLM 损失：

Model&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  | Arch | CLUE AVG |  AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUE WSC2020 | CSL | CMRC | CHID | C3
-- | -- | -- | -- | -- | -- | -- |  -- | -- | -- | -- | -- |  -- |
Metrics |   |   | Acc | Acc | Acc | Acc | Acc | Acc | Acc | Exact/F1 | Acc| Acc
ERNIE 1.0-Large-zh-cw | 24L1024H | <b>79.03</b> | 75.97 |    59.65 |    62.91 |    85.09 |    81.73| 93.09 |    84.53 | 74.22/91.88 | 88.57 | 84.54
ERNIE 3.0-Xbase-zh| 20L1024H | 78.71 | 76.85 |    59.89 |    62.41 |    84.76 |    82.51 |    89.80 |    84.47 |    75.49/92.67 | 86.36 | 84.59
RoBERTa-wwm-ext-large | 24L1024H | 76.61 |    76.00 |    59.33 |    62.02 |    83.88 |    78.81 |    90.79 |    83.67 |    70.58/89.82 |    85.72 |    75.26


<a name="预训练模型贡献"></a>

### 预训练模型贡献
PaddleNLP 为开发者提供了[community](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/community/contribute_models/contribute_awesome_pretrained_models.rst)模块，用户可以上传自己训练的模型，开源给其他用户使用。
使用本文档给出的参数配置，在 CLUECorpusSmall 数据集上训练，可以得到`zhui/ernie-1.0-cluecorpussmall`参数，可直接使用。
```python
model = AutoModelForMaskedLM.from_pretrained('zhui/ernie-1.0-cluecorpussmall')
```

贡献预训练模型的方法，可以参考[贡献预训练模型权重](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/community/contribute_models/contribute_awesome_pretrained_models.rst)教程。

<a name="下游任务微调"></a>

## 3. 下游任务微调

使用训练中产出的 checkpoint，或者 paddlenlp 内置的模型权重，使用本脚本，用户可以快速对当前模型效果进行评估。

### 运行示例
本文档适配了三大主流下游任务，用户可以根据自己的需求，评估自己所需的数据集。

<a name="序列分类"></a>

1. 序列分类
```shell
cd finetune
dataset="chnsenticorp_v2"
python run_seq_cls.py \
    --do_train \
    --do_eval \
    --do_predict \
    --model_name_or_path ernie-1.0-base-zh \
    --dataset $dataset \
    --output_dir ./tmp/$dataset
```

<a name="Token 分类"></a>

2. Token 分类
```shell
cd finetune
dataset="peoples_daily_ner"
python run_ner.py \
    --do_train \
    --do_eval \
    --do_predict \
    --model_name_or_path ernie-1.0-base-zh \
    --dataset $dataset \
    --output_dir ./tmp/$dataset
```

<a name="阅读理解"></a>

3. 阅读理解
```shell
cd finetune
dataset="cmrc2018"
python run_qa.py \
    --do_train \
    --do_eval \
    --model_name_or_path ernie-1.0-base-zh \
    --dataset $dataset \
    --output_dir ./tmp/$dataset
```


<a name="预测部署"></a>

## 4. 预测部署
以中文文本情感分类问题为例，介绍一下从模型 finetune 到部署的过程。

与之前的 finetune 参数配置稍有区别，此处加入了一些配置选项。

- do_export，开启模型导出功能
- eval_steps/save_steps 评估和保存的 step 间隔
- metric_for_best_model  模型效果的比较指标。（次选项生效，需要 save_steps 为 eval_steps 的倍数）
- save_total_limit 最多保存的 ckpt 个数。（超过限制数据时，效果更差，或更旧的 ckpt 将被删除）

```shell
cd finetune
# 开始finetune训练并导出模型
dataset="chnsenticorp_v2"
python run_seq_cls.py \
    --do_train \
    --do_eval \
    --do_predict \
    --do_export \
    --model_name_or_path ernie-1.0-base-zh \
    --dataset $dataset \
    --output_dir ./tmp/$dataset \
    --eval_steps 200 \
    --save_steps 200 \
    --metric_for_best_model "eval_accuracy" \
    --load_best_model_at_end \
    --save_total_limit 3 \

```
训练完导出模型之后，可以用于部署，`deploy/seq_cls_infer.py`文件提供了 python 部署预测示例。可执行以下命令运行部署示例：

```shell
python deploy/seq_cls_infer.py --model_dir tmp/chnsenticorp_v2/export/ --device cpu --backend paddle
```

运行后预测结果打印如下：
```text
WARNING: Logging before InitGoogleLogging() is written to STDERR
W0301 08:25:37.617117 58742 analysis_config.cc:958] It is detected that mkldnn and memory_optimize_pass are enabled at the same time, but they are not supported yet. Currently, memory_optimize_pass is explicitly disabled
[INFO] fastdeploy/runtime/runtime.cc(266)::CreatePaddleBackend    Runtime initialized with Backend::PDINFER in Device::CPU.
Batch id: 0, example id: 0, sentence: 这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般, label: negative, negative prob: 0.9999, positive prob: 0.0001.
Batch id: 1, example id: 0, sentence: 怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片！开始还怀疑是不是赠送的个别现象，可是后来发现每张DVD后面都有！真不知道生产商怎么想的，我想看的是猫和老鼠，不是米老鼠！如果厂家是想赠送的话，那就全套米老鼠和唐老鸭都赠送，只在每张DVD后面添加一集算什么？？简直是画蛇添足！！, label: negative, negative prob: 0.9998, positive prob: 0.0002.
Batch id: 2, example id: 0, sentence: 还稍微重了点，可能是硬盘大的原故，还要再轻半斤就好了。其他要进一步验证。贴的几种膜气泡较多，用不了多久就要更换了，屏幕膜稍好点，但比没有要强多了。建议配赠几张膜让用用户自己贴。, label: negative, negative prob: 0.9999, positive prob: 0.0001.
......
```

更多关于部署的情况可以参考[ERNIE 1.0 模型 Python 部署示例](finetune/deploy/README.md)。

<a name="参考文献"></a>

## 5. 参考文献
- [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/pdf/1904.09223.pdf)
