# ERNIE: Enhanced Representation through kNowledge IntEgration

ERNIE是百度开创性提出的基于知识增强的持续学习语义理解框架，它将大数据预训练与多源丰富知识相结合，通过持续学习技术，不断吸收海量文本数据中词汇、结构、语义等方面的知识，实现模型效果不断进化。

ERNIE在情感分析、文本匹配、自然语言推理、词法分析、阅读理解、智能问答等16个公开数据集上全面显著超越世界领先技术，在国际权威的通用语言理解评估基准GLUE上，得分首次突破90分，获得全球第一。
相关创新成果也被国际顶级学术会议AAAI、IJCAI收录。
同时，ERNIE在工业界得到了大规模应用，如搜索引擎、新闻推荐、广告系统、语音交互、智能客服等。

ERNIE 1.0 通过建模海量数据中的词、实体及实体关系，学习真实世界的语义知识。相较于 BERT 学习原始语言信号，ERNIE 直接对先验语义知识单元进行建模，增强了模型语义表示能力。

这里我们举个例子：
```
Learnt by BERT ：哈 [mask] 滨是 [mask] 龙江的省会，[mask] 际冰 [mask] 文化名城。
Learnt by ERNIE：[mask] [mask] [mask] 是黑龙江的省会，国际 [mask] [mask] 文化名城。
```
在 BERT 模型中，我们通过『哈』与『滨』的局部共现，即可判断出『尔』字，模型没有学习与『哈尔滨』相关的任何知识。而 ERNIE 通过学习词与实体的表达，使模型能够建模出『哈尔滨』与『黑龙江』的关系，学到『哈尔滨』是 『黑龙江』的省会以及『哈尔滨』是个冰雪城市。


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
├── run_gb512_s1m.sh        训练启动shell脚本，batch size 512. max steps 100w
├── run_gb512_s1m_static.sh
├── run_gb512_s1m_trainer.sh
├── run_pretrain.py         训练启动python脚本
├── run_pretrain_static.py
└── run_pretrain_trainer.py
```
## 环境依赖

- visualdl
- pybind11

安装命令 `pip install visualdl pybind11`


## 中文预训练
ERNIE预训练采用的是MLM（Mask Language Model）的训练方式，采用WWM（Whole Word Mask）方式，对于完整语义单元的Token，会同时进行Mask。整体的训练损失loss是mlm_loss + nsp_loss。

本样例为用户提供了高效的训练流程，支持动态文本mask，自动断点训练重启等功能。
用户可以根据自己的需求，灵活修改mask方式。具体可以参考修改`data_tools/dataset_utils.py`中`create_masked_lm_predictions`函数。
用户可以设置`checkpoint_steps`，间隔`checkpoint_steps`数，即保留最新的checkpoint到`model_last`文件夹。重启训练时，程序默认从最新checkpoint重启训练，学习率、数据集都可以恢复到checkpoint时候的状态。

### 数据准备
数据下载部分请参考[data_tools]目录，根据文档中`CLUECorpusSmall 数据集处理教程`，下载数据。下载好后:

解压文件
```shell
unzip comment2019zh_corpus.zip -d  clue_corpus_small_14g/comment2019zh_corpus
unzip news2016zh_corpus.zip    -d  clue_corpus_small_14g/news2016zh_corpus
unzip webText2019zh_corpus.zip -d  clue_corpus_small_14g/webText2019zh_corpus
unzip wiki2019zh_corpus.zip    -d  clue_corpus_small_14g/wiki2019zh_corpus
```
将txt文件转换为jsonl格式
```
python data_tools/trans_to_json.py  --input_path ./clue_corpus_small_14g --output_path clue_corpus_small_14g.jsonl
```
现在我们得到了jsonl格式的数据集，下面是针对训练任务的数据集应用，此处以ernie为例。
```
python -u  data_tools/create_pretraining_data.py \
    --model_name ernie-1.0-base-zh \
    --tokenizer_name ErnieTokenizer \
    --input_path clue_corpus_small_14g.jsonl \
    --split_sentences\
    --chinese \
    --cn_whole_word_segment \
    --cn_seg_func jieba \
    --output_prefix clue_corpus_small_14g_20220104 \
    --workers 48 \
    --log_interval 10000
```
数据共有文档`15702702`条左右，由于分词比较耗时，大概一小时左右可以完成。在当前目录下产出训练所需数据。
```
clue_corpus_small_14g_20220104_ids.npy
clue_corpus_small_14g_20220104_idx.npz
```

###  开始训练

将制作好的数据`clue_corpus_small_14g_20220104_ids.npy,clue_corpus_small_14g_20220104_idx.npz`移动到input_dir中，即可开始训练。
这里以8卡训练为例任务脚本为例：
```
python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/ernie-1.0-dp8-gb512/log" \
    run_pretrain.py \
    --model_type "ernie" \
    --model_name_or_path "ernie-1.0-base-zh" \
    --input_dir "./data" \
    --output_dir "output/ernie-1.0-dp8-gb512" \
    --split 949,50,1 \
    --max_seq_len 512 \
    --micro_batch_size 64 \
    --use_amp true \
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

其中参数释义如下：
- `model_name_or_path` 要训练的模型或者之前训练的checkpoint。
- `input_dir` 指定输入文件，可以使用目录，指定目录时将包括目录中的所有文件。
- `output_dir` 指定输出文件。
- `split` 划分数据集为train、valid、test的比例。整个数据集会按照这个比例划分数据。默认1/1000的数据为test，当样本数太少时，请修改此比例。
- `max_seq_len` 输入文本序列的长度。
- `micro_batch_size` 单卡batch size大小，比如此处单卡bs=64, 采用8卡训练`global_batch_size=64*8=512`。
- `use_amp` 开启混合精度策略。
- `max_lr` 训练学习率。
- `min_lr` 学习率衰减的最小值。
- `max_steps` 最大训练步数。
- `save_steps` 保存模型间隔。默认保存地址格式为`output_dir/model_50000`(5w 步时的权重)。
- `checkpoint_steps` 模型checkpoint间隔，用于模型断点重启训练。默认地址为`output_dir/model_last`.
- `weight_decay` 权重衰减参数。
- `warmup_rate` 学习率warmup参数。
- `grad_clip` 梯度裁剪范围。
- `logging_freq` 日志输出间隔。
- `num_workers` DataLoader采样进程，当数据输入为瓶颈时，可尝试提高采样进程数目。
- `eval_freq` 模型评估间隔。
- `device` 训练设备，默认为GPU。
- `share_folder` 多机训练时，如果多机input_dir为挂载的同一个nfs网络位置，可以开启次选项，多机共享同一份数据。


注：
- 训练支持断点重启，直接启动即可，程序会找到最新的checkpoint(`output_dir/model_last`)，开始重启训练。请确保重启的训练配置与之前相同。
- visualdl的日志在 `./output/ernie-1.0-dp8-gb512/train_log/xxx` 中。


### CLUECorpusSmall 数据集训练效果

使用创建好的训练clue_corpus_small_14g数据集。使用本训练脚本, batch_size=512, max_steps=100w，[详细训练日志](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=3fddf650db14b9319f9dc3a91dfe4ac6)

最终训练loss结果：

<img width="400" alt="image" src="https://user-images.githubusercontent.com/16911935/167784987-3e51a2ae-df3d-4be6-bacf-0a20e9c272b7.png">

<img width="400" alt="image" src="https://user-images.githubusercontent.com/16911935/167785241-0da271ec-0cd9-446d-a425-64022098a271.png">

|Loss | Train | Validation |
|-|-|-|
|loss |2.59 | 2.48 |
|lm_loss|2.48 | 2.38 |
|sop_loss|0.11 | 0.10 |

训练集 lm_loss=2.48 左右, 验证集 lm_loss=2.38 左右。

使用训练好的模型参数，在下游任务重进行finetune。这里报告部分数据集上的finetune结果：

CLUE评估结果：

Model | Arch | CLUE AVG |  AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUEWSC2020 | CSL
-- | -- | -- | -- | -- | -- | -- |  -- | -- | --
Metrics |   |   |   | Acc | Acc | Acc | Acc | Acc | Acc | Acc
ERNIE-1.0 Base | 12L768H | 73.78 |  74.95 | 58.73 | 61.37 | 81.77 | 75.46 | 81.25 | 82.93
ERINE-1.0-cluecorpussmall | 12L768H | 73.24(-0.54) | 74.26 | 57.24 | 60.79 | 81.15 | 76.64 | 81.25 | 81.33

注:
- `ERNIE-1.0 Base`官方预训练参数，采用的训练配置是batch_size=1024、steps=100w，
- `ERINE-1.0-cluecorpussmall`复现版本，采用的是batch_size=512、steps=100w。

### 预训练模型贡献
PaddleNLP为开发者提供了[community](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/community/contribute_models/contribute_awesome_pretrained_models.rst)模块，用户可以上传自己训练的模型，开源给其他用户使用。
使用本文档给出的参数配置，在CLUECorpusSmall数据集上训练，可以得到`zhui/ernie-1.0-cluecorpussmall`参数，可直接使用。
```python
model = AutoModelForMaskedLM.from_pretrained('zhui/ernie-1.0-cluecorpussmall')
```

贡献预训练模型的方法，可以参考[贡献预训练模型权重](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/community/contribute_models/contribute_awesome_pretrained_models.rst)教程。


## 下游任务finetune

使用训练中产出的checkpoint，或者paddlenlp内置的模型权重，使用本脚本，用户可以快速对当前模型效果进行评估。

### 运行示例
本文档适配了三大主流下游任务，用户可以根据自己的需求，评估自己所需的数据集。

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

2. Token分类
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


## 预测部署
以中文文本情感分类问题为例，介绍一下从模型finetune到部署的过程。

与之前的finetune参数配置稍有区别，此处加入了一些配置选项。

- do_export，开启模型导出功能
- eval_steps/save_steps 评估和保存的step间隔
- metric_for_best_model  模型效果的比较指标。（次选项生效，需要save_steps为eval_steps的倍数）
- save_total_limit 最多保存的ckpt个数。（超过限制数据时，效果更差，或更旧的ckpt将被删除）

```shell
cd finetune
# 开始finetune训练
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
# 预测结果
python deploy/predict_chnsenticorp.py --model_dir=./tmp/chnsenticorp_v2/export
```
训练完，导出模型之后，可以用于部署，`deploy/predict_chnsenticorp.py`文件提供了python部署预测示例。
运行后预测结果打印如下：
```text
Data: 东西不错，不过有人不太喜欢镜面的，我个人比较喜欢，总之还算满意。   Label: positive
Data: 房间不错,只是上网速度慢得无法忍受,打开一个网页要等半小时,连邮件都无法收。另前台工作人员服务态度是很好，只是效率有得改善。          Label: positive
Data: 挺失望的,还不如买一本张爱玲文集呢,以<色戒>命名,可这篇文章仅仅10多页,且无头无尾的,完全比不上里面的任意一篇其它文章.         Label: negative
```
更多关于部署的情况可以参考[此处](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification/pretrained_models#%E6%A8%A1%E5%9E%8B%E9%A2%84%E6%B5%8B)。

## 参考文献
- [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/pdf/1904.09223.pdf)
