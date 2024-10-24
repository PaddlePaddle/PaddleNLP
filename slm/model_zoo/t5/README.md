# T5

[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683v3.pdf)

## 摘要

迁移学习在自然语言处理(NLP)中已经成为一种强大的技术。迁移学习首先在数据丰富的任务上进行预训练，然后在下游任务上进行调整。迁移学习的有效性引起了不同的方法、方法和实践。在本文中，我们通过引入一个统一的框架，将所有基于文本的语言问题转换为文本到文本的格式，来探索自然语言处理的迁移学习技术。我们的系统研究比较了数十项语言理解任务的训练前目标、架构、未标记数据集、迁移方法和其他因素。通过将我们的探索与规模和我们的新"Colossal Clean Crawled Corpus"数据集相结合，我们在摘要、问答、文本分类等许多基准测试中取得了最先进的结果。为了促进 NLP 迁移学习的未来工作，我们发布了我们的数据集、预训练模型和代码。

本项目是 T5在 Paddle 2.x 上的开源实现，包含了`模型权重`转换代码和`GLUE 任务`的微调代码。

## 快速开始

### 预训练

本项目致力于 t5模型的预训练，从数据下载，数据转化，模型训练，流程开源开放，可复现。

接下来将从下面几个方面，详细介绍整个数据制作全流程，从零开始，构建一个预训练模型。

#### 1. 数据准备

数据流是预训练的非常重要的，[预处理文档](../../../llm/tools/preprocess/README.md)提供了整体的数据变动的流程示意，用户可以查看数据制作的细节文档。

在数据 ID 化步骤中，我们需要配置 tokenzer_name，选择 t5模型对应的 tokenizer；通过下面脚本转化，我们可以得到处理好的预训练数据，token ids:[`t5_openwebtext.bin`](https://paddlenlp.bj.bcebos.com/models/transformers/t5/data/t5_openwebtext.bin), 文章索引信息[`t5_openwebtext.idx`](https://paddlenlp.bj.bcebos.com/models/transformers/t5/data/t5_openwebtext.idx).（这里提供了一个处理好的预训练数据，可点击链接下载）

```shell
python -u  create_pretraining_data.py \
    --model_name t5-small \
    --tokenizer_name T5Tokenizer \
    --data_format JSON \
    --input_path openwebtext/2020-04.jsonl.zst \
    --split_sentences \
    --output_prefix t5_openwebtext  \
    --workers 1 \
    --log_interval 5 \
    --data_impl mmap
```

#### 2. 开始训练

**路径配置**

- 主要配置输入输出目录
- 这里的`tokenizer_name_or_path`请设置为内置的 tokenizer，如`t5-small`等。
- 这里的 `input_dir` 设置输入数据集路径，例如配置`input_dir "./data"`即可。

**启动训练**：这里启动的是单机8卡任务，整体全局的 batch_size 512 (64*8)。如果指定 ips 参数，进行多机运行，如 `python3 -u  -m paddle.distributed.launch  --gpus "0,1,2,3,4,5,6,7" --ips 192.168.1.101,192.168.1.101 `

```shell
python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "./log" \
    t5_run_pretrain_trainer.py \
    --model_type "t5" \
    --model_name_or_path "t5-small" \
    --tokenizer_name_or_path "${vocab_dir}" \
    --input_dir "${data_dir}" \
    --output_dir "${base_dir}" \
    --split 10,5,1 \
    --max_seq_length 512 \
    --max_seq_length_dec 128 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --learning_rate 0.0001 \
    --min_learning_rate 0.00001 \
    --max_steps 20000 \
    --save_steps 5000 \
    --weight_decay 0.01 \
    --decay_steps 9900 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 10\
    --dataloader_num_workers 4 \
    --eval_steps 100 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --do_train \
    --do_eval \
    --seed 1234 \
    --device "gpu" \
    --data_impl "mmap"
```

其中参数释义如下：

- `model_name_or_path` 要训练的模型或者之前训练的 checkpoint。
- `tokenizer_name_or_path` 模型词表文件所在的文件夹(对于 ernie，词表文件名一般命名为 vocab.txt)，或者 PaddleNLP 内置 tokenizer 的名字。
- `input_dir` 指定输入文件，可以使用目录，指定目录时将包括目录中的所有文件。
- `output_dir` 指定输出文件。
- `split` 划分数据集为 train、valid、test 的比例。整个数据集会按照这个比例划分数据。默认`split=949,50,1`, 使用1/1000的数据为 test，当样本数太少时，增大测试的样本数目。
- `max_seq_len` 输入文本序列的长度，默认值`512`。
- `fp16_opt_level` 混合精度策略，支持 O1 自动混合精度，O2 pure fp16精度训练。
- `max_steps` 最大训练步数。训练不支持通过`epoch`控制，第一次制造数据 index 时候，日志会显示数据会被计算的 epoch 数，请注意查看。
- `save_steps` 保存模型间隔。默认保存地址格式为`output_dir/model_50000`(5w 步时的权重)。
- `weight_decay` 权重衰减参数。
- `warmup_rate` 学习率 warmup 参数。
- `max_grad_norm` 梯度裁剪范围。
- `logging_steps` 日志输出间隔。
- `dataloader_num_workers` DataLoader 采样进程，当数据输入为瓶颈时，可尝试提高采样进程数目。
- `eval_steps` 模型评估间隔。
- `device` 训练设备，默认为 GPU。
- `data_impl` 指定输入文件数据制作类型，默认为 mmap，可指定 mmap 或 lazy。mmap 格式在读入数据时会建立内存映射，lazy 格式在读入数据时直接从文件读取。

### GLUE 任务

### 执行 Fine-tunning

启动 rte 分类任务的 Fine-tuning 的方式如下：

```shell
python run_glue.py \
    --model_name_or_path t5-base \
    --task_name rte \
    --max_seq_length 256 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_radio 0.1 \
    --num_train_epochs 10 \
    --logging_steps 100 \
    --save_steps 100 \
    --seed 42 \
    --scheduler_type linear \
    --output_dir outputs/rte/ \
    --device gpu
```

其中参数释义如下：

- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录的地址。
- `task_name` GLUE 任务名称，可从选["cola","sst-2","mrpc","sts-b","qqp","mnli", "rte", "qnli"]选择。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `train_batch_size` 表示训练时的样本数目。
- `eval_batch_size` 表示验证时的样本数目。
- `learning_rate` 表示基础学习率大小，将于 learning rate scheduler 产生的值相乘作为当前学习率。
- `warmup_radio` warmup 比率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `seed` 表示随机种子。
- `scheduler_type` scheduler 类型，可选 linear 和 cosine，默认 linear。
- `output_dir` 表示模型保存路径。
- `device` 表示训练使用的设备，可选 cpu、gpu 或 npu。

使用 trainer 进行 Fine-tuning:

```shell
python -m paddle.distributed.launch --gpus "0,1,2,3" run_glue_trainer.py \
    --model_name_or_path t5-base \
    --task_name rte \
    --max_seq_length 256 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 64 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --num_train_epochs 10 \
    --eval_steps 200 \
    --logging_steps 20 \
    --save_steps 200 \
    --save_total_limit 3 \
    --metric_for_best_model "eval_accuracy" \
    --fp16 false \
    --fp16_opt_level "O1" \
    --recompute true \
    --sharding "stage1" \
    --overwrite_output_dir \
    --disable_tqdm true \
    --output_dir outputs/rte/
```

具体参数含义请参见: https://paddlenlp.readthedocs.io/zh/latest/trainer.html

###### t5-base 模型在 GLUE 开发集上的结果：

| Model          | cola  | sst-2 | mrpc  | sts-b   | qqp   | mnli  | qnli  | rte   | mean    |
| -------------- | ----- | ----- | ----- | ------- | ----- | ----- | ----- | ----- | ------- |
|                | mcc   | acc   | acc   | pearson | acc   | acc   | acc   | acc   |         |
| T5-base-Paddle | 61.74 | 95.18 | 90.44 | 90.09   | 91.60 | 87.18 | 93.56 | 81.95 | 86.4675 |

###### t5_v1_1-base 模型在 GLUE 开发集上的结果：

使用`run_glue_trainer.py`运行，由于`t5_v1_1-base`没有在 glue 任务上进行训练过，直接生成 label 的策略需要的训练时间需要更长。

| Model               | cola    | sst-2 | mrpc  | sts-b   | qqp   | mnli  | qnli   | rte   |
| ------------------- | ------- | ----- | ----- | ------- | ----- | ----- | ------ | ----- |
|                     | mcc     | acc   | acc   | pearson | acc   | acc   | acc    | acc   |
| T5-v1_1-base Paddle | 47.6845 | 94.38 | 84.31 | 87.74   | 88.05 | 85.39 | 90.518 | 65.70 |
| epoch               | 100     | 10    | 100   | 100     | 3     | 3     | 10     | 100   |

注：

- 直接生成 label 的 finetune 方式难度较大，前期基本学习如何正确生成 label 标签，后期才学习分类任务。
- 生成的 label 标签设计，标签差异大一些，效果会更好一些。
- `qqp`,`mnli`数据集适当增大训练 epoch 数，可以取得更好效果。

### CLUE 任务

使用 trainer 进行 Fine-tuning:

```shell
python -m paddle.distributed.launch --gpus "0,1,2,3" run_clue_trainer.py \
    --model_name_or_path Langboat/mengzi-t5-base-mt \
    --task_name cluewsc2020 \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 64 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --num_train_epochs 100 \
    --eval_steps 200 \
    --logging_steps 20 \
    --save_steps 200 \
    --save_total_limit 3 \
    --metric_for_best_model "eval_accuracy" \
    --fp16 false \
    --fp16_opt_level "O1" \
    --recompute true \
    --sharding "stage1" \
    --overwrite_output_dir \
    --disable_tqdm true \
    --output_dir outputs/clue/cluewsc2020
```

###### Langboat/mengzi-t5-base-mt 模型在 CLUE 开发集上的结果：

| Model                      | afqmc | tnews | iflytek | cmnli | ocnli | cluewsc2020 | csl   |
| :------------------------- | :---- | :---- | :------ | :---- | :---- | :---------- | :---- |
|                            | acc   | acc   | acc     | acc   | acc   | acc         | acc   |
| Langboat/mengzi-t5-base-mt | 74.44 | 58.47 | 61.14   | 80.97 | 75.76 | 79.61       | 84.47 |
| epoch                      | 10    | 10    | 10      | 10    | 10    | 100         | 10    |



### GLUE Demo 测试

```sh
python glue_demo.py
input text: sst2 sentence: contains no wit , only labored gags
label: negative
==================================================
input text: sst2 sentence: that loves its characters and communicates something rather beautiful about human nature
label: positive
==================================================
input text: cola sentence: Mickey looked it up.
label: acceptable
==================================================
input text: sst2 sentence: remains utterly satisfied to remain the same throughout
label: positive
==================================================
input text: sst2 sentence: a well-made and often lovely depiction of the mysteries of friendship
label: positive
==================================================
```

### Zero shot Demo 测试 [参考自 Langboat/mengzi-zero-shot](https://github.com/Langboat/mengzi-zero-shot)

```sh
python zero_shot_demo.py
```

当前**zero shot**时输入的构造方法如下表所示。

| **任务类型**         | **prompt 构造（其中{s}代表句子输**入）                        |
| -------------------- | ------------------------------------------------------------ |
| **实体抽取**         | “{s}”找出上述句子中的实体和他们对应的类别                    |
| **语义相似度**       | “{s1}”和“{s2}”这两句话是在说同一件事吗?                      |
| **金融关系抽取**     | “{s}”中的“{e1}”和“{e2}”是什么关系？答:                       |
| **广告文案生成**     | 请根据以下产品信息设计广告文案。商品信息:{s}                 |
| **医学领域意图分类** | 问题:“{s}”。此问题的医学意图是什么？选项：病情诊断，病因分析，治疗方案，就医建议，指标解读，疾病描述，后果表述，注意事项，功效作用，医疗费用。 |
| **评论情感分类**     | 评论:{s}。请判断该条评论所属类别(积极或消极)并填至空格处。回答： |
| **评论对象抽取**     | 评论:{s}.这条评论的评价对象是谁？                            |
| **新闻分类**         | “{s}”是什么新闻频道写的？选项：故事，文化，娱乐，体育，财经，房产，汽车，教育，科技，军事，旅游，国际，股票，农业，电竞。答： |

```
input_text: “导致泗水的砭石受到追捧，价格突然上涨。而泗水县文化市场综合执法局颜鲲表示，根据监控”找出上述句子中的实体和他们对应的类别
output: 泗水县文化市场综合执法局:政府,颜鲲:姓名
==================================================
input_text: “你好，我还款银行怎么更换”和“怎么更换绑定还款的卡”这两句话是在说同一件事吗?
output: 是
==================================================
input_text: “为打消市场顾虑,工行两位洋股东——美国运通和安联集团昨晚做出承诺,近期不会减持工行H股。”中的“工行”和“美国运通”是什么关系？答:
output: 被持股
==================================================
input_text: 请根据以下产品信息设计广告文案。商品信息:类型-裤，版型-宽松，风格-潮，风格-复古，风格-文艺，图案-复古，裤型-直筒裤，裤腰型-高腰，裤口-毛边
output: 这款牛仔裤采用高腰直筒的版型设计,搭配宽松的裤型,穿着舒适又显潮流感。而裤脚的毛边设计,增添几分复古文艺的气息。
==================================================
input_text: 问题:“呼气试验阳性什么意思”。此问题的医学意图是什么？选项：病情诊断，病因分析，治疗方案，就医建议，指标解读，疾病描述，后果表述，注意事项，功效作用，医疗费用。
output: 指标解读
==================================================
input_text: 评论:房间很一般，小，且让人感觉脏，隔音效果差，能听到走廊的人讲话，走廊光线昏暗，旁边没有什么可吃。请判断该条评论所属类别(积极或消极)并填至空格处。回答：
output: 消极
==================================================
input_text: 评论:灵水的水质清澈，建议带个浮潜装备，可以看清湖里的小鱼。.这条评论的评价对象是谁？
output: 灵水
==================================================
input_text: “懒人适合种的果树：长得多、好打理，果子多得都得送邻居吃”是什么新闻频道写的？选项：故事，文化，娱乐，体育，财经，房产，汽车，教育，科技，军事，旅游，国际，股票，农业，电竞。答：
output: 农业
==================================================
```

# Reference

```bibtex
@article{2020t5,
  author  = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
  title   = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {140},
  pages   = {1-67},
  url     = {http://jmlr.org/papers/v21/20-074.html}
}
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
