# GPT2 微调

本教程主要针对于 GLUE (General Language Understanding Evaluation) benchmark 中的数据集进行微调，涉及到分类和回归任务。

## 下载 GPT345M 预训练模型
```
# 如果已经下载可以忽略
mkdir -p ckpt
wget -O ckpt/GPT_345M.tar.gz https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz
tar -xzf ckpt/GPT_345M.tar.gz -C ckpt/
```

## 快速体验运行

```
# cd path/to/PaddleFleetX
# bash projects/gpt/finetune_gpt_345M_single_card.sh taskname [split]

# taskname 可选: CoLA, SST2, MRPC, QQP, STSB, MNLI, QNLI, RTE, WNLI
# 例如 bash projects/gpt/finetune_gpt_345M_single_card.sh CoLA

# 注：当数据集为 MNLI 时，验证集有两种，分别是 dev_matched 和 dev_mismatched，
# 其他数据集，只有一种验证集，因此不用选择
# 可以通过 bash projects/gpt/finetune_gpt_345M_single_card.sh MNLI dev_matched
# 或者 bash projects/gpt/finetune_gpt_345M_single_card.sh MNLI dev_mismatched
# 进行 finetune 训练

bash projects/gpt/finetune_gpt_345M_single_card.sh SST2
```

## GLUE benchmark 数据集

GLUE benchmark 包含 9 个数据集，分别是 **CoLA**、**SST-2**、**MRPC**、**QQP**、**STS-B**、**MNLI**、**QNLI**、**RTE**、**WNLI**，涉及到 **自然语言推断**，**文本蕴含**，**情感分析**，**语义相似** 等任务，整体可以归位 3 类，分别是单句任务：CoLA、SST-2；相似性：MRPC、QQP、STS-B；释义：MNLI、QNLI、RTE、WNLI。

以下介绍载自 [huggingface](https://huggingface.co/datasets/nyu-mll/glue).

* CoLA: The Corpus of Linguistic Acceptability consists of English acceptability judgments drawn from books and journal articles on linguistic theory. Each example is a sequence of words annotated with whether it is a grammatical English sentence.
* SST-2: The Stanford Sentiment Treebank consists of sentences from movie reviews and human annotations of their sentiment. The task is to predict the sentiment of a given sentence. We use the two-way (positive/negative) class split, and use only sentence-level labels.
* MRPC: The Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005) is a corpus of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.
* QQP: The Quora Question Pairs2 dataset is a collection of question pairs from the community question-answering website Quora. The task is to determine whether a pair of questions are semantically equivalent.
* STS-B: The Semantic Textual Similarity Benchmark (Cer et al., 2017) is a collection of sentence pairs drawn from news headlines, video and image captions, and natural language inference data. Each pair is human-annotated with a similarity score from 1 to 5.
* MNLI: The Multi-Genre Natural Language Inference Corpus is a crowdsourced collection of sentence pairs with textual entailment annotations. Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral). The premise sentences are gathered from ten different sources, including transcribed speech, fiction, and government reports. We use the standard test set, for which we obtained private labels from the authors, and evaluate on both the matched (in-domain) and mismatched (cross-domain) section. We also use and recommend the SNLI corpus as 550k examples of auxiliary training data.
* QNLI: The Stanford Question Answering Dataset is a question-answering dataset consisting of question-paragraph pairs, where one of the sentences in the paragraph (drawn from Wikipedia) contains the answer to the corresponding question (written by an annotator). We convert the task into sentence pair classification by forming a pair between each question and each sentence in the corresponding context, and filtering out pairs with low lexical overlap between the question and the context sentence. The task is to determine whether the context sentence contains the answer to the question. This modified version of the original task removes the requirement that the model select the exact answer, but also removes the simplifying assumptions that the answer is always present in the input and that lexical overlap is a reliable cue.
* RTE: The Recognizing Textual Entailment (RTE) datasets come from a series of annual textual entailment challenges. We combine the data from RTE1 (Dagan et al., 2006), RTE2 (Bar Haim et al., 2006), RTE3 (Giampiccolo et al., 2007), and RTE5 (Bentivogli et al., 2009).4 Examples are constructed based on news and Wikipedia text. We convert all datasets to a two-class split, where for three-class datasets we collapse neutral and contradiction into not entailment, for consistency.
* WNLI: The Winograd Schema Challenge (Levesque et al., 2011) is a reading comprehension task in which a system must read a sentence with a pronoun and select the referent of that pronoun from a list of choices. The examples are manually constructed to foil simple statistical methods: Each one is contingent on contextual information provided by a single word or phrase in the sentence. To convert the problem into sentence pair classification, we construct sentence pairs by replacing the ambiguous pronoun with each possible referent. The task is to predict if the sentence with the pronoun substituted is entailed by the original sentence. We use a small evaluation set consisting of new examples derived from fiction books that was shared privately by the authors of the original corpus. While the included training set is balanced between two classes, the test set is imbalanced between them (65% not entailment). Also, due to a data quirk, the development set is adversarial: hypotheses are sometimes shared between training and development examples, so if a model memorizes the training examples, they will predict the wrong label on corresponding development set example. As with QNLI, each example is evaluated separately, so there is not a systematic correspondence between a model's score on this task and its score on the unconverted original task. We call converted dataset WNLI (Winograd NLI).


## 微调相关类

### `GPTForSequenceClassification`
在 GPT 模型输出的 logits 基础上，增加一个分类层，并且用正态分布对新增的层参数进行初始化。

```
self.score = nn.Linear(self.gpt.hidden_size, num_classes, bias_attr=False)

from paddle.nn.initializer import Normal
normal_ = Normal(std=self.gpt.initializer_range)
normal_(self.score.weight)
```

### `GPTFinetuneModule`
该类继承自`BasicModule`，负责微调模型的初始化以及逻辑计算的类，需要实现几个重要的函数，下面给出两个具体的示例。

* `__init__`: 负责初始化 loss 函数以及评测指标函数。
* `get_model`: 负责微调类 `GPTForSequenceClassification`、`GPTTokenizer` 初始化以及预训练模型的加载。

## 超参数
微调训练也需要一套完整的超参数，但是微调涉及的核心超参数并不多。

### Engine

| 参数字段         | 参数含义                          |
|------------------|-----------------------------------|
| run_mode         | 运行的模式，需要设置为 epoch 方式 |
| num_train_epochs | 需要 finetune 的 epoch 数         |

```
Engine:
  run_mode: epoch
  num_train_epochs: 3 # WNLI 和 MRPC 数据集比较小，因此 `num_train_epochs=5`。
```

### Model

| 参数字段         | 参数含义                                      |
|------------------|-----------------------------------------------|
| module           | 需要设置为 "GPTFinetuneModule"                |
| name             | 需要设置为 "GPT"                              |
| num_classes      | finetune 时的类别数，根据语料库以及任务来设定 |
| pretrained       | 预训练的权重文件路径前缀，去掉 ".pdparams"    |
| loss.train.name  | finetune 时的训练损失函数类名                 |
| loss.eval.name   | finetune 时的验证损失函数类名                 |
| metric.eval.name | finetune 时的验证评估函数类名                 |

微调时，不同任务对应的类别数 和 loss 函数以及评测指标不同，因此需要通过配置来改变设置。
```
Model:
  module: "GPTFinetuneModule"
  name: "GPT"
  num_classes: 2 # 1 or 2 or 3
  pretrained: 'path/to/pretrained_model'

  loss:
    train:
      name: 'CrossEntropyLoss'
    eval:
      name: 'CrossEntropyLoss'

  metric:
    eval:
      name: 'Accuracy'
```

### Optimizer 和 LRScheduler

| 参数字段         | 参数含义                                                                           |
|------------------|------------------------------------------------------------------------------------|
| name             | 优化器类名                                                                         |
| weight_decay     | 权重衰减值                                                                         |
| beta1            | FusedAdamW 的 beta1                                                                |
| beta2            | FusedAdamW 的 beta2                                                                |
| epsilon          | FusedAdamW 的 epsilon                                                              |
| multi_precision  | 当使用 FP16 O2 级别时，是否开启参数使用多精度表示                                  |
| tensor_fusion    | 是否开启 tensor_fusion                                                             |
| lr.name          | 学习率调整策略类名                                                                 |
| lr.warmup        | 当参数时小数时，表示 warmup 步数占总步数的比例，如果是整数时，则表示 warmup 的步数 |
| lr.learning_rate | 初始化学习率值                                                                     |

注：这里的超参会跟随优化器类的不同而不同，可以自行查看优化器类和学习率调整策略类初始化函数需要设置的超参数设定。

```
Optimizer:
  name: FusedAdamW
  weight_decay: 0.0
  beta1: 0.9
  beta2: 0.999
  epsilon: 1e-6
  multi_precision: True
  tensor_fusion: False
  lr:
    name: LinearDecayWithWarmup
    warmup: 0.1
    learning_rate: 2e-5
```

### Data

| 参数字段      | 参数含义                                              |
|---------------|-------------------------------------------------------|
| Train.dataset | 描述 finetune 时的数据集                              |
| Train.sampler | 描述 dataloader 所需要的 batch sampler                |
| Train.loader  | 描述 dataloader 所需要的相关信息，例如 num_workers 等 |

注：数据集的设定会根据不同任务不同语料库不同而设定不同，例如 `split` 字段，不同数据集是有不同的设定，请参考所需要 finetune 的数据集初始化函数。

```
Data:
  Train:
    dataset:
      name: SST2
      root: ./dataset/SST-2/
      split: 'train'
      max_length: 128
    sampler:
      name: DistributedBatchSampler
      batch_size: 32
      shuffle: True
      drop_last: True
    loader:
      num_workers: 4
      return_list: False

  Eval:
    dataset:
      name: SST2
      root: ./dataset/SST-2/
      split: 'dev'
      max_length: 128
    sampler:
      name: DistributedBatchSampler
      batch_size: 32
      shuffle: False
      drop_last: False
    loader:
      num_workers: 4
      return_list: False
```

## 运行

GLUE benchmark 上的语料库 finetune，大部分设置相同，可以同享一份，只有少量区别处需要改变，因此可以通过超参数的覆盖方式来设置。

数据集加载时会自动判断是否已经缓存下载，如果未缓存下载会自行下载，请保证网络的畅通。当自动下载失败时，可以尝试多次以及检查是否有代理设置等。当下载失败时，也可以自己下载及解压到对应的目录中。

以下是 GLUE benchmark 上的每个语料库的 finetune 单机单卡启动命令：

### CoLA 数据集
```
python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
  -o Data.Train.dataset.name=CoLA \
  -o Data.Train.dataset.root=./dataset/cola_public/ \
  -o Data.Eval.dataset.name=CoLA \
  -o Data.Eval.dataset.root=./dataset/cola_public/ \
  -o Data.Eval.dataset.split=dev \
  -o Model.metric.train.name=Mcc \
  -o Model.metric.eval.name=Mcc
  -o Model.num_classes=2
```

### SST2 数据集
```
python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
  -o Data.Train.dataset.name=SST2 \
  -o Data.Train.dataset.root=./dataset/SST-2/ \
  -o Data.Eval.dataset.name=SST2 \
  -o Data.Eval.dataset.root=./dataset/SST-2/ \
  -o Data.Eval.dataset.split=dev \
  -o Model.num_classes=2
```

### MRPC 数据集
```
python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
  -o Engine.num_train_epochs=5 \
  -o Data.Train.dataset.name=MRPC \
  -o Data.Train.dataset.root=./dataset/MRPC/ \
  -o Data.Eval.dataset.name=MRPC \
  -o Data.Eval.dataset.root=./dataset/MRPC/ \
  -o Data.Eval.dataset.split=test \
  -o Model.num_classes=2 \
  -o Model.metric.train.name=AccuracyAndF1 \
  -o Model.metric.eval.name=AccuracyAndF1
```

### QQP 数据集
```
python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
  -o Data.Train.dataset.name=QQP \
  -o Data.Train.dataset.root=./dataset/QQP/ \
  -o Data.Eval.dataset.name=QQP \
  -o Data.Eval.dataset.root=./dataset/QQP/ \
  -o Data.Eval.dataset.split=dev \
  -o Model.num_classes=2 \
  -o Model.metric.train.name=AccuracyAndF1 \
  -o Model.metric.eval.name=AccuracyAndF1
```

### STSB 数据集
```
python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
  -o Data.Train.dataset.name=STSB \
  -o Data.Train.dataset.root=./dataset/STS-B/ \
  -o Data.Eval.dataset.name=STSB \
  -o Data.Eval.dataset.root=./dataset/STS-B/ \
  -o Data.Eval.dataset.split=dev \
  -o Model.num_classes=1 \
  -o Model.metric.train.name=PearsonAndSpearman \
  -o Model.metric.eval.name=PearsonAndSpearman \
  -o Model.loss.train.name=MSELoss \
  -o Model.loss.eval.name=MSELoss
```

### MNLI 数据集

注：MNLI 数据集验证集分为 `dev_matched` 和 `dev_mismatched`，目前暂不支持两个集合同时评测，如果要评测两种验证集，有两种方法：

* 分别 finetune 2次，Data.Eval.dataset.split 设置不同的验证集
* 保存 finetune 后的 checkpoint，在不同验证集上离线评测。


```
python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
  -o Data.Train.dataset.name=MNLI \
  -o Data.Train.dataset.root=./dataset/multinli_1.0 \
  -o Data.Eval.dataset.name=MNLI \
  -o Data.Eval.dataset.root=./dataset/multinli_1.0 \
  -o Data.Eval.dataset.split=dev_matched \
  -o Model.num_classes=3
```

### QNLI 数据集
```
python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
  -o Data.Train.dataset.name=QNLI \
  -o Data.Train.dataset.root=./dataset/QNLI/ \
  -o Data.Eval.dataset.name=QNLI \
  -o Data.Eval.dataset.root=./dataset/QNLI/ \
  -o Data.Eval.dataset.split=dev \
  -o Model.num_classes=2
```

### RTE 数据集
```
python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
  -o Data.Train.dataset.name=RTE \
  -o Data.Train.dataset.root=./dataset/RTE/ \
  -o Data.Eval.dataset.name=RTE \
  -o Data.Eval.dataset.root=./dataset/RTE/ \
  -o Data.Eval.dataset.split=dev \
  -o Model.num_classes=2
```

### WNLI 数据集
```
python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
  -o Engine.num_train_epochs=5 \
  -o Data.Train.dataset.name=WNLI \
  -o Data.Train.dataset.root=./dataset/WNLI/ \
  -o Data.Eval.dataset.name=WNLI \
  -o Data.Eval.dataset.root=./dataset/WNLI/ \
  -o Data.Eval.dataset.split=dev \
  -o Model.num_classes=2
```


## 运行结果

以下的指标是通过 [GPT_345M](https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz) 预训练模型 finetune 得到的结果，仅作为参考。

| Corpus | Task                | Domanin            | Metric                       | Result          |
|--------|---------------------|--------------------|------------------------------|-----------------|
| CoLA   | acceptability       | Misc.              | Matthews corr                | 0.60471         |
| SST-2  | sentiment           | Movie reviews      | Accuracy                     | 0.93005         |
| MNLI   | NLI                 | Misc.              | Matched acc./Mismatched acc. | 0.84238/0.84815 |
| QNLI   | QA/NLI              | Wikipedia          | Accuracy                     | 0.90445         |
| RTE    | NLI                 | News, Wikipedia    | Accuracy                     | 0.70397         |
| WNLI   | coreference         | Books              | Accuracy                     | 0.40845         |
| MRPC   | paraphrase          | News               | Accuracy/F1                  | 0.81913/0.87022 |
| QQP    | paraphrase          | social QA question | Accuracy/F1                  | 0.86087/0.81055 |
| STS-B  | sentence similarity | Misc.              | Pearson/Spearman corr.       | 0.85797/0.85824 |
