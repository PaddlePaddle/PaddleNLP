# 无监督检索式问答系统

**目录**
- [无监督检索式问答系统](#无监督检索式问答系统)
  - [简介](#简介)
    - [项目优势](#项目优势)
  - [方案介绍](#方案介绍)
    - [流程图](#流程图)
    - [技术方案](#技术方案)
    - [代码结构说明](#代码结构说明)
  - [快速体验](#快速体验)
    - [运行环境和安装说明](#运行环境和安装说明)
    - [数据说明](#数据说明)
    - [快速体验无监督检索式问答系统](#快速体验无监督检索式问答系统)
  - [可视化无监督检索式问答系统](#可视化无监督检索式问答系统)
    - [离线问答对语料构建](#离线问答对语料构建)
    - [基于 Pipelines 构建问答系统](#基于 Pipelines 构建问答系统)
  - [自定义模型](#自定义模型)
    - [数据准备](#数据准备)
    - [模型微调](#模型微调)
      - [答案抽取](#答案抽取)
      - [问题生成](#问题生成)
      - [过滤模型](#过滤模型)
      - [语义索引和召回模型](#语义索引和召回模型)
      - [排序模型](#排序模型)
  - [References](#References)

## 简介
问答（QA）系统中最关键的挑战之一是标记数据的稀缺性，这是因为对目标领域获取问答对或常见问答对（FAQ）的成本很高，需要消耗大量的人力和时间。由于上述制约，这导致检索式问答系统落地困难，解决此问题的一种方法是依据问题上下文或大量非结构化文本自动生成的 QA 问答对。

在此背景下，无监督检索式问答系统（即问答对自动生成智能检索式问答），基于 PaddleNLP[问题生成](../../../examples/question_generation/README.md)、[UIE](../../../model_zoo/uie/README.md)、[检索式问答](https://github.com/PaddlePaddle/PaddleNLP/blob/release/2.8/applications/question_answering/supervised_qa/faq_finance/README.md)，支持以非结构化文本形式为上下文自动生成 QA 问答对，生成的问答对语料可以通过无监督的方式构建检索式问答系统。

若开发者已有 FAQ 语料，请参考[supervised_qa](https://github.com/PaddlePaddle/PaddleNLP/blob/release/2.8/applications/question_answering/supervised_qa)。

### 项目优势
具体来说，本项目具有以下优势：

+ 低成本
    + 可通过自动生成的方式快速大量合成 QA 语料，大大降低人力成本
    + 可控性好，合成语料和语义检索问答解耦合，可以人工筛查和删除合成的问答对，也可以添加人工标注的问答对

+ 低门槛
    + 手把手搭建无监督检索式问答系统
    + 无需相似 Query-Query Pair 标注数据也能构建问答系统

+ 效果好
    + 可通过自动问答对生成提升问答对语料覆盖度，缓解中长尾问题覆盖较少的问题
    + 业界领先的检索预训练模型: RocketQA Dual Encoder
    + 针对无标注数据场景的领先解决方案: 检索预训练模型 + 增强的无监督语义索引微调

+ 端到端
    + 提供包括问答语料生成、索引库构建、模型服务部署、WebUI 可视化一整套端到端智能问答系统能力
    + 支持对 Txt、Word、PDF、Image 多源数据上传，同时支持离线、在线 QA 语料生成和 ANN 数据库更新

## 方案介绍
<!-- ### 评估指标
**问答对生成**：问答对生成使用的指标是软召回率 Recall@K，
**语义索引**：语义索引使用的指标是 Recall@K，表示的是预测的前 topK（从最后的按得分排序的召回列表中返回前 K 个结果）结果和语料库中真实的前 K 个相关结果的重叠率，衡量的是检索系统的查全率。 -->
### 流程图
本项目的流程图如下，对于给定的非结构化文本，我们首先通过答案抽取、问题生成、以及往返过滤模块，得到大量语料相关的问答对。针对这些得到的问答对，用户可以通过可以人工筛查和删除的方式来调整生成的问答对，也可以进一步添加人工标注的问答对。随后开发者就可以通过语义索引模块，来构建向量索引库。在构造完索引库之后，我们就可以通过召回模块和排序模块对问答对进行查询，得到最终的查询结果。

<div align="center">
    <img width="700" alt="image" src="https://user-images.githubusercontent.com/20476674/211868709-2ac0932d-c48b-4f87-b1cf-1f2665e5a64e.png">
</div>

### 技术方案
由于涉及较多的模块，本项目将基于 PaddleNLP Pipelines 进行模块的组合和项目的构建。PaddleNLP Pipelines 是一个端到端 NLP 流水线系统框架，它可以通过插拔式组件产线化设计来构建一个完整的无监督问答系统。具体来说，我们的技术方案包含以下方面：

**答案抽取**：我们基于 UIE 训练了一个答案抽取模型，该答案抽取模型接收“答案”作为提示词，该模型可以用来对潜在的答案信息进行挖掘抽取，我们同时提供了训练好的模型权重`uie-base-answer-extractor`。

**问题生成**：我们基于中文预训练语言模型 UNIMO-Text、模版策略和大规模多领域问题生成数据集训练了一个通用点问题生成预训练模型`unimo-text-1.0-question-generation`。

**往返过滤**：我们采用过生成（overgenerate）的策略生成大量的潜在答案和问题，并通过往返过滤的方式针对生成的过量问答对进行过滤得到最终的问答对。我们的往返过滤模块需要训练一个有条件抽取式问答模型<sup>3</sup>。

**语义索引**：针对给定问答对语料，我们基于 RocketQA（即`rocketqa-zh-base-query-encoder`）对问答对进行语义向量化，并通过 ElasticSearch 的 ANN 服务构建索引库。

**召回排序**：给定用户查询，我们基于 RocketQA 的 query-encoder 和 cross-encoder 分别进行召回和排序操作，得到目标的问答对，从而返回给用户查询结果。

**Pipelines**：由于本项目设计的模块较多，我们使用 PaddleNLP Pipelines 进行模块的组合和项目的构建。大体来说，我们的 Pipelines 包含两个具体的 pipeline 和三个服务。两个 pipeline 分别是 qa_generation_pipeline 和 dense_faq_pipeline；三个服务分别是基于 ElasticSearch 的 ANN 在线索引库服务，基于 RestAPI 的模型后端服务以及基于 Streamlit 的前端 WebUI 服务。


## 快速体验
### 运行环境和安装说明
基于 Pipelines 构建问答系统需要安装 paddle-pipelines 依赖，使用 pip 安装命令如下：
```bash
# pip一键安装
pip install --upgrade paddle-pipelines -i https://pypi.tuna.tsinghua.edu.cn/simple
```
或者进入 pipelines 目录下，针对源码进行安装：
```bash
# 源码进行安装
cd PaddleNLP/pipelines/
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup.py install
```

### 数据说明
我们以提供的纯文本文件[source_file.txt](https://paddlenlp.bj.bcebos.com/applications/unsupervised_qa/source_file.txt)为例，系统将每一条都视为一个上下文并基于此生成多个问答对，并基于此构建索引库，该文件可直接下载放入`data`，开发者也可以使用自己的文件。

### 快速体验无监督检索式问答系统
开发者可以通过如下命令快速体验无监督智能检索问答系统的效果，系统将自动根据提供的纯文本文件构建问答对语料库，并基于生成的问答对语料库构造索引库。
我们建议在 GPU 环境下运行本示例，运行速度较快，运行命令如下：
```bash
# GPU环境下运行示例
# 设置1个空闲的GPU卡，此处假设0卡为空闲GPU
export CUDA_VISIBLE_DEVICES=0
python run_pipelines_example.py --device gpu --source_file data/source_file.txt --doc_dir data/my_data --index_name faiss_index --retriever_batch_size 16
```
关键参数释义如下：
- `device`: 使用的设备，默认为'gpu'，可选择['cpu', 'gpu']。
- `source_file`: 源文件路径，指定该路径将自动为其生成问答对至`doc_dir`。
- `doc_dir`: 生成的问答对语料保存的位置，系统将根据该位置自动构建检索数据库，默认为'data/my_data'。
- `index_name`: FAISS 的 ANN 索引名称，默认为'faiss_index'。
- `retriever_batch_size`: 构建 ANN 索引时的批量大小，默认为16。

如果只有 CPU 机器，可以通过--device 参数指定 cpu 即可, 运行耗时较长，运行命令如下：
```bash
# CPU环境下运行示例
unset CUDA_VISIBLE_DEVICES
python run_pipelines_example.py --device cpu --source_file data/source_file.txt --doc_dir data/my_data --index_name faiss_index --retriever_batch_size 16
```




## 可视化无监督检索式问答系统
开发者可以基于 Pipelines 进一步构建 Web 可视化的无监督检索式问答系统，其效果如下，
<div align="center">
    <img src="https://user-images.githubusercontent.com/20476674/199488926-c64d3f4e-8117-475f-afe6-b02088105d09.gif" >
</div>

<!-- ## 基于 Paddle-Serving 构建问答系统
### 环境依赖
安装方式：`pip install -r requirements.txt` -->

### 离线问答对语料构建
这一部分介绍如何离线构建问答对语料，同时我们我们也在 Pipeline 中集成了在线问答对语料。
#### 数据说明
我们以提供的纯文本文件[source_file.txt](https://paddlenlp.bj.bcebos.com/applications/unsupervised_qa/source_file.txt)为例，系统将每一条都视为一个上下文并基于此生成多个问答对，随后系统将根据这些问答对构建索引库，该文件可直接下载放入`data`，开发者也可以使用自己的文件。

#### 问答对生成
对于标准场景的问答对可以直接使用提供的预训练模型实现零样本（zero-shot）问答对生成。对于细分场景开发者可以根据个人需求训练[自定义模型](#自定义模型)，加载自定义模型进行问答对生成，以进一步提升效果。

生成问答对语料的命令如下：
```shell
export CUDA_VISIBLE_DEVICES=0
python -u run_qa_pairs_generation.py \
    --source_file_path=data/source_file.txt \
    --target_file_path=data/target_file.json \
    --answer_generation_model_path=uie-base-answer-extractor-v1 \
    --question_generation_model_path=unimo-text-1.0-question-generation \
    --filtration_model_path=uie-base-qa-filter-v1 \
    --batch_size=8 \
    --a_max_answer_candidates=10 \
    --a_prompt='答案' \
    --a_position_prob=0.01  \
    --q_num_return_sequences=3 \
    --q_max_question_length=50 \
    --q_decode_strategy=sampling \
    --q_top_k=5 \
    --q_top_p=1 \
    --do_filtration \
    --f_filtration_position_prob=0.01 \
    --do_debug
```
关键参数释义如下：
- `source_file_path` 源文件路径，源文件中每一行代表一条待生成问答对的上下文文本。
- `target_file_path` 目标文件路径，生成的目标文件为 json 格式。
- `answer_generation_model_path` 要加载的答案抽取模型的路径，可以是 PaddleNLP 提供的预训练模型，或者是本地模型 checkpoint 路径。如果使用 PaddleNLP 提供的预训练模型，可以选择下面其中之一。
   | 可选预训练模型               |
   |------------------------------|
   | uie-base-answer-extractor-v1 |

- `question_generation_model_path` 要加载的问题生成模型的路径，可以是 PaddleNLP 提供的预训练模型，或者是本地模型 checkpoint 路径。如果使用 PaddleNLP 提供的预训练模型，可以选择下面其中之一。
   | 可选预训练模型                                 |
   |------------------------------------------------|
   | unimo-text-1.0-question-generation             |
   | unimo-text-1.0-dureader_qg                     |
   | unimo-text-1.0-question-generation-dureader_qg |

- `filtration_model_path` 要加载的过滤模型的路径，可以是 PaddleNLP 提供的预训练模型，或者是本地模型 checkpoint 路径。如果使用 PaddleNLP 提供的预训练模型，可以选择下面其中之一。
   | 可选预训练模型        |
   |-----------------------|
   | uie-base-qa-filter-v1 |

- `batch_size` 使用 taskflow 时的批处理大小，请结合机器情况进行调整，默认为8。
- `a_max_answer_candidates` 答案抽取阶段，每个输入的最大返回答案候选数，默认为5。
- `a_prompt` 答案抽取阶段，使用的提示词，以","分隔，默认为"答案"。
- `a_position_prob` 答案抽取阶段，置信度阈值，默认为0.01。
- `q_num_return_sequences` 问题生成阶段，返回问题候选数，在使用"beam_search"解码策略时它应该小于`q_num_beams`，默认为3。
- `q_max_question_length` 问题生成阶段，最大解码长度，默认为50。
- `q_decode_strategy` 问题生成阶段，解码策略，默认为"sampling"。
- `q_top_k` 问题生成阶段，使用"sampling"解码策略时的 top k 值，默认为5。
- `q_top_p` 问题生成阶段，使用"sampling"解码策略时的 top p 值，默认为0。
- `q_num_beams` 问题生成阶段，使用"beam_search"解码策略时的 beam 大小，默认为6。
- `do_filtration` 是否进行过滤。
- `f_filtration_position_prob` 过滤阶段，过滤置信度阈值，默认为0.1。
- `do_debug` 是否进入调试状态，调试状态下将输出过滤掉的生成问答对。

#### 语料转换
执行以下脚本对生成的问答对进行转换，得到语义索引所需要的语料 train.csv、dev.csv、q_corpus.csv、qa_pair.csv：
```shell
python -u run_corpus_preparation.py \
    --source_file_path data/target_file.json \
    --target_dir_path data/my_corpus
```
关键参数释义如下：
- `source_file_path` 指示了要转换的训练数据集文件或测试数据集文件，文件格式要求见从本地文件创建数据集部分。指示了要转换的问答对 json 文件路径，生成的目标文件为 json 格式
- `target_dir_path` 输出数据的目标文件夹，默认为"data/my_corpus"。
- `test_sample_num` 构建检索系统时保留的测试样本数目，默认为0。
- `train_sample_num` 构建检索系统时保留的有监督训练样本数目，默认为0。
- `all_sample_num` 构建检索系统时保留的总样本数目，默认为 None，表示保留除了前`test_sample_num`+`train_sample_num`个样本外的所有样本。



<!-- ### 检索模型训练部署
在已有问答语料库和语义检索模型前提下，模型部署首先要把语义检索模型由动态图转换成静态图，然后转换成 serving 的格式，此外还需要基于 Milvus 和问答语料库构建语义检索引擎。

关于如何对语义检索模型进行无监督训练，以及针对给定问答语料库进行模型部署，请参考 faq_system -->

### 基于 Pipelines 构建问答系统
本项目提供了基于 Pipelines 的低成本构建问答对自动生成智能检索问答系统的能力。开发者只需要提供非结构化的纯文本，就可以使用本项目预制的问答对生成模块生成大量的问答对，并基于此快速搭建一个针对自己业务的检索问答系统，并可以提供 Web 可视化产品服务。Web 可视化产品服务支持问答检索、在线问答对生成，在线文件上传和解析，在线索引库更新等功能，用户也可根据需要自行调整。具体的构建流程请参考[Pipelines-无监督智能检索问答系统](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/pipelines/examples/unsupervised-question-answering)。



## 自定义模型
除了使用预置模型外，用户也可以训练并接入自己训练的模型，我们提供了从答案抽取、问题生成、往返过滤的过滤模型，到语义索引、召回、排序各个阶段的定制化训练方案。
### 数据准备
这一部分介绍如何准备和预处理答案抽取、问题生成、过滤模块微调所需的数据。关于如何准备通过无监督方式训练自定义语义索引模型所需的问答对数据，见[离线问答对语料构建](#离线问答对语料构建)。
#### 自定义数据
在许多情况下，我们需要使用本地数据集来微调模型从而得到定制化的能力，让生成的问答对更接近于理想分布，本项目支持使用固定格式本地数据集文件进行微调。

这里我们提供预先标注好的文件样例[train.json](https://paddlenlp.bj.bcebos.com/applications/unsupervised_qa/train.json)和[dev.json](https://paddlenlp.bj.bcebos.com/applications/unsupervised_qa/dev.json)，开发者可直接下载放入`data`目录，此外也可自行构建本地数据集，具体来说，本地数据集主要包含以下文件：
```text
data
├── train.json # 训练数据集文件
├── dev.json # 开发数据集文件
└── test.json # 可选，待预测数据文件
```
本地数据集文件格式如下：
```text
# train.json/dev.json/test.json文件格式：
{
  "context": <context_text>,
  "answer": <answer_text>,
  "question": <question_text>,
}
...
```
本地数据集文件具体样例如下：
```text
train.json/dev.json/test.json文件样例：
{
  "context": "欠条是永久有效的,未约定还款期限的借款合同纠纷,诉讼时效自债权人主张债权之日起计算,时效为2年。 根据《中华人民共和国民法通则》第一百三十五条:向人民法院请求保护民事权利的诉讼时效期间为二年,法律另有规定的除外。 第一百三十七条:诉讼时效期间从知道或者应当知道权利被侵害时起计算。但是,从权利被侵害之日起超过二十年的,人民法院不予保护。有特殊情况的,人民法院可以延长诉讼时效期间。 第六十二条第(四)项:履行期限不明确的,债务人可以随时履行,债权人也可以随时要求履行,但应当给对方必要的准备时间。",
  "answer": "永久有效",
  "question": "欠条的有效期是多久"
}
...
```

#### 数据预处理
执行以下脚本对数据集进行数据预处理，得到接下来答案抽取、问题生成、过滤模块模型微调所需要的数据，注意这里答案抽取、问题生成、过滤模块的微调数据来源于相同的数据集。
```shell
python -u run_data_preprocess.py \
    --source_file_path data/train.json \
    --target_dir data/finetune \
    --do_answer_prompt

python -u run_data_preprocess.py \
  --source_file_path data/dev.json \
  --target_dir data/finetune \
  --do_answer_prompt
```
关键参数释义如下：
- `source_file_path` 指示了要转换的训练数据集文件或测试数据集文件，文件格式要求见[自定义数据](#自定义数据)部分。
- `target_dir` 输出数据的目标文件夹，默认为"data/finetune"。
- `do_answer_prompt` 表示在构造答案抽取数据时是否添加"答案"提示词。
- `do_len_prompt` 表示在构造答案抽取数据时是否添加长度提示词。
- `do_domain_prompt` 表示在构造答案抽取数据时是否添加领域提示词。
- `domain` 表示添加的领域提示词，在`do_domain_prompt`时有效。

**NOTE:** 预处理后的微调用数据将分别位于`target_dir`下的 answer_extraction、question_generation、filtration 三个子文件夹中。

### 模型微调
#### 答案抽取
运行如下命令即可在样例训练集上微调答案抽取模型，用户可以选择基于`uie-base-answer-extractor`进行微调，或者基于`uie-base`等从头开始微调。
```shell
# GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
# 例如使用1号和2号卡，则：`--gpu 1,2`
unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus "1,2" --log_dir log/answer_extraction finetune/answer_extraction_and_roundtrip_filtration/finetune.py \
    --train_path=data/finetune/answer_extraction/train.json \
    --dev_path=data/finetune/answer_extraction/dev.json \
    --save_dir=log/answer_extraction/checkpoints \
    --learning_rate=1e-5 \
    --batch_size=16 \
    --max_seq_len=512 \
    --num_epochs=30 \
    --model=uie-base \
    --seed=1000 \
    --logging_steps=100 \
    --valid_steps=100 \
    --device=gpu
```
关键参数释义如下：
- `train_path`: 训练集文件路径。
- `dev_path`: 验证集文件路径。
- `save_dir`: 模型存储路径，默认为`log/answer_extration/checkpoints`。
- `learning_rate`: 学习率，默认为1e-5。
- `batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
- `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
- `num_epochs`: 训练轮数，默认为30。
- `model`: 选择模型，程序会基于选择的模型进行模型微调，可选有`uie-base-answer-extractor`，`uie-base`,`uie-medium`, `uie-mini`, `uie-micro`和`uie-nano`，默认为`uie-base`。
- `init_from_ckpt`: 用于初始化的模型参数的路径。
- `seed`: 随机种子，默认为1000.
- `logging_steps`: 日志打印的间隔 steps 数，默认10。
- `valid_steps`: evaluate 的间隔 steps 数，默认100。
- `device`: 选用什么设备进行训练，可选 cpu 或 gpu。


通过运行以下命令在样例验证集上进行模型评估：

```shell
python finetune/answer_extraction_and_roundtrip_filtration/evaluate.py \
    --model_path=log/answer_extraction/checkpoints/model_best \
    --test_path=data/finetune/answer_extraction/dev.json  \
    --batch_size=16 \
    --max_seq_len=512 \
    --limit=0.01
```

关键参数释义如下：
- `model_path`: 进行评估的模型文件夹路径，路径下需包含模型权重文件`model_state.pdparams`及配置文件`model_config.json`。
- `test_path`: 进行评估的测试集文件。
- `batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
- `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
- `model`: 选择所使用的模型，可选有`uie-base`, `uie-medium`, `uie-mini`, `uie-micro`和`uie-nano`，默认为`uie-base`。
- `debug`: 是否开启 debug 模式对每个正例类别分别进行评估，该模式仅用于模型调试，默认关闭。
- `limit`: SpanEvaluator 测评指标的`limit`，当概率数组中的最后一个维度大于该值时将返回相应的文本片段；当 limit 设置为0.01时表示关注模型的召回率，也即答案的覆盖率。

#### 问题生成
运行如下命令即可在样例训练集上微调问题生成模型，并在样例验证集上进行验证。
```shell
# GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
# 例如使用1号和2号卡，则：`--gpu 1,2`
unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus "1,2" --log_dir log/question_generation finetune/question_generation/train.py \
    --train_file=data/finetune/question_generation/train.json \
    --predict_file=data/finetune/question_generation/dev.json \
    --save_dir=log/question_generation/checkpoints \
    --output_path=log/question_generation/predict.txt \
    --dataset_name=dureader_qg \
    --model_name_or_path="unimo-text-1.0" \
    --logging_steps=100 \
    --save_steps=500 \
    --epochs=20 \
    --batch_size=16 \
    --learning_rate=1e-5 \
    --warmup_proportion=0.02 \
    --weight_decay=0.01 \
    --max_seq_len=512 \
    --max_target_len=30 \
    --do_train \
    --do_predict \
    --max_dec_len=20 \
    --min_dec_len=3 \
    --num_return_sequences=1 \
    --template=1 \
    --device=gpu
```


关键参数释义如下：
- `gpus` 指示了训练所用的 GPU，使用多卡训练可以指定多个 GPU 卡号，例如 --gpus "0,1"。
- `dataset_name` 数据集名称，用来指定数据集格式，默认为`dureader_qg`。
- `train_file` 本地训练数据地址，数据格式必须与`dataset_name`所指数据集格式相同，默认为 None。
- `predict_file` 本地测试数据地址，数据格式必须与`dataset_name`所指数据集格式相同，默认为 None。
- `model_name_or_path` 指示了 finetune 使用的具体预训练模型，可以是 PaddleNLP 提供的预训练模型，或者是本地的预训练模型。如果使用本地的预训练模型，可以配置本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含 paddle 预训练模型 model_state.pdparams。如果使用 PaddleNLP 提供的预训练模型，可以选择下面其中之一，默认为`unimo-text-1.0`。
   | 可选预训练模型                     |
   |------------------------------------|
   | unimo-text-1.0                     |
   | unimo-text-1.0-large               |
   | unimo-text-1.0-question-generation |

- `save_dir` 表示模型的保存路径。
- `output_path` 表示预测结果的保存路径。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `seed` 表示随机数生成器的种子。
- `epochs` 表示训练轮数。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于 learning rate scheduler 产生的值相乘作为当前学习率。
- `weight_decay` 表示 AdamW 优化器中使用的 weight_decay 的系数。
- `warmup_proportion` 表示学习率逐渐升高到基础学习率（即上面配置的 learning_rate）所需要的迭代数占总步数的比例。
- `max_seq_len` 模型输入序列的最大长度。
- `max_target_len` 模型训练时标签的最大长度。
- `min_dec_len` 模型生成序列的最小长度。
- `max_dec_len` 模型生成序列的最大长度。
- `do_train` 是否进行训练。
- `do_predict` 是否进行预测，在验证集上会自动评估。
- `device` 表示使用的设备，从 gpu 和 cpu 中选择。
- `template` 表示使用的模版，从[0, 1, 2, 3, 4]中选择，0表示不选择模版，1表示使用默认模版。

程序运行时将会自动进行训练和验证，训练过程中会自动保存模型在指定的`save_dir`中。

**【注意】** 如需恢复模型训练，`model_name_or_path`配置本地模型的目录地址即可。


#### 过滤模型
运行如下命令即可在样例训练集上微调答案抽取模型，用户可以选择基于`uie-base-qa-filter`进行微调，或者基于`uie-base`等从头开始微调。
```shell
# GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
# 例如使用1号和2号卡，则：`--gpu 1,2`
unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus "1,2" --log_dir log/filtration finetune/answer_extraction_and_roundtrip_filtration/finetune.py \
    --train_path=data/finetune/filtration/train.json \
    --dev_path=data/finetune/filtration/dev.json \
    --save_dir=log/filtration/checkpoints \
    --learning_rate=1e-5 \
    --batch_size=16 \
    --max_seq_len=512 \
    --num_epochs=30 \
    --model=uie-base \
    --seed=1000 \
    --logging_steps=100 \
    --valid_steps=100 \
    --device=gpu
```
关键参数释义如下：
- `train_path`: 训练集文件路径。
- `dev_path`: 验证集文件路径。
- `save_dir`: 模型存储路径，默认为`log/filtration/checkpoints`。
- `learning_rate`: 学习率，默认为1e-5。
- `batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
- `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
- `num_epochs`: 训练轮数，默认为30。
- `model`: 选择模型，程序会基于选择的模型进行模型微调，可选有`uie-base-qa-filter`，`uie-base`, `uie-medium`, `uie-mini`, `uie-micro`和`uie-nano`，默认为`uie-base`。
- `init_from_ckpt`: 用于初始化的模型参数的路径。
- `seed`: 随机种子，默认为1000.
- `logging_steps`: 日志打印的间隔 steps 数，默认10。
- `valid_steps`: evaluate 的间隔 steps 数，默认100。
- `device`: 选用什么设备进行训练，可选 cpu 或 gpu。


通过运行以下命令在样例验证集上进行模型评估：

```shell
python finetune/answer_extraction_and_roundtrip_filtration/evaluate.py \
    --model_path=log/filtration/checkpoints/model_best \
    --test_path=data/finetune/filtration/dev.json  \
    --batch_size=16 \
    --max_seq_len=512 \
    --limit=0.5
```

关键参数释义如下：
- `model_path`: 进行评估的模型文件夹路径，路径下需包含模型权重文件`model_state.pdparams`及配置文件`model_config.json`。
- `test_path`: 进行评估的测试集文件。
- `batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
- `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
- `model`: 选择所使用的模型，可选有`uie-base`, `uie-medium`, `uie-mini`, `uie-micro`和`uie-nano`，默认为`uie-base`。
- `debug`: 是否开启 debug 模式对每个正例类别分别进行评估，该模式仅用于模型调试，默认关闭。
- `limit`: SpanEvaluator 测评指标的`limit`，当概率数组中的最后一个维度大于该值时将返回相应的文本片段。

#### 语义索引和召回模型
我们的语义索引和召回模型是基于 RocketQA 的 QueryEncoder 训练的双塔模型，该模型用于语义索引和召回阶段，分别进行语义向量抽取和相似度召回。除使用预置模型外，如果用户想训练并接入自己的模型，模型训练可以参考[FAQ Finance](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/applications/question_answering/supervised_qa/faq_finance)。

#### 排序模型
我们的排序模型是基于 RocketQA 的 CrossEncoder 训练的单塔模型，该模型用于搜索的排序阶段，对召回的结果进行重新排序的作用。关于排序的定制训练，可以参考[CrossEncoder](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/applications/neural_search/ranking/cross_encoder)。

## References
[1] Zheng, Chujie, and Minlie Huang. "Exploring prompt-based few-shot learning for grounded dialog generation." arXiv preprint arXiv:2109.06513 (2021).

[2] Li, Wei, et al. "Unimo: Towards unified-modal understanding and generation via cross-modal contrastive learning." arXiv preprint arXiv:2012.15409 (2020).

[3] Puri, Raul, et al. "Training question answering models from synthetic data." arXiv preprint arXiv:2002.09599 (2020).

[4] Lewis, Patrick, et al. "Paq: 65 million probably-asked questions and what you can do with them." Transactions of the Association for Computational Linguistics 9 (2021): 1098-1115.

[5] Alberti, Chris, et al. "Synthetic QA corpora generation with roundtrip consistency." arXiv preprint arXiv:1906.05416 (2019).
