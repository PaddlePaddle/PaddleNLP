# 问答对自动生成智能检索式问答


**目录**
- [问答对自动生成智能检索式问答](#问答对自动生成智能检索式问答)
  - [简介](#简介)
    - [项目优势](#项目优势)
  <!-- - [开箱即用](#开箱即用)
  - [效果展示](#效果展示) -->
  - [方案介绍](#方案介绍)
    - [技术方案](#技术方案)
    <!-- - [评估指标](#评估指标) -->
    - [代码结构说明](#代码结构说明)
  - [系统构建](#系统构建)
    - [环境依赖](#环境依赖)
    - [问答对生成](#问答对生成)
      - [数据处理](#数据处理)
        - [数据准备](#数据准备)
        - [数据预处理](#数据预处理)
      - [模型微调](#模型微调)
        - [答案抽取](#答案抽取)
        - [问题生成](#问题生成)
        - [过滤模型](#过滤模型)
      - [语料生成](#语料生成)
    - [语义索引](#语义索引)
      - [无监督训练](#无监督训练)
      - [评估](#评估)
      - [模型部署](#模型部署)
        - [动转静导出](#动转静导出)
        - [问答检索引擎](#问答检索引擎)
        - [Paddle-Serving部署](#Paddle-Serving部署)
      - [整体流程](#整体流程)
  - [References](#references)

## 简介
问答（QA）系统中最关键的挑战之一是标记数据的稀缺性，这是因为对目标领域获取问答对或常见问答对（FAQ）的成本很高，需要消耗大量的人力和时间。由于上述制约，这导致问答系统落地困难，解决此问题的一种方法是依据问题上下文或大量非结构化文本自动生成的QA问答对。

在此背景下，本项目，即问答对自动生成智能检索式问答，基于PaddleNLP[问题生成](../../../examples/question_generation/README.md)、[UIE](../../../model_zoo/uie/README.md)、[检索式问答](../faq_system/README.md)，支持以非结构化文本为上下文自动生成QA问答对，生成的问答对语料可以通过无监督的方式构建检索式问答系统。

若已有FAQ语料，请参考[faq_system](../faq_system)或[faq_finance](../faq_finance)。

### 项目优势
具体来说，本项目具有以下优势：


+ 低代价
    + 可通过自动生成的方式快速大量合成QA语料，大大降低人力成本
    + 可控性好，合成语料和语义检索问答松耦合，可以人工筛查和删除合成的问答对，也可以添加人工标注的问答对

+ 低门槛
    + 手把手搭建无监督检索式FAQ System
    + 无需相似Query-Query Pair标注数据也能构建FAQ System

+ 效果好
    + 可通过自动问答对生成提升问答对语料覆盖度，缓解中长尾问题覆盖较少的问题
    + 业界领先的检索预训练模型: RocketQA Dual Encoder
    + 针对无标注数据场景的领先解决方案: 检索预训练模型 + 增强的无监督语义索引微调

+ 性能快
    + 基于Paddle Inference 快速抽取向量
    + 基于Milvus 快速查询和高性能建库
    + 基于Paddle Serving 高性能部署

<!--
## 效果展示
## 开箱即用 -->

## 方案介绍
### 技术方案
**问答对生成**：问答对生成主要由基于UIE的答案抽取模型、基于UNIMO-Text的问题生成模型，以及过滤模型三个模块构成，我们在一个大规模多领域的问题生成数据集上分别对三者进行预训练，用户可直接使用提供的预训练参数生成问答对，或针对特定任务进行微调。

**语义索引**：针对给定问答对语料，我们基于RocketQA提供了一个融合SimCSE和WR（word reptition）策略的无监督的解决方案。

<!-- ### 评估指标
**问答对生成**：问答对生成使用的指标是软召回率Recall@K，
**语义索引**：语义索引使用的指标是Recall@K，表示的是预测的前topK（从最后的按得分排序的召回列表中返回前K个结果）结果和语料库中真实的前K个相关结果的重叠率，衡量的是检索系统的查全率。 -->
### 代码结构说明
以下是本项目主要代码结构及说明：

```text
├── deploy # 部署
│   ├── paddle_inference # PaddleInference高性能推理部署
│   │   ├── inference_unimo_text.py # 推理部署脚本
│   │   └── README.md # 说明文档
│   └── paddle_serving
│       ├── config.yml # 配置文件
│       ├── pipeline_client.py # 客户端程序
│       ├── pipeline_service.py # 服务器程序
│       └── README.md # 说明文档
├── export_model.py # 动态图参数导出静态图参数脚本
├── train.py # 训练脚本
├── predict.py # 预测评估脚本
├── utils.py # 工具函数脚本
└── README.md # 说明文档
```

## 系统构建

### 环境依赖
- nltk
- evaluate
- tqdm

安装方式：`pip install -r requirements.txt`

### 问答对生成
对于标准场景的问答对可以直接使用提供的预训练模型实现零样本（zero-shot）问答对生成，对于细分场景我们推荐使用轻定制功能（标注少量数据进行模型微调）以进一步提升效果。下面通过疫情政务问答的例子展示如何通过5条训练数据进行问答对生成微调。

#### 数据处理
这一部分介绍如何准备和预处理模型微调所需的数据。
##### 数据准备
###### 数据集加载
[**DuReader_QG**数据集](https://www.luge.ai/#/luge/dataDetail?id=8)是一个中文问题生成数据集，我们使用该数据集作为应用案例进行实验。**DuReader_QG**中的数据主要由由上下文、问题、答案3个主要部分组成，该数据集可以作为问答对生成模型微调的原始数据。

为了方便用户快速测试，PaddleNLP Dataset API内置了DuReader_QG数据集，一键即可完成数据集加载，示例代码如下：

```python
from paddlenlp.datasets import load_dataset
train_ds, dev_ds = load_dataset('dureader_qg', splits=('train', 'dev'))
```

###### 从本地文件创建数据集
在许多情况下，我们需要使用本地数据集来微调模型从而得到定制化的能力，让生成的问答对更接近于理想分布，本项目支持使用固定格式本地数据集文件进行微调。
使用本地文件，只需要在模型训练时指定`train_file` 为本地训练数据地址，`predict_file` 为本地测试数据地址即可。

本地数据集目录结构如下：

```text
data/
├── train.json # 训练数据集文件
├── dev.json # 开发数据集文件
└── test.json # 可选，待预测数据文件
```
本地数据集文件格式如下：
- train.json/dev.json/test.json 文件格式：
```text
{
  "context": <context_text>,
  "answer": <answer_text>,
  "question": <question_text>,
}
...
```
- train.json/dev.json/test.json 文件样例：
```text
{
  "context": "欠条是永久有效的,未约定还款期限的借款合同纠纷,诉讼时效自债权人主张债权之日起计算,时效为2年。 根据《中华人民共和国民法通则》第一百三十五条:向人民法院请求保护民事权利的诉讼时效期间为二年,法律另有规定的除外。 第一百三十七条:诉讼时效期间从知道或者应当知道权利被侵害时起计算。但是,从权利被侵害之日起超过二十年的,人民法院不予保护。有特殊情况的,人民法院可以延长诉讼时效期间。 第六十二条第(四)项:履行期限不明确的,债务人可以随时履行,债权人也可以随时要求履行,但应当给对方必要的准备时间。",
  "answer": "永久有效",
  "question": "欠条的有效期是多久"
}
...
```

更多数据集读取格式详见[数据集加载](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_load.html#)和[自定义数据集](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html)。

##### 数据预处理
执行以下脚本对数据集进行数据预处理，得到接下来答案抽取、问题生成、过滤模块模型微调所需要的数据，注意这里答案抽取、问题生成、过滤模块的微调数据来源于相同的数据集。

#### 模型微调
##### 答案抽取

##### 问题生成
运行如下命令即可在样例训练集上进行finetune，并在样例验证集上进行验证。
```shell
# GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
# 例如使用1号和2号卡，则：`--gpu 1,2`
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "1,2" --log_dir ./unimo/finetune/log train.py \
    --dataset_name=dureader_qg \
    --model_name_or_path="unimo-text-1.0" \
    --save_dir=./unimo/finetune/checkpoints \
    --output_path ./unimo/finetune/predict.txt \
    --logging_steps=100 \
    --save_steps=500 \
    --epochs=20 \
    --batch_size=16 \
    --learning_rate=1e-5 \
    --warmup_propotion=0.02 \
    --weight_decay=0.01 \
    --max_seq_len=512 \
    --max_target_len=30 \
    --do_train \
    --do_predict \
    --max_dec_len=20 \
    --min_dec_len=3 \
    --num_return_sequences=1 \
    --adversarial_training=None \
    --template=1 \
    --device=gpu
```


关键参数释义如下：
- `gpus` 指示了训练所用的GPU，使用多卡训练可以指定多个GPU卡号，例如 --gpus "0,1"。
- `dataset_name` 数据集名称，默认为`dureader_qg`。
- `train_file` 本地训练数据地址，数据格式必须与`dataset_name`所指数据集格式相同，默认为None。
- `predict_file` 本地测试数据地址，数据格式必须与`dataset_name`所指数据集格式相同，默认为None。
- `model_name_or_path` 指示了finetune使用的具体预训练模型，可以是PaddleNLP提供的预训练模型，或者是本地的预训练模型。如果使用本地的预训练模型，可以配置本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含paddle预训练模型model_state.pdparams。如果使用PaddleNLP提供的预训练模型，可以选择下面其中之一。
   | 可选预训练模型        |
   |---------------------------------|
   | unimo-text-1.0      |
   | unimo-text-1.0-large |

   <!-- | T5-PEGASUS |
   | ernie-1.0 |
   | ernie-gen-base-en |
   | ernie-gen-large-en |
   | ernie-gen-large-en-430g | -->

- `save_dir` 表示模型的保存路径。
- `output_path` 表示预测结果的保存路径。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `seed` 表示随机数生成器的种子。
- `epochs` 表示训练轮数。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `weight_decay` 表示AdamW优化器中使用的weight_decay的系数。
- `warmup_propotion` 表示学习率逐渐升高到基础学习率（即上面配置的learning_rate）所需要的迭代数占总步数的比例。
- `max_seq_len` 模型输入序列的最大长度。
- `max_target_len` 模型训练时标签的最大长度。
- `min_dec_len` 模型生成序列的最小长度。
- `max_dec_len` 模型生成序列的最大长度。
- `do_train` 是否进行训练。
- `do_predict` 是否进行预测，在验证集上会自动评估。
- `device` 表示使用的设备，从gpu和cpu中选择。
- `template` 表示使用的模版，从[0, 1, 2, 3, 4]中选择，0表示不选择模版，1表示使用默认模版。

程序运行时将会自动进行训练和验证，训练过程中会自动保存模型在指定的`save_dir`中。如：

```text
./unimo/finetune/checkpoints
├── model_1000
│   ├── model_config.json
│   ├── model_state.pdparams
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
└── ...
```

**NOTE:** 如需恢复模型训练，`model_name_or_path`配置本地模型的目录地址即可。

微调的baseline模型在dureader_qg验证集上有如下结果(指标为BLEU-4)：

|       model_name        | DuReaderQG |
| :-----------------------------: | :-----------: |
|    unimo-text-1.0-dureader_qg-template1    | 41.08 |

##### 过滤模型

#### 语料生成

运行下方脚本可以使用训练好的模型进行预测。

```shell
export CUDA_VISIBLE_DEVICES=0
python -u predict.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=your_model_path \
    --output_path=./predict.txt \
    --logging_steps=100 \
    --batch_size=16 \
    --max_seq_len=512 \
    --max_target_len=30 \
    --do_predict \
    --max_dec_len=20 \
    --min_dec_len=3 \
    --template=1 \
    --device=gpu
```
关键参数释义如下：
- `output_path` 表示预测输出结果保存的文件路径，默认为./predict.txt。
- `model_name_or_path` 指示了finetune使用的具体预训练模型，可以是PaddleNLP提供的预训练模型，或者是本地的微调好的预训练模型。如果使用本地的预训练模型，可以配置本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含paddle预训练模型model_state.pdparams。


### 语义索引
#### 无监督训练

```
python -u -m paddle.distributed.launch --gpus '0' \
    train.py \
    --device gpu \
    --save_dir ./checkpoints/ \
    --batch_size 64 \
    --learning_rate 5E-5 \
    --epochs 3 \
    --save_steps 50 \
    --max_seq_length 64 \
    --dropout 0.2 \
    --output_emb_size 256 \
    --dup_rate 0.3 \
    --train_set_file "./data/train.csv"
```

参数含义说明

* `device`: 使用 cpu/gpu 进行训练
* `save_dir`: 模型存储路径
* `batch_size`: 训练的batch size的大小
* `learning_rate`: 训练的学习率的大小
* `epochs`: 训练的epoch数
* `save_steps`： 模型存储 checkpoint 的间隔 steps 个数
* `max_seq_length`: 输入序列的最大长度
* `dropout`: SimCSE的dropout参数
* `output_emb_size`: Transformer 顶层输出的文本向量维度
* `dup_rate` : SimCSE的 Word reptition 策略的重复率
* `train_set_file`: 训练集文件

也可以使用下面的bash脚本：

```
sh scripts/train.sh
```
#### 评估

效果评估分为 4 个步骤:

a. 获取Doc端Embedding

基于语义索引模型抽取出Doc样本库的文本向量。

b. 采用hnswlib对Doc端Embedding建库

使用 ANN 引擎构建索引库(这里基于 [hnswlib](https://github.com/nmslib/hnswlib) 进行 ANN 索引)

c. 获取question的Embedding并查询相似结果

基于语义索引模型抽取出评估集 *Source Text* 的文本向量，在第 2 步中建立的索引库中进行 ANN 查询，召回 Top10 最相似的 *Target Text*, 产出评估集中 *Source Text* 的召回结果 `recall_result` 文件。

d. 评估

基于评估集 `test.csv` 和召回结果 `recall_result` 计算评估指标 Recall@k，其中k取值1，5，10。

运行如下命令进行 ANN 建库、召回，产出召回结果数据 `recall_result`

```
python -u -m paddle.distributed.launch --gpus "0" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "checkpoints/model_150/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 64 \
        --recall_num 10 \
        --similar_text_pair "data/test_pair.csv" \
        --corpus_file "data/corpus.csv"
```
参数含义说明
* `device`: 使用 cpu/gpu 进行训练
* `recall_result_dir`: 召回结果存储目录
* `recall_result_file`: 召回结果的文件名
* `params_path`： 待评估模型的参数文件名
* `hnsw_m`: hnsw 算法相关参数，保持默认即可
* `hnsw_ef`: hnsw 算法相关参数，保持默认即可
* `output_emb_size`: Transformer 顶层输出的文本向量维度
* `recall_num`: 对 1 个文本召回的相似文本数量
* `similar_text_pair`: 由相似文本对构成的评估集
* `corpus_file`: 召回库数据 corpus_file

也可以使用下面的bash脚本：

```
sh scripts/run_build_index.sh
```

run_build_index.sh还包含cpu和gpu运行的脚本，默认是gpu的脚本

接下来，运行如下命令进行效果评估，产出Recall@1, Recall@5, Recall@10 指标:
```
python -u evaluate.py \
        --similar_text_pair "data/test_pair.csv" \
        --recall_result_file "./recall_result_dir/recall_result.txt" \
        --recall_num 10
```
也可以使用下面的bash脚本：

```
sh scripts/evaluate.sh
```
输出如下的结果：

```
recall@1=83.784
recall@5=94.995
recall@10=96.997
```

参数含义说明
* `similar_text_pair`: 由相似文本对构成的评估集 semantic_similar_pair.tsv
* `recall_result_file`: 针对评估集中第一列文本 *Source Text* 的召回结果
* `recall_num`: 对 1 个文本召回的相似文本数量

#### 模型部署
模型部署模块首先要把动态图转换成静态图，然后转换成serving的格式。
##### 动转静导出
首先把动态图模型转换为静态图：

```
python export_model.py --params_path checkpoints/model_150/model_state.pdparams --output_path=./output
```
也可以运行下面的bash脚本：

```
sh scripts/export_model.sh
```
##### 问答检索引擎
模型准备结束以后，开始搭建 Milvus 的语义检索引擎，用于语义向量的快速检索，本项目使用[Milvus](https://milvus.io/)开源工具进行向量检索，Milvus 的搭建教程请参考官方教程  [Milvus官方安装教程](https://milvus.io/cn/docs/v1.1.1/milvus_docker-cpu.md)本案例使用的是 Milvus 的1.1.1 CPU版本，建议使用官方的 Docker 安装方式，简单快捷。


Milvus 搭建完系统以后就可以插入和检索向量了，首先生成 embedding 向量，每个样本生成256维度的向量：

```
python feature_extract.py \
        --model_dir=./output \
        --corpus_file "data/corpus.csv"
```
其中 output 目录下存放的是召回的 Paddle Inference 静态图模型。

然后向搭建好的 Milvus 系统插入向量：

```
python vector_insert.py
```
##### Paddle-Serving部署
Paddle Serving 的安装可以参考[Paddle Serving 安装文档](https://github.com/PaddlePaddle/Serving#installation)。需要在服务端和客户端安装相关的依赖，安装完依赖后就可以执行下面的步骤。


首先把生成的静态图模型导出为 Paddle Serving的格式，命令如下：

```
python export_to_serving.py \
    --dirname "output" \
    --model_filename "inference.get_pooled_embedding.pdmodel" \
    --params_filename "inference.get_pooled_embedding.pdiparams" \
    --server_path "./serving_server" \
    --client_path "./serving_client" \
    --fetch_alias_names "output_embedding"
```

参数含义说明
* `dirname`: 需要转换的模型文件存储路径，Program 结构文件和参数文件均保存在此目录。
* `model_filename`： 存储需要转换的模型 Inference Program 结构的文件名称。如果设置为 None ，则使用 `__model__` 作为默认的文件名
* `params_filename`: 存储需要转换的模型所有参数的文件名称。当且仅当所有模型参数被保>存在一个单独的二进制文件中，它才需要被指定。如果模型参数是存储在各自分离的文件中，设置它的值为 None
* `server_path`: 转换后的模型文件和配置文件的存储路径。默认值为 serving_server
* `client_path`: 转换后的客户端配置文件存储路径。默认值为 serving_client
* `fetch_alias_names`: 模型输出的别名设置，比如输入的 input_ids 等，都可以指定成其他名字，默认不指定
* `feed_alias_names`: 模型输入的别名设置，比如输出 pooled_out 等，都可以重新指定成其他模型，默认不指定

也可以运行下面的 bash 脚本：
```
sh scripts/export_to_serving.sh
```

启动 Pipeline Server:

```
cd deploy/python/
python web_service.py
```

启动客户端调用 Server, 使用 POST的方式：

向服务端发送 POST 请求示例：

```
curl -X POST -k http://localhost:8090/ernie/prediction -d '{"key": ["0"], "value": ["宁夏针对哪些人员开通工伤保障绿色通道?"]}'
```

也可以使用 rpc的方式：

首先修改rpc_client.py中需要预测的样本：

```
list_data = [
    "湖北省为什么鼓励缴费人通过线上缴费渠道缴费？",
    "佛山市救助站有多少个救助床位"
]
```
然后运行：

```
python rpc_client.py
```
#### 整体流程
问答系统使用了Client Server的模式，即抽取向量的模型部署在服务端，然后启动客户端（Client）端去访问。


```
python run_system.py
```
代码内置的测试用例为：

```
list_data = ["嘉定区南翔镇实行双门长制“门长”要求落实好哪些工作？"]
```

会输出如下的结果：

```
......
Extract feature time to cost :0.01161503791809082 seconds
Search milvus time cost is 0.004535675048828125 seconds
嘉定区南翔镇实行双门长制“门长”要求落实好哪些工作？      拦、查、问、测、记 1.2107588152551751e-12
上海市黄浦区老西门街道建立的党建责任区包干机制内容是什么？      街道工作人员担任楼宇联络员，分片区对接商务楼宇所属的物业公司，引导楼宇企业共同落实严防严控任务 0.4956303834915161
上海市街道执行“四个统一”具体指什么？    统一由居委会干部在统一时间（每周三、五下午），递交至统一地点（社区事务受理服务中心专设窗口），街道统一收集至後台 0.6684658527374268
怀柔区城管委在加强监督检查方面是如何落实的？    严格落实四方责任，保证每周2~3次深入环卫、电、气、热、公共自行车、垃圾处置等单位进行巡查，督促企业做好防疫工作，协调复工复产中存在的问题，确保安全复工复产有效落实。 0.7147952318191528
华新镇“亮牌分批复工”工作方案具体内容是什么？    所有店铺一律先贴“红牌”禁止经营，经相关部门审批後，再换贴“蓝牌”准许复工。 0.7162970900535583
.....
```
输出的结果包括特征提取和检索的时间，还包含检索出来的问答对。


## References
Zheng, Chujie, and Minlie Huang. "Exploring prompt-based few-shot learning for grounded dialog generation." arXiv preprint arXiv:2109.06513 (2021).
Li, Wei, et al. "Unimo: Towards unified-modal understanding and generation via cross-modal contrastive learning." arXiv preprint arXiv:2012.15409 (2020).
