# 政务问答检索式 FAQ System

 **目录**

* [1. 场景概述](#场景概述)
* [2. 系统特色](#系统特色)
* [3. 政务问答系统方案](#政务问答系统方案)
* [4. 动手实践——搭建自己的端到端检索式问答系统](#动手实践——搭建自己的端到端检索式问答系统)


<a name="场景概述"></a>

## 1. 场景概述

政府工作人员往往要做很多政策解读等工作，费时费力还耗费大量的人力，在政府内部，工作人员往往积累了很多问答对，但是不知道怎么构建一个问答系统来辅助工作人员提升日常工作效率，简化工作流程。

<a name="系统特色"></a>

## 2. 系统特色

+ 低门槛
    + 手把手搭建检索式 FAQ System
    + 无需相似 Query-Query Pair 标注数据也能构建 FAQ System
+ 效果好
    + 业界领先的检索预训练模型: RocketQA Dual Encoder
    + 针对无标注数据场景的领先解决方案: 检索预训练模型 + 增强的无监督语义索引微调

+ 性能快
    + 基于 Paddle Inference 快速抽取向量
    + 基于 Milvus 快速查询和高性能建库
    + 基于 Paddle Serving 高性能部署

<a name="政务问答系统方案"></a>

## 3. 政务问答系统方案

### 3.1 技术方案和评估指标

#### 3.1.1 技术方案

**语义索引**：针对政务问答只有问答对的场景，我们提供了一个 融合SimCSE 和 WR (word reptition)策略的无监督的解决方案。

#### 3.1.2 评估指标

* 该政务问答系统使用的指标是 Recall@K，表示的是预测的前topK（从最后的按得分排序的召回列表中返回前K个结果）结果和语料库中真实的前 K 个相关结果的重叠率，衡量的是检索系统的查全率。


### 3.2 数据说明

数据集来源于疫情政务问答比赛数据，包括源文本，问题和答案。

|  阶段 |模型 |   训练集 | 评估集（用于评估模型效果） | 召回库 |
| ------------ | ------------ |------------ | ------------ | ------------ |
|  召回 |  SimCSE  |  4000 | 1000 | 5000 |

其中评估集的问题对的构造使用了中英文回译的方法，总共有1000条评估集，其中500条数据使用的是百度翻译的API，详情请参考[百度翻译](https://fanyi-api.baidu.com/?fr=simultaneous)，另外500条数据使用了SimBERT模型生成的同义句。


```
├── data  # 数据集
    ├── train.csv  # 无监督训练集
    ├── test_pair.csv  # 测试集，用于评估模型的效果
    ├── corpus.csv # 构建召回的数据，用于评估模型的召回效果
    ├── qa_pair.csv # 问答对，问题对应的答案
```
数据集的下载链接为: [faq_data](https://paddlenlp.bj.bcebos.com/applications/faq_data.zip)

### 3.3 代码说明

```
|—— data.py # 数据读取、数据转换等预处理逻辑
|—— model.py # SimCSE模型
|—— train.py # SimCSE训练主脚本
|—— ann_util.py # Ann 建索引库相关函数
|—— config.py # Milvus 配置文件
|—— evaluate.py # 召回评估文件
|—— recall.py # 基于训练好的语义索引模型，从召回库中召回给定文本的相似文本
|—— export_model.py # 动态图转换成静态图
|—— export_to_serving.py # 静态图转 Serving
|—— feature_extract.py # 批量提取文本的特征向量
|—— milvus_util.py # Milvus的插入和召回类
|—— vector_insert.py # 向 Milvus 引擎插入向量的函数
|—— run_system.py # Client Server 模式客户端，向 server 发送文本，得到向量后，利用milvus引擎进行检索
|—— scripts
    |—— export_model.sh  # 动态图转换成静态图脚本
    |—— evaluate.sh # 评估 bash 版本
    |—— run_build_index.sh # 构建索引 bash 版本
    |—— train.sh  # 训练 bash 版本
    |—— feature_extract.sh  # 向量抽取 bash 版本
    |—— export_to_serving.sh  # Paddle Inference 转 Serving 的 bash 脚本
|—— deploy
    |—— python
        |—— rpc_client.py # Paddle Serving 的 Client 端
        |—— web_service.py # Paddle Serving 的 Serving 端
        |—— config_nlp.yml # Paddle Serving 的配置文件
```

### 3.3 效果评估

|  模型 |  Recall@1 |Recall@10 |
| ------------ | ------------ |--------- |
|  ERNIE1.0 + SimCSE |  68.068     | 85.686|
|  RocketQA  |  81.381 | 96.997|
|  RocketQA + SimCSE  |  83.283 | 97.297|
|  RocketQA + SimCSE + WR |  **83.584** | **97.497**|

<a name="动手实践——搭建自己的端到端检索式问答系统"></a>

## 4. 动手实践——搭建自己的端到端检索式问答系统

### 4.1 无监督训练

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

### 4.2  评估

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

## 4.3 模型部署

模型部署模块首先要把动态图转换成静态图，然后转换成serving的格式。

### 动转静导出

首先把动态图模型转换为静态图：

```
python export_model.py --params_path checkpoints/model_150/model_state.pdparams --output_path=./output
```
也可以运行下面的bash脚本：

```
sh scripts/export_model.sh
```

### 问答检索引擎

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

### Paddle Serving 部署

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

## 4.4 问答系统整个流程

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
