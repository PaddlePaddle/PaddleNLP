# 保险智能问答

 **目录**

* [1. 项目介绍](#项目介绍)
* [2. 系统特色](#系统特色)
* [3. 保险智能问答系统方案](#保险问答系统方案)
* [4. 动手实践——搭建自己的端到端检索式问答系统](#动手实践——搭建自己的端到端检索式问答系统)
* [5. 模型优化](#模型优化)
* [6. 参考文献](#参考文献)

<a name="项目介绍"></a>

## 1. 项目介绍

智能问答是获取信息和知识的更直接、更高效的方式之一，传统的信息检索方法智能找到相关的文档，而智能问答能够直接找到精准的答案，极大的节省了人们查询信息的时间。问答按照技术分为基于阅读理解的问答和检索式的问答，阅读理解的问答是在正文中找到对应的答案片段，检索式问答则是匹配高频的问题，然后把答案返回给用户。本项目属于检索式的问答，问答的领域用途很广，比如搜索引擎，小度音响等智能硬件，政府，金融，银行，电信，电商领域的智能客服，聊天机器人等。

- 本方案是场景的定制化的方案，用户可以使用自己的数据训练一个特定场景的方案。另外，想快速体验FAQ智能问答系统请参考Pipelines的实现[FAQ智能问答](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/examples/FAQ)

- 本项目的详细教程请参考（包括数据和代码实现）[aistudio教程](https://aistudio.baidu.com/aistudio/projectdetail/3882519)

<a name="系统特色"></a>

## 2. 系统特色

+ 低门槛
    + 手把手搭建检索式保险智能问答
    + 无需相似 Query-Query Pair 标注数据也能构建保险智能问答
+ 效果好
    + 业界领先的检索预训练模型: RocketQA Dual Encoder
    + 针对无标注数据场景的领先解决方案: 检索预训练模型 + 增强的无监督语义索引微调

+ 性能快
    + 基于 Paddle Inference 快速抽取向量
    + 基于 Milvus 快速查询和高性能建库
    + 基于 Paddle Serving 高性能部署

<a name="保险问答系统方案"></a>

## 3. 保险智能问答系统方案

### 3.1 技术方案和评估指标

#### 3.1.1 技术方案

**语义索引**：针对保险等金融领域的问答只有问答对的场景，我们提供了一个在SimCSE的基础上融合WR (word reptition)策略，同义词策略，R-Drop策略的无监督的解决方案。

#### 3.1.2 评估指标

* 该保险智能问答系统使用的指标是 Recall@K，表示的是预测的前topK（从最后的按得分排序的召回列表中返回前K个结果）结果和语料库中真实的前 K 个相关结果的重叠率，衡量的是检索系统的查全率。

### 3.2 数据说明

#### 3.2.1 预置数据介绍

数据集来源于Github开源的保险的问答数据，包括源用户的问题和相应的回复。

|  阶段 |模型 |   训练集 | 评估集（用于评估模型效果） | 召回库 |
| ------------ | ------------ |------------ | ------------ | ------------ |
|  召回 |  SimCSE  |  3030 | 758 | 3788 |

其中训练集的问题-问题对的构造使用了同义词替换的方法，详情请参考[nlpcda](https://github.com/425776024/nlpcda)

评估集的问题对的构造使用了中英文回译的方法，数据使用的是百度翻译的API，详情请参考[百度翻译](https://fanyi-api.baidu.com/?fr=simultaneous)

【注意】：数据集是基于Github开源数据进行了处理得到的，如果有任何侵权问题，请及时联系，我们会第一时间进行删除。

```
├── data  # 数据集
    ├── train.csv  # 无监督训练集
    ├── train_aug.csv # 同义词替换后构造的训练集
    ├── test_pair.csv  # 测试集，用于评估模型的效果
    ├── corpus.csv # 构建召回的数据，用于评估模型的召回效果
    ├── qa_pair.csv # 问答对，问题对应的答案
```
数据集的下载链接为: [faq_finance](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/baoxianzhidao/intro.ipynb)

#### 3.2.2 数据格式

训练需要规定格式的本地数据集，需要准备训练集文件`train.csv`或者`train_aug.csv`，测试集`test_pair.csv`，召回集文件`corpus.csv`,问答对 `qa_pair.csv`。

用于无监督训练的训练集的格式如下：

```
文本1
文本2
...
```
训练集合`train.csv`的文件样例：

```
家里有社保，还有必要买重疾险吗？
工地买了建工险，出了事故多长时间上报保险公司有效
请问下哆啦a保值不值得买呢？不晓得保障多不多
自由职业办理养老保险是否划算
工伤七级如果公司不干了,怎么赔我
普通意外险的保障范围都有哪些？
......
```
除此之外，也可以使用数据增强的格式，训练方式是类似有监督的构造句子对。数据增强的文件格式如下:

```
文本1 \t 增强文本1
文本2 \t 增强文本2
```
增强数据集`train_aug.csv`的格式如下：

```
工伤七级如果公司不干了,怎么赔我	工伤七级如果企业不干了,怎生赔我
普通意外险的保障范围都有哪些？	一般性意外险的保障范围都有哪些？
重疾险赔付三次和赔付一次的区别	重疾险赔偿三次和赔偿一次的区别
。。。。。
```

测试集合`test_pair.csv`是问句对，具体格式如下：

```
句子1 \t 句子2
句子3 \t 句子4
```
其中句子1和句子2是相似的句子，只是表达方式不同，或者进行了一定程度的变形，但实际表达的语义是一样的。

测试集的文件样例：

```
车险如何计算	如何计算汽车保险
农民买养老保险怎么买	农民如何购买养老保险
车险必买哪几项	你必须购买哪些汽车保险
...
```
召回集合`corpus.csv`主要作用是检验测试集合的句子对能否被正确召回，它的构造主要是提取测试集的第二列的句子，然后加入很多无关的句子，用来检验模型能够正确的从这些文本中找出测试集合对应的第二列的句子，格式如下：

```
如何办理企业养老保险
如何为西班牙购买签证保险？
康慧宝需要买多少？
如果另一方对车辆事故负有全部责任，并且拒绝提前支付维修费，该怎么办
准备清明节去新兴坡旅游，什么样的旅游保险好？
你能从国外账户购买互助基金吗？
什么是海上保险？有哪些海上保险？
....
```

问答对集合`qa_pair.csv`包含的是整个项目的问题和对应的答案,，具体格式如下：

```
问题1 \t 答案1
问题2 \t 答案2
......
```
问答对集合示例：

```
既然强制运输保险有浮动费率制度，有商业保险吗？	商业车险也有的。关于汽车商业险的费率在全国每个省都是不一样的，在同一地区，费率也会变化。一般1年、2-4年、4-6年、费率都不同。新车第一年的费率会比较高，2-4是相对比较优惠，4-6会再上涨，不同类型的汽车费率也不同。商业车险保费浮动比例与其他公司相比都是差不多的，一般销售保费浮动比例是这样的：上年赔款1次，保费打7折；上年赔款2次，保费打8折；上年赔款3次，保费上浮15%；上年赔款4次，保费上浮51%；上年赔款5次以上，保费上浮69%。该公司的有关人士表示，如果上年赔款次数超过了7次，续保时可能会遭拒。目前的研究意见规定中加大了车险保费与赔款记录相关系数的浮动区间，并与交通违章情况挂钩，若车主少违章少出险则保费最多可打5折，反之则保费最高可上浮至现行标准的4.5倍。
汇鑫安儿童保险的保费是否也与性别有关	有关系，女宝宝会比男宝宝要多一点。如0岁男宝宝趸交是130.4元，3年期交是43.7元，5年期交是27元；而0岁女宝宝趸交是131.6元，3年期交是44.1元，5年期交是27.2元。
在中国，哪个品牌的餐饮照明比较好？	一般来说美尔家比较可靠吧,有保障
......
```


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
|—— milvus_ann_search.py # 向 Milvus 引擎插入向量的函数
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

以下实验结果使用的是模型是`rocketqa-zh-dureader-query-encoder`：

|  模型 |  Recall@1 |Recall@5 |Recall@10 |
| ------------ | ------------ |--------- |--------- |
|  RocketQA + SimCSE |  82.827 | 93.791| 96.169|
|  RocketQA + SimCSE + WR |  82.695 | 93.791| 96.301|
|  RocketQA + SimCSE + WR + 同义词 |  85.205 | 93.923| 95.509|
|  RocketQA + SimCSE + 同义词 + RDrop |  **85.469** | **94.716**| **96.433**|

<a name="动手实践——搭建自己的端到端检索式问答系统"></a>

## 4. 动手实践——搭建自己的端到端检索式问答系统

### 4.1 环境安装

在运行下面的代码之前，安装相关的依赖，运行下面的命令：

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4.2 模型训练

SimCSE可以使用2种方式进行训练，即有监督训练和无监督训练，区别在于无监督训练不需要标注数据集，有监督训练需要标注好问句对，下面是无监督的执行方式。

#### 无监督训练

无监督训练执行下面的方式，可以选择`train.csv`，纯无监督文本，或者数据增强的数据`train_aug.csv`，然后执行下面的命令：

```
python -u -m paddle.distributed.launch --gpus='0' \
	train.py \
	--device gpu \
	--model_name_or_path rocketqa-zh-base-query-encoder \
	--save_dir ./checkpoints/ \
	--batch_size 64 \
	--learning_rate 5E-5 \
	--epochs 3 \
	--save_steps 50 \
	--eval_steps 50 \
	--max_seq_length 64 \
	--dropout 0.2 \
	--output_emb_size 256 \
	--dup_rate 0.1 \
	--rdrop_coef 0.1 \
	--train_set_file "./data/train_aug.csv"
```

参数含义说明

* `device`: 使用 cpu/gpu 进行训练
* `save_dir`: 模型存储路径
* `model_name_or_path`: 预训练语言模型名，用于模型的初始化
* `batch_size`: 训练的batch size的大小
* `learning_rate`: 训练的学习率的大小
* `epochs`: 训练的epoch数
* `is_unsupervised`:是否使用无监督的训练方式
* `save_steps`： 模型存储 checkpoint 的间隔 steps 个数
* `max_seq_length`: 输入序列的最大长度
* `dropout`: SimCSE的dropout参数
* `output_emb_size`: Transformer 顶层输出的文本向量维度
* `dup_rate` : SimCSE的 Word reptition 策略的重复率
* `train_set_file`: 训练集文件
* `rdrop_coef`: R-Drop的系数

也可以使用下面的bash脚本：

```
sh scripts/train.sh
```

### 4.3  评估

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
        --params_path "checkpoints/model_100/model_state.pdparams" \
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
* `model_name_or_path`: 预训练语言模型名，用于模型的初始化
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
recall@1=84.941
recall@5=94.452
recall@10=96.433
```

参数含义说明
* `similar_text_pair`: 由相似文本对构成的评估集 semantic_similar_pair.tsv
* `recall_result_file`: 针对评估集中第一列文本 *Source Text* 的召回结果
* `recall_num`: 对 1 个文本召回的相似文本数量

### 4.4 模型部署

模型部署模块首先要把动态图转换成静态图，然后转换成serving的格式。

#### 动转静导出

首先把动态图模型转换为静态图：

```
python export_model.py --params_path checkpoints/model_100/model_state.pdparams \
                       --output_path=./output \
                       	--model_name_or_path rocketqa-zh-base-query-encoder
```
也可以运行下面的bash脚本：

```
sh scripts/export_model.sh
```

#### 问答检索引擎

模型准备结束以后，开始搭建 Milvus 的语义检索引擎，用于语义向量的快速检索，本项目使用[Milvus](https://milvus.io/)开源工具进行向量检索，Milvus 的搭建教程请参考官方教程  [Milvus官方安装教程](https://milvus.io/docs/v2.1.x/install_standalone-docker.md)本案例使用的是 Milvus 的2.1 版本，建议使用官方的 Docker-Compose 安装方式，简单快捷。


Milvus 搭建完系统以后就可以插入和检索向量了，首先生成 embedding 向量，每个样本生成256维度的向量：

```
python feature_extract.py \
        --model_dir=./output \
        --model_name_or_path rocketqa-zh-base-query-encoder \
        --corpus_file "data/corpus.csv"
```
其中 output 目录下存放的是召回的 Paddle Inference 静态图模型。

也可以运行下面的bash脚本：

```
sh scripts/feature_extract.sh
```

然后向搭建好的 Milvus 系统插入向量：

```
python milvus_ann_search.py --data_path data/qa_pair.csv \
                            --embedding_path corpus_embedding.npy \
                            --batch_size 100000 \
                            --insert
```

另外，Milvus提供了可视化的管理界面，可以很方便的查看数据，安装地址为[Attu](https://github.com/zilliztech/attu).


#### Paddle Serving 部署

Paddle Serving 的安装可以参考[Paddle Serving 安装文档](https://github.com/PaddlePaddle/Serving#installation)。需要在服务端和客户端安装相关的依赖，用pip安装Paddle Serving的依赖如下：

```
pip install paddle-serving-client==0.8.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install paddle-serving-app==0.8.3 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 如果是CPU部署，只需要安装CPU Server
pip install paddle-serving-server==0.8.3 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 如果是GPU Server，需要确认环境再选择执行哪一条，推荐使用CUDA 10.2的包
# CUDA10.2 + Cudnn7 + TensorRT6（推荐）
pip install paddle-serving-server-gpu==0.8.3.post102 -i https://pypi.tuna.tsinghua.edu.cn/simple
# CUDA10.1 + TensorRT6
pip install paddle-serving-server-gpu==0.8.3.post101 -i https://pypi.tuna.tsinghua.edu.cn/simple
# CUDA11.2 + TensorRT8
pip install paddle-serving-server-gpu==0.8.3.post112 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
更详细的安装信息请参考[链接](https://github.com/PaddlePaddle/Serving/blob/v0.9.0/doc/Install_Linux_Env_CN.md)，安装完依赖后就可以执行下面的步骤。首先把生成的静态图模型导出为 Paddle Serving的格式，命令如下：

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
python web_service.py --model_name_or_path rocketqa-zh-base-query-encoder
```

启动客户端调用 Server, 使用 POST的方式：

向服务端发送 POST 请求示例：

```
curl -X POST -k http://localhost:8090/ernie/prediction -d '{"key": ["0"], "value": ["买了社保，是不是就不用买商业保险了?"]}'
```

也可以使用 rpc的方式：

首先修改rpc_client.py中需要预测的样本：

```
list_data = [
    "买了社保，是不是就不用买商业保险了？",
]
```
然后运行：

```
python rpc_client.py
```

对于Windows用户，启动下面的Pipeline Server:

```
python web_service_windows.py --model_name_or_path rocketqa-zh-base-query-encoder
```

启动客户端调用 Server, 使用 POST的方式(Windows不支持RPC的调用方式)，首先修改http_client.py中需要预测的样本：

```
data = {"feed": ["买了社保，是不是就不用买商业保险了？"], "fetch": ["output_embedding"]}
```
然后运行：
```
python http_client.py
```

### 4.5 问答系统整个流程

问答系统使用了Client Server的模式，即抽取向量的模型部署在服务端，然后启动客户端（Client）端去访问。


```
python run_system.py
```
代码内置的测试用例为：

```
list_data = ["买了社保，是不是就不用买商业保险了？"]
```

会输出如下的结果：

```
......
PipelineClient::predict pack_data time:1663127450.1656108
PipelineClient::predict before time:1663127450.166227
Extract feature time to cost :0.017495155334472656 seconds

=== start connecting to Milvus     ===
=== Connect collection faq_finance ===
Search milvus time cost is 0.18691015243530273 seconds
如果你买社会保险，你不需要买商业保险吗？ 社保是基础的，就是我们通常说的“五险”包括：基本养老保险、基本医疗保险、失业保险、工伤保险和生育保险。而商业保险则是保障。 0.32494643330574036
已有社会保险还需要买商业保险吗 社保是社会保险的简称社会保险是指国家为了预防和分担年老失业疾病以及死亡等社会风险实现社会安全而强制社会多数成员参加的具有所得重分配功能的非营利性的社会安全制度主要包括基本医疗保险基本养老保险工伤保险失业保险生育保险五大类险种，商业保险是社保的一个补充，如果有足够的经济条件可以进行购买。1、社保覆盖面广，不存在拒保问题，但是保障较低，只能满足基本的保障需求。社保中的医疗保险，住院一般可报70%。而且这70%的医疗费，限于扣除起付线标准后。而且，在社保规定用药和规定项目内。许多检查费、专家诊疗、高新尖诊疗技术，社保都是不报的。这就需配合必要的商业保险了。2、另外，社保医疗是出院后报的，商业医保中的重疾险是确诊后就可以给钱，可以弥补很多家庭没钱治的困境；3、商业保险可以选择购买更高的保额，社保则很有限；社保医疗只是补偿医药费，而没有住院期间的收入损失补偿，商业医疗就有住院补贴。总之，建议在有了社保后，再购买适合自己的寿险，加上意外险、住院医疗、重疾医疗保险，就是非常的完善的保障了。 0.38041722774505615
.....
```
输出的结果包括特征提取和检索的时间，还包含检索出来的问答对。


<a name="模型优化"></a>

## 5. 模型优化

### 5.1 有监督训练[优化步骤，可选]

无监督的方式对模型的提升有限，如果需要继续提升模型，则需要标注数据。构造类似`train_aug.csv`中的句子对，只需要构造相似句子对即可，不需要构造不相似的句子对。

```
python -u -m paddle.distributed.launch --gpus='0' \
	train.py \
	--device gpu \
	--model_name_or_path rocketqa-zh-base-query-encoder \
	--save_dir ./checkpoints/ \
	--batch_size 64 \
	--learning_rate 5E-5 \
	--epochs 3 \
	--save_steps 50 \
	--eval_steps 50 \
	--max_seq_length 64 \
	--dropout 0.2 \
	--output_emb_size 256 \
	--dup_rate 0.1 \
	--rdrop_coef 0.1 \
	--train_set_file "./data/train_aug.csv"
```

其他步骤同上，只是使用的数据集是有监督数据。


## 6.参考文献

[1] Tianyu Gao, Xingcheng Yao, Danqi Chen: [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821). EMNLP (1) 2021: 6894-6910
