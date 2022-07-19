# 基于检索的文本分类方法

 **目录**

* [1. 基于语义索引的分类任务介绍](#基于语义索引的分类任务介绍)
* [2. 环境依赖](#环境依赖)
* [3. 代码结构](#代码结构)
* [4. 数据准备](#数据准备)
* [5. 模型训练](#模型训练)
* [6. 模型预测](#模型预测)
* [7. 模型部署](#模型部署)

<a name="基于语义索引的分类任务介绍"></a>

# 1.基于语义索引的分类任务介绍

以前的分类任务中，标签信息作为无实际意义，独立存在的one-hot编码形式存在，这种做法会潜在的丢失标签的语义信息，本方案把文本分类任务中的标签信息转换成含有语义信息的语义向量，将文本分类任务转换成向量检索和匹配的任务。这样做的好处是对于一些类别标签不是很固定的场景，或者需要经常有一些新增类别的需求的情况非常合适。另外，对于一些新的相关的分类任务，这种方法也不需要模型重新学习或者设计一种新的模型结构来适应新的任务。总的来说，这种基于检索的文本分类方法能够有很好的拓展性，能够利用标签里面包含的语义信息，不需要重新进行学习。

本方案是基于语义索引模型的分类，语义索引模型的目标是：给定输入文本，模型可以从海量候选召回库中**快速、准确**地召回一批语义相关文本。如果召回的文本带有类别标签，则可以把召回文本的类别标签作为给定输入文本的类别。本方案使用双塔模型，训练阶段引入In-batch Negatives  策略，使用hnswlib建立索引库，进行召回测试。最后利用召回的结果使用 Accuracy 指标来评估语义索引模型的分类的效果。


**效果评估**

|  模型 |  Accuracy | 策略简要说明|
| ------------ | ------------ |--------- |
|  ernie-3.0-medium-zh |  50.580 | ernie-3.0-medium-zh多分类，5个epoch，对于新增类别需要重新训练|
|  In-batch Negatives + RocketQA |  49.755 | Inbatch-negative有监督训练，标签当作召回集，对新增类别不需要重新训练|
|  In-batch Negatives + RocketQA + 投票|  **51.756** | Inbatch-negative有监督训练，训练集当作召回集，对新增类别，需要至少一条的数据放入召回库中|

<a name="环境依赖"></a>

## 2. 环境依赖

推荐使用GPU进行训练，在预测阶段使用CPU或者GPU均可。

**环境依赖**
* python >= 3.6.2
* paddlepaddle >= 2.3.1
* paddlenlp >= 2.3.4
* hnswlib >= 0.5.2
* visualdl >= 2.2.2

```
pip install -r requirements.txt
```

<a name="代码结构"></a>

## 3. 代码结构

```
|—— data.py # 数据读取、数据转换等预处理逻辑
|—— base_model.py # 语义索引模型基类
|—— train.py # In-batch Negatives 策略的训练主脚本
|—— model.py # In-batch Negatives 策略核心网络结构
|—— ann_util.py # Ann 建索引库相关函数

|—— recall.py # 基于训练好的语义索引模型，从召回库中召回给定文本的相似文本
|—— evaluate.py # 根据召回结果和评估集计算评估指标
|—— predict.py # 给定输入文件，计算文本 pair 的相似度
|—— export_model.py # 动态图转换成静态图
|—— scripts
    |—— export_model.sh  # 动态图转换成静态图脚本
    |—— predict.sh  # 预测 bash 版本
    |—— evaluate.sh # 评估 bash 版本
    |—— run_build_index.sh # 构建索引 bash 版本
    |—— train.sh  # 训练 bash 版本
    |—— export_to_serving.sh  # Paddle Inference 转 Serving 的 bash 脚本
    |—— run.sh # 构建Milvus向量的 bash 版本
|—— utils
    ├── config.py # Milvus 的配置文件
    ├── feature_extract.py # 向量抽取文件
    ├── milvus_util.py # Milvus 的配置文件
|—— deploy
    |—— python
        |—— predict.py # PaddleInference
        |—— deploy.sh # Paddle Inference 部署脚本
        |—— rpc_client.py # Paddle Serving 的 Client 端
        |—— web_service.py # Paddle Serving 的 Serving 端
        |—— config_nlp.yml # Paddle Serving 的配置文件
|—— export_to_serving.py # 静态图转 Serving
```

<a name="数据准备"></a>

## 4. 数据准备

### 数据集说明

我们百科知识问答的数据来构建训练集，验证集。

**训练集** 和 **验证集** 格式一致，训练集30k条，测试集10k条，每行由文本的标题，内容和类别标签组成，以tab符分割，第一列是问题的标题，第二列是问题的描述，第三列由问题的类型。样例数据如下:

```
打算带辛苦了一年半的爸爸妈妈去周边度假，预计三天两夜的行程，要环? 	打算带辛苦了一年半的爸爸妈妈去周边度假，预计三天两夜的行程，要环境好点的，更要好，要适合带宝宝全家旅行，放松休闲型的，大家有什么好推荐啊？	娱乐-度假旅游
CPU每秒执行的指令数CPU频率3.0G，执行一条指令需要1.5 	频率3.0G，执行一条指令需要1.5个周期，每秒执行的指令数为？是20亿吗？	电脑/网络-硬件
自学笛子用哪本教材比较好？竖笛和横笛。。 	竖笛和横笛。。	文化/艺术-器乐/声乐
```

**召回库** 召回库的构建有2种方式，第一种是把所有的类别标签当成召回库，第二种是把训练集当成召回集合，我们以第一种为例，其样例数据如下：
```
体育/运动-游泳
电脑/网络-程序设计
教育/科学-外语学习-日语
教育/科学-职业教育-会计资格考试
健康-内科-呼吸内科
生活-美食/烹饪-餐厅/酒店
健康-整形美容-四肢整形
教育/科学-理工学科-心理学
商业/理财-保险
教育/科学-理工学科-生物学
```

### 数据集下载

数据集选择的是百科问答数据集的一个子集，问答数据集详情请参考[nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus)

- [baike_qa_category](https://paddlenlp.bj.bcebos.com/applications/baike_qa_category.zip)

```
├── data # 数据集目录
    ├── corpus.txt # 用于测试的召回库
    ├── val_data.txt # 召回测试集
    ├── train_data.txt  # 召回训练集
```

<a name="模型训练"></a>

### 单机单卡训练/单机多卡训练

这里采用单机多卡方式进行训练，通过如下命令，指定 GPU 0,1,2,3 卡, 基于 In-batch Negatives 策略训练模型，数据量比较小，几分钟就可以完成。如果采用单机单卡训练，只需要把`--gpus`参数设置成单卡的卡号即可。

如果使用CPU进行训练，则需要吧`--gpus`参数去除，然后吧`device`设置成cpu即可，详细请参考train.sh文件的训练设置

然后运行下面的命令使用GPU训练，得到语义索引模型：

```
root_path=inbatch
data_path=data
python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
    train.py \
    --device gpu \
    --save_dir ./checkpoints/${root_path} \
    --batch_size 24 \
    --learning_rate 5E-5 \
    --epochs 100 \
    --output_emb_size 0 \
    --save_steps 50 \
    --max_seq_length 384 \
    --warmup_proportion 0.0 \
    --margin 0.2 \
    --recall_result_dir "recall_result_dir" \
    --recall_result_file "recall_result.txt" \
    --train_set_file ${data_path}/train_data.txt \
    --corpus_file ${data_path}/corpus.txt   \
    --similar_text_pair ${data_path}/val_data.txt \
    --evaluate True
```

参数含义说明

* `device`: 使用 cpu/gpu 进行训练
* `save_dir`: 模型存储路径
* `batch_size`: 训练的batch size的大小
* `learning_rate`: 训练的学习率的大小
* `epochs`: 训练的epoch数
* `output_emb_size`: Transformer 顶层输出的文本向量维度
* `save_steps`： 模型存储 checkpoint 的间隔 steps 个数
* `max_seq_length`: 输入序列的最大长度
* `margin`: 正样本相似度与负样本之间的目标 Gap
* `train_set_file`: 训练集文件
* `evaluate`: 是否开启边训练边评估模型训练效果，默认开启
* `recall_result_dir`: 召回结果存储目录
* `recall_result_file`: 召回结果的文件名
* `hnsw_m`: hnsw 算法相关参数，保持默认即可
* `hnsw_ef`: hnsw 算法相关参数，保持默认即可
* `recall_num`: 对 1 个文本召回的相似文本数量
* `similar_text_pair`: 由相似文本对构成的评估集
* `corpus_file`: 召回库数据 corpus_file
* `similar_text_pair`: 由相似文本对构成的评估集

也可以使用bash脚本：

```
sh scripts/train.sh
```

<a name="模型预测"></a>

## 6. 模型预测

我们可以基于语义索引模型计算文本和标签的语义相似度。


### 开始预测

加载训练的语义索引模型，然后计算文本和标签的语义相似度:

```
root_dir="checkpoints/inbatch/model_best"
python -u -m paddle.distributed.launch --gpus "0" \
    predict.py \
    --device gpu \
    --params_path "${root_dir}/model_state.pdparams" \
    --output_emb_size 256 \
    --batch_size 128 \
    --max_seq_length 64 \
    --text_pair_file "data/val_data.txt"
```

参数含义说明
* `device`: 使用 cpu/gpu 进行训练
* `params_path`： 预训练模型的参数文件名
* `output_emb_size`: Transformer 顶层输出的文本向量维度
* `text_pair_file`: 由文本 Pair 构成的待预测数据集

也可以运行下面的bash脚本：

```
sh scripts/predict.sh
```
predict.sh文件包含了cpu和gpu运行的脚本，默认是gpu运行的脚本

产出如下结果
```
0.8835686445236206
0.8856571316719055
0.025357507169246674
0.06661193817853928
......
```

<a name="部署"></a>

## 7. 模型部署

### 动转静导出

首先把动态图模型转换为静态图：

```
python export_model.py --params_path checkpoints/inbatch/model_best/model_state.pdparams --output_path=./output
```
也可以运行下面的bash脚本：

```
sh scripts/export_model.sh
```

### Paddle Inference预测

预测既可以抽取向量也可以计算两个文本的相似度。

修改id2corpus的样本：

```
# 抽取向量
id2corpus = {0: {"text_a": "CPU每秒执行的指令数CPU频率3.0G，执行一条指令需要1.5", "text_b": "频率3.0G，执行一条指令需要1.5个周期，每秒执行的指令数为？是20亿吗？"}}
# 计算文本和类别的相似度
corpus_list = [{"text_a": "CPU每秒执行的指令数CPU频率3.0G，执行一条指令需要1.5", "text_b": "频率3.0G，执行一条指令需要1.5个周期，每秒执行的指令数为？是20亿吗？",'label':'电脑/网络-硬件'},
    {"text_a": "CPU每秒执行的指令数CPU频率3.0G，执行一条指令需要1.5", "text_b": "频率3.0G，执行一条指令需要1.5个周期，每秒执行的指令数为？是20亿吗？",'label':'商业/理财-股票'}]

```

然后使用PaddleInference

```
python deploy/python/predict.py --model_dir=./output
```
也可以运行下面的bash脚本：

```
sh deploy.sh
```
最终输出的是256维度的特征向量和句子对的预测概率：

```
(1, 256)
[[-1.79586373e-02  5.01679294e-02 -4.85644415e-02 -3.94791961e-02
   1.17289117e-02 -2.43468843e-02 -3.92385870e-02  1.08875111e-01
  -4.46584150e-02  3.84004302e-02  1.13016561e-01 -3.85019854e-02
   5.57234883e-02  4.88484185e-03  2.14174646e-03 -6.24658801e-02
   ....

[0.9252777099609375, -0.08459024876356125]
```

### 向量引擎

模型准备结束以后，开始搭建 Milvus 的向量检索引擎，用于文本语义向量的快速检索，本项目使用[Milvus](https://milvus.io/)开源工具进行向量检索，Milvus 的搭建教程请参考官方教程  [Milvus官方安装教程](https://milvus.io/cn/docs/v1.1.1/milvus_docker-cpu.md)本案例使用的是 Milvus 的1.1.1 CPU版本，建议使用官方的 Docker 安装方式，简单快捷。


Milvus 搭建完系统以后就可以插入和检索向量了，首先生成 embedding 向量，每个样本生成768维度的向量：

```
CUDA_VISIBLE_DEVICES=0 python utils/feature_extract.py \
        --data_name train \
        --model_dir ./output \
        --output_dir data \
        --corpus_file "./data/corpus.txt"
```
其中 output 目录下存放的是召回的 Paddle Inference 静态图模型。

然后向搭建好的 Milvus 系统插入向量：

```
python utils/vector_insert.py \
                    --vector_path ./data/corpus_embedding.npy
```
也可以直接运行：

```bash
sh scripts/run.sh
```

### Paddle Serving部署

Paddle Serving 的详细文档请参考 [Pipeline_Design](https://github.com/PaddlePaddle/Serving/blob/v0.7.0/doc/Python_Pipeline/Pipeline_Design_CN.md)和[Serving_Design](https://github.com/PaddlePaddle/Serving/blob/v0.7.0/doc/Serving_Design_CN.md),首先把静态图模型转换成Serving的格式：

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

Paddle Serving的部署有两种方式，第一种方式是Pipeline的方式，第二种是C++的方式，下面分别介绍这两种方式的用法：

#### Pipeline方式

启动 Pipeline Server:

```
python web_service.py
```

启动客户端调用 Server, 使用 POST的方式：

向服务端发送 POST 请求示例：

```
curl -X POST -k http://localhost:8080/ernie/prediction -d '{"key": ["0"], "value": ["{\"text_a\":\"这边的话是这的置业顾问。你好。你好，几位？\",\"text_b\":\"你好，几位？\"}"]}'
```

也可以使用 rpc的方式：
首先修改rpc_client.py中需要预测的样本：

```
list_data = [
    {"text_a": "CPU每秒执行的指令数CPU频率3.0G，执行一条指令需要1.5", "text_b": "频率3.0G，执行一条指令需要1.5个周期，每秒执行的指令数为？是20亿吗？"}
]
```
然后运行：

```
python rpc_client.py
```
模型的输出为：

```
PipelineClient::predict pack_data time:1658132894.7838476
PipelineClient::predict before time:1658132894.7843828
time to cost :0.015886306762695312 seconds
['output_embedding']
(1, 256)
[[-1.79586373e-02  5.01679294e-02 -4.85644415e-02 -3.94791961e-02
   1.17289117e-02 -2.43468843e-02 -3.92385870e-02  1.08875111e-01
  -4.46584150e-02  3.84004302e-02  1.13016561e-01 -3.85019854e-02
   5.57234883e-02  4.88484185e-03  2.14174646e-03 -6.24658801e-02
   ......
```

可以看到客户端发送了1条文本，返回这个 embedding 向量

### 3.4 分类流程

基于检索的分类系统使用了Client Server的模式，即抽取向量的模型部署在服务端，然后启动客户端（Client）端去访问。

```
python run_system.py
```
代码内置的测试用例为：

```
list_data = [{"text_a": "我是一个多情善感的小男孩！我想翻译成英文，谢谢！", "text_b": "我想成英文，谢谢！"}]
```
会输出如下的结果：

```
......
PipelineClient::predict pack_data time:1658221288.6802082
PipelineClient::predict before time:1658221288.680842
Extract feature time to cost :0.014887571334838867 seconds
Search milvus time cost is 0.039603233337402344 seconds
{'text_a': '我是一个多情善感的小男孩！我想翻译成英文，谢谢！', 'text_b': '我想成英文，谢谢！'} 教育/科学-外语学习-英语翻译 0.2987557053565979
{'text_a': '我是一个多情善感的小男孩！我想翻译成英文，谢谢！', 'text_b': '我想成英文，谢谢！'} 教育/科学-外语学习 0.4460639953613281
{'text_a': '我是一个多情善感的小男孩！我想翻译成英文，谢谢！', 'text_b': '我想成英文，谢谢！'} 教育/科学-出国/留学 0.594388484954834
{'text_a': '我是一个多情善感的小男孩！我想翻译成英文，谢谢！', 'text_b': '我想成英文，谢谢！'} 教育/科学-外语学习-法语 0.8342819213867188
.....
```
输出的结果包括特征提取和检索的时间，还包含检索出来文本和对应的标签，通过设定阈值等方式可以得到最终的标签。

## Reference

[1] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih, Dense Passage Retrieval for Open-Domain Question Answering, Preprint 2020.
