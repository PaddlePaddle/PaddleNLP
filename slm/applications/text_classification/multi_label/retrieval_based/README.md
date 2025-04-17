# 基于检索的多标签文本分类方法

 **目录**

* [1. 基于语义索引的多标签分类任务介绍](#基于语义索引的多标签分类任务介绍)
* [2. 代码结构说明](#代码结构说明)
* [3. 环境准备](#环境准备)
* [4. 数据准备](#数据准备)
* [5. 模型训练](#模型训练)
* [6. 模型评估](#模型训练)
* [7. 模型预测](#模型预测)
* [8. 模型部署](#模型部署)
* [9. 分类流程](#分类流程)

<a name="基于语义索引的多标签分类任务介绍"></a>

# 1.基于语义索引的多标签分类任务介绍

以前的分类任务中，标签信息作为无实际意义，独立存在的 one-hot 编码形式存在，这种做法会潜在的丢失标签的语义信息，本方案把文本分类任务中的标签信息转换成含有语义信息的语义向量，将文本分类任务转换成向量检索和匹配的任务。这样做的好处是对于一些类别标签不是很固定的场景，或者需要经常有一些新增类别的需求的情况非常合适。另外，对于一些新的相关的分类任务，这种方法也不需要模型重新学习或者设计一种新的模型结构来适应新的任务。总的来说，这种基于检索的文本分类方法能够有很好的拓展性，能够利用标签里面包含的语义信息，不需要重新进行学习。这种方法可以应用到相似标签推荐，文本标签标注，金融风险事件分类，政务信访分类等领域。

本方案是基于语义索引模型的分类，语义索引模型的目标是：给定输入文本，模型可以从海量候选召回库中**快速、准确**地召回一批语义相关文本。基于语义索引的多标签分类方法有两种，第一种方法是直接把标签变成召回库，即把输入文本和标签的文本进行匹配，第二种是利用召回的文本带有类别标签，把召回文本的类别标签作为给定输入文本的类别。本方案使用双塔模型，训练阶段引入 In-batch Negatives  策略，使用 hnswlib 建立索引库，并把标签作为召回库，进行召回测试。最后利用召回的结果使用 Accuracy 指标来评估语义索引模型的分类的效果。

**注意** 基于语义索引的文本分类的标签在预测过程中会抽取成向量，所以标签需要文本的形式，不能是 ID 形式的标签。

<a name="代码结构说明"></a>

## 2. 代码结构说明

```
|—— data.py # 数据读取、数据转换等预处理逻辑
|—— base_model.py # 语义索引模型基类
|—— train.py # In-batch Negatives 策略的训练主脚本
|—— model.py # In-batch Negatives 策略核心网络结构

|—— recall.py # 基于训练好的语义索引模型，从召回库中召回给定文本的相似文本
|—— evaluate.py # 根据召回结果和评估集计算评估指标
|—— metric.py # Macro F1和Micro F1评估指标
|—— predict.py # 给定输入文件，计算文本 pair 的相似度
|—— export_model.py # 动态图转换成静态图
|—— export_to_serving.py # 静态图转 Serving
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

```

<a name="环境准备"></a>

## 3. 环境准备

推荐使用 GPU 进行训练，在预测阶段使用 CPU 或者 GPU 均可。

**环境依赖**
* python >= 3.7
* paddlepaddle >= 2.3.1
* paddlenlp >= 2.3.4
* hnswlib >= 0.5.2
* visualdl >= 2.2.2

```
pip install -r requirements.txt
```

<a name="数据准备"></a>

## 4. 数据准备

训练需要准备指定格式的本地数据集,如果没有已标注的数据集，可以参考[文本分类任务 doccano 数据标注使用指南](../../doccano.md)进行文本分类数据标注。

**指定格式本地数据集目录结构**

```
├── data # 数据集目录
    ├── label.txt # 标签集
    ├── dev.txt # 验证集
    ├── test.txt # 测试集
    ├── train.txt  # 训练集
```

**训练、开发、测试数据集**

train.txt(训练数据集文件)， dev.txt(开发数据集文件)，test.txt(可选，测试数据集文件)，文件中文本与标签类别名用 tab 符`'\t'`分隔开，由于文本有多个标签，需要把文本拆分成多个单标签的形式，即多行文本标签对。训练集指用于训练模型的数据；开发集指用于评测模型表现的数据，可以根据模型在开发集上的精度调整训练参数和模型；测试集用于测试模型表现，没有测试集时可以使用开发集代替。

- train.txt/test.txt 文件格式：
```text
<文本>'\t'<标签1>
<文本>'\t'<标签2>
...
```
- train.txt/test.txt 文件样例：
```text
茄子怎么做才好吃？怎么做才好吃？	生活
茄子怎么做才好吃？怎么做才好吃？	美食/烹饪
茄子怎么做才好吃？怎么做才好吃？	烹饪方法
...
```
- dev.txt 文件格式：
```
<文本>'\t'<标签1>,<标签2>,<标签3>
<文本>'\t'<标签2>,<标签5>
```
- dev.txt 文件样例：
```text
克隆人的思想和原体是一样的吗？	教育/科学,理工学科,生物学
家用燃气灶哪个牌子好些?最好是高效节能的~最好是高效的~	生活,购物
咨询下，黛莱美面膜怎么样，那里有咨询下，黛莱美怎么样，那里有	生活,美容/塑身,护肤
...
```

**分类标签**

label.txt(分类标签文件)记录数据集中所有标签集合，每一行为一个标签名。
- label.txt 文件格式：
```text
<标签>
<标签>
...
```
- label.txt 文件样例：
```text
魔力宝贝
完美游戏
地球科学
美食/烹饪
肝胆外科
动漫
婚嫁
历史话题
新生儿
...
```

<a name="模型训练"></a>

## 5. 模型训练

我们使用百科知识问答的数据来构建训练集，开发集。

**训练集（train.txt）** 和 **开发集(dev.txt)** 格式一致，训练集30k 条，开发集10k 条，每行由文本的标题，内容和类别标签组成，以 tab 符分割，第一列是问题的标题和问题的描述拼接，剩下的列问题的类别。
**召回库（label.txt）** 类别的数量是323类，召回标签库的构建有2种方式，第一种是把所有的类别标签当成召回库，第二种是把训练集当成召回集合，我们以第一种为例。

数据集选择的是百科问答数据集的一个子集，问答数据集详情请参考[nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus)

- [baike_qa_category](https://paddlenlp.bj.bcebos.com/applications/baike_qa_multilabel.zip)

```
wget https://paddlenlp.bj.bcebos.com/applications/baike_qa_multilabel.zip
unzip  baike_qa_category.zip
```

### 单机单卡训练/单机多卡训练

这里采用单机多卡方式进行训练，通过如下命令，指定 GPU 0,1 卡;如果采用单机单卡训练，只需要把`--gpus`参数设置成单卡的卡号即可。

如果使用 CPU 进行训练，则需要吧`--gpus`参数去除，然后吧`device`设置成 cpu 即可，详细请参考 train.sh 文件的训练设置

然后运行下面的命令使用 GPU 训练，得到语义索引模型：

```
root_path=inbatch
data_path=data
python -u -m paddle.distributed.launch --gpus "0,1" \
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
    --train_set_file ${data_path}/train.txt \
    --corpus_file ${data_path}/label.txt   \
    --similar_text_pair_file ${data_path}/dev.txt \
    --evaluate True
```

参数含义说明

* `device`: 使用 cpu/gpu 进行训练
* `save_dir`: 模型存储路径
* `batch_size`: 训练的 batch size 的大小
* `learning_rate`: 训练的学习率的大小
* `epochs`: 训练的 epoch 数
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

也可以使用 bash 脚本：

```
sh scripts/train.sh
```

<a name="模型评估"></a>

## 6. 模型评估

评估脚本命令如下：
```
python -u evaluate.py \
        --similar_text_pair "data/dev.txt" \
        --recall_result_file "./recall_result_dir/recall_result.txt" \
        --label_path data/label.txt
```
也可以使用 bash 脚本：

```
sh scripts/evaluate.sh
```
会得到如下的输出结果：

```
Micro fl score: 99.30934877025769
Macro f1 score: 78.20694877991563
```

<a name="模型预测"></a>

## 7. 模型预测

我们可以基于语义索引模型计算文本和标签的语义相似度。


### 开始预测

加载训练的语义索引模型，然后计算文本和标签的语义相似度:

```
root_dir="checkpoints/inbatch/model_best"
python -u -m paddle.distributed.launch --gpus "0" \
    predict.py \
    --device gpu \
    --params_path "${root_dir}/model_state.pdparams" \
    --model_name_or_path rocketqa-zh-dureader-query-encoder \
    --output_emb_size 0 \
    --batch_size 128 \
    --max_seq_length 384 \
    --text_pair_file "data/test.txt"
```

参数含义说明
* `device`: 使用 cpu/gpu 进行训练
* `params_path`： 预训练模型的参数文件名
* `output_emb_size`: Transformer 顶层输出的文本向量维度
* `model_name_or_path`: 预训练模型，用于模型和`Tokenizer`的参数初始化。
* `text_pair_file`: 由文本 Pair 构成的待预测数据集

也可以运行下面的 bash 脚本：

```
sh scripts/predict.sh
```
predict.sh 文件包含了 cpu 和 gpu 运行的脚本，默认是 gpu 运行的脚本

产出如下结果
```
0.8841502070426941
0.7834227681159973
0.04591505229473114
0.15116563439369202
......
```

<a name="部署"></a>

## 8. 模型部署

模型部署分为：动转静导出，向量引擎，Paddle Inference 推理， Paddle Serving 服务化这几个部分。为了提升预测速度，通常需要把训练好的模型转换成静态图，然后就可以使用 Paddle Inference 静态图进行推理，向量引擎则是存放标签的向量的形式，方便快速检索，另外，Paddle Inference 可以进一步对模型服务化，即使用 Paddle Serving 进行服务化，这样可以通过 HTTP 或者 RPC 的方式进行调用。Paddle Serving 的服务化形式有 Pipeline 和 C++两种形式，Pipeline 灵活一点，方便进行修改，C++部署更麻烦一点，但 C++的部署形式效率更高。

### 动转静导出

首先把动态图模型转换为静态图：

```
python export_model.py \
                 --params_path checkpoints/inbatch/model_best/model_state.pdparams \
                 --model_name_or_path rocketqa-zh-dureader-query-encoder \
                 --output_path=./output
```
也可以运行下面的 bash 脚本：

```
sh scripts/export_model.sh
```
### Paddle Inference 预测

预测既可以抽取向量也可以计算两个文本的相似度。

修改 deploy/python/predict.py 中的 id2corpus 和 corpus_list 的样本：

```
# 抽取向量
id2corpus = {0: {"sentence": "快要到期的“我的资料”怎么续日期？"}}
# 计算文本和类别的相似度
corpus_list = [{
        "sentence": "快要到期的“我的资料”怎么续日期？",
        'label': '互联网'
    }, {
        "sentence": "快要到期的“我的资料”怎么续日期？",
        'label': '游戏'
    }]

```

然后使用 PaddleInference

```
python deploy/python/predict.py --model_dir=./output
```
也可以运行下面的 bash 脚本：

```
sh deploy.sh
```
最终输出的是256维度的特征向量和句子对的预测概率：

```
(1, 768)
[[ 0.01510728 -0.03822846 -0.0350773  -0.02304687  0.04219331 -0.04335611
  -0.03983097  0.04164692  0.04074539 -0.02351343  0.04246496 -0.02563381
   ....

[0.8133385181427002, 0.1509452909231186]
```

### 向量引擎

模型准备结束以后，开始搭建 Milvus 的向量检索引擎，用于文本语义向量的快速检索，本项目使用[Milvus](https://milvus.io/)开源工具进行向量检索，Milvus 的搭建教程请参考官方教程  [Milvus 官方安装教程](https://milvus.io/cn/docs/v1.1.1/milvus_docker-cpu.md)本案例使用的是 Milvus 的1.1.1 CPU 版本，建议使用官方的 Docker 安装方式，简单快捷。


Milvus 搭建完系统以后就可以插入和检索向量了，首先生成 embedding 向量，每个样本生成768维度的向量：

```
CUDA_VISIBLE_DEVICES=0 python utils/feature_extract.py \
        --data_name label \
        --model_dir ./output \
        --output_dir data \
        --corpus_file "./data/label.txt"
```
其中 output 目录下存放的是召回的 Paddle Inference 静态图模型。

修改 utils/config.py 的配置 ip 和端口，本项目使用的是8530端口，而 Milvus 默认的是19530，需要根据情况进行修改：
```
MILVUS_HOST='your milvus ip'
MILVUS_PORT = 8530
```

然后向搭建好的 Milvus 系统插入向量：

```
python utils/vector_insert.py \
                    --vector_path ./data/label_embedding.npy
```
也可以直接运行：

```bash
sh scripts/run.sh
```

### Paddle Serving 部署

Paddle Serving 的安装可以参考[Paddle Serving 安装文档](https://github.com/PaddlePaddle/Serving#installation)。需要在服务端和客户端安装相关的依赖，用 pip 安装 Paddle Serving 的依赖如下：

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
更详细的安装信息请参考[链接](https://github.com/PaddlePaddle/Serving/blob/v0.9.0/doc/Install_Linux_Env_CN.md)，安装完依赖后就可以执行下面的步骤。首先把生成的静态图模型导出为 Paddle Serving 的格式，命令如下:

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

Paddle Serving 的部署采用 Pipeline 的方式，如果用户有对性能有更高的要求，可以采用 C++的部署形式，请参考[Neural Search](../../../neural_search/recall/in_batch_negative/#c%E7%9A%84%E6%96%B9%E5%BC%8F)：

#### Pipeline 方式

启动 Pipeline Server:

```
cd deploy/python/
python web_service.py
```

启动客户端调用 Server, 使用 POST 的方式：

向服务端发送 POST 请求示例：

```
curl -X POST -k http://localhost:8090/ernie/prediction -d '{"key": ["0"], "value": ["{\"sentence\": \"中国农业大学怎么样？可以吗？\"}"]}'
```

也可以使用 rpc 的方式：
首先修改 rpc_client.py 中需要预测的样本：

```
list_data = [{
    "sentence": "中国农业大学怎么样？可以吗？"
}]
```
然后运行：

```
python rpc_client.py
```
模型的输出为：

```
PipelineClient::predict pack_data time:1658988633.3673246
PipelineClient::predict before time:1658988633.3678396
time to cost :0.014188766479492188 seconds
['output_embedding']
(1, 768)
[[-0.06491912 -0.0133915   0.00937684  0.01285653 -0.02468005  0.03528611
   0.0623698  -0.06062918  0.02238894 -0.05348937  0.02161925  0.04480227
   ......
```

可以看到客户端发送了1条文本，返回这个 embedding 向量

<a name="分类流程"></a>

## 9. 分类流程

为了演示基于检索的文本分类流程，我们使用下面的 python 脚本来完成整个流程，该分类系统使用了 Client Server 的模式，即抽取向量的模型部署在服务端，然后启动客户端（Client）端去访问，得到分类的结果。

```
python run_system.py
```
代码内置的测试用例为：

```
list_data = [{"sentence": "中国农业大学怎么样？可以吗？"}]
```
会输出如下的结果：

```
......
PipelineClient::predict pack_data time:1658988661.507715
PipelineClient::predict before time:1658988661.5081818
Extract feature time to cost :0.02322244644165039 seconds
Search milvus time cost is 0.06801486015319824 seconds
{'sentence': '中国农业大学怎么样？可以吗？'} 教育/科学 0.6138255596160889
{'sentence': '中国农业大学怎么样？可以吗？'} 院校信息 0.9188833236694336
```
输出的结果包括特征提取和检索的时间，还包含检索出来文本和对应的标签，通过设定阈值等方式可以得到最终的标签。

## Reference

[1] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih, Dense Passage Retrieval for Open-Domain Question Answering, Preprint 2020.
