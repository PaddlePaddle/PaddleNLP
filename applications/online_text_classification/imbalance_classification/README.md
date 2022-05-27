# 基于语义索引的文本分类

 **目录**

* [1. 场景概述](#场景概述)
* [2. 基于语义检索的分类方案](#基于语义检索的分类方案)
* [3. 动手实践——搭建自己的端到端检索式文本分类系统](#动手实践——搭建自己的端到端检索式文本分类系统)  


<a name="场景概述"></a>

## 1. 场景概述

语义索引不仅可以用于搜索，还可以把它进行改造应用到文本分类领域，由于语义索引技术是标签无关的方法，所以非常适合标签体系的变动的情况，在一些场景下，分类的标签需要经常的变动。可以应用到相似标签推荐，文本标签标注，金融风险事件分类，政务信访分类等领域。


<a name="基于语义检索的分类方案"></a>

## 2. 基于语义检索的分类方案

### 2.1 数据说明

数据集来源于[NLPCC对话行为分析](https://aistudio.baidu.com/aistudio/competition/detail/162/0/introduction)的比赛数据集，经过处理，一定的数据处理，最终的数据情况如下：

|   训练集[未标注]|   训练集 | 评估集 |
|------------ |------------ | ------------ |
|  411359 |  1625 | 29508 |

```
├── data  # 数据集
    ├── train_unsupervised.txt  # 无监督训练集
    ├── val_data.txt # 验证集
    ├── train_data.txt  # 训练集
    ├── label_level1.txt # 一级类别标签
    ├── label_level2.txt # 二级类别标签
```
数据集的下载链接为: [dialogue_data](https://paddlenlp.bj.bcebos.com/applications/dialogue_data.zip)

### 2.2 代码说明

```
|—— data.py # 数据读取、数据转换等预处理逻辑
|—— model.py # SimCSE模型
|—— train.py # SimCSE训练主脚本
|—— ann_util.py # Ann 建索引库相关函数
|—— config.py # Milvus 配置文件
|—— evaluate.py # 召回评估文件
|—— recall.py # 基于训练好的语义索引模型，从召回库中召回给定文本的相似文本
|__ generate_tags.py # 标签生成函数
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
    |—— generate_tags.sh  # 标签聚合生成
|—— deploy
    |—— python
        |—— rpc_client.py # Paddle Serving 的 Client 端
        |—— web_service.py # Paddle Serving 的 Serving 端
        |—— config_nlp.yml # Paddle Serving 的配置文件
```

### 2.3 效果评估

|  模型 |  一级分类精度 | 二级分类精度 |
| ------------ | ------------ | ------------ |
|  rocketQA检索，768维的向量 |  47.09 | 25.14 |
|  RocketQA + SimCSE |  **49.82** |  32.03 |
|  RocketQA + SimCSE + WR |  48.31 |  **36.51** |

<a name="动手实践——搭建自己的端到端检索式问答系统"></a>

## 3. 动手实践——搭建自己的端到端检索式问答系统

### 3.1 无监督训练

```
python -u -m paddle.distributed.launch --gpus '3' \
	train.py \
	--device gpu \
	--save_dir ./checkpoints/ \
	--batch_size 64 \
	--learning_rate 5E-5 \
	--epochs 1 \
	--save_steps 10 \
	--eval_steps 100 \
	--max_seq_length 128 \
	--dropout 0.1 \
    --output_emb_size 0 \
    --dup_rate 0.1 \
	--train_set_file "./data/train_data.txt"
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

### 3.2  评估

效果评估分为 4 个步骤:

a. 获取Doc端Embedding

基于语义索引模型抽取出Doc样本库的文本向量。

b. 采用hnswlib对Doc端Embedding建库

使用 ANN 引擎构建索引库(这里基于 [hnswlib](https://github.com/nmslib/hnswlib) 进行 ANN 索引)

c. 获取question的Embedding并查询相似结果

基于语义索引模型抽取出评估集 *Source Text* 的文本向量，在第 2 步中建立的索引库中进行 ANN 查询，召回 Top10 最相似的 *Target Text*, 产出评估集中 *Source Text* 的召回结果 `recall_result` 文件。

d. 评估

基于评估集 `val_data.txt` 和召回结果 `recall_result` 计算评估指标Accuracy.

运行如下命令进行 ANN 建库、召回，产出召回结果数据 `recall_result`

```
text_file='./data/val_data.txt'
corpus_file='./data/train_data.txt'
python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "checkpoints/model_50/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 128 \
        --similar_text_pair "${text_file}" \
        --corpus_file "${corpus_file}"
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

run_build_index.sh还包含cpu和gpu运行的脚本，默认是gpu的脚本。

接下来生成数据的标签：

```
python generate_result.py \
                        --recall_path ./recall_result_dir/recall_result.txt \
                        --tag_path ./data/answer_tag.txt \
                        --threshold 0.3
```
参数含义说明
* `recall_path`: 召回结果的文件
* `tag_path`: 保存标签的路径
* `threshold`: 设置检索文本的阈值

也可以使用下面的bash脚本：

```
sh scripts/generate.sh
```

接下来，运行如下命令进行效果评估，产出Recall@1, Recall@5, Recall@10 指标:
```
python -u evaluate.py \
        --predict_file "./data/answer_tag.txt" \
        --target_file "./data/val_data.txt"
```
参数含义说明
* `predict_file`: 预测的标签文件
* `target_file`: 真实的标签文件

也可以使用下面的bash脚本：

```
sh scripts/evaluate.sh
```
输出如下的结果：

```
1级分类
0.4941032940219601
2级分类
0.3481767656228819
```



## 3.3 模型部署

模型部署模块首先要把动态图转换成静态图，然后转换成serving的格式。

### 动转静导出

首先把动态图模型转换为静态图：

```
python export_model.py \
                --params_path checkpoints/model_60/model_state.pdparams \
                --output_path=./output
```
也可以运行下面的bash脚本：

```
sh scripts/export_model.sh
```

### 分类引擎

模型准备结束以后，开始搭建 Milvus 的向量检索引擎，用于文本语义向量的快速检索，本项目使用[Milvus](https://milvus.io/)开源工具进行向量检索，Milvus 的搭建教程请参考官方教程  [Milvus官方安装教程](https://milvus.io/cn/docs/v1.1.1/milvus_docker-cpu.md)本案例使用的是 Milvus 的1.1.1 CPU版本，建议使用官方的 Docker 安装方式，简单快捷。


Milvus 搭建完系统以后就可以插入和检索向量了，首先生成 embedding 向量，每个样本生成768维度的向量：

```
CUDA_VISIBLE_DEVICES=2 python feature_extract.py \
        --data_name train \
        --model_dir ./output \
        --output_dir data \
        --corpus_file "./data/train_data.txt"
```
其中 output 目录下存放的是召回的 Paddle Inference 静态图模型。

然后向搭建好的 Milvus 系统插入向量：

```
python vector_insert.py \
                    --vector_path ./data/train_embedding.npy
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
curl -X POST -k http://localhost:8090/ernie/prediction -d '{"key": ["0"], "value": ["{\"text_a\":\"这边的话是这的置业顾问。你好。你好，几位？\",\"text_b\":\"你好，几位？\"}"]}'
```

也可以使用 rpc的方式：

首先修改rpc_client.py中需要预测的样本：

```
list_data = [
    {"text_a":"这边的话是这的置业顾问。你好。你好，几位？","text_b":"你好，几位？"}
]
```
然后运行：

```
python rpc_client.py
```

## 3.4 分类系统整个流程

问答系统使用了Client Server的模式，即抽取向量的模型部署在服务端，然后启动客户端（Client）端去访问。


```
python run_system.py
```
代码内置的测试用例为：

```
list_data = [{"text_a":"这边的话是这的置业顾问。你好。你好，几位？","text_b":"你好，几位？"}]
```

会输出如下的结果：

```
......
Extract feature time to cost :0.020164012908935547 seconds
Search milvus time cost is 0.0042116641998291016 seconds
{'text_a': '这边的话是这的置业顾问。你好。你好，几位？', 'text_b': '你好，几位？', 'label': '破冰介绍\t开场接待'} 1.0
{'text_a': '啊扫这边。哎呀好好好哎先生，这边请，我来帮您介绍一下啊。呃我是这边的置业顾问。', 'text_b': '呃我是这边的置业顾问。', 'label': '破冰介绍\t开场接待'} 0.6129058599472046
{'text_a': '来这边。嗯要的。我是置业顾问。', 'text_b': '我是置业顾问。', 'label': '破冰介绍\t开场接待'} 0.5949944257736206
{'text_a': '嗯。这位是你的置业顾问。来这边吧。', 'text_b': '来这边吧。', 'label': '破冰介绍\t开场接待'} 0.5796182155609131
.....
```
输出的结果包括特征提取和检索的时间，还包含检索出来文本和对应的标签，通过设定阈值等方式可以得到最终的标签。
