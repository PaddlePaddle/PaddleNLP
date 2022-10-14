# 使用预训练模型Fine-tune完成中文文本分类任务


在2017年之前，工业界和学术界对NLP文本处理依赖于序列模型[Recurrent Neural Network (RNN)](../rnn).

<p align="center">
<img src="http://colah.github.io/posts/2015-09-NN-Types-FP/img/RNN-general.png" width="40%" height="30%"> <br />
</p>


[paddlenlp.seq2vec是什么? 瞧瞧它怎么完成情感分析](https://aistudio.baidu.com/aistudio/projectdetail/1283423)教程介绍了如何使用`paddlenlp.seq2vec`表征文本语义。

近年来随着深度学习的发展，模型参数的数量飞速增长。为了训练这些参数，需要更大的数据集来避免过拟合。然而，对于大部分NLP任务来说，构建大规模的标注数据集非常困难（成本过高），特别是对于句法和语义相关的任务。相比之下，大规模的未标注语料库的构建则相对容易。为了利用这些数据，我们可以先从其中学习到一个好的表示，再将这些表示应用到其他任务中。最近的研究表明，基于大规模未标注语料库的预训练模型（Pretrained Models, PTM) 在NLP任务上取得了很好的表现。

近年来，大量的研究表明基于大型语料库的预训练模型（Pretrained Models, PTM）可以学习通用的语言表示，有利于下游NLP任务，同时能够避免从零开始训练模型。随着计算能力的发展，深度模型的出现（即 Transformer）和训练技巧的增强使得 PTM 不断发展，由浅变深。


<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/327f44ff3ed24493adca5ddc4dc24bf61eebe67c84a6492f872406f464fde91e" width="60%" height="50%"> <br />
</p>

本图片来自于：https://github.com/thunlp/PLMpapers

本示例展示了以ERNIE([Enhanced Representation through Knowledge Integration](https://arxiv.org/abs/1904.09223))代表的预训练模型如何Finetune完成中文文本分类任务。

## 模型简介

本项目针对中文文本分类问题，开源了一系列模型，供用户可配置地使用：

+ BERT([Bidirectional Encoder Representations from Transformers](https://arxiv.org/abs/1810.04805))中文模型，简写`bert-base-chinese`， 其由12层Transformer网络组成。
+ ERNIE[ERNIE 3.0 Titan: Exploring Larger-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/abs/2112.12731)，支持ERNIE 3.0-Medium 中文模型（简写`ernie-3.0-medium-zh`）和 ERNIE 3.0-Base-zh 等 ERNIE 3.0 轻量级中文模型。
+ RoBERTa([A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692))，支持 24 层 Transformer 网络的 `roberta-wwm-ext-large` 和 12 层 Transformer 网络的 `roberta-wwm-ext`。

| 模型  | dev acc | test acc |
| ---- | ------- | -------- |
| bert-base-chinese  | 0.93833 | 0.94750 |
| bert-wwm-chinese | 0.94583 | 0.94917 |
| bert-wwm-ext-chinese | 0.94667 | 0.95500 |
| ernie-1.0-base-zh  | 0.94667  | 0.95333  |
| ernie-tiny  | 0.93917  | 0.94833 |
| roberta-wwm-ext  | 0.94750  | 0.95250 |
| roberta-wwm-ext-large | 0.95250 | 0.95333 |
| rbt3 | 0.92583 | 0.93250 |
| rbtl3 | 0.9341 | 0.93583 |


## 快速开始

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
pretrained_models/
├── deploy # 部署
│   └── python
│       └── predict.py # python预测部署示例
│   └── serving
│       ├── client.py # 客户端预测脚本
│       └── export_servable_model.py # 导出Serving模型及其配置
├── export_model.py # 动态图参数导出静态图参数脚本
├── predict.py # 预测脚本
├── README.md # 使用说明
└── train.py # 训练评估脚本
```

### 模型训练

我们以中文情感分类公开数据集ChnSentiCorp为示例数据集，可以运行下面的命令，在训练集（train.tsv）上进行模型训练，并在开发集（dev.tsv）验证
```shell
$ unset CUDA_VISIBLE_DEVICES
$ python -m paddle.distributed.launch --gpus "0" train.py --device gpu --save_dir ./checkpoints --use_amp False
```

可支持配置的参数：

* `save_dir`：可选，保存训练模型的目录；默认保存在当前目录checkpoints文件夹下。
* `dataset`：可选，xnli_cn，chnsenticorp 可选，默认为chnsenticorp数据集。
* `max_seq_length`：可选，ERNIE/BERT模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：可选，Fine-tune的最大学习率；默认为5e-5。
* `weight_decay`：可选，控制正则项力度的参数，用于防止过拟合，默认为0.00。
* `epochs`: 训练轮次，默认为3。
* `valid_steps`: evaluate的间隔steps数，默认100。
* `save_steps`: 保存checkpoints的间隔steps数，默认100。
* `logging_steps`: 日志打印的间隔steps数，默认10。
* `warmup_proption`：可选，学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0.1。
* `init_from_ckpt`：可选，模型参数路径，热启动模型训练；默认为None。
* `seed`：可选，随机种子，默认为1000.
* `device`: 选用什么设备进行训练，可选cpu或gpu。如使用gpu训练则参数gpus指定GPU卡号。
* `use_amp`: 是否使用混合精度训练，默认为False。

代码示例中使用的预训练模型是ERNIE，如果想要使用其他预训练模型如BERT等，只需更换`model` 和 `tokenizer`即可。

```python
# 使用ernie预训练模型
# ernie-3.0-medium-zh
model = AutoModelForSequenceClassification.from_pretrained('ernie-3.0-medium-zh',num_classes=2))
tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')

# 使用bert预训练模型
# bert-base-chinese
model = AutoModelForSequenceClassification.from_pretrained('bert-base-chinese', num_class=2)
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
```
更多预训练模型，参考[transformers](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer)


程序运行时将会自动进行训练，评估，测试。同时训练过程中会自动保存模型在指定的`save_dir`中。
如：
```text
checkpoints/
├── model_100
│   ├── model_config.json
│   ├── model_state.pdparams
│   ├── tokenizer_config.json
│   └── vocab.txt
└── ...
```

**NOTE:**
* 如需恢复模型训练，则可以设置`init_from_ckpt`， 如`init_from_ckpt=checkpoints/model_100/model_state.pdparams`。
* 如需使用ernie-tiny模型，则需要提前先安装sentencepiece依赖，如`pip install sentencepiece`
* 使用动态图训练结束之后，还可以将动态图参数导出成静态图参数，具体代码见export_model.py。静态图参数保存在`output_path`指定路径中。
  运行方式：

```shell
python export_model.py --params_path=./checkpoint/model_900/model_state.pdparams --output_path=./export
```
其中`params_path`是指动态图训练保存的参数路径，`output_path`是指静态图参数导出路径。

导出模型之后，可以用于部署，deploy/python/predict.py文件提供了python部署预测示例。运行方式：

```shell
python deploy/python/predict.py --model_dir=./export
```

### 模型预测

启动预测：
```shell
export CUDA_VISIBLE_DEVICES=0
python predict.py --device 'gpu' --params_path checkpoints/model_900/model_state.pdparams
```

将待预测数据如以下示例：

```text
这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般
怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片
作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。
```

可以直接调用`predict`函数即可输出预测结果。

如

```text
Data: 这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般      Label: negative
Data: 怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片      Label: negative
Data: 作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。      Label: positive
```


## 使用Paddle Serving API进行推理部署

**NOTE：**

使用Paddle Serving服务化部署需要将动态图保存的模型参数导出为静态图Inference模型参数文件。如何导出模型参考上述提到的**导出模型**。

Inference模型参数文件：
| 文件                          | 说明                                   |
|-------------------------------|----------------------------------------|
| inference.pdiparams | 模型权重文件，供推理时加载使用            |
| inference.pdmodel   | 模型结构文件，供推理时加载使用            |


### 依赖安装

* 服务器端依赖：

```shell
pip install paddle-serving-app paddle-serving-client paddle-serving-server
```

如果服务器端可以使用GPU进行推理，则安装server的gpu版本，安装时要注意参考服务器当前CUDA、TensorRT的版本来安装对应的版本：[Serving readme](https://github.com/PaddlePaddle/Serving/tree/v0.6.0)

```shell
pip install paddle-serving-app paddle-serving-client paddle-serving-server-gpu
```

* 客户端依赖：

```shell
pip install paddle-serving-app paddle-serving-client
```

建议在**docker**容器中运行服务器端和客户端以避免一些系统依赖库问题，启动docker镜像的命令参考：[Serving readme](https://github.com/PaddlePaddle/Serving/tree/v0.6.0)

### Serving的模型和配置导出

使用Serving进行预测部署时，需要将静态图inference model导出为Serving可读入的模型参数和配置。运行方式如下：

```shell
python -u deploy/serving/export_servable_model.py \
    --inference_model_dir ./export/ \
    --model_file inference.pdmodel \
    --params_file inference.pdiparams
```

可支持配置的参数：
* `inference_model_dir`： Inference推理模型所在目录，这里假设为 export 目录。
* `model_file`： 推理需要加载的模型结构文件。
* `params_file`： 推理需要加载的模型权重文件。

执行命令后，会在当前目录下生成2个目录：serving_server 和 serving_client。serving_server目录包含服务器端所需的模型和配置，需将其拷贝到服务器端容器中；serving_client目录包含客户端所需的配置，需将其拷贝到客户端容器中。

### 服务器启动server

在服务器端容器中，启动server

```shell
python -m paddle_serving_server.serve \
    --model ./serving_server \
    --port 8090
```
其中：
* `model`： server加载的模型和配置所在目录。
* `port`： 表示server开启的服务端口8090。

如果服务器端可以使用GPU进行推理计算，则启动服务器时可以配置server使用的GPU id

```shell
python -m paddle_serving_server.serve \
    --model ./serving_server \
    --port 8090 \
    --gpu_id 0
```
* `gpu_id`： server使用0号GPU。


### 客服端发送推理请求

在客户端容器中，使用前面得到的serving_client目录启动client发起RPC推理请求。和使用Paddle Inference API进行推理一样。

### 从命令行读取输入数据发起推理请求
```shell
python deploy/serving/client.py \
    --client_config_file ./serving_client/serving_client_conf.prototxt \
    --server_ip_port 127.0.0.1:8090 \
    --max_seq_length 128
```
其中参数释义如下：
- `client_config_file` 表示客户端需要加载的配置文件。
- `server_ip_port` 表示服务器端的ip地址和端口号。ip地址和端口号需要根据实际情况进行更换。
- `max_seq_length` 表示输入的最大句子长度，超过该长度将被截断。
