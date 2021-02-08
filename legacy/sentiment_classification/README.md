# 情感倾向分析

* [模型简介](#模型简介)
* [快速开始](#快速开始)
* [进阶使用](#进阶使用)
* [版本更新](#版本更新)
* [作者](#作者)
* [如何贡献代码](#如何贡献代码)

## 模型简介

情感倾向分析（Sentiment Classification，简称Senta）针对带有主观描述的中文文本，可自动判断该文本的情感极性类别并给出相应的置信度。情感类型分为积极、消极。情感倾向分析能够帮助企业理解用户消费习惯、分析热点话题和危机舆情监控，为企业提供有利的决策支持。可通过 [AI开放平台-情感倾向分析](http://ai.baidu.com/tech/nlp_apply/sentiment_classify) 线上体验。

情感是人类的一种高级智能行为，为了识别文本的情感倾向，需要深入的语义建模。另外，不同领域（如餐饮、体育）在情感的表达各不相同，因而需要有大规模覆盖各个领域的数据进行模型训练。为此，我们通过基于深度学习的语义模型和大规模数据挖掘解决上述两个问题。效果上，我们基于开源情感倾向分类数据集ChnSentiCorp进行评测；此外，我们还开源了百度基于海量数据训练好的模型，该模型在ChnSentiCorp数据集上fine-tune之后（基于开源模型进行Finetune的方法请见下面章节），可以得到更好的效果。具体数据如下所示：

| 模型 | dev | test | 模型（finetune） |dev | test |
| :------| :------ | :------ | :------ |:------ | :------
| BOW | 89.8% | 90.0% | BOW |91.3% | 90.6% |
| CNN | 90.6% | 89.9% | CNN |92.4% | 91.8% |
| LSTM | 90.0% | 91.0% | LSTM |93.3% | 92.2% |
| GRU | 90.0% | 89.8% | GRU |93.3% | 93.2% |
| BI-LSTM | 88.5% | 88.3% | BI-LSTM |92.8% | 91.4% |
| ERNIE | 95.1% | 95.4% | ERNIE |95.4% | 95.5% |
| ERNIE+BI-LSTM | 95.3% | 95.2% | ERNIE+BI-LSTM |95.7% | 95.6% |

## 快速开始

### 安装说明

1. PaddlePaddle 安装

   本项目依赖于 PaddlePaddle Fluid 1.7 及以上版本，请参考 [安装指南](http://www.paddlepaddle.org/#quick-start) 进行安装

2. 代码安装

   克隆代码库到本地

   ```shell
   git clone https://github.com/PaddlePaddle/models.git
   cd models/PaddleNLP/sentiment_classification
   ```

3. 环境依赖

   Python 2 的版本要求 2.7.15+，Python 3 的版本要求 3.5.1+/3.6/3.7，其它环境请参考 PaddlePaddle [安装说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/beginners_guide/install/index_cn.html) 部分的内容

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
.
├── senta_config.json           # 配置文件
├── config.py                   # 配置文件读取接口
├── inference_model.py		     # 保存 inference_model 的脚本
├── inference_ernie_model.py	# 保存 inference_ernie__model 的脚本
├── reader.py                   # 数据读取接口
├── run_classifier.py           # 项目的主程序入口，包括训练、预测、评估
├── run.sh                      # 训练、预测、评估运行脚本
├── run_ernie_classifier.py     # 基于ERNIE表示的项目的主程序入口
├── run_ernie.sh                # 基于ERNIE的训练、预测、评估运行脚本
├── utils.py                    # 其它功能函数脚本
```

### 数据准备

#### **自定义数据**

训练、预测、评估使用的数据可以由用户根据实际的应用场景，自己组织数据。数据由两列组成，以制表符分隔，第一列是以空格分词的中文文本（分词预处理方法将在下文具体说明），文件为utf8编码；第二列是情感倾向分类的类别（0表示消极；1表示积极），注意数据文件第一行固定表示为"text_a\tlabel"

```text
特 喜欢 这种 好看的 狗狗                 1
这 真是 惊艳 世界 的 中国 黑科技          1
环境 特别 差 ，脏兮兮 的，再也 不去 了     0
```

注：PaddleNLP 项目提供了分词预处理脚本（在preprocess目录下），可供用户使用，具体使用方法如下：

```shell
python tokenizer.py --test_data_dir ./test.txt.utf8 --batch_size 1 > test.txt.utf8.seg
#其中test.txt.utf8为待分词的文件，一条文本数据一行，utf8编码，分词结果存放在test.txt.utf8.seg文件中。
```

#### 公开数据集

下载经过预处理的数据，文件解压之后，senta_data目录下会存在训练数据（train.tsv）、开发集数据（dev.tsv）、测试集数据（test.tsv）以及对应的词典（word_dict.txt）

```shell
wget https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz
tar -zxvf sentiment_classification-dataset-1.0.0.tar.gz
```

```text
.
├── train.tsv				# 训练集
├── train.tsv           # 验证集
├── test.tsv				# 测试集
├── word_dict.txt			# 词典
```

### 单机训练

基于示例的数据集，可以运行下面的命令，在训练集（train.tsv）上进行模型训练，并在开发集（dev.tsv）验证
```shell
# BOW、CNN、LSTM、BI-LSTM、GRU模型
sh run.sh train
# ERNIE、ERNIE+BI-LSTM模型
sh run_ernie.sh train
```
训练完成后，可修改```run.sh```中init_checkpoint参数，进行模型评估和预测

```
"""
# 输出结果示例
Running type options:
  --do_train DO_TRAIN   Whether to perform training. Default: False.
  ...

Model config options:
  --model_type {bow_net,cnn_net,lstm_net,bilstm_net,gru_net,textcnn_net}
                        Model type to run the task. Default: bilstm_net.
  --init_checkpoint INIT_CHECKPOINT
                        Init checkpoint to resume training from. Default: .
  --checkpoints SAVE_CHECKPOINT_DIR
                        Directory path to save checkpoints Default: .
...
"""
 ```

本项目参数控制优先级：命令行参数 > ```config.json ``` > 默认值。训练完成后，会在```./save_models``` 目录下生成以 ```step_xxx ``` 命名的模型目录。

### 模型评估

基于上面的预训练模型和数据，可以运行下面的命令进行测试，查看预训练模型在开发集（dev.tsv）上的评测效果
```shell
# BOW、CNN、LSTM、BI-LSTM、GRU模型
sh run.sh eval
# ERNIE、ERNIE+BI-LSTM模型
sh run_ernie.sh eval
```
注：如果用户需要使用预训练的BI-LSTM模型，需要修改run.sh和senta_config.json中的配置。run.sh脚本修改如下：
```shell
MODEL_PATH=senta_model/bilstm_model/
# 在eval()函数中，修改如下参数：
--vocab_path $MODEL_PATH/word_dict.txt
--init_checkpoint $MODEL_PATH/params
```
senta_config.json中需要修改如下：
```shell
# vob_size大小对应为上面senta_model/bilstm_model//word_dict.txt，词典大小
"vocab_size": 1256606
```
如果用户需要使用预训练的ERNIE+BI-LSTM模型，需要修改run_ernie.sh中的配置如下：
```shell
# 在eval()函数中，修改如下参数：
--init_checkpoint senta_model/ernie_bilstm_model/
--model_type "ernie_bilstm"
```
```
"""
# 输出结果示例
Load model from ./save_models/step_100
Final test result:
[test evaluation] avg loss: 0.339021, avg acc: 0.869691, elapsed time: 0.123983 s
"""
```

### 模型推断

利用已有模型，可以运行下面命令，对未知label的数据（test.tsv）进行预测
```shell
# BOW、CNN、LSTM、BI-LSTM、GRU模型
sh run.sh infer
#ERNIE+BI-LSTM模型
sh run_ernie.sh infer
```

```
"""
# 输出结果示例
Load model from ./save_models/step_100
1       0.001659        0.998341
0       0.987223        0.012777
1       0.001365        0.998635
1       0.001875        0.998125
"""
```

### 预训练模型

我们开源了基于海量数据训练好的情感倾向分类模型（基于CNN、BI-LSTM、ERNIE等模型训练），可供用户直接使用，我们提供两种下载方式。

**方式一**：基于PaddleHub命令行工具（PaddleHub[安装方式](https://github.com/PaddlePaddle/PaddleHub)）

```shell
hub download sentiment_classification --output_path ./
tar -zxvf sentiment_classification-1.0.0.tar.gz
```

**方式二**：直接下载脚本

```shell
wget https://baidu-nlp.bj.bcebos.com/sentiment_classification-1.0.0.tar.gz
tar -zxvf sentiment_classification-1.0.0.tar.gz
```

以上两种方式会将预训练的 CNN、BI-LSTM等模型和 ERNIE模型，保存在当前目录下，可直接修改```run.sh```脚本中的```init_checkpoint```参数进行评估、预测。

### 服务部署

为了将模型应用于线上部署，可以利用```inference_model.py```、```inference_ernie_model.py``` 脚本对模型进行裁剪，只保存网络参数及裁剪后的模型。运行命令如下：

```shell
sh run.sh save_inference_model
sh run_ernie.sh save_inference_model
```

#### 服务器部署

请参考PaddlePaddle官方提供的 [服务器端部署](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/advanced_usage/deploy/inference/index_cn.html) 文档进行部署上线。

## 进阶使用

### 背景介绍

传统的情感分类主要基于词典或者特征工程的方式进行分类，这种方法需要繁琐的人工特征设计和先验知识，理解停留于浅层并且扩展泛化能力差。为了避免传统方法的局限，我们采用近年来飞速发展的深度学习技术。基于深度学习的情感分类不依赖于人工特征，它能够端到端的对输入文本进行语义理解，并基于语义表示进行情感倾向的判断。

### 模型概览

本项目针对情感倾向性分类问题，开源了一系列模型，供用户可配置地使用：

+ BOW（Bag Of Words）模型，是一个非序列模型，使用基本的全连接结构；
+ CNN（Convolutional Neural Networks），是一个基础的序列模型，能处理变长序列输入，提取局部区域之内的特征；
+ GRU（Gated Recurrent Unit），序列模型，能够较好地解决序列文本中长距离依赖的问题；
+ LSTM（Long Short Term Memory），序列模型，能够较好地解决序列文本中长距离依赖的问题；
+ BI-LSTM（Bidirectional Long Short Term Memory），序列模型，采用双向LSTM结构，更好地捕获句子中的语义特征；
+ ERNIE（Enhanced Representation through kNowledge IntEgration），百度自研基于海量数据和先验知识训练的通用文本语义表示模型，并基于此在情感倾向分类数据集上进行fine-tune获得。
+ ERNIE+BI-LSTM，基于ERNIE语义表示对接上层BI-LSTM模型，并基于此在情感倾向分类数据集上进行Fine-tune获得；

### 自定义模型

可以根据自己的需求，组建自定义的模型，具体方法如下所示：

1. 定义自己的网络结构

   用户可以在 ```models/classification/nets.py``` 中，定义自己的模型，只需要增加新的函数即可。假设用户自定义的函数名为```user_net```

2. 更改模型配置

   在 ```senta_config.json``` 中需要将 ```model_type``` 改为用户自定义的 ```user_net```

3. 模型训练

   通过```run.sh``` 脚本运行训练、评估、预测。

### 基于 ERNIE 进行 Finetune

ERNIE 是百度自研的基于海量数据和先验知识训练的通用文本语义表示模型，基于 ERNIE 进行 Finetune，能够提升对话情绪识别的效果。

#### 模型训练

需要先下载 ERNIE 模型，使用如下命令：

```shell
mkdir -p pretrain_models/ernie
cd pretrain_models/ernie
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/ERNIE_stable-1.0.1.tar.gz -O ERNIE_stable-1.0.1.tar.gz
tar -zxvf ERNIE_stable-1.0.1.tar.gz
```

然后修改```run_ernie.sh``` 脚本中train 函数的 ```init_checkpoint``` 参数，再执行命令：

```shell
#--init_checkpoint ./pretrain_models/ernie
sh run_ernie.sh train
```

默认使用GPU进行训练，模型保存在 ```./save_models/ernie/```目录下，以 ```step_xxx ``` 命名。

#### 模型评估

根据训练结果，可选择最优的step进行评估，修改```run_ernie.sh``` 脚本中 eval 函数 ```init_checkpoint``` 参数，然后执行

```shell
#--init_checkpoint./save/step_907
sh run_ernie.sh eval

'''
# 输出结果示例
W0820 14:59:47.811139   334 device_context.cc:259] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 9.2, Runtime API Version: 9.0
W0820 14:59:47.815557   334 device_context.cc:267] device: 0, cuDNN Version: 7.3.
Load model from ./save_models/ernie/step_907
Final validation result:
[test evaluation] avg loss: 0.260597, ave acc: 0.907336, elapsed time: 2.383077 s
'''
```

#### 模型推断

修改```run_ernie.sh``` 脚本中 infer 函数 ```init_checkpoint``` 参数，然后执行

```shell
#--init_checkpoint./save/step_907
sh run_ernie.sh infer

'''
# 输出结果示例
Load model from ./save_models/ernie/step_907
Final test result:
1      0.001130      0.998870
0      0.978465      0.021535
1      0.000847      0.999153
1      0.001498      0.998502
'''
```

### 基于 PaddleHub 加载 ERNIE 进行 Finetune

我们也提供了使用 PaddleHub 加载 ERNIE 模型的选项，PaddleHub 是 PaddlePaddle 的预训练模型管理工具，可以一行代码完成预训练模型的加载，简化预训练模型的使用和迁移学习。更多相关的介绍，可以查看 [PaddleHub](https://github.com/PaddlePaddle/PaddleHub)

注意：使用该选项需要先安装PaddleHub，安装命令如下
```shell
pip install paddlehub
```

需要修改```run_ernie.sh```中的配置如下：

```shell
# 在train()函数中，修改--use_paddle_hub选项
--use_paddle_hub true
```

执行以下命令进行 Finetune
```shell
sh run_ernie.sh train
```

Finetune 结束后，进行 eval 或者 infer 时，需要修改 ```run_ernie.sh``` 中的配置如下：
```shell
# 在eval()和infer()函数中，修改--use_paddle_hub选项
--use_paddle_hub true
```

执行以下命令进行 eval 和 infer
```shell
sh run_ernie.sh eval
sh run_ernie.sh infer
```

## 版本更新

2019/08/26 规范化配置的使用，对模块内数据处理代码进行了重构，更新README结构，提高易用性。

2019/06/13 添加PaddleHub调用ERNIE方式。

## 作者

- [liuhao](https://github.com/ChinaLiuHao)

## 如何贡献代码

如果你可以修复某个issue或者增加一个新功能，欢迎给我们提交PR。如果对应的PR被接受了，我们将根据贡献的质量和难度进行打分（0-5分，越高越好）。如果你累计获得了10分，可以联系我们获得面试机会或者为你写推荐信。
