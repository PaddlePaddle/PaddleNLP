# 对话情绪识别

* [模型简介](#模型简介)
* [快速开始](#快速开始)
* [进阶使用](#进阶使用)
* [版本更新](#版本更新)
* [作者](#作者)
* [如何贡献代码](#如何贡献代码)

## 模型简介

对话情绪识别（Emotion Detection，简称EmoTect），专注于识别智能对话场景中用户的情绪，针对智能对话场景中的用户文本，自动判断该文本的情绪类别并给出相应的置信度，情绪类型分为积极、消极、中性。

对话情绪识别适用于聊天、客服等多个场景，能够帮助企业更好地把握对话质量、改善产品的用户交互体验，也能分析客服服务质量、降低人工质检成本。可通过 [AI开放平台-对话情绪识别](http://ai.baidu.com/tech/nlp_apply/emotion_detection) 线上体验。

效果上，我们基于百度自建测试集（包含闲聊、客服）和 nlpcc2014 微博情绪数据集，进行评测，效果如下表所示，此外我们还开源了百度基于海量数据训练好的模型，该模型在聊天对话语料上 Finetune 之后，可以得到更好的效果。

| 模型 | 闲聊 | 客服 | 微博 |
| :------| :------ | :------ | :------ |
| BOW | 90.2% | 87.6% | 74.2% |
| LSTM | 91.4% | 90.1% | 73.8% |
| Bi-LSTM | 91.2%  | 89.9%  | 73.6% |
| CNN | 90.8% |  90.7% | 76.3%  |
| TextCNN |  91.1% | 91.0% | 76.8% |
| BERT | 93.6% | 92.3%  | 78.6%  |
| ERNIE | 94.4% | 94.0% | 80.6% |

同时推荐用户参考[IPython Notebook demo](https://aistudio.baidu.com/aistudio/projectDetail/122291)。

## 快速开始

### 安装说明

1. PaddlePaddle 安装

   本项目依赖于 PaddlePaddle Fluid 1.6 及以上版本，请参考 [安装指南](http://www.paddlepaddle.org/#quick-start) 进行安装。

2. 代码安装

   克隆代码库到本地

   ```shell
   git clone https://github.com/PaddlePaddle/models.git
   cd models/PaddleNLP/emotion_detection
   ```

3. 环境依赖

   Python 2 的版本要求 2.7.15+，Python 3 的版本要求 3.5.1+/3.6/3.7，其它环境请参考 PaddlePaddle [安装说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/beginners_guide/install/index_cn.html) 部分的内容。

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
.
├── config.json             # 配置文件
├── config.py               # 配置文件读取接口
├── download.py             # 下载数据及预训练模型脚本
├── inference_model.py      # 保存 inference_model 的脚本
├── reader.py               # 数据读取接口
├── run_classifier.py       # 项目的主程序入口，包括训练、预测、评估
├── run.sh                  # 训练、预测、评估运行脚本
├── run_ernie_classifier.py # 基于ERNIE表示的项目的主程序入口
├── run_ernie.sh            # 基于ERNIE的训练、预测、评估运行脚本
├── utils.py                # 其它功能函数脚本
```

### 数据准备

#### **自定义数据**

数据由两列组成，以制表符（'\t'）分隔，第一列是情绪分类的类别（0表示消极；1表示中性；2表示积极），第二列是以空格分词的中文文本，如下示例，文件为 utf8 编码。

```text
label   text_a
0   谁 骂人 了 ？ 我 从来 不 骂人 ， 我 骂 的 都 不是 人 ， 你 是 人 吗 ？
1   我 有事 等会儿 就 回来 和 你 聊
2   我 见到 你 很高兴 谢谢 你 帮 我
```

注：PaddleNLP 项目提供了分词预处理脚本（在preprocess目录下），可供用户使用，具体使用方法如下：

```shell
python tokenizer.py --test_data_dir ./test.txt.utf8 --batch_size 1 > test.txt.utf8.seg
```

#### 公开数据集

这里我们提供一份已标注的、经过分词预处理的机器人聊天数据集，只需运行数据下载脚本 ```sh download_data.sh```，或者 ```python download.py dataset``` 运行成功后，会生成文件夹 ```data```，其目录结构如下：

```text
.
├── train.tsv       # 训练集
├── dev.tsv         # 验证集
├── test.tsv        # 测试集
├── infer.tsv       # 待预测数据
├── vocab.txt       # 词典
```

### 单机训练

基于示例的数据集，可以运行下面的命令，在训练集（train.tsv）上进行模型训练，并在开发集（dev.tsv）验证。
```shell
# TextCNN 模型
sh run.sh train
```
其中 ```run.sh``` 默认训练的是 TextCNN 模型，可直接通过 ```run.sh``` 脚本传入```model_type```参数，或者通过修改 ```config.json``` 中的```model_type``` 选择不同的模型，更多参数配置及说明可以运行如下命令查看

 ```shell
python run_classifier.py -h

"""
# 输出结果示例
Running type options:
  --do_train DO_TRAIN   Whether to perform training. Default: False.
  ...

Model config options:
  --model_type {bow_net,cnn_net,lstm_net,bilstm_net,gru_net,textcnn_net}
                        Model type to run the task. Default: textcnn_net.
  --init_checkpoint INIT_CHECKPOINT
                        Init checkpoint to resume training from. Default: .
  --save_checkpoint_dir SAVE_CHECKPOINT_DIR
                        Directory path to save checkpoints Default: .
...
"""
 ```

本项目参数控制优先级：命令行参数 > ```config.json ``` > 默认值。训练完成后，会在```./save_models/textcnn``` 目录下生成以 ```step_xxx ``` 命名的模型目录。

### 模型评估

基于训练的模型，可以运行下面的命令进行测试，查看预训练的模型在测试集（test.tsv）上的评测结果

```shell
# TextCNN 模型
sh run.sh eval

"""
# 输出结果示例
Load model from ./save_models/textcnn/step_756
Final test result:
[test evaluation] avg loss: 0.339021, avg acc: 0.869691, elapsed time: 0.123983 s
"""
```

默认使用的模型```./save_models/textcnn/step_756```，可修改```run.sh```中的 init_checkpoint 参数，选择其它step的模型进行评估。

### 模型推断

利用已有模型，可在未知label的数据集（infer.tsv）上进行预测，得到模型预测结果及各label的概率。
```shell
# TextCNN 模型
sh run.sh infer

"""
# 输出结果示例
Load model from ./save_models/textcnn/step_756
1       0.000776        0.998341        0.000883
0       0.987223        0.003371        0.009406
1       0.000365        0.998635        0.001001
1       0.000455        0.998125        0.001420
"""
```

### 预训练模型

我们开源了基于海量数据训练好的对话情绪识别模型（基于TextCNN、ERNIE等模型训练），可供用户直接使用，我们提供两种下载方式。

**方式一**：基于PaddleHub命令行工具（PaddleHub[安装方式](https://github.com/PaddlePaddle/PaddleHub)）

```shell
mkdir pretrain_models && cd pretrain_models
hub download emotion_detection_textcnn --output_path ./
hub download emotion_detection_ernie_finetune --output_path ./
tar xvf emotion_detection_textcnn-1.0.0.tar.gz
tar xvf emotion_detection_ernie_finetune-1.0.0.tar.gz
```

**方式二**：直接下载脚本

```shell
sh download_model.sh
# 或者
python download.py model
```

以上两种方式会将预训练的 TextCNN 模型和 ERNIE模型，保存在```pretrain_models```目录下，可直接修改```run.sh```脚本中的```init_checkpoint```参数进行评估、预测。

### 服务部署

为了将模型应用于线上部署，可以利用```inference_model.py``` 脚本对模型进行裁剪，只保存网络参数及裁剪后的模型。运行命令如下：

```shell
sh run.sh save_inference_model
```

同时裁剪后的模型使用方法详见```inference_model.py```，测试命令如下:

```shell
python inference_model.py
```


#### 服务器部署

请参考PaddlePaddle官方提供的 [服务器端部署](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/advanced_usage/deploy/inference/index_cn.html) 文档进行部署上线。

## 进阶使用

### 背景介绍

对话情绪识别任务输入是一段用户文本，输出是检测到的情绪类别，包括消极、积极、中性，这是一个经典的短文本三分类任务。

### 模型概览

本项目针对对话情绪识别问题，开源了一系列分类模型，供用户可配置地使用：

+ BOW：Bag Of Words，是一个非序列模型，使用基本的全连接结构；
+ CNN：浅层CNN模型，能够处理变长的序列输入，提取一个局部区域之内的特征；；
+ TextCNN：多卷积核CNN模型，能够更好地捕捉句子局部相关性；
+ LSTM：单层LSTM模型，能够较好地解决序列文本中长距离依赖的问题；
+ BI-LSTM：双向单层LSTM模型，采用双向LSTM结构，更好地捕获句子中的语义特征；
+ ERNIE：百度自研基于海量数据和先验知识训练的通用文本语义表示模型，并基于此在对话情绪分类数据集上进行fine-tune获得。

### 自定义模型

可以根据自己的需求，组建自定义的模型，具体方法如下所示：

1. 定义自己的网络结构

   用户可以在 ```models/classification/nets.py``` 中，定义自己的模型，只需要增加新的函数即可。假设用户自定义的函数名为```user_net```

2. 更改模型配置

   在 ```config.json``` 中需要将 ```model_type``` 改为用户自定义的 ```user_net```

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
1      0.000803      0.998870      0.000326
0      0.976585      0.021535      0.001880
1      0.000572      0.999153      0.000275
1      0.001113      0.998502      0.000385
'''
```

### 基于 PaddleHub 加载 ERNIE 进行 Finetune

我们也提供了使用 PaddleHub 加载 ERNIE 模型的选项，PaddleHub 是 PaddlePaddle 的预训练模型管理工具，可以一行代码完成预训练模型的加载，简化预训练模型的使用和迁移学习。更多相关的介绍，可以查看 [PaddleHub](https://github.com/PaddlePaddle/PaddleHub)

注意：使用该选项需要先安装PaddleHub >= 1.2.0，安装命令如下
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

2019/10/21 PaddlePaddle1.6适配，添加download.py脚本。

2019/08/26 规范化配置的使用，对模块内数据处理代码进行了重构，更新README结构，提高易用性。

2019/06/13 添加PaddleHub调用ERNIE方式。

## 作者

- [chenbingjin](https://github.com/chenbjin)
- [wuzewu](https://github.com/nepeplwu)

## 如何贡献代码

如果你可以修复某个issue或者增加一个新功能，欢迎给我们提交PR。如果对应的PR被接受了，我们将根据贡献的质量和难度进行打分（0-5分，越高越好）。如果你累计获得了10分，可以联系我们获得面试机会或者为你写推荐信。
