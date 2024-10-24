# SKEP: Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis

情感分析旨在自动识别和提取文本中的倾向、立场、评价、观点等主观信息。它包含各式各样的任务，比如句子级情感分类、评价对象级情感分类、观点抽取、情绪分类等。情感分析是人工智能的重要研究方向，具有很高的学术价值。同时，情感分析在消费决策、舆情分析、个性化推荐等领域均有重要的应用，具有很高的商业价值。

情感预训练模型 SKEP（Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis）。SKEP 利用情感知识增强预训练模型， 在14项中英情感分析典型任务上全面超越 SOTA，此工作已经被 ACL 2020录用。SKEP 是百度研究团队提出的基于情感知识增强的情感预训练算法，此算法采用无监督方法自动挖掘情感知识，然后利用情感知识构建预训练目标，从而让机器学会理解情感语义。SKEP 为各类情感分析任务提供统一且强大的情感语义表示。

论文地址：https://arxiv.org/abs/2005.05635

<p align="center">
<img src="https://bj.bcebos.com/paddlenlp/models/transformers/skep/skep.png" width="80%" height="60%"> <br />
</p>

百度研究团队在三个典型情感分析任务，语句级情感分类（Sentence-level Sentiment Classification），评价对象级情感分类（Aspect-level Sentiment Classification）、观点抽取（Opinion Role Labeling），共计14个中英文数据上进一步验证了情感预训练模型 SKEP 的效果。实验表明，下表展示了在模型分别在数据集 SST-2、ChnSentiCorp、SE-ABSA16_PHNS、COTE_DP 上的实验结果，同时标明了各项数据集对应的任务类型、语言类别、下载地址等信息。

<table>
    <tr>
        <td><strong><center>任务</strong></td>
        <td><strong><center>数据集合</strong></td>
        <td><strong><center>语言</strong></td>
        <td><strong><center>指标</strong></td>
        <td><strong><center>SKEP</strong></td>
        <td><strong><center>数据集地址</strong></td>
    </tr>
    <tr>
        <td rowspan="2"><center>语句级情感分类<br /><center>分类</td>
        <td><center>SST-2</td>
        <td><center>英文</td>
        <td><center>ACC</td>
        <td><center>97.60</td>
        <td><center><a href="https://gluebenchmark.com/tasks" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>ChnSentiCorp</td>
        <td><center>中文</td>
        <td><center>ACC</td>
        <td><center>96.08</td>
        <td><center><a href="https://dataset-bj.cdn.bcebos.com/qianyan/ChnSentiCorp.zip" >下载地址</a></td>
    </tr>
    <tr>
        <td rowspan="1"><center>评价对象级<br /><center>情感分类</td>
        <td><center>SE-ABSA16_PHNS</td>
        <td><center>中文</td>
        <td><center>ACC</td>
        <td><center>65.22</td>
        <td><center><a href="http://alt.qcri.org/semeval2016/task5/" >下载地址</a></td>
    </tr>
    <tr>
        <td rowspan="1"><center>观点<br /><center>抽取</td>
        <td><center>COTE_DP</td>
        <td><center>中文</td>
        <td><center>F1</td>
        <td><center>86.30</td>
        <td><center><a href="https://github.com/lsvih/chinese-customer-review" >下载地址</a></td>
    </tr>
</table>


## 快速开始

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
skep/
├── deploy # 部署
│   └── python
│       └── predict.py # python预测部署示例
├── export_model.py # 动态图参数导出静态图参数脚本
├── predict_aspect.py # 对象级的情感分类任务预测脚本
├── predict_opinion.py # 观点抽取任务预测脚本
├── predict_sentence.py # 句子级情感分类任务预测脚本
├── README.md # 使用说明
├── train_aspect.py # 对象级的情感分类任务训练脚本
├── train_opinion.py # 观点抽取任务训练脚本
└── train_sentence.py  # 句子级情感分类任务训练脚本
```

下面以语句级情感分类、评价对象级情感分类，观点抽取等任务类型为例，分别说明相应的训练和测试方式。

### 语句级情感分类
#### 数据下载
本示例采用常用开源数据集 ChnSenticorp 中文数据集、GLUE-SST2英文数据集作为语句级情感分类数据集。这两项数据集已经内置于 PaddleNLP。可以通过以下方式进行加载。

```python
from paddlenlp.datasets import load_dataset

train_ds, dev_ds = load_dataset("chnsenticorp", splits=["train", "dev"])
train_ds, dev_ds = load_dataset("glue", "sst-2", splits=["train", "dev"])
```

#### 模型训练

可以通过如下命令开启语句级情感分析任务训练，需要特别说明的是，如果想要基于数据集 ChnSentiCorp 训练中文情感分析模型，请指定 model_name 为：`skep_ernie_1.0_large_ch`； 基于数据集 GLUE-SST2训练英文情感分析模型请指定 model_name 为：`skep_ernie_2.0_large_en`。下面以中文情感分析为例进行说明。

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" train_sentence.py \
    --model_name "skep_ernie_1.0_large_ch" \
    --device "gpu" \
    --save_dir "./checkpoints" \
    --epochs 3 \
    --max_seq_len 128 \
    --batch_size 16 \
    --learning_rate 5e-5
```

可支持配置的参数：

* `model_name`: 使用预训练模型的名称，可选 skep_ernie_1.0_large_ch 和 skep_ernie_2.0_large_en。
    skep_ernie_1.0_large_ch：是 SKEP 模型在预训练 ernie_1.0_large_ch 基础之上在海量中文数据上继续预训练得到的中文预训练模型;
    skep_ernie_2.0_large_en：是 SKEP 模型在预训练 ernie_2.0_large_en 基础之上在海量英文数据上继续预训练得到的中文预训练模型。
* `save_dir`：可选，保存训练模型的目录；默认保存在当前目录 checkpoints 文件夹下。
* `max_seq_len`：可选，ERNIE/BERT 模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为16。
* `learning_rate`：可选，Fine-tune 的最大学习率；默认为5e-5。
* `weight_decay`：可选，控制正则项力度的参数，用于防止过拟合，默认为0.00。
* `epochs`: 训练轮次，默认为3。
* `init_from_ckpt`：可选，模型参数路径，热启动模型训练；默认为 None。
* `seed`：可选，随机种子，默认为1000.
* `device`: 选用什么设备进行训练，可选 cpu 或 gpu。如使用 gpu 训练则参数 gpus 指定 GPU 卡号。

程序运行时将会自动进行训练，评估，测试。同时训练过程中会自动保存模型在指定的`save_dir`中。

#### 模型预测
使用如下命令进行模型预测：

```shell
export CUDA_VISIBLE_DEVICES=0
python predict_sentence.py \
    --model_name "skep_ernie_1.0_large_ch" \
    --ckpt_dir "checkpoints/model_100" \
    --batch_size 16 \
    --max_seq_len 128 \
    --device "gpu"
```

下面展示了模型的预测示例结果：

```text
Data: 这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般      Label: negative
Data: 怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片      Label: negative
Data: 作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。      Label: positive
```

#### 基于 Taskflow 一键预测
当前 PaddleNLP 已将训练好的 SKEP 中文语句级情感分析模型集成至 Taskflow 中，可以使用 Taskflow 对输入的文本进行一键式情感分析，使用方法如下:

```python
from paddlenlp import Taskflow

senta = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch")
senta("怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片")
'''
[{'text': '这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般', 'label': 'negative', 'score': 0.9894790053367615}]
'''
```

如果想使用自己训练好的模型加载进 Taskflow 进行预测，可以使用参数`task_path`进行指定模型路径，需要注意的是，该路径下需要存放模型文件以及相应的 Tokenizer 文件（训练过程中，已保存这两者相关文件）。

```python
from paddlenlp import Taskflow

senta = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch", task_path="./checkpoints/model_100")
senta("怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片")
'''
[{'text': '这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般', 'label': 'negative', 'score': 0.9686369299888611}]
'''
```

#### 模型部署

使用动态图训练结束之后，还可以将动态图参数导出成静态图参数。在进行模型转换时，需要通过参数`ckpt_dir`指定训练好的模型存放目录，通过`output_path`指定静态图模型参数保存路径，详情请参考 export_model.py。模型转换命令如下：

```shell
export CUDA_VISIBLE_DEVICES=0
python export_model.py \
    --model_name="skep_ernie_1.0_large_ch" \
    --ckpt_dir="./checkpoints/model_100" \
    --output_path="./static/static_graph_params"
```

可以将导出的静态图模型进行部署，deploy/python/predict.py 展示了 python 部署预测示例。运行方式如下：

```shell
export CUDA_VISIBLE_DEVICES=0
python deploy/python/predict.py \
    --model_name="skep_ernie_1.0_large_ch" \
    --model_file="./static/static_graph_params.pdmodel" \
    --params_file="./static/static_graph_params.pdiparams"
```

### 评价对象级情感分类
本节将以数据集 SE-ABSA16_PHNS 为例展示评价对象级的情感分类模型训练和测试。该数据集已内置于 PaddleNLP 中，可以通过语句级情感分类类似方式进行加载。这里不再赘述。下面展示了 SE-ABSA16_PHNS 数据集中的一条数据。

```text
label    text_a    text_b
1    phone#design_features    今天有幸拿到了港版白色iPhone 5真机，试玩了一下，说说感受吧：1. 真机尺寸宽度与4/4s保持一致没有变化，长度多了大概一厘米，也就是之前所说的多了一排的图标。2. 真机重量比上一代轻了很多，个人感觉跟i9100的重量差不多。（用惯上一代的朋友可能需要一段时间适应了）3. 由于目前还没有版的SIM卡，无法插卡使用，有购买的朋友要注意了，并非简单的剪卡就可以用，而是需要去运营商更换新一代的SIM卡。4. 屏幕显示效果确实比上一代有进步，不论是从清晰度还是不同角度的视角，iPhone 5绝对要更上一层，我想这也许是相对上一代最有意义的升级了。5. 新的数据接口更小，比上一代更好用更方便，使用的过程会有这样的体会。6. 从简单的几个操作来讲速度比4s要快，这个不用测试软件也能感受出来，比如程序的调用以及照片的拍摄和浏览。不过，目前水货市场上坑爹的价格，最好大家可以再观望一下，不要急着出手。
```

#### 模型训练

可以通过如下命令开启评价对象级级情感分类任务训练。

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" train_aspect.py \
    --model_name "skep_ernie_1.0_large_ch" \
    --save_dir "./checkpoints" \
    --epochs 50 \
    --max_seq_len 128 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --device "gpu"
```

#### 模型预测
使用如下命令进行模型预测：

```shell
export CUDA_VISIBLE_DEVICES=0
python predict_aspect.py \
    --model_name "skep_ernie_1.0_large_ch" \
    --ckpt_dir "./checkpoints/model_100" \
    --batch_size 16 \
    --max_seq_len 128 \
    --device "gpu"
```

### 观点抽取
本节将以数据集 COTE_DP 为例展示评价对象级的情感分类模型训练和测试。该数据集已内置于 PaddleNLP 中，可以通过语句级情感分类类似方式进行加载。这里不再赘述。下面展示了 COTE_DP 数据中的前3条数据。

```text
label    text_a
重庆老灶火锅    重庆老灶火锅还是很赞的，有机会可以尝试一下！
炉鱼来了    一入店内，就看到招牌特别大的炉鱼来了，餐桌上还摆了五颜六色的小蜡烛，挺有调调的。
外婆家    只能说是聚餐圣地外婆家一个需要提前来取号的地方。
```

#### 模型训练

可以通过如下命令开启评价对象级级情感分类任务训练。

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" train_opinion.py \
    --model_name "skep_ernie_1.0_large_ch" \
    --save_dir "./checkpoints" \
    --epochs 10 \
    --max_seq_len 128 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --device "gpu"
```

#### 模型预测
使用如下命令进行模型预测：

```shell
export CUDA_VISIBLE_DEVICES=0
python predict_opinion.py \
    --model_name "skep_ernie_1.0_large_ch" \
    --ckpt_dir "./checkpoints/model_100" \
    --batch_size 16 \
    --max_seq_len 128 \
    --device "gpu"
```

**备注**：
1. 评价对象级情感分类和观点抽取两类任务的模型部署方式可参考语句级情感分类，这里不再赘述。
2. 评级级情感分类以及观点抽取，暂不支持 skep 模型的 Taskflow 离线模型加载。如需使用此类功能，请参考：[unified_sentiment_analysis](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/applications/sentiment_analysis/unified_sentiment_extraction)。
