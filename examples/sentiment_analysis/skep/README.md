# SKEP: Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis

情感分析旨在自动识别和提取文本中的倾向、立场、评价、观点等主观信息。它包含各式各样的任务，比如句子级情感分类、评价对象级情感分类、观点抽取、情绪分类等。情感分析是人工智能的重要研究方向，具有很高的学术价值。同时，情感分析在消费决策、舆情分析、个性化推荐等领域均有重要的应用，具有很高的商业价值。

情感预训练模型SKEP（Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis）。SKEP利用情感知识增强预训练模型， 在14项中英情感分析典型任务上全面超越SOTA，此工作已经被ACL 2020录用。SKEP是百度研究团队提出的基于情感知识增强的情感预训练算法，此算法采用无监督方法自动挖掘情感知识，然后利用情感知识构建预训练目标，从而让机器学会理解情感语义。SKEP为各类情感分析任务提供统一且强大的情感语义表示。

论文地址：https://arxiv.org/abs/2005.05635

<p align="center">
<img src="https://paddlenlp.bj.bcebos.com/models/transformers/skep/skep.png" width="80%" height="60%"> <br />
</p>

百度研究团队在三个典型情感分析任务，句子级情感分类（Sentence-level Sentiment Classification），评价对象级情感分类（Aspect-level Sentiment Classification）、观点抽取（Opinion Role Labeling），共计14个中英文数据上进一步验证了情感预训练模型SKEP的效果。实验表明，以通用预训练模型ERNIE（内部版本）作为初始化，SKEP相比ERNIE平均提升约1.2%，并且较原SOTA平均提升约2%，具体效果如下表：



<table>
    <tr>
        <td><strong><center>任务</strong></td>
        <td><strong><center>数据集合</strong></td>
        <td><strong><center>语言</strong></td>
        <td><strong><center>指标</strong></td>
        <td><strong><center>原SOTA</strong></td>
        <td><strong><center>SKEP</strong></td>
        <td><strong><center>数据集地址</strong></td>
    </tr>
    <tr>
        <td rowspan="4"><center>句子级情感<br /><center>分类</td>
        <td><center>SST-2</td>
        <td><center>英文</td>
        <td><center>ACC</td>
        <td><center>97.50</td>
        <td><center>97.60</td>
        <td><center><a href="https://gluebenchmark.com/tasks" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>Amazon-2</td>
        <td><center>英文</td>
        <td><center>ACC</td>
        <td><center>97.37</td>
        <td><center>97.61</td>
        <td><center><a href="https://www.kaggle.com/bittlingmayer/amazonreviews/data#" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>ChnSentiCorp</td>
        <td><center>中文</td>
        <td><center>ACC</td>
        <td><center>95.80</td>
        <td><center>96.50</td>
        <td><center><a href="https://ernie.bj.bcebos.com/task_data_zh.tgz" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>NLPCC2014-SC</td>
        <td><center>中文</td>
        <td><center>ACC</td>
        <td><center>78.72</td>
        <td><center>83.53</td>
        <td><center><a href="https://github.com/qweraqq/NLPCC2014_sentiment" >下载地址</a></td>
    </tr>
    <tr>
        <td rowspan="4"><center>评价对象级的<br /><center>情感分类</td>
        <td><center>Sem-L</td>
        <td><center>英文</td>
        <td><center>ACC</td>
        <td><center>81.35</td>
        <td><center>81.62</td>
        <td><center><a href="http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>Sem-R</td>
        <td><center>英文</td>
        <td><center>ACC</td>
        <td><center>87.89</td>
        <td><center>88.36</td>
        <td><center><a href="http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>SE-ABSA16_PHNS</td>
        <td><center>中文</td>
        <td><center>ACC</td>
        <td><center>79.58</td>
        <td><center>82.91</td>
        <td><center><a href="http://alt.qcri.org/semeval2016/task5/" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>SE-ABSA16_CAME</td>
        <td><center>中文</td>
        <td><center>ACC</td>
        <td><center>87.11</td>
        <td><center>90.06</td>
        <td><center><a href="http://alt.qcri.org/semeval2016/task5/" >下载地址</a></td>
    </tr>
    <tr>
        <td rowspan="5"><center>观点<br /><center>抽取</td>
        <td><center>MPQA-H</td>
        <td><center>英文</td>
        <td><center>b-F1/p-F1</td>
        <td><center>83.67/77.12</td>
        <td><center>86.32/81.11</td>
        <td><center><a href="https://mpqa.cs.pitt.edu/corpora/mpqa_corpus/mpqa_corpus_2_0/" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>MPQA-T</td>
        <td><center>英文</td>
        <td><center>b-F1/p-F1</td>
        <td><center>81.59/73.16</td>
        <td><center>83.67/77.53</td>
        <td><center><a href="https://mpqa.cs.pitt.edu/corpora/mpqa_corpus/mpqa_corpus_2_0/" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>COTE_BD</td>
        <td><center>中文</td>
        <td><center>F1</td>
        <td><center>82.17</td>
        <td><center>84.50</td>
        <td><center><a href="https://github.com/lsvih/chinese-customer-review" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>COTE_MFW</td>
        <td><center>中文</td>
        <td><center>F1</td>
        <td><center>86.18</td>
        <td><center>87.90</td>
        <td><center><a href="https://github.com/lsvih/chinese-customer-review" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>COTE_DP</td>
        <td><center>中文</td>
        <td><center>F1</td>
        <td><center>84.33</td>
        <td><center>86.30</td>
        <td><center><a href="https://github.com/lsvih/chinese-customer-review" >下载地址</a></td>
    </tr>
</table>


## 快速开始

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
skep/
├── aspect_sentiment_analysis_predict.py # 对象级的情感分类任务预测脚本
├── aspect_sentiment_analysis_train.py # 评价对象级的情感分类任务训练脚本
├── deploy # 部署
│   └── python
│       └── predict.py # python预测部署示例
├── export_model.py # 动态图参数导出静态图参数脚本
├── keypoint_extraction_predict.py # 观点抽取任务预测脚本
├── keypoint_extraction_train.py # 观点抽取任务训练脚本
├── README.md # 使用说明
├── sentence_sentiment_analysis_predict.py # 句子级情感分类任务预测脚本
└── sentence_sentiment_analysis_train.py # 句子级情感分类任务训练脚本
```


以句子级情感分类任务为例，详细说明SKEP模型在下游任务中该如何使用，其他任务（对象级的情感分类任务、观点抽取任务）使用方式以此类推。

### 模型训练

我们以情感分类公开数据集ChnSentiCorp（中文）、SST-2（英文）为示例数据集，可以运行下面的命令，在训练集（train.tsv）上进行模型训练，并在开发集（dev.tsv）验证
```shell
$ unset CUDA_VISIBLE_DEVICES
$ python -m paddle.distributed.launch --gpus "0" sentence_sentiment_analysis_train.py --model_name "skep_ernie_1.0_large_ch" --device gpu --save_dir ./checkpoints
```

可支持配置的参数：

* `model_name`: 使用预训练模型的名称，可选skep_ernie_1.0_large_ch和skep_ernie_2.0_large_en。
    skep_ernie_1.0_large_ch：是SKEP模型在预训练ernie_1.0_large_ch基础之上在海量中文数据上继续预训练得到的中文预训练模型;
    skep_ernie_2.0_large_en：是SKEP模型在预训练ernie_2.0_large_en基础之上在海量英文数据上继续预训练得到的中文预训练模型。
* `save_dir`：可选，保存训练模型的目录；默认保存在当前目录checkpoints文件夹下。
* `max_seq_length`：可选，ERNIE/BERT模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：可选，Fine-tune的最大学习率；默认为5e-5。
* `weight_decay`：可选，控制正则项力度的参数，用于防止过拟合，默认为0.00。
* `epochs`: 训练轮次，默认为3。
* `warmup_proption`：可选，学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0.1。
* `init_from_ckpt`：可选，模型参数路径，热启动模型训练；默认为None。
* `seed`：可选，随机种子，默认为1000.
* `device`: 选用什么设备进行训练，可选cpu或gpu。如使用gpu训练则参数gpus指定GPU卡号。


```python
model = paddlenlp.transformers.SkepForSequenceClassification.from_pretrained(
    "skep_ernie_1.0_large_ch")
tokenizer = paddlenlp.transformers.SkepTokenizer.from_pretrained(
    "skep_ernie_1.0_large_ch")
```
更多预训练模型，参考[transformers](../../../docs/transformers.md)


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
python export_model.py --model_name="skep_ernie_1.0_large_ch" --params_path=./checkpoint/model_900/model_state.pdparams --output_path=./static_graph_params
```
其中`params_path`是指动态图训练保存的参数路径，`output_path`是指静态图参数导出路径。

导出模型之后，可以用于部署，deploy/python/predict.py文件提供了python部署预测示例。运行方式：

```shell
python deploy/python/predict.py --model_name="skep_ernie_1.0_large_ch" --model_file=static_graph_params.pdmodel --params_file=static_graph_params.pdiparams
```

### 模型预测

启动预测：
```shell
export CUDA_VISIBLE_DEVICES=0
python sentence_sentiment_analysis_predict.py --model_name "skep_ernie_1.0_large_ch" --device 'gpu' --params_path checkpoints/model_900/model_state.pdparams
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
