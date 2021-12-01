# 基于预训练模型 ERNIE-Gram 的单塔文本匹配

我们基于预训练模型 ERNIE-Gram 给出了单塔文本匹配的 2 种训练范式: Point-wise 和 Pair-wise。其中单塔 Point-wise 匹配模型适合直接对文本对进行 2 分类的应用场景: 例如判断 2 个文本是否为语义相似；Pair-wise 匹配模型适合将文本对相似度作为特征之一输入到上层排序模块进行排序的应用场景。

## 模型下载
本项目使用语义匹配数据集万方的真实场景的数据 为训练集 , 基于 ERNIE-Gram 预训练模型热启训练并开源了单塔 Point-wise 语义匹配模型， 用户可以直接基于这个模型对文本对进行语义匹配的 2 分类任务。

| 模型  | dev acc |
| ---- | ------- |
| [ERNIE-Gram-Base](https://paddlenlp.bj.bcebos.com/models/text_matching/ernie_gram_zh_pointwise_matching_model.tar)  | 98.773% |

## 快速开始

### 代码结构说明

以下是本项目主要代码结构及说明：

```
ernie_matching/
├── deply # 部署
|   └── python
|       └── predict.py # python 预测部署示例
├── export_model.py # 动态图参数导出静态图参数脚本
├── model.py # Point-wise & Pair-wise 匹配模型组网
├── data.py # Point-wise & Pair-wise 训练样本的转换逻辑 、Pair-wise 生成随机负例的逻辑
├── train_pointwise.py # Point-wise 单塔匹配模型训练脚本
├── predict_pointwise.py # Point-wise 单塔匹配模型预测脚本，输出文本对是否相似: 0、1 分类
└── train.py # 模型训练评估
```

### 模型训练

数据集使用的是万方的数据集，可以运行下面的命令，在训练集（wanfang_train.csv）上进行单塔 Point-wise 模型训练，并在测试集（wanfang_test.csv）验证。

|  训练集 | 测试集 | 
| ------------ | ------------ | 
 |  59849| 29924 |


```shell
$ unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus "0" train_pointwise.py \
        --device gpu \
        --save_dir ./checkpoints \
        --batch_size 32 \
        --learning_rate 2E-5
```
也可以直接执行下面的命令：

```
sh train.sh
```

可支持配置的参数：

* `save_dir`：可选，保存训练模型的目录；默认保存在当前目录checkpoints文件夹下。
* `max_seq_length`：可选，ERNIE-Gram 模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：可选，Fine-tune的最大学习率；默认为5e-5。
* `weight_decay`：可选，控制正则项力度的参数，用于防止过拟合，默认为0.0。
* `epochs`: 训练轮次，默认为3。
* `warmup_proption`：可选，学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0.0。
* `init_from_ckpt`：可选，模型参数路径，热启动模型训练；默认为None。
* `seed`：可选，随机种子，默认为1000.
* `device`: 选用什么设备进行训练，可选cpu或gpu。如使用gpu训练则参数gpus指定GPU卡号。

代码示例中使用的预训练模型是 ERNIE-Gram，如果想要使用其他预训练模型如 ERNIE, BERT，RoBERTa，Electra等，只需更换`model` 和 `tokenizer`即可。

```python

# 使用 ERNIE-Gram 预训练模型
model = ppnlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')
tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained('ernie-gram-zh')
```
更多预训练模型，参考[transformers](../../../docs/model_zoo/transformers.rst)

程序运行时将会自动进行训练，评估。同时训练过程中会自动保存模型在指定的`save_dir`中。
如：
```text
checkpoints/
├── model_100
│   ├── model_state.pdparams
│   ├── tokenizer_config.json
│   └── vocab.txt
└── ...
```

**NOTE:**
* 如需恢复模型训练，则可以设置`init_from_ckpt`， 如`init_from_ckpt=checkpoints/model_100/model_state.pdparams`。

### 基于动态图模型预测

我们用 万方 的测试集作为预测数据,  测试数据示例如下，：
```text
基于微流控芯片的尿路感染细菌鉴定及抗生素敏感性测试      基于微流控芯片的尿路感染细菌鉴定及抗生素敏感性测试微流控芯片,细菌鉴定,抗生素敏感性测试,快速检测     1
肺炎链球菌脑膜炎        儿童重症监护病房肺炎链球菌感染的化脓性脑膜炎临床分析儿童重症监护室,肺炎链球菌,化脓性脑膜炎,临床特点     1
小学生  学习态度        数学学科中如何培养小学生主动学习的态度学习态度,小学生,教学,培养 1
乙状结肠冗长症的诊断及手术治疗  乙状结肠冗长症的诊断及手术治疗乙状结肠冗长,诊断,治疗    1
电动叉车线控转向系统建模与控制策略研究  电动叉车线控转向系统建模与控制策略研究电动叉车;线控转向;模糊控制;计算机技术     1
基于Python的京东        一种基于Python的商品评论数据智能获取与分析技术商品评论,Python,爬虫,分词,数据分析        1
广西横县华支睾吸虫      广西横县集贸市场淡水鱼虾华支睾吸虫囊蚴感染调查淡水鱼、虾,华支睾吸虫,囊蚴,感染   1
. 从国家自然科学基金申请和评审程序 探讨如何提高申请书质量       从国家自然科学基金申请和评审程序探讨如何提高申请书质量国家自然科学基金,申请书,评审程序,撰写提纲,写作技巧   1
```

启动预测：

```shell
$ unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus "0" \
        predict_pointwise.py \
        --device gpu \
        --params_path "./checkpoints/model_4400/model_state.pdparams"\
        --batch_size 128 \
        --max_seq_length 64 \
        --input_file 'test.tsv'
```
也可以直接执行下面的命令：

```
sh predict.sh
```

输出预测结果如下:
```text
{'query': '基于微流控芯片的尿路感染细菌鉴定及抗生素敏感性测试', 'title': '基于微流控芯片的尿路感染细菌鉴定及抗生素敏感性测试微流控芯片,细菌鉴定,抗生素敏感性测试,快速检测', 'label': 1, 'pred_label': 1}
{'query': '肺炎链球菌脑膜炎', 'title': '儿童重症监护病房肺炎链球菌感染的化脓性脑膜炎临床分析儿童重症监护室,肺炎链球菌,化脓性脑膜炎,临床特点', 'label': 1, 'pred_label': 1}
{'query': '小学生  学习态度', 'title': '数学学科中如何培养小学生主动学习的态度学习态度,小学生,教学,培养', 'label': 1, 'pred_label': 1}
{'query': '乙状结肠冗长症的诊断及手术治疗', 'title': '乙状结肠冗长症的诊断及手术治疗乙状结肠冗长,诊断,治疗', 'label': 1, 'pred_label': 1}
{'query': '电动叉车线控转向系统建模与控制策略研究', 'title': '电动叉车线控转向系统建模与控制策略研究电动叉车;线控转向;模糊控制;计算机技术', 'label': 1, 'pred_label': 1}
```

### 基于静态图部署预测
#### 模型导出
使用动态图训练结束之后，可以使用静态图导出工具 `export_model.py` 将动态图参数导出成静态图参数。 执行如下命令：

`python export_model.py --params_path ernie_ckpt/model_80.pdparams --output_path=./output`

也可以直接执行命令：

```
sh export.sh
```

其中`params_path`是指动态图训练保存的参数路径，`output_path`是指静态图参数导出路径。

#### 预测部署
导出静态图模型之后，可以基于静态图模型进行预测，`deploy/python/predict.py` 文件提供了静态图预测示例。执行如下命令：

`python deploy/python/predict.py --model_dir ./output`

也可以直接执行命令：

```
sh deploy.sh
```

## Reference

[1] Xiao, Dongling, Yu-Kun Li, Han Zhang, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. “ERNIE-Gram: Pre-Training with Explicitly N-Gram Masked Language Modeling for Natural Language Understanding.” ArXiv:2010.12148 [Cs].
