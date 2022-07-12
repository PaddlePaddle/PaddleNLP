# 文本多标签分类任务指南

**目录**
   * [多标签任务介绍](#多标签任务介绍)
   * [代码结构说明](#代码结构说明)
   * [模型微调](#模型微调)
       * [从本地文件创建数据集](#从本地文件创建数据集)
   * [模型预测](#模型预测)
   * [模型静态图导出](#模型预测)
   * [模型裁剪](#模型裁剪)
       * [环境准备](#环境准备)
       * [裁剪API使用](#裁剪API使用)
   * [模型部署](#模型部署)

## 多标签任务介绍

文本多标签分类是自然语言处理（NLP）中常见的文本分类任务，文本多标签分类在各种现实场景中具有广泛的适用性，例如商品分类、网页标签、新闻标注、蛋白质功能分类、电影分类、语义场景分类等。多标签数据集中样本用来自 `n_classes` 个可能类别的 `m` 个标签类别标记，其中 `m` 的取值在 0 到 `n_classes` 之间，这些类别具有不相互排斥的属性。通常，我们将每个样本的标签用One-hot的形式表示，正类用 1 表示，负类用 0 表示。例如，数据集中样本可能标签是A、B和C的多标签分类问题，标签为 \[1,0,1\] 代表存在标签 A 和 C 而标签 B 不存在的样本。

在现实中的案情错综复杂，同一案件可能适用多项法律条文，涉及数罪并罚，需要多标签模型充分学习标签之间的关联性，对文本进行分类预测。CAIL2018—SMALL数据集中罪名预测任务数据来自“中国裁判文书网”公开的刑事法律文书，包括19.6万份文书样例，其中每份数据由法律文书中的案情描述和事实部分组成，包括每个案件被告人被判的罪名，数据集共包含202项罪名，被告人罪名通常涉及一项至多项。以数据集中某一法律文书为例：
```text
"公诉机关指控，2009年12月18日22时许，被告人李某（已判刑）伙同被告人丁某、李某乙、李某甲、杨某某在永吉县岔路河镇夜宴歌厅唱完歌后离开，因之前对该歌厅服务生刘某某心怀不满，遂手持事先准备好的镐把、扎枪再次返回夜宴歌厅，在追赶殴打刘某某过程中，任意损毁歌厅内的笔记本电脑、调音台、麦克接收机等物品。被告人丁某用镐把随意将服务员齐某某头部打伤。经物价部门鉴定，笔记本电脑、调音台、麦克接收机总价值人民币7120.00元；经法医鉴定，齐某某左额部硬膜外血肿，构成重伤。被告人丁某、李某乙、李某甲、杨某某案发后外逃，后主动到公安机关投案。并认为，被告人丁某随意殴打他人，致人重伤，其行为已构成××罪。被告人李某乙、李某甲、杨某某在公共场所持械随意殴打他人，情节恶劣，任意毁损他人财物，情节严重，其行为均已构成××罪，应予惩处。"
```
该案件中被告人涉及故意伤害，寻衅滋事两项罪名。接下来我们将讲解如何利用多标签模型，根据输入文本预测案件所涉及的一个或多个罪名。

## 代码结构说明

以下是本项目主要代码结构及说明：

```text
multi_label/
├── deploy # 部署
│   └── predictor # 导出ONNX模型并基于ONNXRuntime部署
│   │   ├── infer.py # ONNXRuntime推理部署示例
│   │   ├── predictor.py
│   │   └── README.md # 使用说明
│   ├── paddle_serving # 基于Paddle Serving 部署
│   │   ├──config.yml # 分类任务启动服务端的配置文件
│   │   ├──rpc_client.py # 分类任务发送pipeline预测请求的脚本
│   │   ├──service.py # 分类任务启动服务端的脚本
│   │   └── README.md # 使用说明
│   └── triton_serving # 基于Triton server部署
│       ├── README.md # 使用说明
│       ├── seqcls_grpc_client.py # 客户端测试代码
│       └── models # 部署模型
│           ├── seqcls
│           │   └── config.pbtxt
│           ├── seqcls_model
│           │   └──config.pbtxt
│           ├── seqcls_postprocess
│           │   ├── 1
│           │   │   └── model.py
│           │   └── config.pbtxt
│           └── tokenizer
│               ├── 1
│               │   └── model.py
│               └── config.pbtxt
├── train.py # 训练评估脚本
├── predict.py # 预测脚本
├── export_model.py # 动态图参数导出静态图参数脚本
├── utils.py # 工具函数脚本
├── metric.py # metric脚本
├── prune.py # 裁剪脚本
├── prune_trainer.py # 裁剪trainer脚本
├── prune_config.py # 裁剪训练参数配置
├── requirements.txt # 环境依赖
└── README.md # 使用说明
```

## 模型微调

请使用以下命令安装所需依赖

```shell
pip install -r requirements.txt
```

我们以公开数据集CAIL2018—SMALL中罪名预测任务为示例，在训练集上进行模型微调，并在开发集上验证。

单卡训练
```shell
python train.py --early_stop
```

指定GPU卡号/多卡训练
```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" train.py --epochs 100 --early_stop
```
使用多卡训练可以指定多个GPU卡号，例如 --gpus "0,1"

可支持配置的参数：

* `save_dir`：保存训练模型的目录；默认保存在当前目录checkpoint文件夹下。
* `dataset`：训练数据集;默认为"cail2018_small"。
* `dataset_dir`：本地数据集路径，数据集路径中应包含train.txt，dev.txt和label.txt文件;默认为None。
* `task_name`：训练数据集;默认为"charges"。
* `max_seq_length`：ERNIE模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为512。
* `model_name`：选择预训练模型；默认为"ernie-3.0-base-zh"。
* `device`: 选用什么设备进行训练，可选cpu、gpu、xpu、npu。如使用gpu训练，择使用参数gpus指定GPU卡号。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：Fine-tune的最大学习率；默认为3e-5。
* `weight_decay`：控制正则项力度的参数，用于防止过拟合，默认为0.00。
* `early_stop`：选择是否使用早停法(EarlyStopping)；默认为False。
* `early_stop_nums`：在设定的早停训练轮次内，模型在开发集上表现不再上升，训练终止；默认为6。
* `epochs`: 训练轮次，默认为1000。
* `warmup`：是否使用学习率warmup策略；默认为False。
* `warmup_steps`：学习率warmup策略的steps数，如果设为2000，则学习率会在前2000 steps数从0慢慢增长到learning_rate, 而后再缓慢衰减；默认为2000。
* `logging_steps`: 日志打印的间隔steps数，默认5。
* `seed`：随机种子，默认为3。


程序运行时将会自动进行训练，评估，测试。同时训练过程中会自动保存开发集上最佳模型在指定的 `save_dir` 中，保存模型文件结构如下所示：

```text
checkpoint/
├── model_config.json
├── model_state.pdparams
├── tokenizer_config.json
└── vocab.txt
```

**NOTE:**
* 如需恢复模型训练，则可以设置 `init_from_ckpt` ， 如 `init_from_ckpt=checkpoint/model_state.pdparams` 。
* 如需训练中文文本分类任务，只需更换预训练模型参数 `model_name` 。中文训练任务推荐使用"ernie-3.0-base-zh"，更多可选模型可参考[Transformer预训练模型](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer)。

### 从本地文件创建数据集

在许多情况，我们需要使用本地数据集来训练我们的文本分类模型，本项目支持使用固定格式本地数据集文件进行训练。如果需要对本地数据集进行数据标注，可以参考[文本分类任务doccano数据标注使用指南](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/applications/text_classification/doccano.md)进行文本分类数据标注。本项目将以CAIL2018-SMALL数据集罪名预测任务为例进行介绍如何加载本地固定格式数据集进行训练：

```shell
wget https://paddlenlp.bj.bcebos.com/datasets/cail2018_small_charges.tar.gz
tar -zxvf cail2018_small_charges.tar.gz
mv cail2018_small_charges data
```

本地数据集目录结构如下：

```text
data/
├── train.txt # 训练数据集文件
├── dev.txt # 开发数据集文件
├── test.txt # 可选，测试训练集文件
├── label.txt # 分类标签文件
└── data.txt # 可选，待预测数据文件
```

train.txt(训练数据集文件), dev.txt(开发数据集文件), test.txt(可选，测试训练集文件)中输入文本序列与标签数据用`'\t'`分隔开，标签中多个标签之间用`','`逗号分隔开。
- train.txt/dev.txt/test.txt 文件格式：
```text
<输入序列1>'\t'<标签1>','<标签2>
<输入序列2>'\t'<标签1>
...
```
- train.txt/dev.txt/test.txt 文件样例：
```text
灵璧县人民检察院指控：××事实2014年11月29日1时许，被告人彭某甲驾驶辽A×××××小型轿车行驶至辽宁省沈阳市于洪区太湖街红色根据地酒店门口路段时，与前方被害人闻某驾驶的辽A×××××号轿车发生追尾的交通事故。事故发生后，被告人彭某甲与乘坐人韩某下车与闻某发生口角争执，并在一起互相厮打。在厮打过程中，彭某甲与韩某用拳头将闻某面部打伤。经鉴定，闻某的损伤程度为轻伤二级。××事实2015年6月至2015年9月，被告人彭某甲通过其建立的“比特战斗”微信群，将47部淫秽视频文件上传至该微信群供群成员观看。公诉机关针对指控提供了相关书证，证人证言，被害人陈述，被告人供述，鉴定意见，现场勘验检查笔录等证据，公诉机关认为，被告人彭某甲伙同他人故意非法损害公民身体健康，致人轻伤；利用移动通讯终端传播淫秽电子信息，情节严重，其行为已触犯《中华人民共和国刑法》××××、××××、××××之规定，构成××罪、××罪，提请法院依法判处。    故意伤害,[制造、贩卖、传播]淫秽物品,传播淫秽物品
酉阳县人民检察院指控，2014年1月17日1时许，被告人周某某在酉阳县桃花源美食街万州烤鱼店外与田某甲发生口角，随后周某某持刀将在店内的被害人田某某砍伤。经重庆市酉阳县公安局物证鉴定室鉴定，田某某所受伤为轻伤二级。指控的证据有被告人立案决定书，户籍信息，鉴定意见，辨认笔录，被害人田某某的陈述，证人冉某、陈某某等人的证言，周某某的供述与辩解等。公诉机关认为，被告人周某某××他人身体，致人轻伤，其行为触犯了《中华人民共和国刑法》第二百三十四××的规定，犯罪事实清楚，证据确实、充分，应当以××罪追究其刑事责任。周某某在××考验期内发现有其他罪没有判决的，适用《中华人民共和国刑法》××、六十九条。提请依法判决。    故意伤害,[组织、强迫、引诱、容留、介绍]卖淫,[引诱、容留、介绍]卖淫
...
```
label.txt(分类标签文件)记录数据集中所有标签集合，每一行为一个标签名。
- label.txt 文件格式：

```text
<标签名1>
<标签名2>
...
```
- label.txt 文件样例：
```text
故意伤害
盗窃
危险驾驶
非法[持有、私藏][枪支、弹药]
...
```

data.txt(可选，待预测数据文件)

- data.txt 文件格式：

```text
<输入序列1>
<输入序列2>
...
```
- data.txt 文件样例：
```text
经审理查明，2012年4月5日19时许，被告人王某在杭州市下城区朝晖路农贸市场门口贩卖盗版光碟、淫秽光碟时被民警当场抓获，并当场查获其贩卖的各类光碟5515张，其中5280张某属非法出版物、235张某属淫秽物品。上述事实，被告人王某在庭审中亦无异议，且有经庭审举证、质证的扣押物品清单、赃物照片、公安行政处罚决定书、抓获经过及户籍证明等书证；证人胡某、徐某的证言；出版物鉴定书、淫秽物品审查鉴定书及检查笔录等证据证实，足以认定。
榆林市榆阳区人民检察院指控：2015年11月22日2时许，被告人王某某在自己经营的榆阳区长城福源招待所内，介绍并容留杨某向刘某某、白某向乔某某提供性服务各一次
...
```
在训练过程中通过指定数据集路径参数 `dataset_dir` 进行：
单卡训练
```shell
python train.py --early_stop --dataset_dir data
```

指定GPU卡号/多卡训练
```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" train.py --early_stop --dataset_dir data
```
使用多卡训练可以指定多个GPU卡号，例如 --gpus "0,1"

更多数据集读取格式详见[数据集加载](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_load.html#)和[自定义数据集](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html)。


## 模型预测

输入待预测数据和数据标签对照列表，模型预测数据对应的标签

使用默认数据进行预测：
```shell
python predict.py --params_path ./checkpoint/
```
也可以选择使用本地数据文件data/data.txt进行预测：
```shell
python predict.py --params_path ./checkpoint/ --dataset_dir data
```

可支持配置的参数：

* `params_path`：必须，待预测模型和分词器参数文件夹；默认为"./checkpoint/"。
* `dataset_dir`：本地数据集路径，数据集路径中应包含data.txt和label.txt文件;默认为None。
* `max_seq_length`：ERNIE模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为512。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `device`: 选用什么设备进行训练，可选cpu、gpu、xpu、npu；默认为gpu。

## 模型静态图导出

使用动态图训练结束之后，还可以将动态图参数导出成静态图参数，具体代码见[静态图导出脚本](export_model.py)。静态图参数保存在`output_path`指定路径中。运行方式：

```shell
python export_model.py --params_path ./checkpoint/ --output_path ./export
```
可支持配置的参数：

* `params_path`：动态图训练保存的参数路径；默认为"./checkpoint/"。
* `output_path`：静态图图保存的参数路径；默认为"./export"。

程序运行时将会自动导出模型到指定的 `output_path` 中，保存模型文件结构如下所示：

```text
export/
├── float32.pdiparams
├── float32.pdiparams.info
└── float32.pdmodel
```


导出模型之后，可以用于部署，项目提供了[onnxruntime部署预测示例](./deploy/predictor/infer.py),用法详见[ONNX Runtime推理部署](./deploy/predictor/README.md)。

使用内置数据集进行部署：
```shell
python deploy/predictor/infer.py --model_path_prefix ./export/float32
```
也可以选择使用本地数据文件data/data.txt进行部署：
```shell
python deploy/predictor/infer.py --model_path_prefix ./export/float32 --dataset_dir data
```

此外，本项目还提供了基于[Paddle Serving](./deploy/paddle_serving)的服务化部署，用法详见[基于Paddle Serving的服务化部署](./deploy/predictor/README.md)。

## 模型裁剪
### 环境准备

使用裁剪功能需要安装 paddleslim 包

```shell
pip install paddleslim==2.2.2
```

### 裁剪 API 使用
本项目基于 PaddleNLP 的 Trainer API 发布提供了模型裁剪 API。裁剪 API 支持用户对 ERNIE 等Transformers 类下游任务微调模型进行裁剪，用户只需要简单地调用 `prune()` 即可一键启动裁剪和并自动保存裁剪后的模型。

可以这样使用裁剪 API (示例代码只提供了核心调用，如需跑通完整的例子可参考[完整样例脚本](prune.py)):

```python
trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        criterion=criterion)

output_dir = os.path.join(training_args.output_dir, data_args.dataset)

trainer.prune(output_dir, prune_config=DynabertConfig(width_mult=2/3))
```
由于裁剪 API 基于 Trainer，所以首先需要初始化一个 Trainer 实例，对于模型裁剪来说必要传入的参数如下：

* `model`：ERNIE等模型在下游任务中微调后的模型，通过`AutoModelForSequenceClassification.from_pretrained(model_args.params_dir)` 来获取
* `data_collator`：使用 PaddleNLP 预定义好的[DataCollator 类](../../../paddlenlp/data/data_collator.py)，`data_collator` 可对数据进行 `Pad` 等操作,使用方法参考本项目中代码即可
* `train_dataset`：裁剪训练需要使用的训练集
* `eval_dataset`：裁剪训练使用的评估集
* `tokenizer`：模型`model`对应的 `tokenizer`，可使用 `AutoTokenizer.from_pretrained(model_args.params_dir)` 来获取
* `criterion`： 定义criterion计算损失，分类中使用损失函数 paddle.nn.BCEWithLogitsLoss()

然后可以直接调用 `prune` 启动裁剪，其中 `prune` 的参数释义如下：
* `output_dir`：裁剪后模型保存目录
* `prune_config`：裁剪配置，目前裁剪配置仅支持`DynabertConfig`类。

当默认参数不满足需求时，可通过传入参数对裁剪过程进行特殊配置，`DynabertConfig`中可以传的参数有：
* `width_mult_list`：裁剪宽度保留的比例，表示对 `q`、`k`、`v` 以及 `ffn` 权重宽度的保留比例，默认是 `2/3`
* `output_filename_prefix`：裁剪导出模型的文件名前缀，默认是`"float32"`


选择使用默认数据集启动裁剪：
```shell
python prune.py --output_dir ./prune --params_dir ./checkpoint/
```
也可以选择使用本地数据文件启动裁剪：
```shell
python prune.py --output_dir ./prune --params_dir ./checkpoint/ --dataset_dir data
```

可支持配置的参数：
* `TrainingArguments`
  * `output_dir`：必须，保存模型输出和和中间checkpoint的输出目录;默认为 `None` 。
  * `TrainingArguments` 包含了用户需要的大部分训练参数，所有可配置的参数详见[TrainingArguments 参数介绍](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/trainer.md#trainingarguments-%E5%8F%82%E6%95%B0%E4%BB%8B%E7%BB%8D)，示例通过`prune_config.json`对TrainingArguments 参数进行配置

* `DataArguments`
  * `dataset`：训练数据集;默认为cail2018_small数据集。
  * `task_name`：训练数据集任务名;默认为罪名预测任务"charges"。
  * `dataset_dir`：本地数据集路径，需包含train.txt,dev.txt,label.txt;默认为None。
  * `max_seq_length`：ERNIE模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为512。

* `ModelArguments`
  * `params_dir`：待预测模型参数文件夹；默认为"./checkpoint/"。


以上参数都可通过 `python prune.py --dataset xx --params_dir xx` 的方式传入）

程序运行时将会自动进行训练，评估，测试。同时训练过程中会自动保存开发集上最佳模型在指定的 `output_dir` 中，保存模型文件结构如下所示：

```text
prune/
├── 0.6666666666666666
│   ├── float32.pdiparams
│   ├── float32.pdiparams.info
│   ├── float32.pdmodel
│   ├── model_state.pdparams
│   └── model_config.json
└── ...
```

**NOTE:**

1. 目前支持的裁剪策略需要训练，训练时间视下游任务数据量而定，且和微调的训练时间是一个量级；

2. 裁剪类似蒸馏过程，方便起见，可以直接使用微调时的超参。为了进一步提升精度，可以对 `per_device_train_batch_size`、`learning_rate`、`num_train_epochs`、`max_seq_length` 等超参进行 grid search；

3. 模型裁剪主要用于推理部署，因此裁剪后的模型都是静态图模型，只可用于推理部署，不能再通过 `from_pretrained` 导入继续训练。

4. 本项目提供了[onnxruntime部署预测示例](./deploy/predictor/infer.py)，用法详见[ONNX Runtime推理部署](./deploy/predictor/README.md)。

使用内置数据集进行部署：
```shell
python deploy/predictor/infer.py --model_path_prefix ./prune/0.6666666666666666/float32
```
也可以选择使用本地数据文件 data/data.txt 进行部署：
```shell
python deploy/predictor/infer.py --model_path_prefix ./prune/0.6666666666666666/float32 --dataset_dir data
```
5. 本项目提供了基于[Paddle Serving](./deploy/paddle_serving)的服务化部署，用法详见[基于Paddle Serving的服务化部署](./deploy/predictor/README.md)。

## 模型部署


- 服务化部署请参考：[基于Paddle Serving的服务化部署指南](deploy/paddle_serving/README.md)，Paddle Serving支持X86、Arm CPU、NVIDIA GPU、昆仑/昇腾等多种硬件的服务化部署

- ONNXRuntime 部署请参考：[ONNX导出及ONNXRuntime部署指南](deploy/predictor/README.md)

- 基于ONNXRuntime的服务化部署请参考：[基于Triton Inference Server的服务化部署指南](deploy/triton_serving/README.md)
