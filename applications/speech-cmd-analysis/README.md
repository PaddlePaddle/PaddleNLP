# 智能语音指令解析 (Speech Command Analysis)

## 1. 项目说明

**智能语音指令解析**集成了业界领先的语音识别（Automatic Speech Recognition, ASR）、信息抽取（Information Extraction, IE）等技术，打造智能一体化的语音指令系统，广泛应用于智能语音填单、智能语音交互、智能语音检索、手机APP语音唤醒等场景，提高人机交互效率。

其中，**智能语音填单**允许用户通过**口述**的方式记录信息，利用**算法**解析口述内容中的关键信息，完成**自动信息录入**。

#### 场景痛点

- 电话分析：边询问边记录，关键信息遗漏。例如，社区疫情防控信息记录员需要边通电话边记录关键信息，重点信息不突出，人工二次审核成本高。
- 工单生成：特定场景，无法完成文字录入。例如，电力路线巡检工作人员在高空巡检高压电线路，不便即时文字记录，滞后记录可能导致信息遗漏。
- 信息登记：重复性的工作，效率低易出错。例如，某品牌汽车售后客服话务员每天接听约300通电话，重复性工作耗时长，易出错。

针对以上场景，应用[PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)的语音识别技术和[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)的信息抽取技术，可以自动识别和抽取语音中的关键信息，帮助相关人员简化记录流程，提高工作效率和质量。
另外，通过构造小样本优化信息抽取模型，能够获得更加准确的场景定制化效果。

#### 方案选型

- **语音识别模型**
  [Conformer](https://arxiv.org/abs/2005.08100): Anmol Gulati等人在2020年提出的语音识别模型，通过将卷积应用于Transfomer的Encoder层，结合了Transformer提取长序列的优势和卷积提取局部特征的优势，使得语音识别的准确率明显提高。

- **信息抽取模型**
  [UIE](https://arxiv.org/pdf/2203.12277.pdf): Yaojie Lu等人在2022年提出了开放域信息抽取的统一框架，这一框架在实体抽取、关系抽取、事件抽取、情感分析等任务上都有着良好的泛化效果。本范例基于这篇工作的prompt设计思想，实现了以ERNIE为底座的阅读理解型信息抽取模型，用于关键数据抽取。同时，针对不同场景，支持通过构造小样本数据来优化模型效果，快速适配特定的关键信息配置。


## 2. 安装说明

#### 环境要求

- paddlepaddle >= 2.2.0
- paddlenlp >= 2.3.0

安装相关问题可参考[PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)和[PaddleNLP](https://paddlenlp.readthedocs.io/zh/latest/get_started/installation.html)文档。如需下载最新版本的PaddleNLP，可执行以下命令
```shell
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
pip install -r requirements.txt
pip install -e .
```

#### 可选依赖

- 若要使用音频文件格式转换脚本，则需安装依赖``ffmpeg``和``pydub``。

```
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg
./configure
make
make install
pip install pydub
```

## 3. 数据准备

本范例来自于语音报销工单信息录入场景，即从语音中抽取出报销需要的``时间``、``出发地``、``目的地``和``费用``字段。相应的数据集为[语音报销工单数据](https://paddlenlp.bj.bcebos.com/datasets/erniekit/speech-cmd-analysis/audio-expense-account.jsonl)，共50条标注数据，用于信息抽取模型在交通费报销场景下的优化，示例数据如下：
```json
{"id": 39, "text": "10月16日高铁从杭州到上海南站车次d5414共48元", "relations": [], "entities": [{"id": 90, "start_offset": 0, "end_offset": 6, "label": "时间"}, {"id": 77, "start_offset": 9, "end_offset": 11, "label": "出发地"}, {"id": 91, "start_offset": 12, "end_offset": 16, "label": "目的地"}, {"id": 92, "start_offset": 24, "end_offset": 26, "label": "费用"}]}
```

标注数据保存在同一个文本文件中，每条样例占一行且存储为``json``格式，其包含以下字段
- ``id``: 样本在数据集中的唯一标识ID。
- ``text``: 语音报销工单的原始文本数据。
- ``entities``: 数据中包含的实体标签，每个实体标签包含四个字段：
    - ``id``: 实体在数据集中的唯一标识ID，不同样本中的相同实体对应同一个ID。
    - ``start_offset``: 实体的起始token在文本中的下标。
    - ``end_offset``: 实体的结束token在文本中下标的下一个位置。
    - ``label``: 实体的类型。
- ``relations``: 数据中包含的关系标签（在本范例中无关系标签），每个关系标签包含四个字段：
    - ``id``: （关系主语，关系谓语，关系宾语）三元组在数据集中的唯一标识ID，不同样本中的相同三元组对应同一个ID。
    - ``from_id``: 关系主语实体对应的标识ID。
    - ``to_id``: 关系宾语实体对应的标识ID。
    - ``type``: 关系类型。

#### 音频格式转换

在语音报销工单信息录入的场景下，模型的输入为报销工单相关的音频文件。可以根据设备类型，选取合适的录音软件来录制音频文件，保存格式应为``.wav``数据格式。若音频文件格式不符，可以运行以下脚本进行转换：

- 单个文件格式转换

```
python audio_to_wav.py --audio_file sample.m4a --audio_format m4a --save_dir ./audios_wav/
```

- 指定目录下所有文件格式转换

```
python audio_to_wav.py --audio_file ./audios_raw/ --save_dir ./audios_wav/
```

可配置参数包括

- ``audio_file``: 原始音频文件或者所在目录。若设置为目录，则对该目录下所有音频文件进行格式转换。
- ``audio_format``: 原始音频文件格式（可选），支持``mp3``, ``m4a``。若未设置，则根据文件扩展名对支持的两种音频文件进行格式转换。
- ``save_dir``: 转换后``.wav``格式文件的存储目录，文件名称与原始音频保持一致。

#### 自定义数据标注

对于不同的应用场景，关键信息的配置多种多样，直接应用通用信息抽取模型的效果可能不够理想。这时可以标注少量场景相关的数据，利用few-shot learning技术来改进特定场景下的信息抽取效果。

推荐使用数据标注平台[doccano](https://github.com/doccano/doccano) 进行数据标注，具体标注步骤如下

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/164374314-9beea9ad-08ed-42bc-bbbc-9f68eb8a40ee.png />
    <p>图1 doccano数据标注样例图<p/>
</div>

- 定义实体标签类别和关系标签类别。例如针对图一中需要定义的实体标签有`作品名`、`机构名`和`人物名`，关系标签有`出版社名称`和`作者`。
- 依次点击实体标签、文本中对应字段进行实体数据标注；依次点击关系标签、主语实体下的标注圆点、宾语实体下的标注圆点对关系数据进行标注。
- 从``doccano``平台导出``jsonl``格式的数据标注文件，将文件重命名为``doccano.json``后，放在``./data/``目录下。

## 4. 模型训练

针对特定场景下的关键信息配置，需要使用标注数据对通用信息抽取模型进行训练以优化抽取效果。

#### 代码结构

```shell
.
├── speech_cmd_analysis_example.py   # 语音指令解析脚本
├── utils.py                         # 语音转文本脚本
└── uie_finetune                     # 信息抽取模型fine-tune代码目录
    ├── doccano.py                   # doccano平台导出格式到模型训练标准数据格式转换脚本
    ├── model.py                     # 信息抽取模型组网脚本
    ├── metric.py                    # 模型效果验证指标脚本
    ├── train.py                     # 模型训练脚本
    ├── evaluate.py                  # 模型评估脚本
    └── utils.py                     # 数据处理工具
```

#### 数据预处理

- 使用[doccano.py](./uie_finetune/doccano.py)将``doccano``导出的数据格式转换为模型训练需要的标准格式，用于训练。假设导出数据存储在``./data/doccano.json``，执行命令如下

```shell
python ./uie_finetune/doccano.py \
    --doccano_file ./data/doccano.json \
    --save_dir ./data/ext_data \
    --negative_ratio 5
```

可配置参数包括

- ``doccano_file``: 原始数据文件名。文件内容应与``doccano``平台导出格式一致，也即[语音报销工单数据](https://paddlenlp.bj.bcebos.com/datasets/erniekit/speech-cmd-analysis/audio-expense-account.jsonl)的格式。
- ``save_dir``: 训练数据的保存目录。默认按照``6:2:2``的比例将数据划分为训练集、验证集和测试集，分别存储在目录下的``train.txt``、``dev.txt``、``test.txt``文件。
- ``negative_ratio``: 负样本与正样本的比例。使用负样本策略可提升模型效果，负样本数量 = negative_ratio * 正样本数量。


#### 预训练模型参数

下载预训练好的[UIE模型](https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie/model_state.pdparams)，放在`./uie_model/`目录下。

#### 定制化模型训练

运行以下命令，使用单卡训练自定义的UIE模型。

```shell
CUDA_VISIBLE_DEVICES=0 python ./uie_finetune/train.py \
    --train_path ./data/ext_data/train.txt \
    --dev_path ./data/ext_data/dev.txt \
    --save_dir ./checkpoint \
    --learning_rate 1e-5 \
    --batch_size 16 \
    --max_seq_len 512 \
    --num_epochs 50 \
    --init_from_ckpt ./uie_model/model_state.pdparams \
    --seed 1000 \
    --logging_steps 10 \
    --valid_steps 100 \
    --device gpu
```

可配置参数包括

- ``train_path``: 训练集数据文件路径。
- ``dev_path``: 验证集数据文件路径。
- ``save_dir``: 保存训练模型的目录。
- ``init_from_ckpt``: 可选，模型参数路径，热启动模型训练。默认为None。
- ``learning_rate``: 模型训练的学习率的大小。
- ``batch_size``: 每次迭代每张卡上的样本数量。
- ``max_seq_len``: 最大句子长度，超过该长度将被截断。
- ``num_epochs``: 训练轮数。
- ``logging_steps``: 日志打印间隔的step数。
- ``valid_steps``: 在验证集上进行评估间隔的step数。
- ``device``: 模型训练使用的设备，可选cpu或gpu。
- ``seed``: 随机数种子，用于训练过程复现。

#### 定制化模型评估

运行以下命令，对finetune后的UIE模型效果进行评估。

```shell
CUDA_VISIBLE_DEVICES=0 python ./uie_finetune/evaluate.py \
    --model_path ./checkpoint/model_best/model_state.pdparams \
    --test_path ./data/ext_data/test.txt \
    --batch_size 16 \
    --max_seq_len 512
```

可配置参数包括

- ``model_path``: 用于评估的模型参数文件所在路径。
- ``test_path``: 测试集数据文件路径。
- ``batch_size``: 每次评估的样本数量。
- ``max_seq_len``: 最大句子长度，超过该长度将被截断。


## 5. 模型预测

在语音报销工单信息录入场景下，按照第3节中的要求准备好音频文件，执行语音指令解析脚本即可抽取报销需要的``时间``、``出发地``、``目的地``和``费用``字段。具体命令如下

```shell
python speech_cmd_analysis_example.py \
    --audio_file ./audios_wav/sample.wav \
    --uie_model ./checkpoint/model_best/model_state.pdparams \
    --schema ['时间', '出发地', '目的地', '费用']
```

可配置参数包括

- ``audio_file``: ``.wav``格式音频文件路径。
- ``uie_model``: 预测使用的模型参数文件所在路径。默认为None，即使用通用的预训练UIE模型。
- ``schema``: 关键实体信息配置。默认为语音报销工单场景下的四个关键字段。


## 6. 模型部署

在项目中提供了基于Web的部署Demo方案，支持用户在网页录入语音进行预测。用户可根据实际情况参考实现。

[部署图片] **TODO**
