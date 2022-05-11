# 智能语音指令解析 (Speech Command Analysis)

## 1. 项目说明

**智能语音指令解析**集成了业界领先的语音识别（Automatic Speech Recognition, ASR）、信息抽取（Information Extraction, IE）等技术，打造智能一体化的语音指令系统，广泛应用于智能语音填单、智能语音交互、智能语音检索、手机APP语音唤醒等场景，提高人机交互效率。

其中，**智能语音填单**允许用户通过**口述**的方式记录信息，利用**算法**解析口述内容中的关键信息，完成**自动信息录入**。

#### 场景痛点

- 电话分析：边询问边记录，关键信息遗漏。例如，社区疫情防控信息记录员需要边通电话边记录关键信息，重点信息不突出，人工二次审核成本高。
- 工单生成：特定场景，无法完成文字录入。例如，电力路线巡检工作人员在高空巡检高压电线路，不便即时文字记录，滞后记录可能导致信息遗漏。
- 信息登记：重复性的工作，效率低易出错。例如，某品牌汽车售后客服话务员每天接听约300通电话，重复性工作耗时长，易出错。

针对以上场景，应用Baidu大脑AI开放平台[短语音识别标准版](https://ai.baidu.com/tech/speech/asr)和[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)的信息抽取技术，可以自动识别和抽取语音中的关键信息，帮助相关人员简化记录流程，提高工作效率和质量。
另外，通过构造小样本优化信息抽取模型，能够获得更加准确的场景定制化效果。

#### 方案选型

- **语音识别模型**
  Baidu大脑AI开放平台[短语音识别标准版](https://ai.baidu.com/tech/speech/asr)采用领先国际的流式端到端语音语言一体化建模方法，融合百度自然语言处理技术，近场中文普通话识别准确率达98%。根据语音内容理解可以将数字序列、小数、时间、分数、基础运算符正确转换为数字格式，使得识别的数字结果更符合使用习惯，直观自然。

- **信息抽取模型**
  [Universal Information Extraction, UIE](https://arxiv.org/pdf/2203.12277.pdf): Yaojie Lu等人在2022年提出了开放域信息抽取的统一框架，这一框架在实体抽取、关系抽取、事件抽取、情感分析等任务上都有着良好的泛化效果。本应用基于这篇工作的prompt设计思想，提供了以ERNIE为底座的阅读理解型信息抽取模型，用于关键信息抽取。同时，针对不同场景，支持通过构造小样本数据来优化模型效果，快速适配特定的关键信息配置。


## 2. 安装说明

#### 环境要求

- paddlepaddle >= 2.2.0
- paddlenlp >= 2.3.0

安装相关问题可参考[PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)和[PaddleNLP](https://paddlenlp.readthedocs.io/zh/latest/get_started/installation.html)文档。

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

本应用来自于语音报销工单信息录入场景，即员工向公司报销部门提出交通费报销的口头申请，在传统场景下，报销审核人员需要人工将语音转换为文字信息，并从中抽取记录报销需要的``时间``、``出发地``、``目的地``和``费用``字段，而在本应用可以端到端的完成这一工作。相应的数据集为[语音报销工单数据](https://paddlenlp.bj.bcebos.com/datasets/erniekit/speech-cmd-analysis/audio-expense-account.jsonl)，共50条标注数据，用于信息抽取模型在交通费报销场景下的优化，示例数据如下：

```json
{"id": 39, "text": "10月16日高铁从杭州到上海南站车次d5414共48元", "relations": [], "entities": [{"id": 90, "start_offset": 0, "end_offset": 6, "label": "时间"}, {"id": 77, "start_offset": 9, "end_offset": 11, "label": "出发地"}, {"id": 91, "start_offset": 12, "end_offset": 16, "label": "目的地"}, {"id": 92, "start_offset": 24, "end_offset": 26, "label": "费用"}]}
```

其中抽取的目标（schema）表示为：

```python
schema = ['出发地', '目的地', '费用', '时间']
```

标注数据保存在同一个文本文件中，每条样例占一行且存储为``json``格式，其包含以下字段
- ``id``: 样本在数据集中的唯一标识ID。
- ``text``: 语音报销工单的原始文本数据。
- ``entities``: 数据中包含的实体标签，每个实体标签包含四个字段：
    - ``id``: 实体在数据集中的唯一标识ID，不同样本中的相同实体对应同一个ID。
    - ``start_offset``: 实体的起始token在文本中的下标。
    - ``end_offset``: 实体的结束token在文本中下标的下一个位置。
    - ``label``: 实体类型。
- ``relations``: 数据中包含的关系标签（在语音报销工单应用中无关系标签），每个关系标签包含四个字段：
    - ``id``: （关系主语，关系谓语，关系宾语）三元组在数据集中的唯一标识ID，不同样本中的相同三元组对应同一个ID。
    - ``from_id``: 关系主语实体对应的标识ID。
    - ``to_id``: 关系宾语实体对应的标识ID。
    - ``type``: 关系类型。

#### BaiduAI开放平台申请使用

- 注册账号。在[百度智能云](https://console.bce.baidu.com)注册账号并登陆。
- 资源申请。平台提供了免费资源用于功能测试，打开[语音识别控制台](https://console.bce.baidu.com/ai/?fromai=1#/ai/speech/overview/index)，点击``领取免费资源``，勾选短语音识别后点击下方``0元领取``。
- 创建应用。打开语音识别控制台，点击[创建应用](https://console.bce.baidu.com/ai/?fromai=1#/ai/speech/app/create)，填写必选项后点击``立即创建``。
- 获取API Key和Secret Key。打开语音识别控制台，点击[管理应用](https://console.bce.baidu.com/ai/?fromai=1#/ai/speech/app/list)即可查看应用对应的API Key和Secret Key。在运行本应用脚本时，设置这两个参数即可调用该平台的语音识别服务。

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

对于不同的应用场景，关键信息的配置多种多样，直接应用通用信息抽取模型的效果可能不够理想。这时可以标注少量场景相关的数据，利用few-shot learning技术来改进特定场景下的信息抽取效果。在本应用场景中，标注数据为[语音报销工单数据](https://paddlenlp.bj.bcebos.com/datasets/erniekit/speech-cmd-analysis/audio-expense-account.jsonl)。针对其他场景，可使用[doccano](https://github.com/doccano/doccano)平台标注并导出自定义数据。


## 4. 模型训练

针对特定场景下的关键信息配置，需要使用标注数据对通用信息抽取模型进行训练以优化抽取效果。

#### 代码结构

```shell
.
├── audio_to_wav.py           # 音频文件格式转换脚本
├── pipeline.py               # 语音指令解析脚本
├── preprocess.py             # 数据预处理脚本
├── finetune.py               # 信息抽取模型 fine-tune 脚本
├── model.py                  # 信息抽取模型（UIE）组网脚本
└── utils.py                  # 辅助函数
```

#### 数据预处理

下载[语音报销工单数据](https://paddlenlp.bj.bcebos.com/datasets/erniekit/speech-cmd-analysis/audio-expense-account.jsonl)，存储在``./data/``目录下。执行以下脚本，按设置的比例划分数据集，同时构造负样本用于提升模型的学习效果。

```shell
python preprocess.py \
    --input_file ./data/audio-expense-account.jsonl \
    --save_dir ./data/ \
    --negative_ratio 5 \
    --splits 0.2 0.8 0.0 \
    --seed 1000
```

可配置参数包括

- ``input_file``: 标注数据文件名。数据格式应与[语音报销工单数据](https://paddlenlp.bj.bcebos.com/datasets/erniekit/speech-cmd-analysis/audio-expense-account.jsonl)一致。
- ``save_dir``: 训练数据的保存目录，默认存储在``data``目录下。若``splits``为空，则数据存储在``train.txt``文件，若``splits``为长度为3的列表，则数据存储在目录下的``train.txt``、``dev.txt``、``test.txt``文件。
- ``negative_ratio``: 负样本与正样本的比例。使用负样本策略可提升模型效果，负样本数量 = negative_ratio * 正样本数量。
- ``splits``: 划分数据集时训练集、验证集所占的比例。默认为[0.8, 0.1, 0.1]表示按照``8:1:1``的比例将数据划分为训练集、验证集和测试集。
- ``is_shuffle``: 是否对数据集进行随机打散，默认为True。
- ``seed``: 随机种子，默认为1000.


#### 定制化模型训练

运行以下命令，使用单卡训练自定义的UIE模型。

```shell
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --train_path ./data/train.txt \
    --dev_path ./data/dev.txt \
    --save_dir ./checkpoint \
    --model uie-base \
    --learning_rate 1e-5 \
    --batch_size 16 \
    --max_seq_len 512 \
    --num_epochs 50 \
    --seed 1000 \
    --logging_steps 10 \
    --valid_steps 10 \
    --device gpu
```

可配置参数包括

- `train_path`: 训练集文件路径。
- `dev_path`: 验证集文件路径。
- `save_dir`: 模型存储路径，默认为`./checkpoint`。
- ``init_from_ckpt``: 可选，模型参数路径，热启动模型训练。默认为None。
- `learning_rate`: 学习率，默认为1e-5。
- `batch_size`: 批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数，默认为16。
- `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
- `num_epochs`: 训练轮数，默认为100。
- `model`: 选择模型，程序会基于选择的模型进行模型微调，可选有`uie-base`和`uie-tiny`。
- `seed`: 随机种子，默认为1000.
- `logging_steps`: 日志打印的间隔steps数，默认为10。
- `valid_steps`: evaluate的间隔steps数，默认为100。
- `device`: 模型训练使用的设备，可选cpu或gpu。


## 5. 模型预测

预测时使用的schema应与finetune阶段训练数据的schema保持一致以得到更好的效果。在语音报销工单信息录入场景下，
- 首先准备好``.wav``格式的音频文件，例如下载[sample.wav](https://bj.bcebos.com/paddlenlp/applications/speech-cmd-analysis/sample.wav)放在``./audios_wav/``目录下。
- 然后在BaiduAI开放平台创建语音识别应用以获取API Key和Secret Key。
- 最后加载用场景数据finetune后的模型参数，执行语音指令解析脚本即可抽取报销需要的``时间``、``出发地``、``目的地``和``费用``字段。具体命令如下

```shell
python pipeline.py \
    --api_key '4E1BG9lTnlSeIf1NQFlrxxxx' \
    --secret_key '544ca4657ba8002e3dea3ac2f5fxxxxx' \
    --audio_file ./audios_wav/sample.wav \
    --uie_model ./checkpoint/model_best/ \
    --schema '时间' '出发地' '目的地' '费用'
```

可配置参数包括

- ``api_key``: BaiduAI开放平台上创建应用的API Key。
- ``secret_key``: BaiduAI开放平台上创建应用的Secret Key。
- ``audio_file``: ``.wav``格式音频文件路径。
- ``uie_model``: 预测使用的模型参数文件所在路径。默认为None，即使用通用的预训练UIE模型。
- ``schema``: 关键实体信息配置。默认为语音报销工单场景下的四个关键字段。


## 6. 模型部署

在应用中提供了基于Web的部署Demo方案，支持用户在网页录入语音进行预测。用户可根据实际情况参考实现。

![demo](https://user-images.githubusercontent.com/25607475/165510522-a7f5f131-cd3f-4855-8932-6d8b6a7bb913.png)
