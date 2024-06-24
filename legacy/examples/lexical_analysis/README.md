# 词法分析

## 1. 简介

词法分析任务的输入是一个字符串（我们后面使用『句子』来指代它），而输出是句子中的词边界和词性、实体类别。序列标注是词法分析的经典建模方式，我们使用基于 GRU 的网络结构学习特征，将学习到的特征接入 CRF 解码层完成序列标注。模型结构如下所示：<br />

![GRU-CRF-MODEL](https://bj.bcebos.com/paddlenlp/imgs/gru-crf-model.png)

1. 输入采用 one-hot 方式表示，每个字以一个 id 表示
2. one-hot 序列通过字表，转换为实向量表示的字向量序列；
3. 字向量序列作为双向 GRU 的输入，学习输入序列的特征表示，得到新的特性表示序列，我们堆叠了两层双向 GRU 以增加学习能力；
4. CRF 以 GRU 学习到的特征为输入，以标记序列为监督信号，实现序列标注。


## 快速开始

### 数据准备

我们提供了少数样本用以示例输入数据格式。执行以下命令，下载并解压示例数据集：

```bash
python download.py --data_dir ./
```

训练使用的数据可以由用户根据实际的应用场景，自己组织数据。除了第一行是 `text_a\tlabel` 固定的开头，后面的每行数据都是由两列组成，以制表符分隔，第一列是 utf-8 编码的中文文本，以 `\002` 分割，第二列是对应每个字的标注，以 `\002` 分隔。我们采用 IOB2 标注体系，即以 X-B 作为类型为 X 的词的开始，以 X-I 作为类型为 X 的词的持续，以 O 表示不关注的字（实际上，在词性、专名联合标注中，不存在 O ）。示例如下：

```text
除\002了\002他\002续\002任\002十\002二\002届\002政\002协\002委\002员\002,\002马\002化\002腾\002,\002雷\002军\002,\002李\002彦\002宏\002也\002被\002推\002选\002为\002新\002一\002届\002全\002国\002人\002大\002代\002表\002或\002全\002国\002政\002协\002委\002员    p-B\002p-I\002r-B\002v-B\002v-I\002m-B\002m-I\002m-I\002ORG-B\002ORG-I\002n-B\002n-I\002w-B\002PER-B\002PER-I\002PER-I\002w-B\002PER-B\002PER-I\002w-B\002PER-B\002PER-I\002PER-I\002d-B\002p-B\002v-B\002v-I\002v-B\002a-B\002m-B\002m-I\002ORG-B\002ORG-I\002ORG-I\002ORG-I\002n-B\002n-I\002c-B\002n-B\002n-I\002ORG-B\002ORG-I\002n-B\002n-I
```

其中词性和专名类别标签集合如下表，包含词性标签 24 个（小写字母），专名类别标签 4 个（大写字母）。这里需要说明的是，人名、地名、机构名和时间四个类别，存在（PER / LOC / ORG / TIME 和 nr / ns / nt / t）两套标签，被标注为第二套标签的词，是模型判断为低置信度的人名、地名、机构名和时间词。开发者可以基于这两套标签，在四个类别的准确、召回之间做出自己的权衡。

| 标签 | 含义     | 标签 | 含义     | 标签 | 含义     | 标签 | 含义     |
| ---- | -------- | ---- | -------- | ---- | -------- | ---- | -------- |
| n    | 普通名词 | f    | 方位名词 | s    | 处所名词 | t    | 时间     |
| nr   | 人名     | ns   | 地名     | nt   | 机构名   | nw   | 作品名   |
| nz   | 其他专名 | v    | 普通动词 | vd   | 动副词   | vn   | 名动词   |
| a    | 形容词   | ad   | 副形词   | an   | 名形词   | d    | 副词     |
| m    | 数量词   | q    | 量词     | r    | 代词     | p    | 介词     |
| c    | 连词     | u    | 助词     | xc   | 其他虚词 | w    | 标点符号 |
| PER  | 人名     | LOC  | 地名     | ORG  | 机构名   | TIME | 时间     |

### 模型训练

#### 单卡训练

启动方式如下：

```bash
python train.py \
        --data_dir ./lexical_analysis_dataset_tiny \
        --model_save_dir ./save_dir \
        --epochs 10 \
        --batch_size 32 \
        --device gpu \
        # --init_checkpoint ./save_dir/final
```

其中参数释义如下：
- `data_dir`: 数据集所在文件夹路径.
- `model_save_dir`: 训练期间模型保存路径。
- `epochs`: 模型训练迭代轮数。
- `batch_size`: 表示每次迭代**每张卡**上的样本数目。
- `device`: 训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。
- `init_checkpoint`: 模型加载路径，通过设置init_checkpoint可以启动增量训练。

#### 多卡训练

启动方式如下：

```bash
python -m paddle.distributed.launch --gpus "0,1"  train.py \
        --data_dir ./lexical_analysis_dataset_tiny \
        --model_save_dir ./save_dir \
        --epochs 10 \
        --batch_size 32 \
        --device gpu \
        # --init_checkpoint ./save_dir/final
```

### 模型评估

通过加载训练保存的模型，可以对测试集数据进行验证，启动方式如下：

```bash
python eval.py --data_dir ./lexical_analysis_dataset_tiny \
        --init_checkpoint ./save_dir/model_100.pdparams \
        --batch_size 32 \
        --device gpu
```

其中`./save_dir/model_100.pdparams`是训练过程中保存的参数文件，请更换为实际得到的训练保存路径。

### 模型导出

使用动态图训练结束之后，还可以将动态图参数导出成静态图参数，具体代码见export_model.py。静态图参数保存在`output_path`指定路径中。

运行方式：

```shell
python export_model.py --data_dir=./lexical_analysis_dataset_tiny --params_path=./save_dir/model_100.pdparams --output_path=./infer_model/static_graph_params
```

其中`./save_dir/model_100.pdparams`是训练过程中保存的参数文件，请更换为实际得到的训练保存路径。

* `params_path`是指动态图训练保存的参数路径
* `output_path`是指静态图参数导出路径。

导出模型之后，可以用于部署，deploy/predict.py文件提供了python部署预测示例。运行方式：

```shell
python deploy/predict.py --model_file=infer_model/static_graph_params.pdmodel --params_file=infer_model/static_graph_params.pdiparams --data_dir lexical_analysis_dataset_tiny
```

### 模型预测

对无标签数据可以启动模型预测：

```bash
python predict.py --data_dir ./lexical_analysis_dataset_tiny \
        --init_checkpoint ./save_dir/model_100.pdparams \
        --batch_size 32 \
        --device gpu
```

得到类似以下输出：

```txt
(大学, n)(学籍, n)(证明, n)(怎么, r)(开, v)
(电车, n)(的, u)(英文, nz)
(什么, r)(是, v)(司法, n)(鉴定人, vn)
```

### Taskflow一键预测
可以使用PaddleNLP提供的Taskflow工具来对输入的文本进行一键分词，具体使用方法如下:

```python
from paddlenlp import Taskflow

lac = Taskflow("lexical_analysis")
lac("LAC是个优秀的分词工具")
'''
[{'text': 'LAC是个优秀的分词工具', 'segs': ['LAC', '是', '个', '优秀', '的', '分词', '工具'], 'tags': ['nz', 'v', 'q', 'a', 'u', 'n', 'n']}]
'''

lac(["LAC是个优秀的分词工具", "三亚是一个美丽的城市"])
'''
[{'text': 'LAC是个优秀的分词工具', 'segs': ['LAC', '是', '个', '优秀', '的', '分词', '工具'], 'tags': ['nz', 'v', 'q', 'a', 'u', 'n', 'n']},
 {'text': '三亚是一个美丽的城市', 'segs': ['三亚', '是', '一个', '美丽', '的', '城市'], 'tags': ['LOC', 'v', 'm', 'a', 'u', 'n']}
]
'''
```

任务的默认路径为`$HOME/.paddlenlp/taskflow/lexical_analysis/lac/`，默认路径下包含了执行该任务需要的所有文件。

如果希望得到定制化的分词及标注结果，用户也可以通过Taskflow来加载自定义的词法分析模型并进行预测。

通过`task_path`指定用户自定义路径，自定义路径下的文件需要和默认路径的文件一致。

自定义路径包含如下文件（用户自己的模型权重、标签字典）：
```text
custom_task_path/
├── model.pdparams
├── word.dic
├── tag.dic
└── q2b.dic
```

使用Taskflow加载自定义模型进行一键预测：

```python
from paddlenlp import Taskflow

my_lac = Taskflow("lexical_analysis", model_path="./custom_task_path/")
```

更多使用方法请参考[Taskflow文档](../../docs/model_zoo/taskflow.md)。

## 预训练模型

如果您希望使用已经预训练好了的LAC模型完成词法分析任务，请参考：

[Lexical Analysis of Chinese](https://github.com/baidu/lac)

[PaddleHub分词模型](https://www.paddlepaddle.org.cn/hubdetail?name=lac&en_category=LexicalAnalysis)
