# 词法分析

## 1. 简介

词法分析任务的输入是一个字符串（我们后面使用『句子』来指代它），而输出是句子中的词边界和词性、实体类别。序列标注是词法分析的经典建模方式，我们使用基于 GRU 的网络结构学习特征，将学习到的特征接入 CRF 解码层完成序列标注。模型结构如下所示：<br />

![GRU-CRF-MODEL](https://paddlenlp.bj.bcebos.com/imgs/gru-crf-model.png)

1. 输入采用 one-hot 方式表示，每个字以一个 id 表示
2. one-hot 序列通过字表，转换为实向量表示的字向量序列；
3. 字向量序列作为双向 GRU 的输入，学习输入序列的特征表示，得到新的特性表示序列，我们堆叠了两层双向 GRU 以增加学习能力；
4. CRF 以 GRU 学习到的特征为输入，以标记序列为监督信号，实现序列标注。


## 2. 快速开始

### 2.1 环境配置

- Python >= 3.6

- paddlepaddle >= 2.0.0，安装方式请参考 [快速安装](https://www.paddlepaddle.org.cn/install/quick)。

- paddlenlp >= 2.0.0rc, 安装方式：`pip install paddlenlp\>=2.0.0rc`

### 2.2 数据准备

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

### 2.3 模型训练

模型训练支持 CPU 和 GPU，使用 GPU 之前应指定使用的显卡卡号：

```bash
export CUDA_VISIBLE_DEVICES=0 # 支持多卡训练，如使用双卡，可以设置为0,1
```

训练启动方式如下：

```bash
python train.py \
        --data_dir ./lexical_analysis_dataset_tiny \
        --model_save_dir ./save_dir \
        --epochs 10 \
        --batch_size 32 \
        --n_gpu 1 \
        # --init_checkpoint ./save_dir/final
```

其中 data_dir 是数据集所在文件夹路径，init_checkpoint 是模型加载路径，通过设置init_checkpoint可以启动增量训练。

### 2.4 模型评估

通过加载训练保存的模型，可以对测试集数据进行验证，启动方式如下：

```bash
python eval.py --data_dir ./lexical_analysis_dataset_tiny \
        --init_checkpoint ./save_dir/final \
        --batch_size 32 \
        --use_gpu
```

### 2.5 模型预测

对无标签数据可以启动模型预测：

```bash
python predict.py --data_dir ./lexical_analysis_dataset_tiny \
        --init_checkpoint ./save_dir/final \
        --batch_size 32 \
        --use_gpu
```

得到类似以下输出：

```txt
(大学, n)(学籍, n)(证明, n)(怎么, r)(开, v)
(电车, n)(的, u)(英文, nz)
(什么, r)(是, v)(司法, n)(鉴定人, vn)
```


## 预训练模型

如果您希望使用已经预训练好了的LAC模型完成词法分析任务，请参考：

[Lexical Analysis of Chinese](https://github.com/baidu/lac)

[PaddleHub分词模型](https://www.paddlepaddle.org.cn/hubdetail?name=lac&en_category=LexicalAnalysis)
