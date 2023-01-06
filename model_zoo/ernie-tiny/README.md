# 端上语义理解压缩方案

 **目录**
   * [ERNIE 3.0 Tiny v2 介绍](#模型介绍)
   * [预训练模型效果](#模型效果)
   * [代码结构](#代码结构)
   * [开始运行](#开始运行)
       * [任务介绍](#任务介绍)
       * [环境要求](#环境要求)
       * [数据准备](#数据准备)
   * [模型训练](#模型训练)
   * [模型评估](#模型评估)
   * [端上模型压缩方案🔥](#模型压缩)
       * [压缩效果](#压缩效果)
   * [FastDeploy 部署🔥](#FastDeploy部署)
   * [参考文献](#参考文献)


本项目开源了 **ERNIE 3.0 Tiny v2** 预训练模型及 **端上语义理解压缩方案**。

- **ERNIE 3.0 Tiny v2** 预训练模型在 in-domain、out-domain、low-resourced 的下游任务上比 ERNIE 3.0 Tiny v1 平均提升了X。并且 v2 版本新增开源了 3L128H 结构的模型。

- **端上语义理解压缩方案** 以车载语音任务型对话为背景，使用了模型裁剪、量化策略，将模型大小减小了X倍，推理时延降低X倍，内存占用减小X倍，达到XKB。使 ERNIE 类模型能轻松地部署至移动端。由于移动端部署对内存占用的要求比起服务端更高，因此该方案也同样适用于服务端部署。

<a name="模型介绍"></a>

## ERNIE 3.0 Tiny v2 介绍
由于预训练语言模型越来越大，过大的参数量导致了模型难以部署，因此 [ERNIE 3.0 Tiny v1](../ernie-3.0/)（即 ERNIE 3.0 轻量级模型）通过 task-agnostic 知识蒸馏的方式将大模型压缩成开箱即用的小模型，小模型在下游任务上直接微调就能取得不错的效果。然而，蒸馏出来的小模型和教师模型在效果上仍然存在差距，对此我们提出并开源了 **ERNIE 3.0 Tiny v2** 。ERNIE 3.0 Tiny v2 使教师模型参与 **多任务训练**，大大提高了小模型在下游任务上的效果。

### 注入下游知识
ERNIE-Tiny v1 通过 task-agnostic 蒸馏技术将预训练大模型压缩成预训练小模型，然而由于小模型在微调之前没有接触到下游任务的相关知识，导致效果和大模型仍然存在差距。因此我们提出 **ERNIE 3.0 Tiny v2**，通过微调教师模型，让教师模型学习到下游任务的相关知识，进而能够在蒸馏的过程中传导给学生模型。尽管学生模型完全没有见过下游数据，通过预先注入下游知识到教师模型，蒸馏得到的学生模型也能够获取到下游任务的相关知识，进而使下游任务上的效果得到提升。

### 多任务学习提升泛化性
多任务学习已经被证明对增强模型泛化性有显著的效果，例如 MT-DNN、MUPPET、FLAN 等。通过对教师模型加入多下游任务微调，不但能够对教师模型注入下游知识、提高教师模型的泛化性，并且能够通过蒸馏传给学生模型，大幅度提升小模型的泛化性。具体地，我们对教师模型进行了 28 个任务的多任务微调。

ERNIE 3.0 Tiny v2 比起 ERNIE 3.0 Tiny v1 在 in-domain、out-domain、low-resourced 数据上获得显著的提升。

<p align="center">
        <img width="644" alt="image" src="https://user-images.githubusercontent.com/26483581/210303124-c9df89a9-e291-4322-a6a5-37d2c4c1c008.png" title="ERNIE 3.0 Tiny v2">
</p>

<p align="center"> ERNIE 3.0 Tiny v2 训练流程图</p>
<br>

<a name="模型效果"></a>

## 预训练模型效果

本项目开源 **ERNIE 3.0 Tiny _Base_ v2** 、**ERNIE 3.0 Tiny _Medium_ v2** 、 **ERNIE 3.0 Tiny _Mini_ v2** 、 **ERNIE 3.0 Tiny _Micro_ v2** 、 **ERNIE 3.0 Tiny _Nano_ v2**、**ERNIE 3.0 Tiny _Pico_ v2** 六个中文模型：

- [**ERNIE 3.0-Tiny-_Base_-v2**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_tiny_base_v2.pdparams) (_12-layer, 768-hidden, 12-heads_)
- [**ERNIE 3.0-Tiny-_Medium_-v2**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_tiny_medium_v2.pdparams) (_6-layer, 768-hidden, 12-heads_)
- [**ERNIE 3.0-Tiny-_Mini_-v2**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_tiny_mini_v2.pdparams) (_6-layer, 384-hidden, 12-heads_)
- [**ERNIE 3.0-Tiny-_Micro_-v2**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_tiny_micro_v2.pdparams) (_4-layer, 384-hidden, 12-heads_)
- [**ERNIE 3.0-Tiny-_Nano_-v2**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_tiny_nano_v2.pdparams) (_4-layer, 312-hidden, 12-heads_)
- [**ERNIE 3.0-Tiny-_Pico_-v2**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_tiny_pico_v2.pdparams) (_4-layer, 312-hidden, 2-heads_)

ERNIE 3.0 Tiny 模型可以用于文本分类、文本推理、实体抽取、问答等各种 NLU 任务中。下表是 ERNIE 3.0 Tiny 模型在 in-domain、out-domain 和 low-resourced 三类数据集上的效果。其中 CLUE 指标可以通过 [PaddleNLP CLUE Benchmark](../../../examples/benchmark/clue) 复现。

<table>
    <tr>
        <td>Arch</td>
        <td>Model</td>
        <td colspan=11 align=center> In-domain </td>
        <td colspan=3 align=center> Out-domain </td>
        <td colspan=3 align=center> Low-resourced</td>
    </tr>
    <tr>
        <td>-</td>
        <td>-</td>
        <td>avg.</td>
        <td>afqmc</td>
        <td>tnews</td>
        <td>iflytek</td>
        <td>cmnli</td>
        <td>ocnli</td>
        <td>cluewssc2020</td>
        <td>csl</td>
        <td>cmrc2018</td>
        <td>chid</td>
        <td>c3</td>
        <td>avg.</td>
        <td>CANLI</td>
        <td>shopping_10</td>
        <td>avg.</td>
        <td>bustm_few</td>
        <td>eprtmt_few</td>
        <td>csldcp_few</td>
    </tr>
    <tr>
        <td rowspan=2 align=center>12L768H</td>
        <td>ERNIE 3.0 Tiny-Base v1</td>
        <td>76.05</td>
        <td>75.93</td>
        <td>58.26</td>
        <td>61.56</td>
        <td>83.02</td>
        <td>80.10</td>
        <td>86.18</td>
        <td>82.63</td>
        <td>70.71/90.41</td>
        <td>84.26</td>
        <td>77.88</td>
        <td>97.29</td>
        <td>99.31</td>
        <td>95.26</td>
        <td>75.81</td>
        <td>76.09</td>
        <td>89.06</td>
        <td>62.29</td>
    </tr>
    <tr>
        <td><b>ERNIE 3.0 Tiny-Base v2</b></td>
        <td>76.31</td>
        <td>77.43</td>
        <td>59.11</td>
        <td>61.49</td>
        <td>84.56</td>
        <td>81.86</td>
        <td>82.57</td>
        <td>82.50</td>
        <td>68.87/89.96</td>
        <td>83.55</td>
        <td><b>81.16</b></td>
        <td>97.30</td>
        <td>99.22</td>
        <td>95.38</td>
        <td><b>79.00</b></td>
        <td><b>82.50</b></td>
        <td>89.84</td>
        <td>64.65</td>
    </tr>
    <tr>
        <td rowspan=2 align=center>6L768H</td>
        <td>ERNIE 3.0 Tiny-Medium v1</td>
        <td>72.49</td>
        <td>73.37</td>
        <td>57.00</td>
        <td>60.67</td>
        <td>80.64</td>
        <td>76.88</td>
        <td>79.28</td>
        <td>81.60</td>
        <td>65.83/87.30</td>
        <td>79.91</td>
        <td>69.73</td>
        <td>96.99</td>
        <td>99.16</td>
        <td>94.82</td>
        <td>72.16</td>
        <td>69.06</td>
        <td>85.94</td>
        <td>61.48</td>
    </tr>
    <tr>
        <td><b>ERNIE 3.0 Tiny-Medium v2</b></td>
        <td>74.22</td>
        <td>75.88</td>
        <td>57.86</td>
        <td>61.64</td>
        <td>82.89</td>
        <td><b>80.27</b></td>
        <td>79.93</td>
        <td>81.27</td>
        <td>65.86/87.62</td>
        <td>80.75</td>
        <td><b>75.86</b></td>
        <td>97.22</td>
        <td>99.19</td>
        <td>95.24</td>
        <td><b>78.64</b></td>
        <td><b>81.41</b></td>
        <td><b>90.94</b></td>
        <td>63.58</td>
    </tr>
    <tr>
        <td rowspan=2 align=center>6L384H</td>
        <td>ERNIE 3.0 Tiny-Mini v1</td>
        <td>66.90</td>
        <td>71.85</td>
        <td>55.24</td>
        <td>54.48</td>
        <td>77.19</td>
        <td>73.08</td>
        <td>71.05</td>
        <td>79.30</td>
        <td>58.53/81.97</td>
        <td>69.71</td>
        <td>58.60</td>
        <td>96.27</td>
        <td>98.44</td>
        <td>94.10</td>
        <td>66.79</td>
        <td>67.34</td>
        <td>82.97</td>
        <td>50.07</td>
    </tr>
    <tr>
        <td><b>ERNIE 3.0 Tiny-Mini v2</b></td>
        <td>68.67</td>
        <td><b>74.40</b></td>
        <td>56.20</td>
        <td>55.79</td>
        <td>80.17</b></td>
        <td><b>76.75</b></td>
        <td>72.37</td>
        <td>77.77</td>
        <td>54.46/81.42</td>
        <td>71.50</td>
        <td><b>67.27</b></td>
        <td>96.69</td>
        <td>98.69</td>
        <td>94.68</td>
        <td><b>72.46</b></td>
        <td><b>73.75</b></td>
        <td><b>88.12</b></td>
        <td><b>55.50</b></td>
    </tr>
    <tr>
        <td rowspan=2 align=center>4L384H</td>
        <td>ERNIE 3.0 Tiny-Micro v1</td>
        <td>64.21</td>
        <td>71.15</td>
        <td>55.05</td>
        <td>53.83</td>
        <td>74.81</td>
        <td>70.41</td>
        <td>69.08</td>
        <td>76.50</td>
        <td>53.77/77.82</td>
        <td>62.26</td>
        <td>55.53</td>
        <td>95.76</td>
        <td>97.69</td>
        <td>93.83</td>
        <td>65.71</td>
        <td>66.25</td>
        <td>83.75</td>
        <td>47.12</td>
    </tr>
    <tr>
        <td><b>ERNIE 3.0 Tiny-Micro v2</b></td>
        <td>64.05</td>
        <td>72.52</td>
        <td>55.45</td>
        <td>54.33</td>
        <td><b>77.81</b></td>
        <td><b>74.85</b></td>
        <td>66.45</td>
        <td>74.43</td>
        <td>37.50/69.48</td>
        <td>64.89</td>
        <td><b>62.24</b></td>
        <td>96.47</td>
        <td>98.41</td>
        <td>94.52</td>
        <td><b>69.65</b></td>
        <td><b>72.50</b></td>
        <td>84.53</td>
        <td><b>51.93</b></td>
    </tr>
    <tr>
        <td rowspan=2 align=center>4L312H</td>
        <td>ERNIE 3.0 Tiny-Nano v1</td>
        <td>62.97</td>
        <td>70.51</td>
        <td>54.57</td>
        <td>48.36</td>
        <td>74.97</td>
        <td>70.61</td>
        <td>68.75</td>
        <td>75.93</td>
        <td>52.00/76.35</td>
        <td>58.91</td>
        <td>55.11</td>
        <td>71.16</td>
        <td>51.87</td>
        <td>91.35</td>
        <td>53.80</td>
        <td>58.59</td>
        <td>81.41</td>
        <td>21.40</td>
    </tr>
    <tr>
        <td><b>ERNIE 3.0 Tiny-Nano v2</b></td>
        <td>63.71</td>
        <td>72.75</td>
        <td>55.38</td>
        <td>48.90</td>
        <td><b>78.01</b></td>
        <td><b>74.54</b></td>
        <td>66.45</td>
        <td>76.37</td>
        <td>39.70/73.11</td>
        <td><b>63.04</b></td>
        <td><b>61.95</b></td>
        <td><b>96.34</b></td>
        <td><b>98.19</b></td>
        <td><b>94.48</b></td>
        <td><b>68.16</b></td>
        <td><b>72.34</b></td>
        <td><b>87.03</b></td>
        <td><b>45.10</b></td>
    </tr>
    <tr>
        <td rowspan=1 align=center>3L128H2A</td>
        <td><b>ERNIE 3.0 Tiny-Pico v2</b></td>
        <td>49.02</td>
        <td>69.35</td>
        <td>52.50</td>
        <td>21.05</td>
        <td>65.65</td>
        <td>64.03</td>
        <td>63.49</td>
        <td>68.60</td>
        <td>5.96/29.40</td>
        <td>36.77</td>
        <td>42.79</td>
        <td>74.13</td>
        <td>54.97</td>
        <td>93.29</td>
        <td>51.25</td>
        <td>62.34</td>
        <td>79.84</td>
        <td>11.58</td>
    </tr>
</table>


使用 PaddleNLP 只需要一行代码就可以下载并获取 ERNIE 3.0 Tiny v2 预训练模型，之后可以用自己的下游数据下进行微调。

```python

from paddlenlp.transformers import *

tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-tiny-medium-v2-zh")

# 用于分类任务（本项目中的意图识别任务）
seq_cls_model = AutoModelForSequenceClassification.from_pretrained("ernie-3.0-tiny-medium-v2-zh")

# 用于序列标注任务（本项目中的槽位填充任务）
token_cls_model = AutoModelForTokenClassification.from_pretrained("ernie-3.0-tiny-medium-v2-zh")

# 用于阅读理解任务
qa_model = AutoModelForQuestionAnswering.from_pretrained("ernie-3.0-tiny-medium-v2-zh")

```

<a name="代码结构"></a>

## 代码结构

以下是本项目代码结构

```text
.
├── train.py                     # 微调和压缩脚本
├── utils.py                     # 训练工具脚本
├── model.py                     # 模型结构脚本
├── evaluate.py                  # 评估脚本
├── data                         # 数据目录（自定义数据）
│ └── train.txt                  # 训练集（待用户新增）
│ └── dev.txt                    # 验证集（待用户新增）
│ └── intent_label.txt           # 意图标签文件
│ └── slot_label.txt             # 槽位标签文件
├── deploy                       # 部署目录
│ └── README.md                  # Fastdeploy 部署文档
│ └── android                    # 移动端部署目录
│ └── cpp                        # 服务端部署目录（C++）
│ └── python                     # 服务端部署目录（Python）
└── README.md                    # 文档
```

<a name="开始运行"></a>

## 开始运行

<a name="任务介绍"></a>

### 任务介绍

本项目是使用 ERNIE 3.0 Tiny v2 预训练模型移动端部署方案，任务背景是车载语音场景下的口语理解（Spoken Language Understanding，SLU）。本项目包括微调、压缩和部署的全流程。

SLU 任务主要将用户的自然语言表达解析为结构化信息。结构化信息的解析主要包括意图识别和槽位填充两个步骤。

- 数据样例：

```text
- 输入：来一首周华健的花心
- 输出
    - 意图识别任务：music.play
    - 槽位填充任务：来一首<singer>周华健</singer>的<song>花心</song>
```

在本项目中，意图识别和槽位填充任务分别被建模为文本分类和序列标注任务，二者共用一个 ERNIE Tiny 模型，只有最后的任务层是独立的。

- 评价方法：单句意图和槽位被完全正确分类的准确率（Accuracy）。

### 环境要求
- python >= 3.7
- paddlepaddle >= 2.4.1
- paddlenlp >= 2.5
- paddleslim >= 2.4

### 数据准备

本项目使用了 [NLPCC2018 Shared Task 4](http://tcci.ccf.org.cn/conference/2018/taskdata.php) 的数据集，该数据集来源于中文真实商用车载语音任务型对话系统的对话日志。需要说明的一点是，本项目为了使压缩样例更简洁，只考虑了原任务中的意图识别和槽位填充任务，纠错数据被忽略，并且只考虑单句任务。由于公开的测试集没有标签，因此只使用了训练集，并自行分割出训练集和验证集。

训练集的下载地址为[链接](http://tcci.ccf.org.cn/conference/2018/dldoc/trainingdata04.zip)。下载、解压后得到 `corpus.train.txt` 文件，将它移动至本项目中的 `data` 目录，再经过下面的代码按照 4:1 的比例分割出训练集和验证集，得到 `data/train.txt` 和 `data/dev.txt` 两个文件：

```shell
cd data

shuf corpus.train.txt > corpus.train.shuf.txt
num_lines=$(wc -l corpus.train.txt|awk '{print $1}')
head -n $[num_lines/5] corpus.train.txt.shuf > dev.txt
tail -n $[num_lines-num_lines/5] corpus.train.txt.shuf > train.txt

```
执行完后，data 目录应是如下结构：

```text
├── data                         # 数据目录（自定义数据）
│ └── train.txt                  # 训练集
│ └── dev.txt                    # 验证集
│ └── intent_label.txt           # 意图标签文件
│ └── slot_label.txt             # 槽位标签文件
```

由于文件较小，`intent_label.txt` 和 `slot_label.txt` 文件是从 `corpus.train.txt` 文件中提取并上传 git 的，提前写入这两个文件是为了读取数据逻辑更便捷，也便于预测时后处理使用。

<a name="模型训练"></a>

## 模型训练

本项目自定义了继承自 `ErniePretrainedModel` 的模型 `JointErnie`，使意图识别和槽位填充两个任务可以共用一个预训练模型 `ernie-3.0-tiny-nano-v2-zh`，但是各自也分别拥有最后一层独立的全连接层。模型的定义依然可以使用 `from_pretrained` API 传入使用的预训练模型和相关参数。这里也可以按照需求使用 ERNIE 3.0 Tiny v2 的其他大小的模型，如果不知道如何选择，可以对多个大小的模型都进行训练和压缩，最后根据在硬件上的精度、时延、内存占用等指标来选择模型。

```python
from model import JointErnie

model = JointErnie.from_pretrained(
    pretrained_model_name_or_path="ernie-3.0-tiny-nano-v2-zh",
    intent_dim=11,
    slot_dim=32,
)
```

运行下面的脚本，使用 Trainer API 启动训练：

```shell
BS=64
LR=5e-5
EPOCHS=30

export finetuned_model=./output/BS${BS}_LR${LR}_${EPOCHS}EPOCHS
mkdir $finetuned_model

python train.py \
    --device gpu \
    --logging_steps 100 \
    --save_steps 100 \
    --eval_steps 100 \
    --model_name_or_path ernie-3.0-tiny-nano-v2-zh \
    --prune_embeddings \
    --max_vocab_size 6000 \
    --max_seq_length 16  \
    --output_dir $finetuned_model \
    --train_path data/train.txt \
    --dev_path data/dev.txt \
    --intent_label_path data/intent_label.txt \
    --slot_label_path data/slot_label.txt \
    --label_names  'intent_label' 'slot_label' \
    --per_device_eval_batch_size ${BS} \
    --per_device_train_batch_size  ${BS} \
    --learning_rate ${LR} \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --do_train \
    --do_eval \
    --do_export \
    --input_dtype "int32" \
    --disable_tqdm True \
    --overwrite_output_dir \
    --num_train_epochs $EPOCHS \
    --load_best_model_at_end  True \
    --save_total_limit 1 \
    --metric_for_best_model eval_accuracy \
```

可配置参数说明：

* `model_name_or_path`：必须，进行微调使用的预训练模型。可选择的有 "ernie-3.0-tiny-base-v2-zh"、"ernie-3.0-tiny-medium-v2-zh"、"ernie-3.0-tiny-mini-v2-zh"、"ernie-3.0-tiny-micro-v2-zh"、"ernie-3.0-tiny-nano-v2-zh"、"ernie-3.0-tiny-pico-v2-zh"。
* `output_dir`：必须，模型训练后保存的模型目录。
* `prune_embeddings`：可选，模型的 embeddings 是否需要裁剪。如果设置，会按照 `max_seq_length` 以及 `max_vocab_size` 对预训练模型的 `position embeddings` 和 `word_embeddings` 参数进行裁剪，并将新的 model 和 tokenizer 保存至 `${output_dir}/pretrained_model` 下。后续的模型微调会基于 embeddings 裁剪后的模型开始。该策略主要是为了减少部署时模型的内存占用。如果对模型的内存占用要求不高，也可以不设置。
* `max_seq_length`：最大序列长度，是指分词后样本的最大token数，本项目中是 16。如果设置了 `prune_embeddings`，那么会对模型的 `position embeddings` 根据 `max_seq_length` 的值进行裁剪。
* `max_vocab_size`：词表裁剪后的大小。当设置 `prune_embeddings` 时，会根据词频对预训练模型的词表进行排序，并根据 `max_vocab_size` 大小进行裁剪。
* `train_path`：必须，训练集路径
* `dev_path`：必须，验证集路径
* `intent_label_path`：必须，意图标签文件路径。
* `slot_label_path`：必须，槽位标签文件路径。
* `label_names`：训练集中标签对应的的 key 名称。如果不传入，在训练时 Trainer 可能由于无法区分输入数据和标签造成错误。
* `do_train`:是否进行微调训练，设置该参数表示进行微调训练。
* `do_eval`:是否进行评估，设置该参数表示进行评估。
* `do_export`：是否导出模型，设置该参数表示训练完成后导出预测模型。
* `load_best_model_at_end`：是否在训练结尾导入最好的模型。
* `metric_for_best_model`：选择最好模型的 metric 名称。
* `per_device_train_batch_size`：训练集训练过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为 32。
* `per_device_eval_batch_size`：开发集评测过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为 32。
* `learning_rate`：训练最大学习率。
* `num_train_epochs`: 训练轮次，使用早停法时可以选择 100；默认为10。
* `logging_steps`: 训练过程中日志打印的间隔 steps 数，默认100。
* `save_steps`: 训练过程中保存模型 checkpoint 的间隔 steps 数，默认100。
* `weight_decay`：除了所有 bias 和 LayerNorm 权重之外，应用于所有层的权重衰减数值。可选；默认为 0.0；
* `input_dtype`：模型输入张量的数据类型。默认是 `int64`。
* `device`: 训练设备，可选择 'cpu'、'gpu' 其中的一种；默认为 'gpu'。


<a name="模型评估"></a>

## 模型评估
- 动态图

使用动态图进行评估，可以直接使用 [模型训练](#模型训练) 中的评估脚本，取消设置 `--do_train` 和 `--do_export` 并保留设置 `--do_eval`，并将 `--model_name_or_path` 设置成微调后的模型路径即可。

- 静态图

如果使用静态图进行评估或者预测，可以参考脚本 `evaluate.py`，参考下面的命令启动评估：

```shell
python evaluate.py  \
    --device gpu \
    --model_name_or_path output/BS64_LR5e-5_30EPOCHS/checkpoint-7700/ \
    --infer_prefix output/BS64_LR5e-5_30EPOCHS/infer_model \
    --output_dir ./ \
    --test_path data/dev.txt \
    --intent_label_path data/intent_label.txt \
    --slot_label_path data/slot_label.txt \
    --max_seq_length 16  \
    --per_device_eval_batch_size 512 \
    --do_eval
```

* `model_name_or_path`：动态图模型的目录，主要用于加载 tokenizer。
* `infer_prefix`：预测模型的路径（目录+前缀）。例如当 `infer_prefix` 为 `output/infer_model` 时，代表预测模型和参数文件分别为 `output/infer_model.pdmodel` 和 `output/infer_model.pdiparams`。
* `test_path` ：评估所用文件路径名；
* `do_eval`，是否输出评价指标的结果。如果设置，脚本会开启评估模式，最终会输出精度评价指标的值。如果不设置，则会输出模型后处理后的结果。例如：

```text
- 输入：放一首刘德华的音乐
- 输出：
    {'intent': 'music.play', 'confidence': array([0.9984201], dtype=float32)}
    {'value': [[{'slot': 'singer', 'entity': '刘德华', 'pos': [3, 5]}]]}
```

<a name="模型压缩"></a>

## 模型压缩

尽管 ERNIE 3.0 Tiny v2 已提供了效果不错的轻量级模型可以微调后直接使用，但如果有模型部署上线的需求，想要进一步压缩模型体积，降低推理时延，可使用本项目的 **端上语义理解压缩方案** 对上一步微调后的模型进行压缩，为了方便实现，[模型压缩 API](../../../docs/compression.md) 已提供了以下压缩功能。

端上模型压缩流程如下图所示：

<p align="center">
        <img width="1000" alt="image" src="https://user-images.githubusercontent.com/26483581/211022166-0558371b-c5b2-4a7a-a019-674f0a321ccf.png" title="compression plan">
</p>
<br>
在本项目中，运行下面的脚本，可对上面微调后的模型进行压缩：

```shell
EPOCHS=10

python train.py \
    --device gpu \
    --logging_steps 100 \
    --save_steps 100 \
    --eval_steps 100 \
    --model_name_or_path $finetuned_model/checkpoint-6700 \
    --output_dir $finetuned_model \
    --train_path data/train.txt \
    --dev_path data/dev.txt \
    --intent_label_path data/intent_label.txt \
    --slot_label_path data/slot_label.txt \
    --label_names  'intent_label' 'slot_label' \
    --max_seq_length 16  \
    --per_device_eval_batch_size ${BS} \
    --per_device_train_batch_size  ${BS} \
    --learning_rate ${LR} \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --input_dtype "int32" \
    --do_compress \
    --strategy 'dynabert+qat+embeddings' \
    --disable_tqdm True \
    --num_train_epochs $EPOCHS \
    --save_total_limit 1 \
    --metric_for_best_model eval_accuracy \
```

可配置参数说明：

* `strategy`：压缩策略，本案例中推荐使用`"dynabert+qat+embeddings"`，这是一个策略组合，由 `"dynabert"`、`"qat"`、`"embeddings"` 组成。其中`"dynabert"` 是一种裁剪策略，能直接对模型宽度进行裁剪，从而直接减少参数量，需要训练；`"qat"` 是一种量化方法，用于将模型中矩阵乘(底层是 matmul_v2 算子)的权重及激活值的数据类型由 FP32 转成 INT8，并使模型精度尽量保持无损，需要训练；`"embeddings"` 则代表 Embedding 量化策略，它将 Embedding API（底层是 lookup_table_v2 算子）的权重由 FP32 转成 INT8 存储，而不需要训练。由于词表参数量占比非常大，Embedding 量化能够大幅度减少模型的内存占用，但不会对时延产生正向作用。
* `model_name_or_path`：必须，进行压缩所使用的微调模型。
* `output_dir`：必须，模型训练或者压缩后保存的模型目录；默认为 `None` 。
* `do_compress`：必须。压缩需要通过这个开关来打开。其他的开关`do_train` 、`do_eval`和`do_export` 在此步则不能设置。
* `input_dtype`：模型输入张量的数据类型。默认是 `int64`。

其他参数同训练参数，如`learning_rate`、`num_train_epochs`、`per_device_train_batch_size` 等，是指压缩过程中的训练（`"dynabert"` 裁剪 以及 `"qat"` 量化）时所使用的参数，一般可以和微调时保持一致即可，其中 `num_train_epochs` 可比微调时略小。

### 压缩效果

模型经过压缩后，使用 FastDeploy 在移动端选用 Paddle Lite 作为后端部署，并进行了精度和性能的测试（max_seq_length=16，batch_size=1），得到模型精度、时延、内存占用的数据如下：

| 模型                                | 精度(acc.) | 时延(ms) | 内存占用 Pss (MB)  | 磁盘占用(MB)  |
|-----------------------------------|----------|--------|----------------|-----------|
| 原模型                               | 82.34    | TBD    | TBD            | 69.0      |
| 原模型+裁剪（词表+模型宽度）                   | 82.11    | TBD    | TBD            | 64.0      |
| 原模型+裁剪（词表+模型宽度）+量化（矩阵乘）           | 82.21    | TBD    | TBD            | 11.0      |
| 原模型+裁剪（词表+模型宽度）+量化（矩阵乘+Embedding） | TBD      | TBD    | TBD            | 5.4       |

由此可见，经过压缩后，精度基本无损，性能TBD，内存占用TBD，磁盘占用减小 92.2%。

<a name="FastDeploy部署"></a>

## FastDeplopy 部署

以下动图是 ERNIE 3.0 Tiny v2 意图识别、槽位填充模型部署在 Android App 上推理的效果展示：

<p align="center">
        <img width="200" alt="image" src="https://user-images.githubusercontent.com/26483581/210997849-9d3b7f7f-9363-4a3d-87c9-b29496a6b5b0.gif" title="compression plan">
</p>

针对 ERNIE 3.0 Tiny v2 模型，本项目已提供基于 FastDeploy 的云边端高性能部署示例代码和文档，欢迎参考 [FastDeploy ERNIE Tiny 模型高性能部署](deploy/README.md)。目前，FastDeploy 已支持多种后端：

- 在移动端上支持 `PaddleLite` 后端。

- 在服务端的 GPU 硬件上，支持 `PaddleInference`、`ONNX Runtime`、`Paddle TensorRT` 以及`TensorRT` 后端；在服务端的 CPU 硬件上支持 `PaddleInference`、`ONNX Runtime` 以及 `OpenVINO` 后端；

<a name="参考文献"></a>

## 参考文献
* TBD

* Su W, Chen X, Feng S, et al. ERNIE-Tiny: A Progressive Distillation Framework for Pretrained Transformer Compression[J]. arXiv preprint arXiv:2106.02241, 2021.
