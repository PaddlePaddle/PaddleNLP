# ERNIE 3.0 Tiny: Frustratingly Simple Method to Improve Task-Agnostic Distillation Generalization

 **目录**
   * [ERNIE 3.0 Tiny 介绍](#模型介绍)
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
   * [⚡️ FastDeploy 部署](#FastDeploy 部署)
       * [性能结论](#压缩结论)
   * [参考文献](#参考文献)

本项目开源了 **ERNIE 3.0 Tiny** 预训练模型及 **端上语义理解压缩方案**。

- **ERNIE 3.0 Tiny** 百度 ERNIE 使用 ERNIE-Tiny 系列的知识蒸馏技术，将 ERNIE 3.0 Titan 大模型的能力传递给小模型，产出并开源了易于部署的 ERNIE 3.0 Tiny 系列预训练模型，刷新了中文小模型的 SOTA 成绩。在这些较少参数量的 ERNIE 3.0 Tiny 系列模型中，有一部分可以直接部署在 CPU 上。

- **端上语义理解压缩方案** 在语义理解任务中使用 ERNIE 3.0 Tiny 微调的基础上，我们建议进一步使用包含模型裁剪、量化训练、Embedding 量化等策略的压缩方案，在保持模型精度不降的情况下，可将模型体积减小为原来的 7.8%，达到 5.4 MB，内存占用也随之大幅减小。再经过 [⚡️FastDeploy](https://github.com/PaddlePaddle/FastDeploy) 部署工具，**端到端推理性能**也有显著提升，从而将 ERNIE 3.0 Tiny 模型成功部署至 **📱端侧**。由于端侧部署对内存占用的要求比服务端更高，因此该方案也同样适用于 🖥服务端部署。

<a name="模型介绍"></a>

## ERNIE 3.0 Tiny 介绍

百度 ERNIE 团队在 2021 年底发布了百亿级别大模型 ERNIE 3.0 和千亿级别的大模型 ERNIE 3.0 Titan。为了让大模型的能力能够真正在一线业务发挥威力，ERNIE 团队推出了 ERNIE-Tiny 系列的知识蒸馏技术，通过任务无关蒸馏的方法，产出了多个轻量级模型 ERNIE 3.0 Tiny，刷新了中文小模型的成绩，并使这些模型能够直接在 CPU 上进行预测，大大拓展了 ERNIE 模型的使用场景。

2023 年初，ERNIE 团队进一步开源了 ERNIE 3.0 Tiny 模型的 v2 版本，使教师模型预先**注入下游知识**并参与 **多任务训练**，大大提高了小模型在下游任务上的效果。ERNIE 3.0 Tiny v2 模型在 in-domain、out-domain、low-resource 的下游任务上比 v1 有了进一步的提升，并且 v2 还开源了 3L128H 结构的模型。

### 在线蒸馏技术

在线蒸馏技术在模型学习的过程中周期性地将知识信号传递给若干个学生模型同时训练，从而在蒸馏阶段一次性产出多种尺寸的学生模型。相对传统蒸馏技术，该技术极大节省了因大模型额外蒸馏计算以及多个学生的重复知识传递带来的算力消耗。

这种新颖的蒸馏方式利用了文心大模型的规模优势，在蒸馏完成后保证了学生模型的效果和尺寸丰富性，方便不同性能需求的应用场景使用。此外，由于文心大模型的模型尺寸与学生模型差距巨大，模型蒸馏难度极大甚至容易失效。为此，通过引入了助教模型进行蒸馏的技术，利用助教作为知识传递的桥梁以缩短学生模型和大模型表达空间相距过大的问题，从而促进蒸馏效率的提升。

<p align="center">
        <img width="644" alt="image" src="https://user-images.githubusercontent.com/1371212/168516904-3fff73e0-010d-4bef-adc1-4d7c97a9c6ff.png" title="ERNIE 3.0 Online Distillation">
</p>

<br>

### 注入下游知识
ERNIE 3.0 Tiny v1 通过在线蒸馏技术将预训练大模型压缩成预训练小模型，然而由于小模型在微调之前没有接触到下游任务的相关知识，导致效果和大模型仍然存在差距。因此 ERNIE 团队进一步提出 **ERNIE 3.0 Tiny v2**，通过微调教师模型，让教师模型学习到下游任务的相关知识，进而能够在蒸馏的过程中传导给学生模型。尽管学生模型完全没有见过下游数据，通过预先注入下游知识到教师模型，蒸馏得到的学生模型也能够获取到下游任务的相关知识，进而使下游任务上的效果得到提升。

### 多任务学习提升泛化性
多任务学习已经被证明对增强模型泛化性有显著的效果，例如 MT-DNN、MUPPET、FLAN 等。通过对教师模型加入多下游任务微调，不但能够对教师模型注入下游知识、提高教师模型的泛化性，并且能够通过蒸馏传给学生模型，大幅度提升小模型的泛化性。具体地，我们对教师模型进行了 28 个任务的多任务微调。

<p align="center">
        <img width="644" alt="image" src="https://user-images.githubusercontent.com/26483581/210303124-c9df89a9-e291-4322-a6a5-37d2c4c1c008.png" title="ERNIE 3.0 Tiny v2">
</p>
<br>

因此，ERNIE 3.0 Tiny v2 相比 ERNIE 3.0 Tiny v1 在 in-domain、out-domain、low-resource 数据上都能获得显著的提升。

<a name="模型效果"></a>

## 预训练模型效果

本项目开源 **ERNIE 3.0 Tiny _Base_** 、**ERNIE 3.0 Tiny _Medium_** 、 **ERNIE 3.0 Tiny _Mini_** 、 **ERNIE 3.0 Tiny _Micro_** 、 **ERNIE 3.0 Tiny _Nano_**、**ERNIE 3.0 Tiny _Pico_** 六种结构的中文模型：

- **ERNIE 3.0-Tiny-_Base_**-zh (_12-layer, 768-hidden, 12-heads_)
- **ERNIE 3.0-Tiny-_Medium_**-zh(_6-layer, 768-hidden, 12-heads_)
- **ERNIE 3.0-Tiny-_Mini_**-zh (_6-layer, 384-hidden, 12-heads_)
- **ERNIE 3.0-Tiny-_Micro_**-zh (_4-layer, 384-hidden, 12-heads_)
- **ERNIE 3.0-Tiny-_Nano_**-zh (_4-layer, 312-hidden, 12-heads_)
- **ERNIE 3.0-Tiny-_Pico_**-zh (_3-layer, 128-hidden, 2-heads_)

其中，v2 版本开源了 6 种结构的模型，v1 版本开源了前 5 种结构的模型。

ERNIE 3.0 Tiny 模型可以用于文本分类、文本推理、实体抽取、问答等各种 NLU 任务中。下表是 ERNIE 3.0 Tiny 模型在 in-domain、out-domain 和 low-resource 三类数据集上的效果。其中 CLUE 指标可以通过 [PaddleNLP CLUE Benchmark](../../examples/benchmark/clue) 复现。

<table>
    <tr>
        <td>Arch</td>
        <td>Model</td>
        <td colspan=8 align=center> In-domain </td>
        <td colspan=3 align=center> Out-domain </td>
        <td colspan=4 align=center> Low-resource</td>
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
        <td>cluewsc2020</td>
        <td>csl</td>
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
        <td>ERNIE 3.0 Tiny-Base-v1-zh</td>
        <td>75.38</td>
        <td>75.93</td>
        <td>58.26</td>
        <td>61.56</td>
        <td>83.02</td>
        <td>80.10</td>
        <td>86.18</td>
        <td>82.63</td>
        <td>97.29</td>
        <td>99.31</td>
        <td>95.26</td>
        <td>75.81</td>
        <td>76.09</td>
        <td>89.06</td>
        <td>62.29</td>
    </tr>
    <tr>
        <td><b>ERNIE 3.0 Tiny-Base-v2-zh</b></td>
        <td>75.93</td>
        <td>77.43</td>
        <td>59.11</td>
        <td>61.49</td>
        <td>84.56</td>
        <td>81.86</td>
        <td>84.54</td>
        <td>82.50</td>
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
        <td>ERNIE 3.0 Tiny-Medium-v1-zh</td>
        <td>72.78</td>
        <td>73.37</td>
        <td>57.00</td>
        <td>60.67</td>
        <td>80.64</td>
        <td>76.88</td>
        <td>79.28</td>
        <td>81.60</td>
        <td>96.99</td>
        <td>99.16</td>
        <td>94.82</td>
        <td>72.16</td>
        <td>69.06</td>
        <td>85.94</td>
        <td>61.48</td>
    </tr>
    <tr>
        <td><b>ERNIE 3.0 Tiny-Medium-v2-zh</b></td>
        <td>74.25</td>
        <td>75.88</td>
        <td>57.86</td>
        <td>61.64</td>
        <td>82.89</td>
        <td><b>80.27</b></td>
        <td>79.93</td>
        <td>81.27</td>
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
        <td>ERNIE 3.0 Tiny-Mini-v1-zh</td>
        <td>68.88</td>
        <td>71.85</td>
        <td>55.24</td>
        <td>54.48</td>
        <td>77.19</td>
        <td>73.08</td>
        <td>71.05</td>
        <td>79.30</td>
        <td>96.27</td>
        <td>98.44</td>
        <td>94.10</td>
        <td>66.79</td>
        <td>67.34</td>
        <td>82.97</td>
        <td>50.07</td>
    </tr>
    <tr>
        <td><b>ERNIE 3.0 Tiny-Mini-v2-zh</b></td>
        <td>70.49</td>
        <td><b>74.40</b></td>
        <td>56.20</td>
        <td>55.79</td>
        <td>80.17</b></td>
        <td><b>76.75</b></td>
        <td>72.37</td>
        <td>77.77</td>
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
        <td>ERNIE 3.0 Tiny-Micro-v1-zh</td>
        <td>67.26</td>
        <td>71.15</td>
        <td>55.05</td>
        <td>53.83</td>
        <td>74.81</td>
        <td>70.41</td>
        <td>69.08</td>
        <td>76.50</td>
        <td>95.76</td>
        <td>97.69</td>
        <td>93.83</td>
        <td>65.71</td>
        <td>66.25</td>
        <td>83.75</td>
        <td>47.12</td>
    </tr>
    <tr>
        <td><b>ERNIE 3.0 Tiny-Micro-v2-zh</b></td>
        <td>67.98</td>
        <td>72.52</td>
        <td>55.45</td>
        <td>54.33</td>
        <td><b>77.81</b></td>
        <td><b>74.85</b></td>
        <td>66.45</td>
        <td>74.43</td>
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
        <td>ERNIE 3.0 Tiny-Nano-v1-zh</td>
        <td>66.24</td>
        <td>70.51</td>
        <td>54.57</td>
        <td>48.36</td>
        <td>74.97</td>
        <td>70.61</td>
        <td>68.75</td>
        <td>75.93</td>
        <td>71.16</td>
        <td>51.87</td>
        <td>91.35</td>
        <td>53.80</td>
        <td>58.59</td>
        <td>81.41</td>
        <td>21.40</td>
    </tr>
    <tr>
        <td><b>ERNIE 3.0 Tiny-Nano-v2-zh</b></td>
        <td>67.77</td>
        <td>72.75</td>
        <td>55.38</td>
        <td>48.90</td>
        <td><b>78.01</b></td>
        <td><b>74.54</b></td>
        <td>68.42</td>
        <td>76.37</td>
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
        <td><b>ERNIE 3.0 Tiny-Pico-v2-zh</b></td>
        <td>57.81</td>
        <td>69.35</td>
        <td>52.50</td>
        <td>21.05</td>
        <td>65.65</td>
        <td>64.03</td>
        <td>63.49</td>
        <td>68.60</td>
        <td>74.13</td>
        <td>54.97</td>
        <td>93.29</td>
        <td>51.25</td>
        <td>62.34</td>
        <td>79.84</td>
        <td>11.58</td>
    </tr>
</table>

ERNIE 3.0 Tiny v2 多任务学习、在线蒸馏方案效果显著，刷新了中文小模型的 SOTA 成绩。具体对比数据见如下模型 **精度-时延** 图，横坐标表示在 Arm CPU（高通 865 芯片）上，基于 Arm v8 arch 测试（batch_size=1, seq_len=32）的推理时延（Latency，单位毫秒），纵坐标是 CLUE 10 个任务上的平均精度（包含文本分类、文本匹配、自然语言推理、代词消歧、阅读理解等任务），其中 CMRC2018 阅读理解任务的评价指标是 Exact Match(EM)，其它任务的评价指标均是 Accuracy。模型名下方标注了模型的参数量。

<p align="center">
        <img width="644" alt="image" src="https://user-images.githubusercontent.com/26483581/218035834-050c04d4-3b59-468a-910b-aabf543d9c98.png" title="">
</p>


图中越靠左上方的模型，精度和性能水平越高。可以看到 ERNIE 3.0 Tiny v2 在同等规模的开源模型中，综合实力领先其他同类型轻量级模型。与 UER/RoBERTa-Base 相比，12L768H 的 ERNIE 3.0-Base 平均精度提升了 4.5 个点，比同等规模的 BERT-Base-Chinese 提升 3.7 个点；6L768H 的 ERNIE 3.0-Medium 相比 12L768H 的 UER/Chinese-RoBERTa 高 2.4，比 BERT-Base-Chinese 高 1.7，并且节省一倍运算时间；另外值得一提的是，这些小模型能够直接部署在 CPU 上。

使用 PaddleNLP 只需要一行代码就可以下载并获取 ERNIE 3.0 Tiny 预训练模型，之后可以用自己的下游数据下进行微调。

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

如果使用 v1 版本模型，只需要把 v2 替换成 v1 即可。

<a name="代码结构"></a>

## 代码结构

以下是本项目代码结构

```text
.
├── run_train.py                 # 微调和压缩脚本
├── run_eval.py                  # 评估脚本
├── utils.py                     # 训练工具脚本
├── model.py                     # 模型结构脚本
├── data                         # 数据目录（自定义数据）
│ └── train.txt                  # 训练集（待用户新增）
│ └── dev.txt                    # 验证集（待用户新增）
│ └── intent_label.txt           # 意图标签文件
│ └── slot_label.txt             # 槽位标签文件
├── deploy                       # 部署目录
│ └── README.md                  # Fastdeploy 部署文档
│ └── android                    # 端侧部署目录
│ └── cpp                        # 服务端部署目录（C++）
│ └── python                     # 服务端部署目录（Python）
└── README.md                    # 文档
```

<a name="开始运行"></a>

## 开始运行

<a name="任务介绍"></a>

### 任务介绍

本项目是使用 ERNIE 3.0 Tiny 预训练模型端侧部署方案，任务背景是车载语音场景下的口语理解（Spoken Language Understanding，SLU）。本项目包括微调、压缩和部署的全流程。

SLU 任务主要将用户的自然语言表达解析为结构化信息。结构化信息的解析主要包括意图识别和槽位填充两个步骤。

- 数据样例：

```text
- 输入：来一首周华健的花心
- 输出
    - 意图识别任务：music.play
    - 槽位填充任务：来一首<singer>周华健</singer>的<song>花心</song>
```

在本项目中，意图识别和槽位填充任务分别被建模为文本分类和序列标注任务，二者共用一个 ERNIE 3.0 Tiny 模型，只有最后的任务层是独立的。

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

shuf corpus.train.txt > corpus.train.txt.shuf
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

本项目自定义了继承自 `ErniePretrainedModel` 的模型 `JointErnie`，使意图识别和槽位填充两个任务可以共用一个预训练模型 `ernie-3.0-tiny-nano-v2-zh`，但是各自也分别拥有最后一层独立的全连接层。模型的定义依然可以使用 `from_pretrained` API 传入使用的预训练模型和相关参数。这里也可以按照需求使用 ERNIE 3.0 Tiny 其他大小的模型，如果不知道如何选择，可以对多个大小的模型都进行训练和压缩，最后根据在硬件上的精度、时延、内存占用等指标来选择模型。

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
mkdir output/BS64_LR5e-5_EPOCHS30

python run_train.py \
    --device gpu \
    --logging_steps 100 \
    --save_steps 100 \
    --eval_steps 100 \
    --model_name_or_path ernie-3.0-tiny-nano-v2-zh \
    --num_train_epochs 30 \
    --per_device_eval_batch_size 64 \
    --per_device_train_batch_size  64 \
    --learning_rate 5e-5 \
    --prune_embeddings \
    --max_vocab_size 6000 \
    --max_seq_length 16  \
    --output_dir output/BS64_LR5e-5_EPOCHS30 \
    --train_path data/train.txt \
    --dev_path data/dev.txt \
    --intent_label_path data/intent_label.txt \
    --slot_label_path data/slot_label.txt \
    --label_names  'intent_label' 'slot_label' \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --do_train \
    --do_eval \
    --do_export \
    --input_dtype "int32" \
    --disable_tqdm True \
    --overwrite_output_dir \
    --load_best_model_at_end  True \
    --save_total_limit 1 \
    --metric_for_best_model eval_accuracy \
```

可配置参数说明：

* `model_name_or_path`：必须，进行微调使用的预训练模型。可选择的有 "ernie-3.0-tiny-base-v2-zh"、"ernie-3.0-tiny-medium-v2-zh"、"ernie-3.0-tiny-mini-v2-zh"、"ernie-3.0-tiny-micro-v2-zh"、"ernie-3.0-tiny-nano-v2-zh"、"ernie-3.0-tiny-pico-v2-zh"。
* `output_dir`：必须，模型训练后保存的模型目录。
* `prune_embeddings`：可选，模型的 embeddings 是否需要裁剪。如果设置，会按照 `max_seq_length` 以及 `max_vocab_size` 对预训练模型的 `position embeddings` 和 `word_embeddings` 参数进行裁剪，并将新的 model 和 tokenizer 保存至 `${output_dir}/pretrained_model` 下。后续的模型微调会基于 embeddings 裁剪后的模型开始。该策略主要是为了减少部署时模型的内存占用。如果对模型的内存占用要求不高，也可以不设置。
* `max_seq_length`：最大序列长度，是指分词后样本的最大 token 数，本项目中是 16。如果设置了 `prune_embeddings`，那么会对模型的 `position embeddings` 根据 `max_seq_length` 的值进行裁剪。
* `max_vocab_size`：词表裁剪后的大小。当设置 `prune_embeddings` 时，会根据词频对预训练模型的词表进行排序，并根据 `max_vocab_size` 大小进行裁剪。
* `train_path`：必须，训练集路径
* `dev_path`：必须，验证集路径
* `intent_label_path`：必须，意图标签文件路径。
* `slot_label_path`：必须，槽位标签文件路径。
* `label_names`：训练集中标签对应的 key 名称。如果不传入，在训练时 Trainer 可能由于无法区分输入数据和标签造成错误。
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

如果使用静态图进行评估或者预测，可以参考脚本 `run_eval.py`，参考下面的命令启动评估：

```shell
python run_eval.py  \
    --device gpu \
    --model_name_or_path output/BS64_LR5e-5_EPOCHS30/checkpoint-7700/ \
    --infer_prefix output/BS64_LR5e-5_EPOCHS30/infer_model \
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

## 🔥端上模型压缩方案

尽管 ERNIE 3.0 Tiny 已提供了效果不错的轻量级模型可以微调后直接使用，但在本项目中，微调后的模型体积是 69.0 MB，内存占用达到 115.72MB，部署至端侧还是存在一定困难。因此当模型有部署上线的需求，想要进一步压缩模型体积，降低推理时延，可使用本项目的 **端上语义理解压缩方案** 对上一步微调后的模型进行压缩。

为了方便实现，[PaddleNLP 模型压缩 API](../../../docs/compression.md) 已提供了以下压缩功能，模型压缩 API 主要是基于 [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) 模型压缩能力，PaddleSlim 是一个专注于深度学习模型压缩的工具库，提供低比特量化、知识蒸馏、稀疏化和模型结构搜索等模型压缩策略，帮助开发者快速实现模型的小型化，欢迎大家使用。

端上模型压缩流程如下图所示：

<p align="center">
        <img width="1000" alt="image" src="https://user-images.githubusercontent.com/26483581/218037457-8b91cac4-e19e-401f-86c8-b64d7247014c.png" title="compression plan">
</p>
<br>

在本项目中，模型压缩和模型训练共用了脚本 `run_train.py`，压缩时需设置 `--do_compress` 开启模型压缩，并取消设置 `--do_train` 关闭普通训练。模型压缩还需要设置 `--strategy` 参数，本项目中选择 `'dynabert+qat+embeddings'` 组合策略。

运行下面的脚本，可对上面微调后的模型进行压缩：

```shell
python run_train.py \
    --do_compress \
    --strategy 'dynabert+qat+embeddings' \
    --num_train_epochs 10 \
    --model_name_or_path output/BS64_LR5e-5_EPOCHS30/checkpoint-6700 \
    --output_dir output/BS64_LR5e-5_EPOCHS30/ \
    --max_seq_length 16  \
    --per_device_eval_batch_size 64 \
    --per_device_train_batch_size  64 \
    --learning_rate 5e-5 \
    --train_path data/train.txt \
    --dev_path data/dev.txt \
    --intent_label_path data/intent_label.txt \
    --slot_label_path data/slot_label.txt \
    --label_names  'intent_label' 'slot_label' \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --input_dtype "int32" \
    --device gpu \
    --logging_steps 100 \
    --save_steps 100 \
    --eval_steps 100 \
    --disable_tqdm True \
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

<a name="压缩效果"></a>

### 压缩效果

| 模型                                                  | 模型精度(acc.) | 模型体积(MB) |
|-------------------------------------------------------|----------------|--------------|
| 原模型                                                | 82.34          | 69.0         |
| 原模型+裁剪（词表+模型宽度）                          | 82.11(-0.23)   | 64.0(-7.2%)  |
| 原模型+裁剪（词表+模型宽度）+量化（矩阵乘）           | 82.21(-0.13)   | 11.0(-84.1%) |
| 原模型+裁剪（词表+模型宽度）+量化（矩阵乘+Embedding） | 82.21(-0.13)   | 5.4(-92.2%)  |

模型经过压缩后，精度基本无损，体积减小了 92.2%，仅有 5.4 MB。到此，算法侧的工作基本完成。

<a name="FastDeploy 部署"></a>

## ⚡️FastDeplopy 部署
能够将深度学习模型部署到性能较低的端侧本身是比较困难的工作，因此在前面我们对小模型做了大量的优化，在精度不降的情况下将 69 MB 的模型压缩至 5.4 MB，但是如果想更好地满足业务上线要求，还需要有部署工具对性能有更多优化。在这里，PaddlePadde 提供了易用高效的云边端推理部署工具 ⚡️FastDeploy，它的 [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite) 后端基于算子融合和常量折叠进行了深度模型优化，使得模型推理速度可有大幅度提升；在麒麟 985 芯片上单条文本的分词的推理时延低于 0.1 毫秒；

因此，本项目基于 FastDeploy 部署工具，完成了 ERNIE 3.0 Tiny 端侧和服务端的高效部署，请参考 [ERNIE 3.0 Tiny 部署文档](deploy/README.md)。以下动图是 ERNIE 3.0 Tiny 意图识别、槽位填充联合模型使用 FastDeploy 部署在 Android App 上推理的效果展示：

<p align="center">
        <img width="200" alt="image" src="https://user-images.githubusercontent.com/26483581/210997849-9d3b7f7f-9363-4a3d-87c9-b29496a6b5b0.gif" title="compression plan">
</p>

想要更多了解 FastDeploy 可参考 [FastDeploy 仓库](https://github.com/PaddlePaddle/FastDeploy)。FastDeploy 是一款全场景、易用灵活、极致高效的 AI 推理部署工具，提供开箱即用的部署体验。它为 NLP 任务提供了一整套完整的部署 Pipeline，提供 ERNIE 3.0 Tiny 模型从文本预处理、推理引擎 Runtime 以及后处理三个阶段所需要的接口模块，开发者可以基于这些接口模块在云、边、端上部署各类常见的 NLP 任务，如文本分类、序列标注、信息抽取等：
- 在 Runtime 阶段，FastDeploy 集成多款硬件以及推理引擎后端，开发者可以设置 `fastdeploy::RuntimeOption` 以完成在不同硬件以及使用不同的推理引擎进行部署。目前，FastDeploy 支持的后端引擎有：
    - 端侧： `Paddle Lite`；
    - 服务端 GPU： `Paddle Inference`、`ONNX Runtime`、`Paddle TensorRT` 以及 `TensorRT`；
    - 服务端 CPU：`Paddle Inference`、`ONNX Runtime` 以及 `OpenVINO`。
- 在后处理阶段，FastDeploy 提供了张量级别的 [数值运算模块](https://baidu-paddle.github.io/fastdeploy-api/cpp/html/namespacefastdeploy_1_1function.html)， 基于该模块可以快速完成各类任务的后处理计算，如文本分类任务的 Softmax 等数值计算。

<a name="性能结论"></a>

### 性能结论

使用 FastDeploy 将压缩后的模型部署在华为 nova 7 Pro （麒麟 985 芯片）上，选用 Paddle Lite 作为后端进行测试，得到不同推理精度下的模型效果、端到端时延（包括前后处理）、内存占用的数据如下：

| 模型                                                      | 模型精度(acc.) | 推理精度  | 端到端时延(ms)  | 内存占用 Pss (MB)  | 模型体积(MB)    |
|-----------------------------------------------------------|----------------|-----------|-----------------|--------------------|-----------------|
| 原模型                                                    | 82.34          | FP32      | 9.90            | 115.72             | 69.0            |
| 原模型                                                    | 82.34(-0.00)   | FP16      | 6.03(1.64x)     | 106.24(-8.2%)      | 69.0(-0.0%)     |
| 原模型+裁剪（词表+模型宽度）                              | 82.11(-0.23)   | FP32      | 7.55(1.31x)     | 59.49(-48.59%)     | 64.0(-7.2%)     |
| 原模型+裁剪（词表+模型宽度）                              | 82.11(-0.23)   | FP16      | 4.68(2.12x)     | 52.23(-54.87%)     | 64.0(-7.2%)     |
| 原模型+裁剪（词表+模型宽度）+量化（矩阵乘）               | 82.21(-0.13)   | FP32+INT8 | 4.57(2.17x)     | 49.17(-57.51%)     | 11.0(-84.1%)    |
| **原模型+裁剪（词表+模型宽度）+量化（矩阵乘+Embedding）** | 82.21(-0.13)   | FP32+INT8 | **4.64(2.13x)** | **43.77(-62.18%)** | **5.4(-92.2%)** |


**测试条件**：max_seq_length=16，batch_size=1，thread_num=1

模型经过压缩后，精度基本无损，体积减小了 92.2%。在以上测试条件下，端到端推理速度达到原来的 2.13 倍，内存占用减小了 62.18%。

<a name="参考文献"></a>

## 参考文献
* Liu W, Chen X, Liu J, et al. ERNIE 3.0 Tiny: Frustratingly Simple Method to Improve Task-Agnostic Distillation Generalization[J]. arXiv preprint arXiv:2301.03416, 2023.
* Su W, Chen X, Feng S, et al. ERNIE-Tiny: A Progressive Distillation Framework for Pretrained Transformer Compression[J]. arXiv preprint arXiv:2106.02241, 2021.
* Wang S, Sun Y, Xiang Y, et al. ERNIE 3.0 Titan: Exploring Larger-scale Knowledge Enhanced Pre-training for Language Understanding and Generation[J]. arXiv preprint arXiv:2112.12731, 2021.
* Sun Y, Wang S, Feng S, et al. ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation[J]. arXiv preprint arXiv:2107.02137, 2021.
