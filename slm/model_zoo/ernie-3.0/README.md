# ERNIE 3.0 轻量级模型

 **目录**
   * [模型介绍](#模型介绍)
       * [在线蒸馏技术](#在线蒸馏技术)
   * [模型效果](#模型效果)
   * [开始运行](#开始运行)
       * [环境要求](#环境要求)
       * [数据准备](#数据准备)
   * [模型训练](#模型训练)
   * [模型预测](#模型预测)
   * [模型压缩](#模型压缩)
       * [环境依赖](#环境依赖)
       * [模型压缩 API 使用](#模型压缩 API 使用)
       * [压缩效果](#压缩效果)
           * [精度测试](#精度测试)
           * [性能测试](#性能测试)
               * [CPU 性能](#CPU 性能)
               * [GPU 性能](#CPU 性能)
   * [部署](#部署)
       * [FastDeploy 部署](#FastDeploy 部署)
           * [Python 部署](#Python 部署)
       * [服务化部署](#服务化部署)
   * [Notebook 教程](#Notebook 教程)
   * [参考文献](#参考文献)

<a name="模型介绍"></a>

## 模型介绍

本次开源的模型是文心大模型 ERNIE 3.0, 文心大模型 ERNIE 3.0 作为百亿参数知识增强的大模型，除了从海量文本数据中学习词汇、结构、语义等知识外，还从大规模知识图谱中学习。 基础上通过**在线蒸馏技术**得到的轻量级模型，模型结构与 ERNIE 2.0 保持一致，相比 ERNIE 2.0 具有更强的中文效果。

相关技术详解可参考文章[《解析全球最大中文单体模型鹏城-百度·文心技术细节》](https://www.jiqizhixin.com/articles/2021-12-08-9)

### 在线蒸馏技术

在线蒸馏技术在模型学习的过程中周期性地将知识信号传递给若干个学生模型同时训练，从而在蒸馏阶段一次性产出多种尺寸的学生模型。相对传统蒸馏技术，该技术极大节省了因大模型额外蒸馏计算以及多个学生的重复知识传递带来的算力消耗。

这种新颖的蒸馏方式利用了文心大模型的规模优势，在蒸馏完成后保证了学生模型的效果和尺寸丰富性，方便不同性能需求的应用场景使用。此外，由于文心大模型的模型尺寸与学生模型差距巨大，模型蒸馏难度极大甚至容易失效。为此，通过引入了助教模型进行蒸馏的技术，利用助教作为知识传递的桥梁以缩短学生模型和大模型表达空间相距过大的问题，从而促进蒸馏效率的提升。

更多技术细节可以参考论文：
- [ERNIE-Tiny: A Progressive Distillation Framework for Pretrained Transformer Compression](https://arxiv.org/abs/2106.02241)
- [ERNIE 3.0 Titan: Exploring Larger-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/abs/2112.12731)

<p align="center">
        <img width="644" alt="image" src="https://user-images.githubusercontent.com/1371212/168516904-3fff73e0-010d-4bef-adc1-4d7c97a9c6ff.png" title="ERNIE 3.0 Online Distillation">
</p>

<a name="模型效果"></a>


### 模型效果

本项目开源 **ERNIE 3.0 _Base_** 、**ERNIE 3.0 _Medium_** 、 **ERNIE 3.0 _Mini_** 、 **ERNIE 3.0 _Micro_** 、 **ERNIE 3.0 _Nano_** 五个模型：

- [**ERNIE 3.0-_Base_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_base_zh.pdparams) (_12-layer, 768-hidden, 12-heads_)
- [**ERNIE 3.0-_Medium_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_medium_zh.pdparams) (_6-layer, 768-hidden, 12-heads_)
- [**ERNIE 3.0-_Mini_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_mini_zh.pdparams) (_6-layer, 384-hidden, 12-heads_)
- [**ERNIE 3.0-_Micro_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_micro_zh.pdparams) (_4-layer, 384-hidden, 12-heads_)
- [**ERNIE 3.0-_Nano_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_nano_zh.pdparams) (_4-layer, 312-hidden, 12-heads_)


下面是 PaddleNLP 中轻量级中文模型的**效果-时延图**。横坐标表示在 IFLYTEK 数据集 (最大序列长度设置为 128) 上测试的延迟（latency，单位：ms），纵坐标是 CLUE 10 个任务上的平均精度（包含文本分类、文本匹配、自然语言推理、代词消歧、阅读理解等任务），其中 CMRC2018 阅读理解任务的评价指标是 Exact Match(EM)，其他任务的评价指标均是 Accuracy。图中越靠**左上**的模型，精度和性能水平越高。

图中模型名下方标注了模型的参数量，测试环境见[性能测试](#性能测试)。

batch_size=32 时，CPU 下的效果-时延图（线程数 1 和 8）：

<table>
    <tr>
        <td><a><img src="https://user-images.githubusercontent.com/26483581/175852121-2798b5c9-d122-4ac0-b4c8-da46b89b5512.png"></a></td>
        <td><a><img src="https://user-images.githubusercontent.com/26483581/175852129-bbe58835-8eec-45d5-a4a9-cc2cf9a3db6a.png"></a></td>
    </tr>
</table>

batch_size=1 时，CPU 下的效果-时延图（线程数 1 和 8）：

<table>
    <tr>
        <td><a><img src="https://user-images.githubusercontent.com/26483581/175852106-658e18e7-705b-4f53-bad0-027281163ae3.png"></a></td>
        <td><a><img src="https://user-images.githubusercontent.com/26483581/175852112-4b89d675-7c95-4d75-84b6-db5a6ea95e2c.png"></a></td>
    </tr>
</table>

batch_size=32 和 1，预测精度为 FP16 时，GPU 下的效果-时延图：

<table>
    <tr>
        <td><a><img src="https://user-images.githubusercontent.com/26483581/175854679-3247f42e-8716-4a36-b5c6-9ce4661b36c7.png"></a></td>
        <td><a><img src="https://user-images.githubusercontent.com/26483581/175854670-57878b34-c213-47ac-b620-aaaec082f435.png"></a></td>
    </tr>
</table>

从图上可看出，ERNIE 3.0 系列轻量级模型在精度和性能上的综合表现已全面领先于 UER-py、Huawei-Noah 以及 HFL 的中文模型。且当 batch_size=1、预测精度为 FP16 时，在 GPU 上宽且浅的模型的推理性能更有优势。

在 CLUE **验证集**上评测指标如下表所示：

<table style="width:100%;" cellpadding="2" cellspacing="0" border="1" bordercolor="#000000">
    <tbody>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;">Arch</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">Model</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">AVG</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">AFQMC</span>
            </td>
            <td style="text-align:center;">
                <span style="font-size:18px;">TNEWS</span>
            </td>
            <td style="text-align:center;">
                <span style="font-size:18px;">IFLYTEK</span>
            </td>
            <td style="text-align:center;">
                <span style="font-size:18px;">CMNLI</span>
            </td>
            <td style="text-align:center;">
                <span style="font-size:18px;">OCNLI</span>
            </td>
            <td style="text-align:center;">
                <span style="font-size:18px;">CLUEWSC2020</span>
            </td>
            <td style="text-align:center;">
                <span style="font-size:18px;">CSL</span>
            </td>
            <td style="text-align:center;">
                <span style="font-size:18px;">CMRC2018</span>
            </td>
            <td style="text-align:center;">
                <span style="font-size:18px;">CHID</span>
            </td>
            <td style="text-align:center;">
                <span style="font-size:18px;">C<sup>3</sup></span>
            </td>
        </tr>
        <tr>
            <td rowspan=3 align=center> 24L1024H </td>
            <td style="text-align:center">
                <span style="font-size:18px">ERNIE 1.0-Large-cw</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>79.03</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">75.97</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">59.65</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>62.91</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>85.09</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>81.73</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>93.09</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>84.53</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>74.22/91.88</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>88.57</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>84.54</b></span>
            </td>
        </tr>
        <tr>
            <td style="text-align:center">
                <span style="font-size:18px">ERNIE 2.0-Large-zh</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">76.90</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>76.23</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>59.33</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">61.91</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">83.85</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">79.93</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">89.82</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">83.23</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">70.95/90.31</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">86.78</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">78.12</span>
            </td>
        </tr>
        <tr>
            <td style="text-align:center">
                <span style="font-size:18px">RoBERTa-wwm-ext-large</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">76.61</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">76.00</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">59.33</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">62.02</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">83.88</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">78.81</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">90.79</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">83.67</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">70.58/89.82</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">85.72</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">75.26</span>
            </td>
        </tr>
        <tr>
            <td rowspan=1 align=center> 20L1024H </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>ERNIE 3.0-Xbase-zh</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>78.39</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>76.16</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>59.55</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>61.87</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>84.40</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>81.73</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>88.82</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>83.60</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>75.99/93.00</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>86.78</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>84.98</b></span>
            </td>
        </tr>
        <tr>
            <td rowspan=9 align=center> 12L768H </td>
            <td style="text-align:center">
                <span style="font-size:18px">
                    <a href="https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_base_zh.pdparams">
                        ERNIE 3.0-Base-zh
                    </a>
                </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">76.05</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">75.93</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">58.26</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">61.56</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">83.02</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>80.10</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">86.18</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">82.63</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">70.71/90.41</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">84.26</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>77.88</b></span>
            </td>
        </tr>
        <tr>
            <td style="text-align:center">
                <span style="font-size:18px">ERNIE 1.0-Base-zh-cw</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>76.47</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>76.07</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">57.86</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">59.91</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>83.41</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">79.58</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>89.91</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>83.42</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>72.88/90.78</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>84.68</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">76.98</span>
            </td>
        </tr>
        <tr>
            <td style="text-align:center">
                <span style="font-size:18px">ERNIE-Gram-zh</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">75.72</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">75.28</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">57.88</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">60.87</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">82.90</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">79.08</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">88.82</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">82.83</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">71.82/90.38</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">84.04</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">73.69</span>
            </td>
        </tr>
        <tr>
            <td style="text-align:center">
                <span style="font-size:18px">Langboat/Mengzi-BERT-Base</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">74.69</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">75.35</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">57.76</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">61.64</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">82.41</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">77.93</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">88.16</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">82.20</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">67.04/88.35</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">83.74</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">70.70</span>
            </td>
        </tr>
        <tr>
            <td style="text-align:center">
                <span style="font-size:18px">ERNIE 2.0-Base-zh</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">74.32</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">75.65</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">58.25</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">61.64</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">82.62</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">78.71</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">81.91</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">82.33</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">66.08/87.46</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">82.78</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">73.19</span>
            </td>
        </tr>
        <tr>
            <td style="text-align:center">
                <span style="font-size:18px">ERNIE 1.0-Base-zh</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">74.17</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">74.84</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>58.91</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>62.25</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">81.68</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">76.58</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">85.20</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">82.77</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">67.32/87.83</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">82.47</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">69.68</span>
            </td>
        </tr>
        <tr>
            <td style="text-align:center">
                <span style="font-size:18px">RoBERTa-wwm-ext</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">74.11</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">74.60</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">58.08</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">61.23</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">81.11</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">76.92</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">88.49</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">80.77</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">68.39/88.50</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">83.43</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">68.03</span>
            </td>
        </tr>
        <tr>
            <td style="text-align:center">
                <span style="font-size:18px">BERT-Base-Chinese</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">72.57</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">74.63</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">57.13</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">61.29</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">80.97</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">75.22</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">81.91</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">81.90</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">65.30/86.53</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">82.01</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">65.38</span>
            </td>
        </tr>
        <tr>
            <td style="text-align:center">
                <span style="font-size:18px">UER/Chinese-RoBERTa-Base</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">71.78</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">72.89</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">57.62</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">61.14</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">80.01</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">75.56</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">81.58</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">80.80</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">63.87/84.95</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">81.52</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">62.76</span>
            </td>
        </tr>
        <tr>
            <td rowspan=1 align=center> 8L512H </td>
            <td style="text-align:center">
                <span style="font-size:18px">UER/Chinese-RoBERTa-Medium</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">67.06</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">70.64</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">56.10</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">58.29</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">77.35</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">71.90</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">68.09</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">78.63</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">57.63/78.91</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">75.13</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">56.84</span>
            </td>
        </tr>
        <tr>
            <td rowspan=5 align=center> 6L768H </td>
            <td style="text-align:center">
                <span style="font-size:18px">
                    <a href="https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_medium_zh.pdparams">
                        ERNIE 3.0-Medium-zh
                    </a>
                </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>72.49</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>73.37</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>57.00</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">60.67</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>80.64</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>76.88</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>79.28</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>81.60</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>65.83/87.30</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>79.91</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>69.73</b></span>
            </td>
        </tr>
        <tr>
            <td style="text-align:center">
                <span style="font-size:18px">HLF/RBT6, Chinese</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">70.06</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">73.45</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">56.82</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">59.64</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">79.36</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">73.32</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">76.64</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">80.67</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">62.72/84.77</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">78.17</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">59.85</span>
            </td>
        </tr>
        <tr>
            <td style="text-align:center">
                <span style="font-size:18px">TinyBERT<sub>6</sub>, Chinese</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">69.62</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">72.22</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">55.70</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">54.48</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">79.12</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">74.07</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">77.63</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">80.17</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">63.03/83.75</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">77.64</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">62.11</span>
            </td>
        </tr>
        <tr>
            <td style="text-align:center">
                <span style="font-size:18px">RoFormerV2 Small</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">68.52</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">72.47</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">56.53</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>60.72</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">76.37</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">72.95</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">75.00</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">81.07</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">62.97/83.64</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">67.66</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">59.41</span>
            </td>
        </tr>
        <tr>
            <td style="text-align:center">
                <span style="font-size:18px">UER/Chinese-RoBERTa-L6-H768</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">67.09</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">70.13</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">56.54</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">60.48</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">77.49</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">72.00</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">72.04</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">77.33</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">53.74/75.52</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">76.73</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">54.40</span>
            </td>
        </tr>
        <tr>
            <td rowspan=1 align=center> 6L384H </td>
            <td style="text-align:center">
                <span style="font-size:18px">
                    <a href="https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_mini_zh.pdparams">
                        ERNIE 3.0-Mini-zh
                    </a>
                </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">66.90</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">71.85</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">55.24</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">54.48</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">77.19</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">73.08</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">71.05</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">79.30</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">58.53/81.97</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">69.71</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">58.60</span>
            </td>
        </tr>
               <tr>
            <td rowspan=1 align=center> 4L768H </td>
            <td style="text-align:center">
                <span style="font-size:18px">HFL/RBT4, Chinese</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">67.42</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">72.41</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">56.50</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">58.95</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">77.34</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">70.78</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">71.05</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">78.23</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">59.30/81.93</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">73.18</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">56.45</span>
            </td>
        </tr>
        <tr>
            <td rowspan=1 align=center> 4L512H </td>
            <td style="text-align:center">
                <span style="font-size:18px">UER/Chinese-RoBERTa-Small</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">63.25</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">69.21</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">55.41</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">57.552</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">73.64</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">69.80</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">66.78</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">74.83</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">46.75/69.69</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">67.59</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">50.92</span>
            </td>
        </tr>
        <tr>
            <td rowspan=1 align=center> 4L384H </td>
            <td style="text-align:center">
                <span style="font-size:18px">
                    <a href="https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_micro_zh.pdparams">
                    ERNIE 3.0-Micro-zh
                    </a>
                </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">64.21</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">71.15</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">55.05</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">53.83</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">74.81</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">70.41</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">69.08</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">76.50</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">53.77/77.82</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">62.26</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">55.53</span>
            </td>
        </tr>
        <tr>
            <td rowspan=2 align=center> 4L312H </td>
            <td style="text-align:center">
                <span style="font-size:18px">
                    <a href="https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_nano_zh.pdparams">
                        ERNIE 3.0-Nano-zh
                    </a>
                </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>62.97</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>70.51</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>54.57</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>48.36</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>74.97</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>70.61</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">68.75</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>75.93</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>52.00/76.35</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>58.91</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>55.11</b></span>
            </td>
        </tr>
        <tr>
            <td style="text-align:center">
            <span style="font-size:18px">TinyBERT<sub>4</sub>, Chinese</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">60.82</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">69.07</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">54.02</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">39.71</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">73.94</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">69.59</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>70.07</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">75.07</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">46.04/69.34</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">58.53</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">52.18</span>
            </td>
        </tr>
        <tr>
            <td rowspan=1 align=center> 4L256H </td>
            <td style="text-align:center">
            <span style="font-size:18px">UER/Chinese-RoBERTa-Mini</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">53.40</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">69.32</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">54.22</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">41.63</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">69.40</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">67.36</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">65.13</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">70.07</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">5.96/17.13</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">51.19</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">39.68</span>
            </td>
        </tr>
        <tr>
            <td rowspan=1 align=center> 3L1024H </td>
            <td style="text-align:center">
                <span style="font-size:18px">HFL/RBTL3, Chinese</span>
            </td>
                <td style="text-align:center">
                <span style="font-size:18px">66.63</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">71.11</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">56.14</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">59.56</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">76.41</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">71.29</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">69.74</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">76.93</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">58.50/80.90</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">71.03</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">55.56</span>
            </td>
        </tr>
        <tr>
            <td rowspan=1 align=center> 3L768H </td>
            <td style="text-align:center">
                <span style="font-size:18px">HFL/RBT3, Chinese</span>
            </td>
                <td style="text-align:center">
                <span style="font-size:18px">65.72</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">70.95</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">55.53</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">59.18</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">76.20</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">70.71</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">67.11</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">76.63</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">55.73/78.63</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">70.26</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">54.93</span>
            </td>
        </tr>
        <tr>
            <td rowspan=1 align=center> 2L128H </td>
            <td style="text-align:center">
                <span style="font-size:18px">UER/Chinese-RoBERTa-Tiny</span>
            </td>
                <td style="text-align:center">
                <span style="font-size:18px">44.45</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">69.02</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">51.47</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">20.28</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">59.95</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">57.73</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">63.82</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">67.43</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">3.08/14.33</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">23.57</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">28.12</span>
            </td>
        </tr>
    <tbody>
</table>
<br />


<a name="代码结构"></a>

## 代码结构
以下是本项目代码结构

```text
.
├── run_seq_cls.py               # 分类任务的微调脚本
├── run_token_cls.py             # 序列标注任务的微调脚本
├── run_qa.py                    # 阅读理解任务的微调脚本
├── compress_seq_cls.py          # 分类任务的压缩脚本
├── compress_token_cls.py        # 序列标注任务的压缩脚本
├── compress_qa.py               # 阅读理解任务的压缩脚本
├── utils.py                     # 训练工具脚本
├── configs                      # 压缩配置文件夹
│ └── default.yml                # 默认配置文件
├── deploy                       # 部署目录
│ └── predictor                  # onnx离线部署
│   └── infer_cpu.py
│   └── infer_gpu.py
│   └── README.md
│   └── requirements_cpu.txt
│   └── requirements_gpu.txt
│ └── simple_serving            # 基于PaddleNLP SimpleServing 服务化部署
│   └── client_qa.py
│   └── client_seq_cls.py
│   └── client_token_cls.py
│   └── README.md
│   └── server_qa.py
│   └── server_seq_cls.py
│   └── server_token_cls.py
│ └── triton_serving           # 基于Triton Serving 服务化部署
│   └── models
│   └── README.md
│   └── seq_cls_grpc_client.py
│   └── token_cls_grpc_client.py
└── README.md                    # 文档

```


<a name="开始运行"></a>
## 开始运行
下面提供以 CLUE 数据集进行模型微调相关训练、预测、部署的代码, CLUE 数据集是中文语言理解测评基准数据集，包括了文本分类、文本推理、实体抽取、问答等相关数据集。

### 环境要求
- python >= 3.7
- paddlepaddle >= 2.3
- paddlenlp >= 2.4
- paddleslim >= 2.4
- evaluate

### 数据准备
此次微调数据主要是以 CLUE benchmark 数据集为主, CLUE benchmark 包括了文本分类、实体抽取、问答三大类数据集，而 CLUE benchmark 数据目前已经集成在 PaddleNLP 的 datasets 里面，可以通过下面的方式来使用数据集

```python
from paddlenlp.datasets import load_dataset

# Load the clue Tnews dataset
train_ds, test_ds = load_dataset('clue', 'tnews', splits=('train', 'test'))

```

<a name="模型训练"></a>
## 模型训练

使用 PaddleNLP 只需要一行代码可以拿到 ERNIE 3.0 系列模型，之后可以在自己的下游数据下进行微调，从而获得具体任务上效果更好的模型。

```python

from paddlenlp.transformers import *

tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")

# 用于分类任务
seq_cls_model = AutoModelForSequenceClassification.from_pretrained("ernie-3.0-medium-zh")

# 用于序列标注任务
token_cls_model = AutoModelForTokenClassification.from_pretrained("ernie-3.0-medium-zh")

# 用于阅读理解任务
qa_model = AutoModelForQuestionAnswering.from_pretrained("ernie-3.0-medium-zh")

```

本项目提供了针对分类（包含文本分类、文本匹配、自然语言推理、代词消歧等任务）、序列标注、阅读理解三大场景下微调的示例脚本，可分别参考 `run_seq_cls.py` 、`run_token_cls.py`、`run_qa.py` 三个脚本，启动方式如下：

```shell
# 分类任务
# 该脚本共支持 CLUE 中 7 个分类任务，超参不全相同，因此分类任务中的超参配置利用 config.yml 配置
# --device 选择训练模型的硬件，可选 cpu/gpu/xpu/npu，默认为 gpu。xpu 为昆仑芯片，npu 为昇腾芯片。
python run_seq_cls.py  --model_name_or_path ernie-3.0-medium-zh  --dataset afqmc --output_dir ./best_models --export_model_dir best_models/ --do_train --do_eval --do_export --config=configs/default.yml

# 序列标注任务
python run_token_cls.py --model_name_or_path ernie-3.0-medium-zh --dataset msra_ner --output_dir ./best_models --export_model_dir best_models/ --do_train --do_eval --do_export --config=configs/default.yml

# 阅读理解任务
python run_qa.py --model_name_or_path ernie-3.0-medium-zh --dataset cmrc2018  --output_dir ./best_models --export_model_dir best_models/ --do_train --do_eval --do_export --config=configs/default.yml
```

<a name="模型预测"></a>
## 模型预测

```shell
# 分类任务
# 该脚本共支持 CLUE 中 7 个分类任务，超参不全相同，因此分类任务中的超参配置利用 config.yml 配置
# --device 选择训练模型的硬件，可选 cpu/gpu/xpu/npu，默认为 gpu。xpu 为昆仑芯片，npu 为昇腾芯片。
python run_seq_cls.py  --model_name_or_path best_models/afqmc/  --dataset afqmc --output_dir ./best_models --do_predict --config=configs/default.yml

# 序列标注任务
python run_token_cls.py  --model_name_or_path best_models/msra_ner/  --dataset msra_ner --output_dir ./best_models --do_predict --config=configs/default.yml

# 阅读理解任务
python run_qa.py --model_name_or_path best_models/cmrc2018/ --dataset cmrc2018  --output_dir ./best_models --do_predict --config=configs/default.yml
```


<a name="模型压缩"></a>

## 模型压缩

尽管 ERNIE 3.0 已提供了效果不错的 6 层、4 层轻量级模型可以微调后直接使用，但如果有模型部署上线的需求，则需要进一步压缩模型体积，可以使用这里提供的一套模型压缩方案及 API 对上一步微调后的模型进行压缩。

<a name="环境依赖"></a>

### 环境依赖

使用裁剪功能需要安装 paddleslim 包

```shell
pip install paddleslim
```

<a name="模型压缩 API 使用"></a>

### 模型压缩 API 使用

本项目使用压缩 API 对任务数据上微调后的模型进行裁剪和量化。用户在传入模型，以及相关的压缩超参数（可选，提供默认选项）后，只需要调用一行 `compress()` 即可一键启动裁剪和量化，并自动保存压缩后的模型进行后续部署。

核心调用方法如下，如需跑通完整的例子可参考本目录下完整样例脚本:

```python

trainer = Trainer(
    model=model,
    args=compression_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    criterion=criterion)

trainer.compress()

```
压缩 API 可以传入的超参数可参考[文档](../../../docs/compression.md)。

本项目提供了压缩 API 在分类（包含文本分类、文本匹配、自然语言推理、代词消歧等任务）、序列标注、阅读理解三大场景下的使用样例，可以分别参考 `compress_seq_cls.py` 、`compress_token_cls.py`、`compress_qa.py`，启动方式如下：

```shell
# 分类任务
# 该脚本共支持 CLUE 中 7 个分类任务，超参不全相同，因此分类任务中的超参配置利用 configs/defalut.yml 配置
python compress_seq_cls.py  --model_name_or_path best_models/afqmc/  --dataset afqmc --output_dir ./best_models/afqmc --config=configs/default.yml

# 序列标注任务
python compress_token_cls.py  --model_name_or_path best_models/msra_ner/  --dataset msra_ner --output_dir ./best_models/msra_ner --config=configs/default.yml

# 阅读理解任务
python compress_qa.py --model_name_or_path best_models/cmrc2018/ --dataset cmrc2018  --output_dir ./best_models/cmrc2018 --config=configs/default.yml

```


<a name="压缩效果"></a>

### 压缩效果

<a name="精度测试"></a>

#### 精度测试

本案例中我们对 ERNIE 3.0-Medium 模型在三类任务上微调后的模型使用压缩 API 进行压缩。压缩后精度如下：

| Model                           | AVG   | AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUEWSC2020 | CSL   | CMRC2018    | MSRA_NER          |
|---------------------------------|-------|-------|-------|---------|-------|-------|-------------|-------|-------------|-------------------|
| ERNIE 3.0-Medium                | 74.87 | 75.35 | 57.45 | 60.18   | 81.16 | 77.19 | 80.59       | 81.93 | 66.95/87.15 | 92.65/93.43/93.04 |
| ERNIE 3.0-Medium+FP16           | 74.87 | 75.32 | 57.45 | 60.22   | 81.16 | 77.22 | 80.59       | 81.90 | 66.95/87.16 | 92.65/93.45/93.05 |
| ERNIE 3.0-Medium+裁剪+FP32      | 74.70 | 75.14 | 57.31 | 60.29   | 81.25 | 77.46 | 79.93       | 81.70 | 65.92/86.43 | 93.10/93.43/93.27 |
| ERNIE 3.0-Medium+裁剪+FP16      | 74.71 | 75.21 | 57.27 | 60.29   | 81.24 | 77.56 | 79.93       | 81.73 | 65.89/86.44 | 93.10/93.43/93.27 |
| ERNIE 3.0-Medium+裁剪+量化+INT8 | 74.44 | 75.02 | 57.26 | 60.37   | 81.03 | 77.25 | 77.96       | 81.67 | 66.17/86.55 | 93.17/93.23/93.20 |
| ERNIE 3.0-Medium+量化+INT8      | 74.10 | 74.67 | 56.99 | 59.91   | 81.03 | 75.05 | 78.62       | 81.60 | 66.32/86.82 | 93.10/92.90/92.70 |

**评价指标说明：** 其中 CLUE 分类任务（AFQMC 语义相似度、TNEWS 文本分类、IFLYTEK 长文本分类、CMNLI 自然语言推理、OCNLI 自然语言推理、CLUEWSC2020 代词消歧、CSL 论文关键词识别）的评价指标是 Accuracy，阅读理解任务 CLUE CMRC2018 的评价指标是 EM (Exact Match) / F1-Score，计算平均值时取 EM，序列标注任务 MSRA_NER 的评价指标是 Precision/Recall/F1-Score，计算平均值时取 F1-Score。

由表可知，`ERNIE 3.0-Medium` 模型经过裁剪和量化后，精度平均下降 0.46，其中裁剪后下降了 0.17，单独量化精度平均下降 0.77。

<a name="性能测试"></a>

#### 性能测试

性能测试的配置如下：

1. 数据集：TNEWS（文本分类）、MSRA_NER（序列标注）、CLUE CMRC2018（阅读理解）

2. 计算卡：T4、CUDA11.2、CuDNN8.2

3. CPU 信息：Intel(R) Xeon(R) Gold 6271C CPU

4. PaddlePaddle 版本：2.3

5. PaddleNLP 版本：2.3

6. 性能数据单位是 QPS。QPS 测试方法：固定 batch size 为 32，测试运行时间 total_time，计算 QPS = total_samples / total_time

7. 精度数据单位：文本分类是 Accuracy，序列标注是 F1-Score，阅读理解是 EM (Exact Match)

<a name="CPU 性能"></a>

##### CPU 性能

测试环境及说明如上，测试 CPU 性能时，线程数设置为12。

|                            | TNEWS 性能   | TNEWS 精度   | MSRA_NER 性能 | MSRA_NER 精度 | CMRC2018 性能 | CMRC2018 精度 |
|----------------------------|--------------|--------------|---------------|---------------|---------------|---------------|
| ERNIE 3.0-Medium+FP32      | 311.95(1.0X) | 57.45        | 90.91(1.0x)   | 93.04         | 33.74(1.0x)   | 66.95         |
| ERNIE 3.0-Medium+INT8      | 600.35(1.9x) | 56.57(-0.88) | 141.00(1.6x)  | 92.64(-0.40)  | 56.51(1.7x)   | 66.23(-0.72)  |
| ERNIE 3.0-Medium+裁剪+FP32 | 408.65(1.3x) | 57.31(-0.14) | 122.13(1.3x)  | 93.27(+0.23)  | 48.47(1.4x)   | 65.55(-1.40)  |
| ERNIE 3.0-Medium+裁剪+INT8 | 704.42(2.3x) | 56.69(-0.76) | 215.58(2.4x)  | 92.39(-0.65)  | 75.23(2.2x)   | 63.47(-3.48)  |


三类任务（分类、序列标注、阅读理解）经过相同压缩过程后，加速比达到 2.3 左右。


<a name="GPU 性能"></a>

##### GPU 性能

|                            | TNEWS 性能    | TNEWS 精度   | MSRA_NER 性能 | MSRA_NER 精度 | CMRC2018 性能 | CMRC2018 精度 |
|----------------------------|---------------|--------------|---------------|---------------|---------------|---------------|
| ERNIE 3.0-Medium+FP32      | 1123.85(1.0x) | 57.45        | 366.75(1.0x)  | 93.04         | 146.84(1.0x)  | 66.95         |
| ERNIE 3.0-Medium+FP16      | 2672.41(2.4x) | 57.45(0.00)  | 840.11(2.3x)  | 93.05(0.01)   | 303.43(2.1x)  | 66.95(0.00)   |
| ERNIE 3.0-Medium+INT8      | 3226.26(2.9x) | 56.99(-0.46) | 889.33(2.4x)  | 92.70(-0.34)  | 348.84(2.4x)  | 66.32(-0.63)  |
| ERNIE 3.0-Medium+裁剪+FP32 | 1424.01(1.3x) | 57.31(-0.14) | 454.27(1.2x)  | 93.27(+0.23)  | 183.77(1.3x)  | 65.92(-1.03)  |
| ERNIE 3.0-Medium+裁剪+FP16 | 3577.62(3.2x) | 57.27(-0.18) | 1138.77(3.1x) | 93.27(+0.23)  | 445.71(3.0x)  | 65.89(-1.06)  |
| ERNIE 3.0-Medium+裁剪+INT8 | 3635.48(3.2x) | 57.26(-0.19) | 1105.26(3.0x) | 93.20(+0.16)  | 444.27(3.0x)  | 66.17(-0.78)  |


三类任务（分类、序列标注、阅读理解）经过裁剪 + 量化后加速比均达到 3 倍左右，所有任务上平均精度损失可控制在 0.5 以内（0.46）。

<a name="部署"></a>

## 部署

我们基于 FastDeploy 为 ERNIE 3.0 提供了多种部署方案，可以满足不同场景下的部署需求，请根据实际情况进行选择。

<a name="FastDeploy 部署"></a>

### FastDeploy 部署

⚡️[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)是一款全场景、易用灵活、极致高效的 AI 推理部署工具，为开发者提供多硬件、多推理引擎后端的部署能力。开发者只需调用一行代码即可随意切换硬件、推理引擎后端。

<div align="center">

<img src="https://user-images.githubusercontent.com/54695910/213087724-7175953a-0e07-4af8-a4a1-5304163da2e0.png" >

</div>

目前 ERNIE 3.0 模型已提供基于 FastDeploy 的部署示例，支持在多款硬件（CPU、GPU、昆仑芯、华为昇腾以及 Graphcore IPU）以及推理引擎后端进行部署。具体的适配的硬件以及推理引擎请参考：[FastDeploy 部署指南](./deploy/README.md)

<a name="Python 部署"></a>

#### Python 部署

Python 部署请参考：[Python 部署指南](./deploy/python/README.md)

<a name="服务化部署"></a>

### 服务化部署

- [FastDeploy Serving 高性能服务化部署指南](./deploy/serving/README.md)
- [PaddleNLP SimpleServing 服务化部署指南](./deploy/simple_serving/README.md)


<a name="参考文献"></a>

## Notebook 教程

- [【快速上手 ERNIE 3.0】中文情感分析实战](https://aistudio.baidu.com/aistudio/projectdetail/3955163)

- [【快速上手 ERNIE 3.0】法律文本多标签分类实战](https://aistudio.baidu.com/aistudio/projectdetail/3996601)

- [【快速上手 ERNIE 3.0】中文语义匹配实战](https://aistudio.baidu.com/aistudio/projectdetail/3986803)

- [【快速上手 ERNIE 3.0】MSRA 序列标注实战](https://aistudio.baidu.com/aistudio/projectdetail/3989073)

- [【快速上手 ERNIE 3.0】机器阅读理解实战](https://aistudio.baidu.com/aistudio/projectdetail/2017189)

- [【快速上手 ERNIE 3.0】对话意图识别](https://aistudio.baidu.com/aistudio/projectdetail/2017202?contributionType=1)


## 参考文献

* Sun Y, Wang S, Feng S, et al. ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation[J]. arXiv preprint arXiv:2107.02137, 2021.

* Su W, Chen X, Feng S, et al. ERNIE-Tiny: A Progressive Distillation Framework for Pretrained Transformer Compression[J]. arXiv preprint arXiv:2106.02241, 2021.

* Wang S, Sun Y, Xiang Y, et al. ERNIE 3.0 Titan: Exploring Larger-scale Knowledge Enhanced Pre-training for Language Understanding and Generation[J]. arXiv preprint arXiv:2112.12731, 2021.
