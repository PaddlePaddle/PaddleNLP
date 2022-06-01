# ERNIE 3.0 轻量级模型

 **目录**
   * [模型介绍](#模型介绍)
   * [模型效果](#模型效果)
   * [微调](#微调)
   * [模型压缩](#模型压缩)
       * [环境依赖](#环境依赖)
       * [模型压缩 API 使用](#模型压缩API使用)
       * [压缩效果](#压缩效果)
           * [精度测试](#精度测试)
           * [性能测试](#性能测试)
               * [CPU 性能](#CPU性能)
               * [GPU 性能](#CPU性能)
   * [部署](#部署)
       * [Python 部署](#Python部署)
       * [服务化部署](#服务化部署)
       * [Paddle2ONNX 部署](#Paddle2ONNX部署)
   * [Notebook教程](#Notebook教程)
   * [参考文献](#参考文献)

## 模型介绍

本次开源的模型是在文心大模型ERNIE 3.0 基础上通过**在线蒸馏技术**得到的轻量级模型，模型结构与 ERNIE 2.0 保持一致，相比 ERNIE 2.0 具有更强的中文效果。

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


## 模型效果

本项目开源 **ERNIE 3.0 _Base_** 、**ERNIE 3.0 _Medium_** 、 **ERNIE 3.0 _Mini_** 、 **ERNIE 3.0 _Micro_** 、 **ERNIE 3.0 _Nano_** 五个模型：

- [**ERNIE 3.0-_Base_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_base_zh.pdparams) (_12-layer, 768-hidden, 12-heads_)
- [**ERNIE 3.0-_Medium_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_medium_zh.pdparams) (_6-layer, 768-hidden, 12-heads_)
- [**ERNIE 3.0-_Mini_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_mini_zh.pdparams) (_6-layer, 384-hidden, 12-heads_)
- [**ERNIE 3.0-_Micro_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_micro_zh.pdparams) (_4-layer, 384-hidden, 12-heads_)
- [**ERNIE 3.0-_Nano_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_nano_zh.pdparams) (_4-layer, 312-hidden, 12-heads_)

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
            <td rowspan=6 align=center> 12L768H </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>ERNIE 3.0-Base-zh</b></span>
                        </td>
                        <td style="text-align:center">
                          <span style="font-size:18px"><b>76.05</b></span>
                        </td>
                        <td style="text-align:center">
                          <span style="font-size:18px"><b>75.93</b></span>
                        </td>
                        <td style="text-align:center">
                          <span style="font-size:18px"><b>58.26</b></span>
                        </td>
                        <td style="text-align:center">
                          <span style="font-size:18px"><b>61.56</b></span>
                        </td>
                        <td style="text-align:center">
                          <span style="font-size:18px"><b>83.02</b></span>
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
                          <span style="font-size:18px"><b>84.26</b></span>
                        </td>  
                        <td style="text-align:center">
                          <span style="font-size:18px"><b>77.88</b></span>
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
                          <span style="font-size:18px"><b>88.82</b></span>
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
                                <span style="font-size:18px">Mengzi-BERT-Base</span>
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
                                <span style="font-size:18px">ERNIE-1.0</span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px">74.17</span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px">74.84</span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px">58.91</span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px">62.25</span>
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
                       <td rowspan=5 align=center> 6L768H </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>ERNIE 3.0-Medium-zh</b></span>
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
                          <span style="font-size:18px"><b>60.67</b></span>
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
                                <span style="font-size:18px">69.74</span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px">73.15</span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px">56.62</span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px">59.68</span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px">79.26</span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px">73.15</span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px">75.00</span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px">80.04</span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px">62.26/84.72</span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px">78.26</span>
                        </td>  
                        <td style="text-align:center">
                                <span style="font-size:18px">59.93</span>
                        </td>
                </tr>
                <tr>
                <td style="text-align:center">
                        <span style="font-size:18px">TinyBERT<sub>6</sub>, Chinese</span>
                        </td>
                        <td style="text-align:center">
                          <span style="font-size:18px">69.58</span>
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
                          <span style="font-size:18px">62.63/83.72</span>
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
                          <span style="font-size:18px">60.72</span>
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
                        <span style="font-size:18px">UER/Chinese-RoBERTa (L6-H768)</span>
                        </td>
                        <td style="text-align:center">
                          <span style="font-size:18px">66.67</span>
                        </td>
                        <td style="text-align:center">
                          <span style="font-size:18px">70.13</span>
                        </td>
                        <td style="text-align:center">
                          <span style="font-size:18px">56.41</span>
                        </td>
                        <td style="text-align:center">
                          <span style="font-size:18px">59.79</span>
                        </td>
                        <td style="text-align:center">
                          <span style="font-size:18px">77.38</span>
                        </td>
                        <td style="text-align:center">
                          <span style="font-size:18px">71.86</span>
                        </td>
                        <td style="text-align:center">
                          <span style="font-size:18px">69.41</span>
                        </td>
                        <td style="text-align:center">
                          <span style="font-size:18px">76.73</span>
                        </td>
                        <td style="text-align:center">
                          <span style="font-size:18px">53.22/75.03</span>
                        </td>
                        <td style="text-align:center">
                          <span style="font-size:18px">77.00</span>
                        </td>
                        <td style="text-align:center">
                          <span style="font-size:18px">54.77</span>
                        </td>
                </tr>
                <tr>
                        <td rowspan=1 align=center> 6L384H </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>ERNIE 3.0-Mini-zh</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>66.90</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>71.85</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>55.24</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>54.48</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>77.19</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>73.08</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>71.05</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>79.30</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>58.53/81.97</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>69.71</b></span>
                        </td>  
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>58.60</b></span>
                        </td>
                </tr>
                <tr>
            <td rowspan=1 align=center> 4L384H </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>ERNIE 3.0-Micro-zh</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>64.21</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>71.15</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>55.05</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>53.83</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>74.81</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>70.41</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>69.08</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>76.50</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>53.77/77.82</b></span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>62.26</b></span>
                        </td>  
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>55.53</b></span>
                        </td>
                </tr>
                <tr>
            <td rowspan=1 align=center> 4L312H </td>
                        <td style="text-align:center">
                                <span style="font-size:18px"><b>ERNIE 3.0-Nano-zh</b></span>
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
                                <span style="font-size:18px"><b>68.75</b></span>
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
        <tbody>
</table>
<br />


以下是本项目目录结构及说明：

```shell
.
├── run_seq_cls.py               # 分类任务的微调脚本
├── run_token_cls.py             # 序列标注任务的微调脚本
├── run_qa.py                    # 阅读理解任务的微调脚本
├── compress_seq_cls.py          # 分类任务的压缩脚本
├── compress_token_cls.py        # 序列标注任务的压缩脚本
├── compress_qa.py               # 阅读理解任务的压缩脚本  
├── config.yml                   # 压缩配置文件
├── infer.py                     # 支持 CLUE 分类、CLUE CMRC2018、MSRA_NER 任务的预测脚本
├── deploy                       # 部署目录
│ └── python
│   └── ernie_predictor.py
│   └── infer_cpu.py
│   └── infer_gpu.py
│   └── README.md
│ └── serving
│   └── seq_cls_rpc_client.py  
│   └── seq_cls_service.py  
│   └── seq_cls_config.yml  
│   └── token_cls_rpc_client.py  
│   └── token_cls_service.py  
│   └── token_cls_config.yml
│   └── README.md
│ └── paddle2onnx
│   └── ernie_predictor.py  
│   └── infer.py
│   └── README.md
└── README.md                    # 文档，本文件

```

<a name="微调"></a>

## 微调

```python

from paddlenlp.transformers import *

tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")

# 用于分类任务
seq_cls_model = AutoModelForSequenceClassification.from_pretrained("ernie-3.0-medium-zh")

# 用于序列标注任务
token_cls_model = AutoModelForTokenClassification.from_pretrained("ernie-3.0-base-zh")

# 用于阅读理解任务
qa_model = AutoModelForQuestionAnswering.from_pretrained("ernie-3.0-base-zh")

```

本项目提供了针对分类、序列标注、阅读理解三大场景下的微调使用样例，可分别参考 `run_seq_cls.py` 、`run_token_cls.py`、`run_qa.py` 三个脚本，启动方式如下：

```shell
# 分类任务
python run_seq_cls.py  --task_name tnews --model_name_or_path ernie-3.0-base-zh --do_train

# 序列标注任务
python run_token_cls.py --task_name msra_ner  --model_name_or_path ernie-3.0-medium-zh --do_train

# 阅读理解任务
python run_qa.py --model_name_or_path ernie-3.0-medium-zh --do_train

```

<a name="模型压缩"></a>

## 模型压缩

<a name="环境依赖"></a>

### 环境依赖

使用裁剪功能需要安装 paddleslim 包

```shell
pip install paddleslim
```

<a name="压缩效果"></a>

### 压缩 效果


<a name="模型压缩API使用"></a>

### 模型压缩 API 使用

本项目基于 PaddleNLP 的 Trainer API 发布提供了模型压缩 API。压缩 API 支持用户对 ERNIE、BERT 等Transformers 类下游任务微调模型进行裁剪、量化。用户只需要简单地调用 `compress()` 即可一键启动裁剪和量化，并自动保存压缩后的模型。


可以这样使用压缩 API (示例代码只提供了核心调用，如需跑通完整的例子可参考下方完整样例脚本):

```python

trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer)

output_dir = os.path.join(model_args.model_name_or_path, "compress")

compress_config = CompressConfig(quantization_config=PTQConfig(
        algo_list=['hist', 'mse'], batch_size_list=[4, 8, 16]),
        DynabertConfig(width_mul_ist=[3/4]))

trainer.compress(
    data_args.dataset,
    output_dir,
    pruning=True, # 开启裁剪
    quantization=True, # 开启量化
    compress_config=compress_config)
```
由于压缩 API 基于 Trainer，所以首先需要初始化一个 Trainer 实例，对于模型压缩来说必要传入的参数如下：

- `model`：ERNIE、BERT 等模型，是在 `task_name` 任务中微调后的模型。以分类模型为例，可通过`AutoModelForSequenceClassification.from_pretrained(model_name_or_path)` 来获取
- `data_collator`：三类任务均可使用 PaddleNLP 预定义好的[DataCollator 类](../../paddlenlp/data/data_collator.py)，`data_collator` 可对数据进行 `Pad` 等操作。使用方法参考本项目中代码即可
- `train_dataset`：裁剪训练需要使用的训练集
- `eval_dataset`：裁剪训练使用的评估集，也是量化使用的校准数据
- `tokenizer`：模型`model`对应的 `tokenizer`，可使用 `AutoTokenizer.from_pretrained(model_name_or_path)` 来获取

然后可以直接调用 `compress` 启动压缩，其中 `compress` 的参数释义如下：

- `task_name`：任务名，例如 `tnews`、`msra_ner`、`clue cmrc2018`等
- `output_dir`：裁剪、量化后的模型保存目录
- `pruning`：是否裁剪，默认为`True`
- `quantization`：是否量化，默认为 `True`
- `compress_config`：压缩配置，需要分别传入裁剪和量化的配置实例。目前裁剪和量化分别仅支持`DynabertConfig`和`PTQConfig`类。当默认参数不满足需求时，可通过传入参数对压缩过程进行特殊配置：

其中，`DynabertConfig`中可以传的参数有：
- `width_mult_list`：裁剪宽度保留的比例，对 6 层模型推荐 `3/4` ，对 12 层模型推荐 `2/3`，表示对 `q`、`k`、`v` 以及 `ffn` 权重宽度的保留比例。默认是 `3/4`
- `output_filename_prefix`：裁剪导出模型的文件名前缀，默认是`"float32"`

`PTQConfig`中可以传的参数有：
- `algo_list`：量化策略列表，目前支持 `KL`, `abs_max`, `min_max`, `avg`, `hist`和`mse`，不同的策略计算量化比例因子的方法不同。建议传入多种策略，可批量得到由多种策略产出的多个量化模型，从中选择最优模型。推荐`hist`, `mse`, `KL`，默认是`["hist"]`
- `batch_size_list`：校准样本数，默认是 `[4]`。并非越大越好，也是一个超参数，建议传入多种校准样本数，可从多个量化模型中选择最优模型。
- `input_dir`：待量化模型的目录。如果是 `None`，当不启用裁剪时，表示待量化的模型是 `Trainer` 初始化的模型；当启用裁剪时，表示待量化的模型是裁剪后导出的模型。默认是`None`
- `input_filename_prefix`：待量化模型文件名前缀，默认是 `"float32"`
- `output_filename_prefix`：导出的量化模型文件名后缀，默认是`"int8"`


本项目还提供了压缩 API 在文本分类、序列标注、阅读理解三大场景下的使用样例，可以分别参考 `compress_seq_cls.py` 、`compress_token_cls.py`、`compress_qa.py`，启动方式如下：

```shell
# 文本分类任务
python compress_seq_cls.py --dataset "clue tnews"  --model_name_or_path best_models/TNEWS  --output_dir ./

# 序列标注任务
python compress_token_cls.py --dataset "msra_ner"  --model_name_or_path best_models/MSRA_NER  --output_dir ./

# 阅读理解任务
python compress_seq_cls.py --dataset "clue cmrc2018"  --model_name_or_path best_models/CMRC2018  --output_dir ./
```

测试模型压缩后的精度：

```shell
# 原模型
python infer.py --task_name tnews --model_path best_models/TNEWS/compress/inference/infer --use_trt
# 裁剪后
python infer.py --task_name tnews --model_path best_models/TNEWS/compress/0.75/float --use_trt
# 量化后
python infer.py --task_name tnews --model_path best_models/TNEWS/compress/0.75/hist16/int8 --use_trt --precision int8

```

**压缩 API 使用TIPS：**

1. 压缩 API 提供裁剪和量化两个功能，如果硬件支持量化模型的部署，建议裁剪和量化都选择。目前支持的裁剪策略需要训练，训练时间视下游任务数据量而定，且和微调的训练时间是一个量级。量化则不需要训练，更快，量化的加速比比裁剪更明显，但是单独量化精度下降可能也更多；

2. 裁剪类似蒸馏过程，方便起见，可以直接使用微调时的超参。为了进一步提升精度，可以对 `batch_size`、`learning_rate`、`epoch`、`max_seq_length` 等超参进行 grid search；

3. 模型压缩主要用于推理部署，因此压缩后的模型都是静态图模型，只可用于推理部署，不能再通过 `from_pretrained` 导入继续训练。

<a name="压缩效果"></a>

### 压缩效果

<a name="精度测试"></a>

#### 精度测试

本案例中我们对 ERNIE 3.0-Medium 模型在三类任务上微调后的模型使用压缩 API 进行压缩。压缩后精度如下：

| Model                           | AVG   | AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUEWSC2020 | CSL   | CMRC2018    | MSRA_NER          |
| ------------------------------- | ----- | ----- | ----- | ------- | ----- | ----- | ----------- | ----- | ----------- | ----------------- |
| ERNIE 3.0-Medium                | 74.87 | 75.35 | 57.45 | 60.18   | 81.16 | 77.19 | 80.59       | 81.93 | 66.95/87.15 | 92.65/93.43/93.04 |
| ERNIE 3.0-Medium+FP16           | 74.87 | 75.32 | 57.45 | 60.22   | 81.16 | 77.22 | 80.59       | 81.90 | 66.95/87.16 | 92.65/93.45/93.05 |
| ERNIE 3.0-Medium+裁剪+FP32      | 74.70 | 75.14 | 57.31 | 60.29   | 81.25 | 77.46 | 79.93       | 81.70 | 65.92/86.43 | 93.10/93.43/93.27 |
| ERNIE 3.0-Medium+裁剪+FP16      | 74.71 | 75.21 | 57.27 | 60.29   | 81.24 | 77.56 | 79.93       | 81.73 | 65.89/86.44 | 93.10/93.43/93.27 |
| ERNIE 3.0-Medium+裁剪+量化+INT8 | 74.44 | 75.02 | 57.26 | 60.37   | 81.03 | 77.25 | 77.96       | 81.67 | 66.17/86.55 | 93.17/93.23/93.20 |
| ERNIE 3.0-Medium+量化+INT8      | 74.10 | 74.67 | 56.99 | 59.91   | 81.03 | 75.05 | 78.62       | 81.60 | 66.32/86.82 | 93.10/92.90/92.70 |

**评价指标说明：** 其中 CLUE 分类任务（AFQMC、TNEWS、IFLYTEK、CMNLI、OCNLI、CLUEWSC2020、CSL）的评价指标是 Accuracy，阅读理解任务 CLUE CMRC2018 的评价指标是 EM (Exact Match) / F1-Score，计算平均值时取 EM，序列标注任务 MSRA_NER 的评价指标是 Precision/Recall/F1-Score，计算平均值时取 F1-Score。

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

<a name="CPU性能"></a>

##### CPU 性能

测试环境及说明如上，测试 CPU 性能时，线程数设置为12。

|                            | TNEWS 性能   | TNEWS 精度   | MSRA_NER 性能 | MSRA_NER 精度 | CMRC2018 性能 | CMRC2018 精度 |
| -------------------------- | ------------ | ------------ | ------------- | ------------- | ------------- | ------------- |
| ERNIE 3.0-Medium+FP32      | 311.95(1.0X) | 57.45        | 90.91(1.0x)   | 93.04         | 33.74(1.0x)   | 66.95         |
| ERNIE 3.0-Medium+INT8      | 600.35(1.9x) | 56.57(-0.88) | 141.00(1.6x)  | 92.64(-0.40)  | 56.51(1.7x)   | 66.23(-0.72)  |
| ERNIE 3.0-Medium+裁剪+FP32 | 408.65(1.3x) | 57.31(-0.14) | 122.13(1.3x)  | 93.27(+0.23)  | 48.47(1.4x)   | 65.55(-1.40)  |
| ERNIE 3.0-Medium+裁剪+INT8 | 704.42(2.3x) | 56.69(-0.76) | 215.58(2.4x)  | 92.39(-0.65)  | 75.23(2.2x)   | 63.47(-3.48)  |


三类任务（分类、序列标注、阅读理解）经过相同压缩过程后，加速比达到 2.3 左右。


<a name="GPU性能"></a>

##### GPU 性能

|                            | TNEWS 性能    | TNEWS 精度   | MSRA_NER 性能 | MSRA_NER 精度 | CMRC2018 性能 | CMRC2018 精度 |
| -------------------------- | ------------- | ------------ | ------------- | ------------- | ------------- | ------------- |
| ERNIE 3.0-Medium+FP32      | 1123.85(1.0x) | 57.45        | 366.75(1.0x)  | 93.04         | 146.84(1.0x)  | 66.95         |
| ERNIE 3.0-Medium+FP16      | 2672.41(2.4x) | 57.45(0.00)  | 840.11(2.3x)  | 93.05(0.01)   | 303.43(2.1x)  | 66.95(0.00)   |
| ERNIE 3.0-Medium+INT8      | 3226.26(2.9x) | 56.99(-0.46) | 889.33(2.4x)  | 92.70(-0.34)  | 348.84(2.4x)  | 66.32(-0.63   |
| ERNIE 3.0-Medium+裁剪+FP32 | 1424.01(1.3x) | 57.31(-0.14) | 454.27(1.2x)  | 93.27(+0.23)  | 183.77(1.3x)  | 65.92(-1.03)  |
| ERNIE 3.0-Medium+裁剪+FP16 | 3577.62(3.2x) | 57.27(-0.18) | 1138.77(3.1x) | 93.27(+0.23)  | 445.71(3.0x)  | 65.89(-1.06)  |
| ERNIE 3.0-Medium+裁剪+INT8 | 3635.48(3.2x) | 57.26(-0.19) | 1105.26(3.0x) | 93.20(+0.16)  | 444.27(3.0x)  | 66.17(-0.78)  |


三类任务（分类、序列标注、阅读理解）经过裁剪 + 量化后加速比均达到 3 倍左右，所有任务上平均精度损失可控制在 0.5以内（0.46）。

<a name="部署"></a>

## 部署
我们为ERNIE 3.0提供了多种部署方案，可以满足不同场景下的部署需求，请根据实际情况进行选择。  
<p align="center">
        <img width="700" alt="image" src="https://user-images.githubusercontent.com/30516196/168466069-e8162235-2f06-4a2d-b78f-d9afd437c620.png">
</p>

<a name="Python部署"></a>

### Python 部署

Python部署请参考：[Python部署指南](./deploy/python/README.md)

<a name="服务化部署"></a>

### 服务化部署

服务化部署请参考：[服务化部署指南](./deploy/serving/README.md)

<a name="Paddle2ONNX部署"></a>

### Paddle2ONNX 部署

ONNX 导出及 ONNXRuntime 部署请参考：[ONNX导出及ONNXRuntime部署指南](./deploy/paddle2onnx/README.md)  


<a name="参考文献"></a>


## Notebook教程

- [【快速上手ERNIE 3.0】中文情感分析实战](https://aistudio.baidu.com/aistudio/projectdetail/3955163)

- [【快速上手ERNIE 3.0】法律文本多标签分类实战](https://aistudio.baidu.com/aistudio/projectdetail/3996601)

- [【快速上手ERNIE 3.0】中文语义匹配实战](https://aistudio.baidu.com/aistudio/projectdetail/3986803)

- [【快速上手ERNIE 3.0】MSRA序列标注实战](https://aistudio.baidu.com/aistudio/projectdetail/3989073)

## 参考文献

* Sun Y, Wang S, Feng S, et al. ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation[J]. arXiv preprint arXiv:2107.02137, 2021.

* Su W, Chen X, Feng S, et al. ERNIE-Tiny: A Progressive Distillation Framework for Pretrained Transformer Compression[J]. arXiv preprint arXiv:2106.02241, 2021.

* Wang S, Sun Y, Xiang Y, et al. ERNIE 3.0 Titan: Exploring Larger-scale Knowledge Enhanced Pre-training for Language Understanding and Generation[J]. arXiv preprint arXiv:2112.12731, 2021.
