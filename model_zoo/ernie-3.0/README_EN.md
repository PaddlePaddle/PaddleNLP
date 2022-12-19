# ERNIE 3.0 Lightweight Model

 **Directory**
   * [An Introduction to the Model](#An Introduction to the Model)
       * [Online Distillation Technology](#Online Distillation Technology)
   * [Model Effect](#Model Effect)
   * [Fine-tuning](#Fine-tuning)
   * [Model Compression](#Model Compression)
       * [Environment Dependencies](#Environment Dependencies)
       * [Model Compression API ](#Model Compression API )
       * [Compression Effect](#Compression Effect)
           * [Accuracy Test](#Accuracy Test)
           * [Performance Test](#Performance Test)
               * [CPU Performance](#CPU Performance)
               * [GPU Performance](#CPU Performance)
   * [Speed Up with FasterTokenizer](#Speed Up with FasterTokenizer)
   * [Deployment](#Deployment)
       * [Python Deployment](#Python Deployment)
       * [Service-oriented Deployment](#Service-oriented Deployment)
       * [Paddle2ONNX Deployment](#Paddle2ONNX Deployment)
   * [Notebook Guide](#Notebook Guide)
   * [References](#References)

<a name="An Introduction to the model"></a>

## An Introduction to the model

The open source model is a lightweight model obtained through **online distillation technology** based on Wenxin Big Models (ERNIE 3.0). Compared with ERNIE 2.0, its Chinese effect is stronger while the model structure remains unchanged.

Related technical details can be found in the article ["Analysis of the world's largest Chinese monolithic model Pengcheng - Baidu - Wenxin technical details"](https://www.jiqizhixin.com/articles/2021-12-08-9)

<a name="Online Distillation Technology"></a>

### Online Distillation Technology

In model learning, the online distillation technology periodically passes knowledge signals to several student models for simultaneous training, aiming to produce student models of multiple sizes at once in the distillation phase. Compared with traditional distillation ones, this technique greatly reduces computing power consumption caused by the extra distillation computation of big models and the repetitive knowledge transfer of multiple student models.

Taking advantage of the scale strength of Wenxin Big Models, this novel method ensures the effect and multiple sizes of student models after distillation, as well as allows their application for different performance requirements. In addition, due to the huge gap between the size of Wenxin Big Models and student models, model distillation can be extremely difficult and even prone to failure. For this reason, the introduced helper model is used as a bridge for knowledge transfer. It can solve the problem of the large expression distance between student models and large models, thus promoting the efficiency of distillation.
More technical details can be found in the following articles:
- [ERNIE-Tiny: A Progressive Distillation Framework for Pretrained Transformer Compression](https://arxiv.org/abs/2106.02241)
- [ERNIE 3.0 Titan: Exploring Larger-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/abs/2112.12731)

<p align="center">
        <img width="644" alt="image" src="https://user-images.githubusercontent.com/1371212/168516904-3fff73e0-010d-4bef-adc1-4d7c97a9c6ff.png" title="ERNIE 3.0 Online Distillation">
</p>

<a name="Model Effect"></a>

## Model Effect

The five open source models **ERNIE 3.0 _Base_** 、**ERNIE 3.0 _Medium_** 、 **ERNIE 3.0 _Mini_** 、 **ERNIE 3.0 _Micro_** 、 **ERNIE 3.0 _Nano_** in this project:

- [**ERNIE 3.0-_Base_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_base_zh.pdparams) (_12-layer, 768-hidden, 12-heads_)
- [**ERNIE 3.0-_Medium_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_medium_zh.pdparams) (_6-layer, 768-hidden, 12-heads_)
- [**ERNIE 3.0-_Mini_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_mini_zh.pdparams) (_6-layer, 384-hidden, 12-heads_)
- [**ERNIE 3.0-_Micro_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_micro_zh.pdparams) (_4-layer, 384-hidden, 12-heads_)
- [**ERNIE 3.0-_Nano_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_nano_zh.pdparams) (_4-layer, 312-hidden, 12-heads_)


Below is the **effect-latency graph** of PaddleNLP lightweight Chinese models. The horizontal axis represents the latency (in ms) tested on the IFLYTEK dataset (maximum sequence length is 128), and the vertical axis represents the average accuracy on the 10 tasks of CLUE (including text classification, text matching, natural language inference, pronoun disambiguation, and reading comprehension). Among them, the evaluation metric for the CMRC2018 reading comprehension is Exact Match (EM), while others use Accuracy for evaluation. The higher the model is to **the upper left**, the higher its level of accuracy and performance.

In the figure, the parameters are marked below the model name, and the test environment is shown in the [performance test](#performance test)。

batch_size=32,the effect-latency graph under CPU (num_threads=1 and num_threads=8)：

<table>
    <tr>
        <td><a><img src="https://user-images.githubusercontent.com/26483581/175852121-2798b5c9-d122-4ac0-b4c8-da46b89b5512.png"></a></td>
        <td><a><img src="https://user-images.githubusercontent.com/26483581/175852129-bbe58835-8eec-45d5-a4a9-cc2cf9a3db6a.png"></a></td>
    </tr>
</table>

batch_size=1, the effect-latency graph under CPU (num_threads=1 and num_threads=8)：

<table>
    <tr>
        <td><a><img src="https://user-images.githubusercontent.com/26483581/175852106-658e18e7-705b-4f53-bad0-027281163ae3.png"></a></td>
        <td><a><img src="https://user-images.githubusercontent.com/26483581/175852112-4b89d675-7c95-4d75-84b6-db5a6ea95e2c.png"></a></td>
    </tr>
</table>

batch_size=32/1, FP16 prediction accuracy, the effect-latency graph under GPU (num_threads=1 and num_threads=8)：

<table>
    <tr>
        <td><a><img src="https://user-images.githubusercontent.com/26483581/175854679-3247f42e-8716-4a36-b5c6-9ce4661b36c7.png"></a></td>
        <td><a><img src="https://user-images.githubusercontent.com/26483581/175854670-57878b34-c213-47ac-b620-aaaec082f435.png"></a></td>
    </tr>
</table>

As can be seen from the figures, the performance of ERNIE 3.0 lightweight models in accuracy and performance has been ahead of UER-py, Huawei-Noah, and Chinese models of HFL. When the batch_size equals 1 with FP16 prediction accuracy, wide and shallow models on GPU perform better in terms of inference.

The metrics on the CLUE **validation set** are shown in the following table：

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


The following is the directory structure and description of this project:：

```shell
.
├── run_seq_cls.py               # Fine-tuning scripts for classification tasks
├── run_token_cls.py             # Fine-tuning scripts for sequence labeling
├── run_qa.py                    # Fine-tuning scripts for reading comprehension
├── compress_seq_cls.py          # Minified scripts for classification tasks 
├── compress_token_cls.py        # Minified scripts for sequence labeling
├── compress_qa.py               # Minified scripts for reading comprehension
├── config.yml                   # Minify configuration files
├── infer.py                     # Prediction scripts for CLUE classification, CLUE CMRC2018, MSRA_NER tasks
├── deploy                       # Deployment directory
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

<a name="Fine-tuning"></a>

## Fine-tuning

The pre-trained models released by ERNIE 3.0 cannot be directly used on downstream tasks. They need fine-tuning using data from specific tasks.

ERNIE 3.0 models can be got with one line of code using PaddleNLP. Then you can fine-tune it using your own downstream data to obtain better models in specific tasks.

```python

from paddlenlp.transformers import *

tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")

# For classification tasks
seq_cls_model = AutoModelForSequenceClassification.from_pretrained("ernie-3.0-medium-zh")

# For sequence labeling
token_cls_model = AutoModelForTokenClassification.from_pretrained("ernie-3.0-medium-zh")

# For reading comprehension
qa_model = AutoModelForQuestionAnswering.from_pretrained("ernie-3.0-medium-zh")

```

The project provides fine-tuned sample scripts targeted at classification (including text classification, text matching, natural language inference, pronoun disambiguation, and other tasks), sequence labeling, and reading comprehension.You can refer to the three scripts `run_seq_cls.py` 、`run_token_cls.py`、`run_qa.py`，Start up as follows：

```shell
# classification tasks
# The script supports seven classification tasks in CLUE. Their hyperparameters are not all the same, so the configuration of hyperparameters in classification tasks is configured by config.yml
python run_seq_cls.py  \
    --task_name tnews \
    --model_name_or_path ernie-3.0-medium-zh \
    --do_train

# sequence labeling
python run_token_cls.py \
    --task_name msra_ner  \
    --model_name_or_path ernie-3.0-medium-zh \
    --do_train \
    --num_train_epochs 3 \
    --learning_rate 0.00005 \
    --save_steps 100 \
    --batch_size 32 \
    --max_seq_length 128 \
    --remove_unused_columns False

# reading comprehension
python run_qa.py \
    --model_name_or_path ernie-3.0-medium-zh \
    --do_train \
    --learning_rate 0.00003 \
    --num_train_epochs 8 \
    --batch_size 24 \
    --max_seq_length 512
```

<a name="Model Compression"></a>

## Model Compression

Although ERNIE 3.0 has provided effective 6-layer and 4-layer lightweight models that can be used directly after fine-tuning, models need to be compressed if models require further deployment and release. You can refer to the model compression schemes and API provided here to process fine-tuned models.

<a name="Environment Dependencies"></a>

### Environment Dependencies

Paddleslim package needs to be installed to use cropping

```shell
pip install paddleslim
```

<a name="Model Compression API"></a>

### Model Compression API

This project uses the compression API to crop and quantify models fine-tuned on the task data. After uploading the model and the associated compression hyperparameters (optional, with default options provided), you can start cropping and quantization with the `compress()`line. Compressed models can be saved automatically for deployment.

The core invocation is as follows. You can refer to the full sample script below this directory if you need to run through the complete example:

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
You should first initialize a Trainer instance to use Trainer-based compression API, and call `compress()` to start compression.

Assuming the preceding code is in the compress.py script, you can call it like this:

```shell
python compress.py \
    --dataset   "clue cluewsc2020"   \
    --model_name_or_path best_models/CLUEWSC2020 \
    --output_dir ./compress_models  \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --width_mult_list 0.75 \
    --batch_size_list 4 8 16 \
    --batch_num_list 1 \
```

Some hyperparameters of model compression can be controlled by passing command-line parameters. The hyperparameters that can be passed by the compression API can be checked in the [document](../../docs/compression.md)。

This project provides examples of using compression API in classification (text classification, text matching, natural language inference, pronoun disambiguation, etc.), sequence labeling, and reading comprehension. Refer to `compress_seq_cls.py` 、`compress_token_cls.py`、`compress_qa.py`，then start up as follows：

```shell
# classification tasks
# This script supports seven classification tasks in CLUE. Their hyperparameters are not all the same, so the configuration of hyperparameters in classification tasks is configured by config.yml
python compress_seq_cls.py \
    --dataset "clue tnews"  \
    --model_name_or_path best_models/TNEWS  \
    --output_dir ./

# sequence labeling
python compress_token_cls.py \
    --dataset "msra_ner"  \
    --model_name_or_path best_models/MSRA_NER \
    --output_dir ./ \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 0.00005 \
    --remove_unused_columns False \
    --num_train_epochs 3

# reading comprehension
python compress_qa.py \
    --dataset "clue cmrc2018" \
    --model_name_or_path best_models/CMRC2018  \
    --output_dir ./ \
    --max_answer_length 50 \
    --max_seq_length 512 \
    --learning_rate 0.00003 \
    --num_train_epochs 8 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \

```

One line of code to verify the accuracy of compressed models:

```shell
# primitive model
python infer.py --task_name tnews --model_path best_models/TNEWS/compress/inference/infer --use_trt
# after clipping
python infer.py --task_name tnews --model_path best_models/TNEWS/compress/0.75/float --use_trt
# after quantization
python infer.py --task_name tnews --model_path best_models/TNEWS/compress/0.75/hist16/int8 --use_trt --precision int8

```
The --model_path parameter requires the path and prefix of the static graph model.


<a name="Compression Effect"></a>

### Compression Effect

<a name="Accuracy Test"></a>

#### Accuracy Test

In this case, we use compression API to compress ERNIE 3.0-Medium models fine-tuned on three types of tasks. The compressed accuracy is as follows:

| Model                           | AVG   | AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUEWSC2020 | CSL   | CMRC2018    | MSRA_NER          |
| ------------------------------- | ----- | ----- | ----- | ------- | ----- | ----- | ----------- | ----- | ----------- | ----------------- |
| ERNIE 3.0-Medium                | 74.87 | 75.35 | 57.45 | 60.18   | 81.16 | 77.19 | 80.59       | 81.93 | 66.95/87.15 | 92.65/93.43/93.04 |
| ERNIE 3.0-Medium+FP16           | 74.87 | 75.32 | 57.45 | 60.22   | 81.16 | 77.22 | 80.59       | 81.90 | 66.95/87.16 | 92.65/93.45/93.05 |
| ERNIE 3.0-Medium+clipping+FP32      | 74.70 | 75.14 | 57.31 | 60.29   | 81.25 | 77.46 | 79.93       | 81.70 | 65.92/86.43 | 93.10/93.43/93.27 |
| ERNIE 3.0-Medium+clipping+FP16      | 74.71 | 75.21 | 57.27 | 60.29   | 81.24 | 77.56 | 79.93       | 81.73 | 65.89/86.44 | 93.10/93.43/93.27 |
| ERNIE 3.0-Medium+clipping+quantization+INT8 | 74.44 | 75.02 | 57.26 | 60.37   | 81.03 | 77.25 | 77.96       | 81.67 | 66.17/86.55 | 93.17/93.23/93.20 |
| ERNIE 3.0-Medium+quantization+INT8      | 74.10 | 74.67 | 56.99 | 59.91   | 81.03 | 75.05 | 78.62       | 81.60 | 66.32/86.82 | 93.10/92.90/92.70 |

**Evaluation indicators：** the indicator of CLUE classification tasks (AFQMC semantic similarity, TNEWS text classification, IFLYTEK long text classification, CMNLI natural language inference, OCNLI natural language inference, CLUEWSC2020 pronoun disambiguation, CSL paper keyword identification) is Accuracy; The indicator of the reading comprehension task CLUE CMRC2018 is EM (Exact Match)/F1-Score, and EM is used to calculate the average; The indicator of the sequence labeling task MSRA_NER is Precision/Recall/F1-Score, and F1-Score is used to calculate the average.
According to the table, the accuracy of `ERNIE 3.0-Medium` models decreases by 0.46 on average with a decrease of 0.17 after clipping and 0.77 after quantization.

<a name="Performance Testing"></a>

#### Performance Testing

The configuration is as follows：

1. Datasets：TNEWS（text classification）、MSRA_NER（sequence labeling）、CLUE CMRC2018（reading comprehension）

2. Computer card：T4、CUDA11.2、CuDNN8.2

3. CPU：Intel(R) Xeon(R) Gold 6271C CPU

4. PaddlePaddle version：2.3

5. PaddleNLP verison：2.3

6. The unit of performance data is QPS. QPS test method: fix batch size to 32, test run time total_time, and calculate QPS = total_samples/total_time

7. The unit of accuracy data: Accuracy for text classification, F1-Score for sequence labeling, and EM (Exact Match) for reading comprehension.

<a name="CPU performance"></a>

##### CPU performance

The testing environment is as above. The number of threads is set to 12 when testing CPU performance.

|                            | TNEWS Performance   | TNEWS Accuracy   | MSRA_NER Performance | MSRA_NER Accuracy | CMRC2018 Performance | CMRC2018 Accuracy |
| -------------------------- | ------------ | ------------ | ------------- | ------------- | ------------- | ------------- |
| ERNIE 3.0-Medium+FP32      | 311.95(1.0X) | 57.45        | 90.91(1.0x)   | 93.04         | 33.74(1.0x)   | 66.95         |
| ERNIE 3.0-Medium+INT8      | 600.35(1.9x) | 56.57(-0.88) | 141.00(1.6x)  | 92.64(-0.40)  | 56.51(1.7x)   | 66.23(-0.72)  |
| ERNIE 3.0-Medium+裁剪+FP32 | 408.65(1.3x) | 57.31(-0.14) | 122.13(1.3x)  | 93.27(+0.23)  | 48.47(1.4x)   | 65.55(-1.40)  |
| ERNIE 3.0-Medium+裁剪+INT8 | 704.42(2.3x) | 56.69(-0.76) | 215.58(2.4x)  | 92.39(-0.65)  | 75.23(2.2x)   | 63.47(-3.48)  |


After the same compression, the speedup ratio of three types of tasks (classification, sequence labeling, and reading comprehension) reaches about 2.3.


<a name="GPU performance"></a>

#####  GPU performance

|                            | TNEWS Performance    | TNEWS Accuracy   | MSRA_NER Performance | MSRA_NER Accuracy | CMRC2018 Performance | CMRC2018 Accuracy |
| -------------------------- | ------------- | ------------ | ------------- | ------------- | ------------- | ------------- |
| ERNIE 3.0-Medium+FP32      | 1123.85(1.0x) | 57.45        | 366.75(1.0x)  | 93.04         | 146.84(1.0x)  | 66.95         |
| ERNIE 3.0-Medium+FP16      | 2672.41(2.4x) | 57.45(0.00)  | 840.11(2.3x)  | 93.05(0.01)   | 303.43(2.1x)  | 66.95(0.00)   |
| ERNIE 3.0-Medium+INT8      | 3226.26(2.9x) | 56.99(-0.46) | 889.33(2.4x)  | 92.70(-0.34)  | 348.84(2.4x)  | 66.32(-0.63   |
| ERNIE 3.0-Medium+clipping+FP32 | 1424.01(1.3x) | 57.31(-0.14) | 454.27(1.2x)  | 93.27(+0.23)  | 183.77(1.3x)  | 65.92(-1.03)  |
| ERNIE 3.0-Medium+clipping+FP16 | 3577.62(3.2x) | 57.27(-0.18) | 1138.77(3.1x) | 93.27(+0.23)  | 445.71(3.0x)  | 65.89(-1.06)  |
| ERNIE 3.0-Medium+clipping+INT8 | 3635.48(3.2x) | 57.26(-0.19) | 1105.26(3.0x) | 93.20(+0.16)  | 444.27(3.0x)  | 66.17(-0.78)  |


After clipping and quantization, the speedup ratio of the three types of tasks (classification, sequence labeling, and reading comprehension) reaches about 3, and the average accuracy loss of all tasks can be controlled within 0.5 (0.46).

<a name="Speed Up with FasterTokenizer"></a>

### Speed Up with FasterTokenizer

PaddlePaddle’s FasterTokenizer is a text processing operator library with high speed. It integrates with Google's LinMaxMatch algorithm that was released in late 2021. This algorithm introduces Aho-Corasick to optimize the time complexity of WordPiece from O(N2) to O(N), and it has been widely used in Google Search. FasterTokenizer takes the lead in speed. The larger the batch_size, the faster it is. For example, with a batch_size of 64, FasterTokenizer Tokenizes words 28 times faster than HuggingFace.

After clipping and quantization of the ERNIE 3.0 lightweight model, when the number of word segmentation threads is set to 4, FasterTokenizer can improve its performance by 2.39 times in IFLYTEK (long text classification dataset with a maximum sequence length of 128) in the NVIDIA Tesla T4 environment. Compared with BERT-Base, the model’s performance is improved by 7.09 times. In the condition of Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz and 8 threads, the performance is improved by 1.27 times, 5.13 times higher than BERT-Base. The speedup effect is demonstrated in the following chart:

<table>
    <tr>
        <td><a><img src="https://user-images.githubusercontent.com/26483581/175452331-bc5ff646-90ee-4377-85a5-d5b073a8e7f9.png"></a></td>
        <td><a><img src="https://user-images.githubusercontent.com/26483581/175452337-e0eff0d3-ed5f-42e7-b06b-caad61f37978.png"></a></td>
    </tr>
</table>

It is simple to use FasterTokenizer. After installing the faster_tokenizer package, you just need to pass `use_fast=True` during tokenizer instantiation. Currently, models like BERT, ERNIE, and TinyBERT are supported on Linux.

Command to install the fast_tokenizer package:

```shell
pip install fast-tokenizer-python
```

Call `fast_tokenizer.set_thread_num` interface to set thread numbers:


```python
# word segmentation threads is set to 4
import fast_tokenizer
fast_tokenizer.set_thread_num(4)
```

Pass one parameter `use_fast=True` to call `from_pretrained`：

```python
from paddlenlp.transformers import AutoTokenizer
AutoTokenizer.from_pretrained("ernie-3.0-medium-zh", use_fast=True)
```

<a name="Deployment"></a>

## Deployment
We provide a variety of deployment options for ERNIE 3.0 to meet various deployment requirements. Please select according to actual situations.
<p align="center">
        <img width="700" alt="image" src="https://user-images.githubusercontent.com/26483581/175260618-610a160c-270c-469a-842c-96871243c4ed.png">
</p>

<a name="Python Deployment"></a>

### Python Deployment

For Python deplyment, please refer to: [Python Deployment Guide](./deploy/python/README.md)

<a name="Service-oriented Deployment"></a>

### Service-oriented Deployment

- [Triton Inference Server Service-oriented Deployment Guide](./deploy/triton/README.md)
- [Paddle Serving Service-oriented Deployment Guide](./deploy/serving/README.md)

<a name="Paddle2ONNX Deployment"></a>

### Paddle2ONNX Deployment

For ONNX Export and ONNXRuntime deployment,please refer to [ONNX Export and ONNXRuntime Deployment Guide](./deploy/paddle2onnx/README.md)


### Paddle Lite Mobile deployment

Coming soon!


<a name="References"></a>


## Notebook Guide

- [【快速上手ERNIE 3.0】中文情感分析实战](https://aistudio.baidu.com/aistudio/projectdetail/3955163)

- [【快速上手ERNIE 3.0】法律文本多标签分类实战](https://aistudio.baidu.com/aistudio/projectdetail/3996601)

- [【快速上手ERNIE 3.0】中文语义匹配实战](https://aistudio.baidu.com/aistudio/projectdetail/3986803)

- [【快速上手ERNIE 3.0】MSRA序列标注实战](https://aistudio.baidu.com/aistudio/projectdetail/3989073)

- [【快速上手ERNIE 3.0】机器阅读理解实战](https://aistudio.baidu.com/aistudio/projectdetail/2017189)

- [【快速上手ERNIE 3.0】对话意图识别](https://aistudio.baidu.com/aistudio/projectdetail/2017202?contributionType=1)


## References

* Sun Y, Wang S, Feng S, et al. ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation[J]. arXiv preprint arXiv:2107.02137, 2021.

* Su W, Chen X, Feng S, et al. ERNIE-Tiny: A Progressive Distillation Framework for Pretrained Transformer Compression[J]. arXiv preprint arXiv:2106.02241, 2021.

* Wang S, Sun Y, Xiang Y, et al. ERNIE 3.0 Titan: Exploring Larger-scale Knowledge Enhanced Pre-training for Language Understanding and Generation[J]. arXiv preprint arXiv:2112.12731, 2021.
