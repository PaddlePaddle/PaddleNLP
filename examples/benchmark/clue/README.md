# CLUE Benchmark

**目录**
   * [CLUE 评测结果](#CLUE评测结果)
   * [一键复现模型效果](#一键复现模型效果)
       * [启动 CLUE 分类任务](#启动CLUE分类任务)
           * [使用 Trainer 启动 CLUE 分类任务](#使用Trainer启动CLUE分类任务)
       * [启动 CLUE 阅读理解任务](#启动CLUE阅读理解任务)
       * [批量启动 Grid Search](#批量启动GridSearch)
           * [环境依赖](#环境依赖)
           * [一键启动方法](#一键启动方法)
           * [Grid Search 脚本说明](#GridSearch脚本说明)
   * [参加 CLUE 竞赛](#参加CLUE竞赛)
       * [分类任务](#分类任务)
       * [阅读理解任务](#阅读理解任务)

[CLUE](https://www.cluebenchmarks.com/) 自成立以来发布了多项 NLP 评测基准，包括分类榜单，阅读理解榜单和自然语言推断榜单等，在学术界、工业界产生了深远影响。是目前应用最广泛的中文语言测评指标之一。详细可参考 [CLUE论文](https://arxiv.org/abs/2004.05986)。

本项目基于 PaddlePaddle 在 CLUE 数据集上对领先的开源预训练模型模型进行了充分评测，为开发者在预训练模型选择上提供参考，同时开发者基于本项目可以轻松一键复现模型效果，也可以参加 CLUE 竞赛取得好成绩。


<a name="CLUE评测结果"></a>

## CLUE 评测结果

使用多种**中文**预训练模型微调在 CLUE 的各验证集上有如下结果：

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
            <td rowspan=2 align=center> 24L1024H </td>
            <td style="text-align:center">
                <span style="font-size:18px">ERNIE 2.0-Large-zh</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>77.03</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>76.41</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>59.67</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>62.29</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">83.82</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>79.69</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">89.14</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>84.10</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>71.48/90.35</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">85.52</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>78.12</b></span>
            </td>
        </tr>
        <tr>
            <td style="text-align:center">
                <span style="font-size:18px">HFL/RoBERTa-wwm-ext-large</span>
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
                <span style="font-size:18px"><b>83.88</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">78.81</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>90.79</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">83.67</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">70.58/89.82</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>85.72</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">75.26</span>
            </td>
        </tr>
        <tr>
            <td rowspan=1 align=center> 20L1024H </td>
            <td style="text-align:center">
                <span style="font-size:18px">ERNIE 3.0-Xbase-zh</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>78.71</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>76.85</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>59.89</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>62.41</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>84.76</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>82.51</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>89.80</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>84.47</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>75.49/92.67</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>86.36</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>84.59</b></span>
            </td>
        </tr>
        <tr>
            <td rowspan=8 align=center> 12L768H </td>
            <td style="text-align:center">
                <span style="font-size:18px">
                    <a href="https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_base_zh.pdparams">
                        ERNIE 3.0-Base-zh
                    </a>
                </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>76.05</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>75.93</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">58.26</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">61.56</span>
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
                <span style="font-size:18px"><b>82.83</b></span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px"><b>71.82/90.38</b></span>
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
                <span style="font-size:18px">ERNIE 2.0-Base-zh</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">74.95</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">76.25</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">58.53</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">61.72</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">83.07</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">78.81</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">84.21</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">82.77</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px">68.22/88.71</span>
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
                <span style="font-size:18px">HFL/RoBERTa-wwm-ext</span>
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

AFQMC（语义相似度）、TNEWS（文本分类）、IFLYTEK（长文本分类）、CMNLI（自然语言推理）、OCNLI（自然语言推理）、CLUEWSC2020（代词消歧）、CSL（论文关键词识别）、CHID（成语阅读理解填空） 和 C<sup>3</sup>（中文多选阅读理解） 任务使用的评估指标均是 Accuracy。CMRC2018（阅读理解） 的评估指标是 EM (Exact Match)/F1，计算每个模型效果的平均值时，取 EM 为最终指标。

其中前 7 项属于分类任务，后面 3 项属于阅读理解任务，这两种任务的训练过程在下面将会分开介绍。

**NOTE：具体评测方式如下**
1. 以上所有任务均基于 Grid Search 方式进行超参寻优。分类任务训练每间隔 100 steps 评估验证集效果，阅读理解任务每隔一个 epoch 评估验证集效果，取验证集最优效果作为表格中的汇报指标。

2. 分类任务 Grid Search 超参范围: batch_size: 16, 32, 64; learning rates: 1e-5, 2e-5, 3e-5, 5e-5；因为 CLUEWSC2020 数据集较小，所以模型在该数据集上的效果对 batch_size 较敏感，所以对 CLUEWSC2020 评测时额外增加了 batch_size = 8 的超参搜索； 因为CLUEWSC2020 和 IFLYTEK 数据集对 dropout 概率值较为敏感，所以对 CLUEWSC2020 和 IFLYTEK 数据集评测时额外增加了 dropout_prob = 0.0 的超参搜索。

3. 阅读理解任务 Grid Search 超参范围：batch_size: 24, 32; learning rates: 1e-5, 2e-5, 3e-5。阅读理解任务均使用多卡训练，其中 Grid Search 中的 batch_size 是指多张卡上的 batch_size 总和。

4. 以上每个下游任务的固定超参配置如下表所示：

| TASK              | AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUEWSC2020 | CSL  | CMRC2018 | CHID | C<sup>3</sup> |
| ----------------- | ----- | ----- | ------- | ----- | ----- | ----------- | ---- | -------- | ---- | ------------- |
| epoch             | 3     | 3     | 3       | 2     | 5     | 50          | 5    | 2        | 3    | 8             |
| max_seq_length    | 128   | 128   | 128     | 128   | 128   | 128         | 256  | 512      | 64   | 512           |
| warmup_proportion | 0.1   | 0.1   | 0.1     | 0.1   | 0.1   | 0.1         | 0.1  | 0.1      | 0.06 | 0.1           |
| num_cards         | 1     | 1     | 1       | 1     | 1     | 1           | 1    | 2        | 4    | 4             |

不同预训练模型在下游任务上做 Grid Search 之后的最优超参（learning_rate、batch_size）如下：

| Model                            | AFQMC   | TNEWS   | IFLYTEK | CMNLI    | OCNLI    | CLUEWSC2020 | CSL     | CMRC2018 | CHID    | C<sup>3</sup> |
| -------------------------------- | ------- | ------- | ------- | -------- | -------- | ----------- | ------- | -------- | ------- | ------------- |
| ERNIE 3.0-Xbase-zh               | 2e-5,16 | 3e-5,32 | 3e-5,32 | 3e-5,64  | 3e-5,64  | 2e-5,32     | 1e-5,16 | 3e-5,24  | 2e-5,24 | 3e-5,24       |
| ERNIE 2.0-Large-zh               | 1e-5,32 | 3e-5,64 | 3e-5,32 | 2e-5,32  | 1e-5,16  | 3e-5,32     | 1e-5,64 | 2e-5,24  | 2e-5,24 | 3e-5,32       |
| HFL/RoBERTa-wwm-ext-large        | 1e-5,32 | 3e-5,32 | 2e-5,32 | 1e-5,16  | 1e-5,16  | 2e-5,16     | 2e-5,16 | 3e-5,32  | 1e-5,24 | 2e-5,24       |
| ERNIE 3.0-Base-zh                | 3e-5,16 | 3e-5,32 | 5e-5,32 | 3e-5,32  | 2e-5,64  | 2e-5,16     | 2e-5,32 | 2e-5,24  | 3e-5,24 | 3e-5,32       |
| ERNIE-Gram-zh                    | 1e-5,16 | 5e-5,16 | 5e-5,16 | 2e-5,32  | 2e-5,64  | 3e-5,16     | 3e-5,64 | 3e-5,32  | 2e-5,24 | 2e-5,24       |
| ERNIE 2.0-Base-zh                | 3e-5,64 | 3e-5,64 | 5e-5,16 | 5e-5,64  | 5e-5,32  | 5e-5,16     | 2e-5,16 | 2e-5,32  | 3e-5,24 | 3e-5,32       |
| Langboat/Mengzi-Bert-Base        | 3e-5,32 | 5e-5,32 | 5e-5,16 | 2e-5,16  | 2e-5,16  | 3e-5,8      | 1e-5,16 | 3e-5,24  | 3e-5,24 | 2e-5,32       |
| ERNIE 1.0-Base-zh                | 3e-5,16 | 3e-5,32 | 5e-5,16 | 5e-5,32  | 3e-5,16  | 2e-5,8      | 2e-5,16 | 3e-5,32  | 3e-5,24 | 3e-5,24       |
| HFL/RoBERTa-wwm-ext              | 3e-5,32 | 3e-5,64 | 5e-5,16 | 3e-5,32  | 2e-5,32  | 3e-5,32     | 2e-5,32 | 3e-5,32  | 2e-5,32 | 3e-5,24       |
| BERT-Base-Chinese                | 2e-5,16 | 5e-5,16 | 5e-5,16 | 5e-5,64  | 3e-5,16  | 3e-5,16     | 1e-5,16 | 3e-5,24  | 2e-5,32 | 3e-5,24       |
| UER/Chinese-RoBERTa-Base         | 2e-5,16 | 5e-5,16 | 5e-5,16 | 2e-5,16  | 3e-5,16  | 3e-5,8      | 2e-5,16 | 3e-5,24  | 3e-5,32 | 3e-5,32       |
| UER/Chinese-RoBERTa-Medium       | 3e-5,32 | 5e-5,64 | 5e-5,16 | 5e-5,32  | 3e-5,32  | 3e-5,16     | 5e-5,32 | 3e-5,24  | 3e-5,24 | 3e-5,32       |
| ERNIE 3.0-Medium-zh              | 3e-5,32 | 3e-5,64 | 5e-5,32 | 2e-5,32  | 1e-5,64  | 3e-5,16     | 2e-5,32 | 3e-5,24  | 2e-5,24 | 1e-5,24       |
| TinyBERT<sub>6</sub>, Chinese    | 1e-5,16 | 3e-5,32 | 5e-5,16 | 5e-5,32  | 3e-5,64  | 3e-5,16     | 3e-5,16 | 3e-5,32  | 3e-5,24 | 2e-5,24       |
| RoFormerV2 Small                 | 5e-5,16 | 2e-5,16 | 5e-5,16 | 5e-5,32  | 2e-5,16  | 3e-5,8      | 3e-5,16 | 3e-5,24  | 3e-5,24 | 3e-5,24       |
| HLF/RBT6, Chinese                | 3e-5,16 | 5e-5,16 | 5e-5,16 | 5e-5,64  | 3e-5,16  | 3e-5,8      | 5e-5,64 | 2e-5,24  | 3e-5,32 | 2e-5,32       |
| UER/Chinese-RoBERTa-L6-H768      | 2e-5,16 | 3e-5,16 | 5e-5,16 | 5e-5,16  | 5e-5,32  | 2e-5,32     | 3e-5,16 | 3e-5,32  | 3e-5,24 | 3e-5,24       |
| ERNIE 3.0-Mini-zh                | 5e-5,64 | 5e-5,64 | 5e-5,16 | 5e-5,32  | 2e-5,16  | 2e-5,8      | 2e-5,16 | 3e-5,24  | 3e-5,24 | 3e-5,24       |
| HFL/RBT4, Chinese                | 5e-5,16 | 5e-5,16 | 5e-5,16 | 5e-5,16  | 2e-5,16  | 2e-5,8      | 2e-5,16 | 3e-5,32  | 3e-5,24 | 3e-5,32       |
| UER/Chinese-RoBERTa-Small        | 2e-5,32 | 5e-5,32 | 5e-5,16 | 5e-5,16  | 5e-5,16  | 2e-5,64     | 5e-5,32 | 3e-5,24  | 3e-5,24 | 3e-5,24       |
| ERNIE 3.0-Micro-zh               | 3e-5,16 | 5e-5,32 | 5e-5,16 | 5e-5,16  | 2e-5,32  | 5e-5,16     | 3e-5,64 | 3e-5,24  | 3e-5,32 | 3e-5,24       |
| ERNIE 3.0-Nano-zh                | 2e-5,32 | 5e-5,16 | 5e-5,16 | 5e-5,16  | 3e-5,16  | 1e-5,8      | 3e-5,32 | 3e-5,24  | 3e-5,24 | 2e-5,24       |
| TinyBERT<sub>4</sub>, Chinese    | 3e-5,32 | 5e-5,16 | 5e-5,16 | 5e-5,16  | 3e-5,16  | 1e-5,16     | 5e-5,16 | 3e-5,24  | 3e-5,24 | 2e-5,24       |
| UER/Chinese-RoBERTa-Mini         | 3e-5,16 | 5e-5,16 | 5e-5,16 | 5e-5,16  | 5e-5,32  | 3e-5,8      | 5e-5,32 | 3e-5,24  | 3e-5,32 | 3e-5,32       |
| HFL/RBTL3, Chinese               | 5e-5,32 | 5e-5,16 | 5e-5,16 | 5e-5,32  | 2e-5,16  | 5e-5,8      | 2e-5,16 | 3e-5,24  | 2e-5,24 | 3e-5,24       |
| HFL/RBT3, Chinese                | 5e-5,64 | 5e-5,32 | 5e-5,16 | 5e-5,16  | 2e-5,16  | 3e-5,16     | 5e-5,16 | 3e-5,32  | 3e-5,24 | 3e-5,32       |
| UER/Chinese-RoBERTa-Tiny         | 5e-5,64 | 5e-5,16 | 5e-5,16 | 5e-5,16  | 5e-5,16  | 5e-5,8      | 5e-5,16 | 3e-5,24  | 3e-5,24 | 3e-5,24       |

其中，`ERNIE 3.0-Base-zh`、`ERNIE 3.0-Medium-zh`、`ERNIE-Gram-zh`、`ERNIE 1.0-Base-zh`、`ERNIE 3.0-Mini-zh`、`ERNIE 3.0-Micro-zh`、`ERNIE 3.0-Nano-zh` 、`HFL/RBT3, Chinese`、`HFL/RBTL3, Chinese`、`HFL/RBT6, Chinese`、`TinyBERT<sub>4</sub>, Chinese`、`UER/Chinese-RoBERTa-Base`、`UER/Chinese-RoBERTa-Mini`、`UER/Chinese-RoBERTa-Small` 在 CLUEWSC2020 处的 dropout_prob 为 0.0，`ERNIE 3.0-Base-zh`、`HLF/RBT6, Chinese`、`Langboat/Mengzi-BERT-Base`、`ERNIE-Gram-zh`、`ERNIE 1.0-Base-zh` 、`TinyBERT6, Chinese`、`UER/Chinese-RoBERTa-L6-H768`、`ERNIE 3.0-Mini-zh`、`ERNIE 3.0-Micro-zh`、`ERNIE 3.0-Nano-zh`、`HFL/RBT3, Chinese`、`HFL/RBT4, Chinese`、`HFL/RBT6, Chinese`、`TinyBERT<sub>4</sub>, Chinese`、`UER/Chinese-RoBERTa-Medium`、`UER/Chinese-RoBERTa-Base`、`UER/Chinese-RoBERTa-Mini`、`UER/Chinese-RoBERTa-Tiny`、`UER/Chinese-RoBERTa-Small` 在 IFLYTEK 处的 dropout_prob 为 0.0。

<a name="一键复现模型效果"></a>

## 一键复现模型效果

这一节将会对分类、阅读理解任务分别展示如何一键复现本文的评测结果。

<a name="启动CLUE分类任务"></a>

### 启动 CLUE 分类任务
以 CLUE 的 TNEWS 任务为例，启动 CLUE 任务进行 Fine-tuning 的方式如下：

```shell
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=TNEWS
export LR=3e-5
export BS=32
export EPOCH=6
export MAX_SEQ_LEN=128
export MODEL_PATH=ernie-3.0-medium-zh

cd classification
mkdir ernie-3.0-medium-zh
python -u ./run_clue_classifier.py \
    --model_name_or_path ${MODEL_PATH} \
    --task_name ${TASK_NAME} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --batch_size ${BS}   \
    --learning_rate ${LR} \
    --num_train_epochs ${EPOCH} \
    --logging_steps 100 \
    --seed 42  \
    --save_steps  100 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-8 \
    --output_dir ${MODEL_PATH}/models/${TASK_NAME}/${LR}_${BS}/ \
    --device gpu  \
    --dropout 0.1 \
    --gradient_accumulation_steps 1 \
    --save_best_model True \
    --do_train \

```

另外，如需评估，传入参数 `--do_eval` 即可，如果只对读入的 checkpoint 进行评估不训练，则不需传入 `--do_train`。

其中参数释义如下：

- `model_name_or_path` 指示了 Fine-tuning 使用的具体预训练模型，可以是 PaddleNLP 提供的预训练模型，可以选择[Transformer预训练模型汇总](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer) 中相对应的中文预训练权重。注意 CLUE 任务应选择中文预训练权重。
- `task_name` 表示 Fine-tuning 的分类任务，当前支持 AFQMC、TNEWS、IFLYTEK、OCNLI、CMNLI、CSL、CLUEWSC2020。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于 learning rate scheduler 产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `save_best_model` 是否保存在评估集上效果最好的模型，默认为 True
- `output_dir` 表示模型保存路径。
- `device` 表示训练使用的设备, 'gpu' 表示使用GPU, 'xpu' 表示使用百度昆仑卡, 'cpu' 表示使用 CPU。

Fine-tuning 过程将按照 `logging_steps` 和 `save_steps` 的设置打印出如下日志：

```
global step 100/20010, epoch: 0, batch: 99, rank_id: 0, loss: 2.734340, lr: 0.0000014993, speed: 8.7969 step/s
eval loss: 2.720359, acc: 0.0827, eval done total : 25.712125062942505 s
global step 200/20010, epoch: 0, batch: 199, rank_id: 0, loss: 2.608563, lr: 0.0000029985, speed: 2.5921 step/s
eval loss: 2.652753, acc: 0.0945, eval done total : 25.64827537536621 s
global step 300/20010, epoch: 0, batch: 299, rank_id: 0, loss: 2.555283, lr: 0.0000044978, speed: 2.6032 step/s
eval loss: 2.572999, acc: 0.112, eval done total : 25.67190170288086 s
global step 400/20010, epoch: 0, batch: 399, rank_id: 0, loss: 2.631579, lr: 0.0000059970, speed: 2.6238 step/s
eval loss: 2.476962, acc: 0.1697, eval done total : 25.794789791107178 s
```

<a name="使用Trainer启动CLUE分类任务"></a>

#### 使用 Trainer 启动 CLUE 分类任务
PaddleNLP 提供了 Trainer API，本示例新增了`run_clue_classifier_trainer.py`脚本供用户使用。

```
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=TNEWS
export LR=3e-5
export BS=32
export EPOCH=6
export MAX_SEQ_LEN=128
export MODEL_PATH=ernie-3.0-medium-zh

cd classification
mkdir ernie-3.0-medium-zh

python -u ./run_clue_classifier_trainer.py \
    --model_name_or_path ${MODEL_PATH} \
    --dataset "clue ${TASK_NAME}" \
    --max_seq_length ${MAX_SEQ_LEN} \
    --per_device_train_batch_size ${BS}   \
    --per_device_eval_batch_size ${BS}   \
    --learning_rate ${LR} \
    --num_train_epochs ${EPOCH} \
    --logging_steps 100 \
    --seed 42  \
    --save_steps 100 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-8 \
    --output_dir ${MODEL_PATH}/models/${TASK_NAME}/${LR}_${BS}/ \
    --device gpu  \
    --do_train \
    --do_eval \
    --metric_for_best_model "eval_accuracy" \
    --load_best_model_at_end \
    --save_total_limit 3 \
```
大部分参数含义如上文所述，这里简要介绍一些新参数:
- `dataset`, 同上文`task_name`，此处为小写字母。表示 Fine-tuning 的分类任务，当前支持 afamc、tnews、iflytek、ocnli、cmnli、csl、cluewsc2020。
- `per_device_train_batch_size` 同上文`batch_size`。训练时，每次迭代**每张卡**上的样本数目。
- `per_device_eval_batch_size` 同上文`batch_size`。评估时，每次迭代**每张卡**上的样本数目。
- `warmup_ratio` 同上文`warmup_proportion`，warmup步数占总步数的比例。
- `metric_for_best_model` 评估时，最优评估指标。
- `load_best_model_at_end` 训练结束时，时候加载评估结果最好的 ckpt。
- `save_total_limit` 保存的ckpt数量的最大限制

<a name="启动CLUE阅读理解任务"></a>

### 启动 CLUE 阅读理解任务
以 CLUE 的 C<sup>3</sup> 任务为例，多卡启动 CLUE 任务进行 Fine-tuning 的方式如下：

```shell

cd mrc

MODEL_PATH=ernie-3.0-medium-zh
BATCH_SIZE=6
LR=2e-5

python -m paddle.distributed.launch --gpus "0,1,2,3" run_c3.py \
    --model_name_or_path ${MODEL_PATH} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --max_seq_length 512 \
    --num_train_epochs 8 \
    --do_train \
    --warmup_proportion 0.1 \
    --gradient_accumulation_steps 3 \

```
需要注意的是，如果显存无法容纳所传入的 `batch_size`，可以通过传入 `gradient_accumulation_steps` 参数来模拟该 `batch_size`。

<a name="批量启动GridSearch"></a>

### 批量启动 Grid Search

<a name="环境依赖"></a>

#### 环境依赖

Grid Search 需要在 GPU 环境下进行，需要注意的是 C<sup>3</sup> 任务需要显存大于 16 GB，最好是在显存 32 GB的环境下启动。

Grid Search 中的 GPU 调度需要依赖 pynvml 库，pynvml 库提供了 GPU 管理的 Python 接口。可启动以下命令进行安装 pynvml：

```shell
pip install pynvml
```

<a name="一键启动方法"></a>

#### 一键启动方法

运行下面一句命令即可启动 Grid Search 任务。前期需要注意数据集是否正常下载，否则训练任务不会正式启动。
脚本默认不保存模型，如需保存每个超参数下最好的模型，需要修改 Python 脚本中的 `--save_best_models` 参数为 True。

```shell
cd grid_search_tools

# 这里 ernie-3.0-base-zh 是模型名，也可以传用户自定义的模型目录
# 自定义的模型目录需要有 model_config.json, model_state.pdparams, tokenizer_config.json 和 vocab.txt 四个文件
python grid_seach.py ernie-3.0-base-zh

```

确认模型所有任务训练完成后，可以调用脚本 `extract_result.sh` 一键抽取 Grid Search 结果，打印出每个任务的最佳结果和对应的超参数，例如：

```shell
bash extract_result.sh ernie-3.0-base-zh
```
```text
AFQMC	TNEWS	IFLYTEK	CMNLI	OCNLI	CLUEWSC2020	CSL	CMRC2018	CHID	C3
75.93	58.26	61.56	83.02	80.10	86.18	82.63	70.71/90.41	84.26	77.88
====================================================================
Best hyper-parameters list:
====================================================================
TASK	result	(lr, batch_size, dropout_p)
AFQMC	75.93	(3e-05,16,0.1)
TNEWS	58.26	(3e-05,32,0.1)
IFLYTEK	61.56	(5e-05,32,0.0)
CMNLI	83.02	(3e-05,32,0.1)
OCNLI	80.10	(2e-05,64,0.1)
CLUEWSC2020	86.18	(2e-05,16,0.0)
CSL	82.63	(2e-05,32,0.1)
CMRC2018	70.71/90.41	(2e-05,24,0.1)
CHID	84.26	(3e-05,24,0.1)
C3	77.88	(3e-05,32,0.1)
```

另外，如遇意外情况（如机器重启）导致训练中断，可以直接再次启动 `grid_search.py` 脚本，之前已完成（输出完整日志）的任务则会直接跳过。

<a name="GridSearch脚本说明"></a>

#### Grid Search 脚本说明

本节介绍 grid_search_tools 目录下各个脚本的功能：

- `grid_search.py` Grid Search 任务入口脚本，该脚本负责调度 GPU 资源，可自动将 7 个分类任务、3 个阅读理解下所有超参数对应的任务完成，训练完成后会自动调用抽取结果的脚本 `extract_result.sh` 打印出所有任务的最佳结果和对应的超参。
- `warmup_dataset_and_model.py` 首次运行时，该脚本完成模型下载（如需）、数据集下载，阅读理解任务数据预处理、预处理文件缓存等工作，再次运行则会检查这些文件是否存在，存在则跳过。该脚本由 `grid_search.py` 在 Grid Search 训练前自动调用，预处理 cache 文件生成后，后面所有训练任务即可加载缓存文件避免重复进行数据预处理。如果该阶段任务失败，大多需要检查网络，解决之后需重启 `grid_search.py`，直到训练正常开始。该脚本也可手动调用，需要 1 个参数，模型名称或目录。该脚本在使用 Intel(R) Xeon(R) Gold 6271C CPU 且 `--num_proc`默认为 4 的情况下需约 30 分钟左右完成，可以更改 `run_mrc.sh` 中的 `--num_proc` 参数以改变生成 cache 的进程数。需要注意的是，若改变 num_proc，之前的缓存则不能再使用，该脚本会重新处理数据并生成新的 cache，cache 相关内容可查看[datasets.Dataset.map文档](https://huggingface.co/docs/datasets/v2.0.0/package_reference/main_classes?highlight=map#datasets.Dataset.map)。
- `extract_result.sh` 从日志抽取每个任务的最佳结果和对应的最佳超参并打印，`grid_search.py` 在完成训练任务后会自动调用，也可手动调用，需要 1 个参数：模型名称或目录。手动调用前需要确认训练均全部完成，并且保证该目录下有分类和阅读理解所有任务的日志。
- `run_mrc.sh` 阅读理解任务的启动脚本。
- `run_cls.sh` 分类任务的启动脚本。


<a name="参加CLUE竞赛"></a>

## 参加 CLUE 竞赛

对各个任务运行预测脚本，汇总多个结果文件压缩之后，即可提交至 CLUE 官网进行评测。

下面 2 小节会分别介绍分类、阅读理解任务产生预测结果的方法。

<a name="分类任务"></a>

### 分类任务

以 TNEWS 为例，可以直接使用脚本 `classification/run_clue_classifier.py` 对单个任务进行预测，注意脚本启动时需要传入参数 `--do_predict`。假设 TNEWS 模型所在路径为 `${TNEWS_MODEL}`，运行如下脚本可得到模型在测试集上的预测结果，预测结果会写入地址 `${OUTPUT_DIR}/tnews_predict.json`。

```
cd classification
OUTPUT_DIR=results
mkdir ${OUTPUT_DIR}

python run_clue_classifier.py \
    --task_name TNEWS \
    --model_name_or_path ${TNEWS_MODEL}  \
    --output_dir ${OUTPUT_DIR} \
    --do_predict \
```
<a name="阅读理解任务"></a>

### 阅读理解任务

以 C<sup>3</sup> 为例，直接使用 `mrc/run_c3.py`对该任务进行预测，注意脚本启动时需要传入参数 `--do_predict`。假设 C<sup>3</sup> 模型所在路径为 `${C3_MODEL}`，运行如下脚本可得到模型在测试集上的预测结果，预测结果会写入地址 `${OUTPUT_DIR}/c311_predict.json`。

```shell
cd mrc
OUTPUT_DIR=results
mkdir ${OUTPUT_DIR}

python run_c3.py \
    --model_name_or_path ${C3_MODEL} \
    --output_dir ${OUTPUT_DIR} \
    --do_predict \
```
