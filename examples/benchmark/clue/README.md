# CLUE Benchmark

[CLUE](https://www.cluebenchmarks.com/) 自成立以来发布了多项 NLP 评测基准，包括分类榜单，阅读理解榜单和自然语言推断榜单等，在学术界、工业界产生了深远影响。是目前应用最广泛的中文语言测评指标之一。详细可参考 [CLUE论文](https://arxiv.org/abs/2004.05986)。

本项目基于 PaddlePaddle 在 CLUE 数据集上对领先的开源预训练模型模型进行了充分评测，为开发者在预训练模型选择上提供参考，同时开发者基于本项目可以轻松一键复现模型效果，也可以参加 CLUE 竞赛取得好成绩。

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
                        <td rowspan=1 align=center> 24L1024H </td>
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
			<td rowspan=6 align=center> 12L768H </td>
                        <td style="text-align:center">
                                <span style="font-size:18px">ERNIE 3.0-Base-zh</span>
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
                                <span style="font-size:18px">ERNIE 3.0-Medium-zh</span>
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
                                <span style="font-size:18px">ERNIE 3.0-Mini-zh</span>
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
                                <span style="font-size:18px">ERNIE 3.0-Micro-zh</span>
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
                                <span style="font-size:18px">ERNIE 3.0-Nano-zh</span>
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


AFQMC、TNEWS、IFLYTEK、CMNLI、OCNLI、CLUEWSC2020、CSL 、CHID 和 C<sup>3</sup> 任务使用的评估指标均是 Accuracy。CMRC2018 的评估指标是 EM (Exact Match)/F1，计算每个模型效果的平均值时，取 EM 为最终指标。

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
| RoBERTa-wwm-ext-large            | 1e-5,32 | 3e-5,32 | 2e-5,32 | 1e-5,16  | 1e-5,16  | 2e-5,16     | 2e-5,16 | 3e-5,32  | 1e-5,24 | 2e-5,24       |
| ERNIE 3.0-Base-zh                | 3e-5,16 | 3e-5,32 | 5e-5,32 | 3e-5,32  | 2e-5,64  | 2e-5,16     | 2e-5,32 | 2e-5,24  | 3e-5,24 | 3e-5,32       |
| ERNIE-Gram-zh                    | 1e-5,16 | 5e-5,16 | 5e-5,16 | 2e-5,32  | 2e-5,64  | 3e-5,16     | 3e-5,64 | 3e-5,32  | 2e-5,24 | 2e-5,24       |
| Mengzi-Bert-Base                 | 3e-5,32 | 5e-5,32 | 5e-5,16 | 2e-5,16  | 2e-5,16  | 3e-5,8      | 1e-5,16 | 3e-5,24  | 3e-5,24 | 2e-5,32       |
| ERNIE 1.0                        | 3e-5,16 | 3e-5,32 | 5e-5,16 | 5e-5，32 | 3e-5，16 | 2e-5,8      | 2e-5,16 | 3e-5,32  | 3e-5,24 | 3e-5,24       |
| RoBERTa-wwm-ext                  | 3e-5,32 | 3e-5,64 | 5e-5,16 | 3e-5,32  | 2e-5,32  | 3e-5,32     | 2e-5,32 | 3e-5,32  | 2e-5,32 | 3e-5,24       |
| BERT-Base-Chinese                | 2e-5,16 | 5e-5,16 | 5e-5,16 | 5e-5,64  | 3e-5,16  | 3e-5,16     | 1e-5,16 | 3e-5,24  | 2e-5,32 | 3e-5,24       |
| ERNIE 3.0-Medium-zh              | 3e-5,32 | 3e-5,64 | 5e-5,32 | 2e-5,32  | 1e-5,64  | 3e-5,16     | 2e-5,32 | 3e-5,24  | 2e-5,24 | 1e-5,24       |
| TinyBERT<sub>6</sub> ,Chinese    | 1e-5,16 | 3e-5,32 | 5e-5,16 | 5e-5,32  | 3e-5,64  | 3e-5,16     | 3e-5,16 | 8,3e-5   | 3e-5,24 | 2e-5,24       |
| RoFormerV2 Small                 | 5e-5,16 | 2e-5,16 | 5e-5,16 | 5e-5,32  | 2e-5,16  | 3e-5,8      | 3e-5,16 | 3e-5,24  | 3e-5,24 | 3e-5,24       |
| HLF/RBT6, Chinese                | 3e-5,16 | 5e-5,16 | 5e-5,16 | 5e-5,64  | 3e-5,32  | 3e-5,32     | 3e-5,16 | 3e-5,32  | 3e-5,24 | 3e-5,24       |
| UER/Chinese-RoBERTa (L6-H768) | 2e-5,16 | 5e-5,32 | 5e-5,16 | 5e-5,32  | 3e-5,16  | 5e-5,8      | 3e-5,16 | 3e-5,24  | 3e-5,24 | 3e-5,32       |
| ERNIE 3.0-Mini-zh                | 5e-5,64 | 5e-5,64 | 5e-5,16 | 5e-5,32  | 2e-5,16  | 2e-5,8      | 2e-5,16 | 3e-5,24  | 3e-5,24 | 3e-5,24       |
| ERNIE 3.0-Micro-zh               | 3e-5,16 | 5e-5,32 | 5e-5,16 | 5e-5,16  | 2e-5,32  | 5e-5,16      | 3e-5,64 | 3e-5,24  | 3e-5,32 | 3e-5,24       |
| ERNIE 3.0-Nano-zh                | 2e-5,32 | 5e-5,16 | 5e-5,16 | 5e-5,16  | 3e-5,16  | 1e-5,8      | 3e-5,32 | 3e-5,24  | 3e-5,24 | 2e-5,24       |


其中，`ERNIE 3.0-Base-zh`、`ERNIE 3.0-Medium-zh`、`ERNIE-Gram-zh`、`ERNIE 1.0`、`ERNIE 3.0-Mini-zh`、`ERNIE 3.0-Micro-zh`、`ERNIE 3.0-Nano-zh` 在 CLUEWSC2020 处的 dropout_prob 为 0.0，`ERNIE 3.0-Base-zh`、`HLF/RBT6, Chinese`、`Mengzi-BERT-Base`、`ERNIE-Gram-zh`、`ERNIE 1.0` 、`TinyBERT6, Chinese`、`UER/Chinese-RoBERTa (L6-H768)`、`ERNIE 3.0-Mini-zh`、`ERNIE 3.0-Micro-zh`、`ERNIE 3.0-Nano-zh` 在 IFLYTEK 处的 dropout_prob 为 0.0。


## 一键复现模型效果

这一节将会对分类、阅读理解任务分别展示如何一键复现本文的评测结果。

### 启动 CLUE 分类任务
以 CLUE 的 TNEWS 任务为例，启动 CLUE 任务进行 Fine-tuning 的方式如下：

```shell
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=TNEWS
export LR=3e-5
export BS=32
export EPOCH=6
export MAX_SEQ_LEN=128
export MODEL_PATH=ernie-3.0-base-zh

cd classification
mkdir ernie-3.0-base-zh
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
#### 使用Trainer启动 CLUE 分类任务
PaddleNLP 提供了 Trainer API，本示例新增了`run_clue_classifier_trainer.py`脚本供用户使用。

```
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=TNEWS
export LR=3e-5
export BS=32
export EPOCH=6
export MAX_SEQ_LEN=128
export MODEL_PATH=ernie-3.0-base-zh

cd classification
mkdir ernie-3.0-base-zh

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

### 启动 CLUE 阅读理解任务
以 CLUE 的 C<sup>3</sup> 任务为例，多卡启动 CLUE 任务进行 Fine-tuning 的方式如下：

```shell

cd mrc

MODEL_PATH=ernie-3.0-base-zh
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

### 批量启动 Grid Search

#### 环境说明

脚本中的默认参数是针对 32G 显卡的机器设置的，如果显卡是 16G 或者更小，可以在`grid_search_tools/run_mrc.sh`中增加每个任务的`${grd_accu_steps}`，`${grd_accu_steps}` 对应于 Python 脚本中的 `--gradient_accumulation_steps` 参数，该参数代表梯度累加的步数。即，实际训练的时候，会以 batch_size 为 `batch_size / ${grd_accu_steps}` 训练 `${grd_accu_steps}` 步来模拟原 batch_size 下的训练。因此也需要保证原 batch_size 能被 `${grd_accu_steps}` 整除。

#### 使用说明
- `run_all_cls.sh` 分类任务批量启动脚本入口，需要 2 个参数：模型名称或目录、模式 id
    - 模式 0、1、2 均代表独立的 1 组 4 卡实验，需要在不同的 4 张卡上运行（见下方分类任务的方法二）
    - 模型 3 代表 同时起 12 组 8 卡实验（见下方分类任务的方法一）
    - 对于 32G 8 GPUs 卡机器，base 模型可以选模式 3，即 1 组 8 卡，可同时完成所有分类任务。而 large 模型需要分别起模式 0、1、2 三组实验
- `run_all_mrc.sh` 阅读理解任务批量启动脚本入口，需要 2 个参数：模型名称或目录、模式id
    - 模式 0、1、2 分别代表起 4 卡 CHID，4 卡 C<sup>3</sup>，2 卡 CMRC2018 实验（见下方阅读理解任务的方法二）
    - 模式 3 代表 起 8 卡实验，任务会完成 CHID、C<sup>3</sup>、CMRC2018 任务（见下方阅读理解任务的方法一）
    - 对于 32G 8 GPUs 卡机器，base 模型可以选模式 3
- `extract_acc.sh` 从日志抽取每个任务的最佳结果，需要 1 个参数：模型名称或目录。调用前需要确认训练均全部完成，并且该目录下有分类和阅读理解任务所有的日志。可以把打印出的 10 个任务的结果直接复制到表格中。例如：

```
AFQMC	TNEWS	IFLYTEK	CMNLI	OCNLI	CLUEWSC2020	CSL	CMRC2018	CHID	C3
75.3    557.45  60.18   81.16   77.19   79.28   81.93   65.83/87.29     80.00   70.36
```


```shell
cd grid_search_tools
# 7 个分类任务
# 方法一：3 代表模式 3，1 组 8 卡实验（适用于 base 及更小的模型）
bash run_all_cls.sh ernie-3.0-nano-zh 3 # 第二个参数 3 就代表模式 3

# 方法二：0、1、2 分别代表模式 0、1、2，代表起了 3 组 4 卡实验（适用于较大的模型）
bash run_all_cls.sh ernie-3.0-nano-zh 0
bash run_all_cls.sh ernie-3.0-nano-zh 1
bash run_all_cls.sh ernie-3.0-nano-zh 2

# 3 个阅读理解任务
# 对于32G 8 GPUs 卡机器，base 模型可以选模式 3

# 方法一：3 代表模式3，1组8卡实验
bash run_all_mrc.sh ernie-3.0-nano-zh 3

# 方法二：对于较大模型，分别起3组实验：
bash run_all_mrc.sh ernie-3.0-nano-zh 0
bash run_all_mrc.sh ernie-3.0-nano-zh 1
bash run_all_mrc.sh ernie-3.0-nano-zh 2
```

确认模型所有任务训练完成后，可以调用脚本 `extract_acc.sh` 一键抽取 Grid Search 结果

```shell
sh extract_acc.sh ernie-3.0-nano-zh
```

## 参加 CLUE 竞赛

对各个任务运行预测脚本，汇总多个结果文件压缩之后，即可提交至 CLUE 官网进行评测。

下面 2 小节会分别介绍分类、阅读理解任务产生预测结果的方法。

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
