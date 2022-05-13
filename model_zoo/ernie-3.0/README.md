# ERNIE 3.0 轻量级模型

## 模型介绍
TBD
<p align="center">
        <img width="644" alt="image" src="https://user-images.githubusercontent.com/1371212/168254282-e6901f9a-fd9a-4bbd-a5b6-e847b7f90bb4.png">
</p>

## 模型效果

本项目开源 **ERNIE 3.0 _base_*** 和 **ERNIE 3.0 _medium_**两个模型：

- [**ERNIE 3.0-_Base_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_base_zh.pdparams) (_12-layer, 768-hidden, 12-heads_)
- [**ERNIE 3.0-_Medium_**](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_medium_zh.pdparams) (_6-layer, 768-hidden, 12-heads_)

在 CLUE **验证集**上评测指标如下表所示：
<table style="width:100%;" cellpadding="2" cellspacing="0" border="1" bordercolor="#000000">
        <tbody>
                <tr>
                        <td style="text-align:center">
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
                                <span style="font-size:18px">Mengzi-Bert-Base</span>
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
                       <td rowspan=2 align=center> 6L768H </td>
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
                                <span style="font-size:18px">RBT6, Chinese</span>
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
        <tbody>
</table>
<br />



## ERNIE 3.0 模型使用

```python

from paddlenlp.transformers import *

tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")

# 用于分类任务
seq_cls_model = AutoModelForSequenceClassificaion.from_pretrained("ernie-3.0-medium-zh")

# 用于命名实体识别任务
token_cls_model = AutoModelForTokenClassification.from_pretrained("ernie-3.0-medium-zh")

# 用于阅读理解任务
qa_model = AutoModelForQuestionAnswering.from_pretrained("ernie-3.0-medium-zh")

```
ERNIE 3.0 提供了针对分类、命名实体识别、阅读理解三大场景下的微调使用样例，分别参考 `run_seq_cls.py` 、`run_token_cls.py`、`run_qa.py` 三个脚本，启动方式如下：

```shell
# 分类任务
python run_seq_cls.py  --task_name tnews --model_name_or_path ernie-3.0-base-zh --do_train

# 命名实体识别任务
python run_token_cls.py --task_name msra_ner  --model_name_or_path ernie-3.0-medium-zh --do_train

# 阅读理解任务
python run_qa.py --model_name_or_path ernie-3.0-medium-zh --do_train

```

## 模型压缩及推理部署

### 模型压缩 API 及使用

ERNIE 3.0 基于 PaddleNLP 的 Trainer API 发布提供了模型压缩 API。压缩 API 支持用户对 ERNIE、BERT 等Transformers 类下游任务微调模型进行裁剪、量化。用户只需要简单地调用 `compress()` API 即可一键启动裁剪和量化，并自动保存压缩后的模型。

首先需要安装 paddleslim 包

```shell
pip install paddleslim
```

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
    algo_list=['hist', 'mse'], batch_size_list=[4, 8, 16]))

trainer.compress(
    data_args.dataset,
    output_dir,
    pruning=True,
    quantization=True,
    compress_config=compress_config)
```

并且，ERNIE 3.0 还提供了压缩 API 在分类、命名实体识别、阅读理解三大场景下的使用样例，可以分别参考 `compress_seq_cls.py` 、`compress_token_cls.py`、`compress_qa.py`，启动方式如下：

```shell
# 分类任务
python compress_seq_cls.py --dataset "clue tnews"  --model_name_or_path best_models/TNEWS  --output_dir ./

# 命名实体识别任务
python compress_token_cls.py --dataset "msra_ner"  --model_name_or_path best_models/MSRA_NER  --output_dir ./

# 阅读理解任务
python compress_seq_cls.py --dataset "cmrc2018"  --model_name_or_path best_models/CMRC2018  --output_dir ./
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


### 压缩精度

这三类任务使用压缩 API 压缩精度如下：


|      | Model                      | AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUEWSC2020 | CSL   | CMRC2018    | MSRA_NER          | AVG   |
| ---- | -------------------------- | ----- | ----- | ------- | ----- | ----- | ----------- | ----- | ----------- | ----------------- | ----- |
| 1    | ERNIE 3.0-Medium           | 75.35 | 57.45 | 60.18   | 81.16 | 77.19 | 80.59       | 81.93 | 66.67/87.49 | 93.04/92.68/93.43 | 74.75 |
| 2    | ERNIE 3.0-Medium量化       |       |       |         |       |       |             |       |             |                   |       |
| 3    | ERNIE 3.0-Medium裁剪       | 75.14 | 57.31 | 60.29   | 81.25 | 77.46 | 79.93       | 81.7  | 65.67/86.55 | 93.10/92.77/93.43 | 74.65 |
| 4    | ERNIE 3.0-Medium裁剪、量化 |       |       |         |       |       |             |       |             |                   |       |

### 性能提升

压缩后的性能：

GPU:
TBD

CPU:
TBD

**压缩 API 使用TIPS：**

1. 压缩 API 提供裁剪和量化两个过程，建议两种都选择，裁剪需要训练，训练时间视下游任务数据量而定且和微调是一个量级的。量化不需要训练，更快；因此也可以只选择量化；

2. 裁剪类似蒸馏过程，方便起见，可以直接使用微调时的超参。为了进一步提升精度，可以对 `batch_size`、`learning_rate`、`epoch`、`max_seq_length` 等超参进行 grid search；

3. 模型压缩主要用于推理部署，因此压缩后的模型都是静态图模型，只可用于预测，不能再通过 `from_pretrained` 导入继续训练。

### 推理部署

TBD

#### Paddle Serving 部署

TBD

## Reference

* Sun Y, Wang S, Feng S, et al. ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation[J]. arXiv preprint arXiv:2107.02137, 2021.

* Su W, Chen X, Feng S, et al. ERNIE-Tiny: A Progressive Distillation Framework for Pretrained Transformer Compression[J]. arXiv preprint arXiv:2106.02241, 2021.

* Wang S, Sun Y, Xiang Y, et al. ERNIE 3.0 Titan: Exploring Larger-scale Knowledge Enhanced Pre-training for Language Understanding and Generation[J]. arXiv preprint arXiv:2112.12731, 2021.