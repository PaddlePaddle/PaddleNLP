[English](README.md) | 简体中文

# ERNIE-Layout

 **目录**

- [1. 模型介绍](#1)
- [2. 开箱即用](#2)
  - [HuggingFace web demo](#21)
  - [应用场景展示](#22)
  - [Taskflow](#23)
- [3. Benchmark 模型效果](#3)
- [4. 模型微调](#4)
  - [4.1 文档信息抽取任务](#41)
  - [4.2 文档视觉问答任务](#42)
  - [4.3 文档图像分类任务](#43)
- [5. 部署](#5)
  - [5.1 静态图导出](#51)
  - [5.2 Python 部署](#52)

<a name="1"></a>

## 1. 模型介绍


ERNIE-Layout 以文心文本大模型 ERNIE 为底座，融合文本、图像、布局等信息进行跨模态联合建模，创新性引入布局知识增强，提出阅读顺序预测、细粒度图文匹配等自监督预训练任务，升级空间解偶注意力机制，在各数据集上效果取得大幅度提升，相关工作[ERNIE-Layout: Layout-Knowledge Enhanced Multi-modal Pre-training for Document Understanding](http://arxiv.org/abs/2210.06155)已被 EMNLP 2022 Findings 会议收录[1]。考虑到文档智能在多语种上商用广泛，依托 PaddleNLP 对外开源业界最强的多语言跨模态文档预训练模型 ERNIE-Layout。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195091552-86a2d174-24b0-4ddf-825a-4503e0bc390b.png height=450 width=1000 hspace='10'/>
</div>

<a name="2"></a>

## 2. 开箱即用

<a name="21"></a>

#### HuggingFace web demo

🧾 通过[Huggingface 网页](https://huggingface.co/spaces/PaddlePaddle/ERNIE-Layout)体验 DocPrompt 功能：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195749427-864d7744-1fd1-455e-99c6-53a260776483.jpg height=700 width=1100 hspace='10'/>
</div>

<a name="22"></a>

#### 应用场景展示

- 发票抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/196118171-fd3e49a0-b9f1-4536-a904-c48f709a2dec.png height=350 width=1000 hspace='10'/>
</div>

- 海报抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195610368-04230855-62de-439e-b708-2c195b70461f.png height=600 width=1000 hspace='15'/>
</div>

- 网页抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195611613-bdbe692e-d7f2-4a2b-b548-1a933463b0b9.png height=350 width=1000 hspace='10'/>
</div>


- 表格抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195610692-8367f1c8-32c2-4b5d-9514-a149795cf609.png height=350 width=1000 hspace='10'/>
</div>


- 试卷抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195823294-d891d95a-2ef8-4519-be59-0fedb96c00de.png height=700 width=1000 hspace='10'/>
</div>


- 英文票据多语种（中、英、日、泰、西班牙、俄语）抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195610820-7fb88608-b317-45fc-a6ab-97bf3b20a4ac.png height=400 width=1000 hspace='15'/>
</div>

- 中文票据多语种（中简、中繁、英、日、德语）抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/209898223-71f09f4d-a394-4a4d-91fa-33784d1cec18.jpg height=350 width=1000 hspace='15'/>
</div>

- Demo 图片可在此[下载](https://bj.bcebos.com/paddlenlp/taskflow/document_intelligence/demo.zip)

<a name="23"></a>

#### Taskflow

通过``paddlenlp.Taskflow``三行代码调用 DocPrompt 功能，具备多语言文档抽取问答能力，部分应用场景展示如下：

- 输入格式

```
[
  {"doc": "./invoice.jpg", "prompt": ["发票号码是多少?", "校验码是多少?"]},
  {"doc": "./resume.png", "prompt": ["五百丁本次想要担任的是什么职位?", "五百丁是在哪里上的大学?", "大学学的是什么专业?"]}
]
```

默认使用 PaddleOCR 进行 OCR 识别，同时支持用户通过``word_boxes``传入自己的 OCR 结果，格式为``List[str, List[float, float, float, float]]``。

```
[
  {"doc": doc_path, "prompt": prompt, "word_boxes": word_boxes}
]
```

- 支持单条、批量预测

  - 支持本地图片路径输入

  <div align="center">
      <img src=https://user-images.githubusercontent.com/40840292/194748579-f9e8aa86-7f65-4827-bfae-824c037228b3.png height=800 hspace='20'/>
  </div>

  ```python
  >>> from pprint import pprint
  >>> from paddlenlp import Taskflow

  >>> docprompt = Taskflow("document_intelligence")
  >>> pprint(docprompt([{"doc": "./resume.png", "prompt": ["五百丁本次想要担任的是什么职位?", "五百丁是在哪里上的大学?", "大学学的是什么专业?"]}]))
  [{'prompt': '五百丁本次想要担任的是什么职位?',
    'result': [{'end': 7, 'prob': 1.0, 'start': 4, 'value': '客户经理'}]},
  {'prompt': '五百丁是在哪里上的大学?',
    'result': [{'end': 37, 'prob': 1.0, 'start': 31, 'value': '广州五百丁学院'}]},
  {'prompt': '大学学的是什么专业?',
    'result': [{'end': 44, 'prob': 0.82, 'start': 38, 'value': '金融学(本科）'}]}]
  ```

  - http 图片链接输入

  <div align="center">
      <img src=https://user-images.githubusercontent.com/40840292/194748592-e20b2a5f-d36b-46fb-8057-86755d188af0.jpg height=400 hspace='10'/>
  </div>

  ```python
  >>> from pprint import pprint
  >>> from paddlenlp import Taskflow

  >>> docprompt = Taskflow("document_intelligence")
  >>> pprint(docprompt([{"doc": "https://bj.bcebos.com/paddlenlp/taskflow/document_intelligence/images/invoice.jpg", "prompt": ["发票号码是多少?", "校验码是多少?"]}]))
  [{'prompt': '发票号码是多少?',
    'result': [{'end': 2, 'prob': 0.74, 'start': 2, 'value': 'No44527206'}]},
  {'prompt': '校验码是多少?',
    'result': [{'end': 233,
                'prob': 1.0,
                'start': 231,
                'value': '01107 555427109891646'}]}]
  ```

- 可配置参数说明
  * `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
  * `lang`：选择 PaddleOCR 的语言，`ch`可在中英混合的图片中使用，`en`在英文图片上的效果更好，默认为`ch`。
  * `topn`: 如果模型识别出多个结果，将返回前 n 个概率值最高的结果，默认为1。


<a name="3"></a>

## 3. Benchmark 模型效果

- 开源数据集介绍

  |   数据集   |  任务类型   | 语言 | 说明 |
  | --------- | ---------- | --- | ---- |
  | FUNSD     | 文档信息抽取 | 英文 | - |
  | XFUND-ZH  | 文档信息抽取 | 中文 | - |
  | DocVQA-ZH | 文档视觉问答 | 中文 | [DocVQA-ZH](http://ailab.aiwin.org.cn/competitions/49)已停止榜单提交，因此我们将原始训练集进行重新划分以评估模型效果，划分后训练集包含4,187张图片，验证集包含500张图片，测试集包含500张图片。 |
  | RVL-CDIP (sampled)  | 文档图像分类 | 英文 | RVL-CDIP 原始数据集共包含400,000张图片，由于数据集较大训练较慢，为验证文档图像分类的模型效果故进行降采样，采样后的训练集包含6,400张图片，验证集包含800张图片，测试集包含800张图片。 |

- 评测结果

  在文档智能领域主流开源数据集的**验证集**上评测指标如下表所示：

  |         Model      |    FUNSD  | RVL-CDIP (sampled)  | XFUND-ZH  | DocVQA-ZH |
  | ------------------ | --------- | --------- | --------- | --------- |
  | LayoutXLM-Base     |   86.72   |   **90.88**   |   86.24   |   66.01   |
  | ERNIE-LayoutX-Base | **89.31** | 90.29 | **88.58** | **69.57** |

- 具体评测方式

  - 以上所有任务均基于 Grid Search 方式进行超参寻优。FUNSD 和 XFUND-ZH 每间隔 100 steps 评估验证集效果，评价指标为 F1-Score。
    RVL-CDIP 每间隔2000 steps 评估验证集效果，评价指标为 Accuracy。DocVQA-ZH 每间隔10000 steps 评估验证集效果，取验证集最优效果作为表格中的汇报指标，评价指标为 ANLS（计算方法参考 https://arxiv.org/pdf/1907.00490.pdf）。

  - 以上每个下游任务的超参范围如下表所示：

    | Hyper Parameters  |  FUNSD  | RVL-CDIP (sampled)  | XFUND-ZH | DocVQA-ZH |
    | ----------------- | ------- | -------- | -------- | --------- |
    | learning_rate     | 5e-6, 1e-5, 2e-5, 5e-5 | 5e-6, 1e-5, 2e-5, 5e-5 | 5e-6, 1e-5, 2e-5, 5e-5 | 5e-6, 1e-5, 2e-5, 5e-5 |
    | batch_size        | 1, 2, 4 |  8, 16, 24   | 1, 2, 4 |   8, 16, 24  |
    | warmup_ratio      |     -   | 0, 0.05, 0.1 |    -    | 0, 0.05, 0.1 |

    FUNSD 和 XFUND-ZH 使用的 lr_scheduler_type 策略是 constant，因此不对 warmup_ratio 进行搜索。

  - 文档信息抽取任务 FUNSD 和 XFUND-ZH 采用最大步数（max_steps）的微调方式，分别为10000 steps 和20000 steps；文档视觉问答 DocVQA-ZH 的 num_train_epochs 为6；文档图像分类 RVL-CDIP 的 num_train_epochs 为20。

- 最优超参

  不同预训练模型在下游任务上做 Grid Search 之后的最优超参（learning_rate、batch_size、warmup_ratio）如下：

  |         Model      |     FUNSD    |   RVL-CDIP (sampled)   |   XFUND-ZH   |   DocVQA-ZH |
  | ------------------ | ------------ | ------------ | ------------ | ----------- |
  | LayoutXLM-Base     |  1e-5, 2, _  | 1e-5, 8, 0.1 |  1e-5, 2, _  | 2e-5. 8, 0.1 |
  | ERNIE-LayoutX-Base |  2e-5, 4, _  | 1e-5, 8, 0.  |  1e-5, 4, _  | 2e-5. 8, 0.05 |


<a name="4"></a>

## 4. 模型微调

- 请执行以下命令进行安装项目依赖

```
pip install -r requirements.txt
```

<a name="41"></a>

#### 4.1 文档信息抽取任务

- FUNSD 训练

```shell
python -u run_ner.py \
  --model_name_or_path ernie-layoutx-base-uncased \
  --output_dir ./ernie-layoutx-base-uncased/models/funsd/ \
  --dataset_name funsd \
  --do_train \
  --do_eval \
  --max_steps 10000 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --pattern ner-bio \
  --preprocessing_num_workers 4 \
  --overwrite_cache false \
  --use_segment_box \
  --doc_stride 128 \
  --target_size 1000 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --learning_rate 2e-5 \
  --lr_scheduler_type constant \
  --gradient_accumulation_steps 1 \
  --seed 1000 \
  --metric_for_best_model eval_f1 \
  --greater_is_better true \
  --overwrite_output_dir
```

- XFUND-ZH 训练

```shell
python -u run_ner.py \
  --model_name_or_path ernie-layoutx-base-uncased \
  --output_dir ./ernie-layoutx-base-uncased/models/xfund_zh/ \
  --dataset_name xfund_zh \
  --do_train \
  --do_eval \
  --lang "ch" \
  --max_steps 20000 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --pattern ner-bio \
  --preprocessing_num_workers 4 \
  --overwrite_cache false \
  --use_segment_box \
  --doc_stride 128 \
  --target_size 1000 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --learning_rate 1e-5 \
  --lr_scheduler_type constant \
  --gradient_accumulation_steps 1 \
  --seed 1000 \
  --metric_for_best_model eval_f1 \
  --greater_is_better true \
  --overwrite_output_dir
```

<a name="42"></a>

#### 4.2 文档视觉问答任务

- DocVQA-ZH 训练

```shell
python3 -u run_mrc.py \
  --model_name_or_path ernie-layoutx-base-uncased \
  --output_dir ./ernie-layoutx-base-uncased/models/docvqa_zh/ \
  --dataset_name docvqa_zh \
  --do_train \
  --do_eval \
  --lang "ch" \
  --num_train_epochs 6 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.05 \
  --weight_decay 0 \
  --eval_steps 10000 \
  --save_steps 10000 \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --pattern "mrc" \
  --use_segment_box false \
  --return_entity_level_metrics false \
  --overwrite_cache false \
  --doc_stride 128 \
  --target_size 1000 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 2e-5 \
  --preprocessing_num_workers 32 \
  --save_total_limit 1 \
  --train_nshard 16 \
  --seed 1000 \
  --metric_for_best_model anls \
  --greater_is_better true \
  --overwrite_output_dir
```

<a name="43"></a>

#### 4.3 文档图像分类任务

- RVL-CDIP 训练

```shell
python3 -u run_cls.py \
    --model_name_or_path ernie-layoutx-base-uncased \
    --output_dir ./ernie-layoutx-base-uncased/models/rvl_cdip_sampled/ \
    --dataset_name rvl_cdip_sampled \
    --do_train \
    --do_eval \
    --num_train_epochs 20 \
    --lr_scheduler_type linear \
    --max_seq_length 512 \
    --warmup_ratio 0.05 \
    --weight_decay 0 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --pattern "cls" \
    --use_segment_box \
    --return_entity_level_metrics false \
    --overwrite_cache false \
    --doc_stride 128 \
    --target_size 1000 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-5 \
    --preprocessing_num_workers 32 \
    --train_nshard 16 \
    --seed 1000 \
    --metric_for_best_model acc \
    --greater_is_better true \
    --overwrite_output_dir
```

<a name="5"></a>

## 5. 部署

<a name="51"></a>

#### 5.1 静态图导出

使用动态图训练结束之后，还可以将动态图参数导出为静态图参数，静态图模型将用于**后续的推理部署工作**。具体代码见[静态图导出脚本](export_model.py)，静态图参数保存在`output_path`指定路径中。运行方式：


- 导出在 FUNSD 上微调后的模型：

```shell
python export_model.py --task_type ner --model_path ./ernie-layoutx-base-uncased/models/funsd/ --output_path ./ner_export
```

- 导出在 DocVQA-ZH 上微调后的模型：

```shell
python export_model.py --task_type mrc --model_path ./ernie-layoutx-base-uncased/models/docvqa_zh/ --output_path ./mrc_export
```

- 导出在 RVL-CDIP(sampled)上微调后的模型：

```shell
python export_model.py --task_type cls --model_path ./ernie-layoutx-base-uncased/models/rvl_cdip_sampled/ --output_path ./cls_export
```

- 可支持配置的参数：
* `model_path`：动态图训练保存的参数路径；默认为"./checkpoint/"。
* `output_path`：静态图图保存的参数路径；默认为"./export"。

- 程序运行时将会自动导出模型到指定的 `output_path` 中，保存模型文件结构如下所示：

```text
export/
├── inference.pdiparams
├── inference.pdiparams.info
└── inference.pdmodel
```

<a name="52"></a>

#### 5.2 Python 部署

导出静态图模型之后可用于部署，项目提供了文档信息抽取、文档视觉问答和文档图像分类三大场景下的使用示例，详见[ERNIE-Layout Python 部署指南](./deploy/python/README_ch.md)。


<a name="References"></a>

## References

- [ERNIE-Layout: Layout-Knowledge Enhanced Multi-modal Pre-training for Document Understanding](http://arxiv.org/abs/2210.06155)

- [ICDAR 2019 Competition on Scene Text Visual Question Answering](https://arxiv.org/pdf/1907.00490.pdf)

- [XFUND dataset](https://github.com/doc-analysis/XFUND)

- [FUNSD dataset](https://guillaumejaume.github.io/FUNSD/)

- [RVL-CDIP dataset](https://adamharley.com/rvl-cdip/)

- [保险文本视觉认知问答竞赛](http://ailab.aiwin.org.cn/competitions/49)
