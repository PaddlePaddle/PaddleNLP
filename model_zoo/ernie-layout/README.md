English | [简体中文](README_ch.md)

# ERNIE-Layout

 **content**

- [ERNIE-Layout](#ERNIE-Layout)
  - [1. Model Instruction](#1)
  - [2. Out-of-Box](#2)
      - [Gradio web demo](#21)
      - [Demo show](#22)
      - [Taskflow](#23)
  - [3. Model Performance](#3)
  - [4. Fine-tuning Examples](#4)
      - [4.1 Document Information Extraction](#41)
      - [4.2 Document Visual Question Answering](#42)
      - [4.3 Document Visual Classification](#43)
  - [5. Deploy](#5)
      - [5.1 Inference Model Export](#51)
      - [5.2 Python Deploy](#52)
  - [References](#references)

<a name="1"></a>

## 1. Model Instruction
Recent years have witnessed the rise and success of pre-training techniques in visually-rich document understanding. However, most existing methods lack the systematic mining and utilization of layout-centered knowledge, leading to sub-optimal performances. In this paper, we propose ERNIE-Layout, a novel document pre-training solution with layout knowledge enhancement in the whole workflow, to learn better representations that combine the features from text, layout, and image. Specifically, we first rearrange input sequences in the serialization stage, and then present a correlative pre-training task, reading order prediction, to learn the proper reading order of documents. To improve the layout awareness of the model, we integrate a spatial-aware disentangled attention into the multi-modal transformer and a replaced regions prediction task into the pre-training phase. Experimental results show that ERNIE-Layout achieves superior performance on various downstream tasks, setting new state-of-the-art on key information extraction, document image classification, and document question answering datasets. [Related work](http://arxiv.org/abs/2210.06155) accpeted by EMNLP 2022(Findings). To expand the scope of commercial applications for document intelligence, we propose a multilingual multi-modal document model ERNIE-Layout through PaddleNLP.

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195091552-86a2d174-24b0-4ddf-825a-4503e0bc390b.png height=400 hspace='10'/>
</div>

<a name="2"></a>

## 2. Out-of-Box

<a name="21"></a>

#### Gradio web demo

Gradio web demo is available [here](https://huggingface.co/spaces/PaddlePaddle/ERNIE-Layout)

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195117247-01a9caf5-3394-42b9-bfec-4a1c316a6990.png height=400 hspace='10'/>
</div>

<a name="22"></a>

#### Demo show

- 发票文档抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195603240-a3cb664e-2548-441f-a75c-9e0b9eebb023.png height=400 hspace='10'/>
</div>

- 海报文档抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195603422-a0a1905c-057f-4cac-9b45-52651eb7452a.png height=750 hspace='25'/>
</div>

- 网页文档抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195603536-b5e0107d-c39f-4bee-9d78-072a53072d68.png height=500 hspace='15'/>
</div>


- 表格文档抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195603626-5f1bed89-1467-491f-811d-296f2623e179.png height=500 hspace='15'/>
</div>

- 英文票据多语种（中、英、日、泰、西班牙、俄语）抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195603876-151894d4-a570-4a63-97cc-528b722c2b01.png height=500 hspace='15'/>
</div>

- 中文票据多语种（中简、中繁、英、日、法语）抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195603730-49fe0da0-b6c5-43e7-bbe5-9b7d2dd03f9a.png height=300 hspace='10'/>
</div>

<a name="23"></a>

#### Taskflow

- Input Format

```
[
  {"doc": "./invoice.jpg", "prompt": ["发票号码是多少?", "校验码是多少?"]},
  {"doc": "./resume.png", "prompt": ["五百丁本次想要担任的是什么职位?", "五百丁是在哪里上的大学?", "大学学的是什么专业?"]}
]
```

Default to use PaddleOCR, you can also use your own OCR result via ``word_boxes``, the data format is ``List[str, List[float, float, float, float]]``。

```
[
  {"doc": doc_path, "prompt": prompt, "word_boxes": word_boxes}
]
```

- Support single and batch input

  - Image from local path

  <div align="center">
      <img src=https://user-images.githubusercontent.com/40840292/194748579-f9e8aa86-7f65-4827-bfae-824c037228b3.png height=800 hspace='20'/>
  </div>

  ```python
  >>> from pprint import pprint
  >>> from paddlenlp import Taskflow

  >>> docprompt = Taskflow("document_intelligence")
  >>> docprompt([{"doc": "./resume.png", "prompt": ["五百丁本次想要担任的是什么职位?", "五百丁是在哪里上的大学?", "大学学的是什么专业?"]}])
  [{'prompt': '五百丁本次想要担任的是什么职位?',
    'result': [{'end': 183, 'prob': 1.0, 'start': 180, 'value': '客户经理'}]},
  {'prompt': '五百丁是在哪里上的大学?',
    'result': [{'end': 38, 'prob': 1.0, 'start': 32, 'value': '广州五百丁学院'}]},
  {'prompt': '大学学的是什么专业?',
    'result': [{'end': 45, 'prob': 0.74, 'start': 39, 'value': '金融学(本科）'}]}]
  ```

  - Image from http link

  <div align="center">
      <img src=https://user-images.githubusercontent.com/40840292/195605071-02d4f3ab-ef2d-43c1-9bf0-ffdc017d4f92.png height=400 hspace='10'/>
  </div>

  ```python
  >>> from pprint import pprint
  >>> from paddlenlp import Taskflow

  >>> docprompt = Taskflow("document_intelligence", lang="en")
  >>> docprompt([{"doc": "https://bj.bcebos.com/paddlenlp/taskflow/document_intelligence/images/book.png", "prompt": ["What is the name of the author of 'The Adventure Zone: The Crystal Kingdom’?", "What type of book cover does The Adventure Zone: The Crystal Kingdom have?", "For Rage, who is the author listed as?"]}])
  [{'prompt': "What is the name of the author of 'The Adventure Zone: The "
              'Crystal Kingdom’?',
    'result': [{'end': 39,
                'prob': 0.99,
                'start': 22,
                'value': 'Clint McElroy. Carey Pietsch, Griffn McElroy, Travis '
                        'McElroy'}]},
  {'prompt': 'What type of book cover does The Adventure Zone: The Crystal '
              'Kingdom have?',
    'result': [{'end': 51, 'prob': 1.0, 'start': 51, 'value': 'Paperback'}]},
  {'prompt': 'For Rage, who is the author listed as?',
    'result': [{'end': 93, 'prob': 1.0, 'start': 91, 'value': 'Bob Woodward'}]}]
  ```

- Parameter Description
  * `batch_size`: number of input in each batch, default to 1.
  * `lang`: PaddleOCR language, `en` is better to English images, default to `ch`.
  * `topn`: return the top n results with highest probability, default to 1.


<a name="3"></a>

## 3. Model Performance

- Dataset

  |   Dataset   |  Task   | Language | Note |
  | --------- | ---------- | --- | ---- |
  | FUNSD     | 文档信息抽取 | English | - |
  | XFUND-ZH  | 文档信息抽取 | Chinese | - |
  | DocVQA-ZH | 文档视觉问答 | Chinese | [DocVQA-ZH](http://ailab.aiwin.org.cn/competitions/49)已停止榜单提交，因此我们将原始训练集进行重新划分以评估模型效果，划分后训练集包含4,187张图片，验证集包含500张图片，测试集包含500张图片。 |
  | RVL-CDIP (sampled)  | 文档图像分类 | English | RVL-CDIP原始数据集共包含400,000张图片，由于数据集较大训练较慢，为验证文档图像分类的模型效果故进行降采样，采样后的训练集包含6,400张图片，验证集包含800张图片，测试集包含800张图片。 |

- Results

  |         Model      |    FUNSD  | RVL-CDIP (sampled)  | XFUND-ZH  | DocVQA-ZH |
  | ------------------ | --------- | --------- | --------- | --------- |
  | LayoutXLM-Base     |   86.72   |   **90.88**   |   86.24   |   66.01   |
  | ERNIE-LayoutX-Base | **89.31** | 90.29 | **88.58** | **69.57** |

- Evaluation Methods

  - 以上所有任务均基于Grid Search方式进行超参寻优。FUNSD和XFUND-ZH每间隔 100 steps 评估验证集效果，评价指标为Accuracy。
    RVL-CDIP每间隔2000 steps评估验证集效果，评价指标为F1-Score。DocVQA-ZH每间隔10000 steps评估验证集效果，取验证集最优效果作为表格中的汇报指标，评价指标为ANLS（计算方法参考https://arxiv.org/pdf/1907.00490.pdf）。

  - 以上每个下游任务的超参范围如下表所示：

    | Hyper Parameters  |  FUNSD  | RVL-CDIP (sampled)  | XFUND-ZH | DocVQA-ZH |
    | ----------------- | ------- | -------- | -------- | --------- |
    | learning_rate     | 5e-6, 1e-5, 2e-5, 5e-5 | 5e-6, 1e-5, 2e-5, 5e-5 | 5e-6, 1e-5, 2e-5, 5e-5 | 5e-6, 1e-5, 2e-5, 5e-5 |
    | batch_size        | 1, 2, 4 |  8, 16, 24   | 1, 2, 4 |   8, 16, 24  |
    | warmup_ratio      |     -   | 0, 0.05, 0.1 |    -    | 0, 0.05, 0.1 |

    FUNSD和XFUND-ZH使用的lr_scheduler_type策略是constant，因此不对warmup_ratio进行搜索。

  - 文档信息抽取任务FUNSD和XFUND-ZH采用最大步数（max_steps）的微调方式，分别为10000 steps和20000 steps；文档视觉问答DocVQA-ZH的num_train_epochs为6；文档图像分类RVL-CDIP的num_train_epochs为20。

- Best Hyper Parameter

  不同预训练模型在下游任务上做Grid Search之后的最优超参（learning_rate、batch_size、warmup_ratio）如下：

  |         Model      |     FUNSD    |   RVL-CDIP (sampled)   |   XFUND-ZH   |   DocVQA-ZH |
  | ------------------ | ------------ | ------------ | ------------ | ----------- |
  | LayoutXLM-Base     |  1e-5, 2, _  | 1e-5, 8, 0.1 |  1e-5, 2, _  | 2e-5. 8, 0.1 |
  | ERNIE-LayoutX-Base |  2e-5, 4, _  | 1e-5, 8, 0.  |  1e-5, 4, _  | 2e-5. 8, 0.05 |


<a name="4"></a>

## 4. Fine-tuning Examples

- Installation

```
pip install -r requirements.txt
```

<a name="41"></a>

#### 4.1 Document Information Extraction

- FUNSD Train

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

- XFUND-ZH Train

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

#### 4.2 Document Visual Question Answering

- DocVQA-ZH Train

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

#### 4.3 Document Visual Classification

- RVL-CDIP Train

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

## 5. Deploy

<a name="51"></a>

#### 5.1 Inference Model Export

After fine-tuning, you can also export the inference model via [Model Export Script](export_model.py), the inference model will be saved in the `output_path` you specified.

- Export the model fine-tuned on FUNSD

```shell
python export_model.py --task_type ner --model_path ./ernie-layoutx-base-uncased/models/funsd/ --output_path ./ner_export
```

-Export the model fine-tuned on DocVQA-ZH

```shell
python export_model.py --task_type mrc --model_path ./ernie-layoutx-base-uncased/models/docvqa_zh/ --output_path ./mrc_export
```

- Export the model fine-tuned on RVL-CDIP(sampled)

```shell
python export_model.py --task_type cls --model_path ./ernie-layoutx-base-uncased/models/rvl_cdip_sampled/ --output_path ./cls_export
```

- Parameter Description
  * `model_path`：the save directory of dygraph model parameters, default to "./checkpoint/"。
  * `output_path`：the save directory of static graph model parameters, default to "./export"。

- Directory

  ```text
  export/
  ├── inference.pdiparams
  ├── inference.pdiparams.info
  └── inference.pdmodel
  ```

<a name="52"></a>

#### 5.2 Python Deploy

We provide the deploy example on Document information extraction, DocVQA and Document image classification, please follow the [ERNIE-Layout Python Deploy Guide](./deploy/python/README.md)


<a name="References"></a>

## References

- [ERNIE-Layout: Layout-Knowledge Enhanced Multi-modal Pre-training for Document Understanding](http://arxiv.org/abs/2210.06155)

- [ICDAR 2019 Competition on Scene Text Visual Question Answering](https://arxiv.org/pdf/1907.00490.pdf)

- [XFUND dataset](https://github.com/doc-analysis/XFUND)

- [FUNSD dataset](https://guillaumejaume.github.io/FUNSD/)

- [RVL-CDIP dataset](https://adamharley.com/rvl-cdip/)

- [Competition of Insurance Document Visual Cognition Question Answering](http://ailab.aiwin.org.cn/competitions/49)
