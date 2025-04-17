English | [简体中文](README_ch.md)

# ERNIE-Layout

 **content**

- [ERNIE-Layout](#ERNIE-Layout)
  - [1. Model Instruction](#1)
  - [2. Out-of-Box](#2)
      - [HuggingFace web demo](#21)
      - [Demo show](#22)
      - [Taskflow](#23)
  - [3. Model Performance](#3)
  - [4. Fine-tuning Examples](#4)
      - [4.1 Key Information Extraction](#41)
      - [4.2 Document Question Answering](#42)
      - [4.3 Document Image Classification](#43)
  - [5. Deploy](#5)
      - [5.1 Inference Model Export](#51)
      - [5.2 Python Deploy](#52)

<a name="1"></a>

## 1. Model Instruction
Recent years have witnessed the rise and success of pre-training techniques in visually-rich document understanding. However, most existing methods lack the systematic mining and utilization of layout-centered knowledge, leading to sub-optimal performances. In this paper, we propose ERNIE-Layout, a novel document pre-training solution with layout knowledge enhancement in the whole workflow, to learn better representations that combine the features from text, layout, and image. Specifically, we first rearrange input sequences in the serialization stage, and then present a correlative pre-training task, reading order prediction, to learn the proper reading order of documents. To improve the layout awareness of the model, we integrate a spatial-aware disentangled attention into the multi-modal transformer and a replaced regions prediction task into the pre-training phase. Experimental results show that ERNIE-Layout achieves superior performance on various downstream tasks, setting new state-of-the-art on key information extraction, document image classification, and document question answering datasets.

[The work](http://arxiv.org/abs/2210.06155) is accepted by EMNLP 2022 (Findings). To expand the scope of commercial applications for document intelligence, we release the multilingual model of ERNIE-Layout through PaddleNLP.

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195091552-86a2d174-24b0-4ddf-825a-4503e0bc390b.png height=450 width=1000 hspace='10'/>
</div>

<a name="2"></a>

## 2. Out-of-Box

<a name="21"></a>

#### HuggingFace web demo

🧾 HuggingFace web demo is available [here](https://huggingface.co/spaces/PaddlePaddle/ERNIE-Layout)

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195749427-864d7744-1fd1-455e-99c6-53a260776483.jpg height=700 width=1100 hspace='10'/>
</div>

<a name="22"></a>

#### Demo show

- Invoice VQA

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/196118171-fd3e49a0-b9f1-4536-a904-c48f709a2dec.png height=350 width=1000 hspace='10'/>
</div>

- Poster VQA

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195610368-04230855-62de-439e-b708-2c195b70461f.png height=600 width=1000 hspace='15'/>
</div>

- WebPage VQA

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195611613-bdbe692e-d7f2-4a2b-b548-1a933463b0b9.png height=350 width=1000 hspace='10'/>
</div>


- Table VQA

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195610692-8367f1c8-32c2-4b5d-9514-a149795cf609.png height=350 width=1000 hspace='10'/>
</div>


- Exam Paper VQA

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195823294-d891d95a-2ef8-4519-be59-0fedb96c00de.png height=700 width=1000 hspace='10'/>
</div>


- English invoice VQA by multilingual(CH, EN, JP, Th, ES, RUS) prompt

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195615523-05d05aba-3bc3-415d-a836-ad1a5d3db56e.png height=400 width=1000 hspace='15'/>
</div>

- Chinese invoice VQA by multilingual(CHS, CHT, EN, JP, DE) prompt

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/209898193-c09a87fe-29cd-4e22-8e67-a5f281a99871.jpg height=350 width=1000 hspace='15'/>
</div>

- Demo images are available [here](https://bj.bcebos.com/paddlenlp/taskflow/document_intelligence/demo.zip)

<a name="23"></a>

#### Taskflow

- Input Format

```
[
  {"doc": "./book.png", "prompt": ["What is the name of the author of 'The Adventure Zone: The Crystal Kingdom’?", "What type of book cover does The Adventure Zone: The Crystal Kingdom have?", "For Rage, who is the author listed as?"]},
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

  - Image from local path

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

- Parameter Description
  * `batch_size`: number of input of each batch, default to 1.
  * `lang`: PaddleOCR language, `en` is better to English images, default to `ch`.
  * `topn`: return the top n results with highest probability, default to 1.


<a name="3"></a>

## 3. Model Performance

- Dataset

  |   Dataset   |  Task   | Language | Note |
  | --------- | ---------- | --- | ---- |
  | FUNSD     | Key Information Extraction | English | - |
  | XFUND-ZH  | Key Information Extraction | Chinese | - |
  | DocVQA-ZH | Document Question Answering | Chinese | The submission of the competition of [DocVQA-ZH](http://ailab.aiwin.org.cn/competitions/49) is now closed so we split original dataset into three parts for model evluation. There are 4,187 training images, 500 validation images, and 500 test images.|
  | RVL-CDIP (sampled)  | Document Image Classification | English | The RVL-CDIP dataset consists of 400,000 grayscale images in 16 classes, with 25,000 images per class. Because of the original dataset is large and slow for training, so we downsampling from it. The sampled dataset consist of 6,400 training images, 800 validation images, and 800 test images. |

- Results

  |         Model      |    FUNSD  | RVL-CDIP (sampled)  | XFUND-ZH  | DocVQA-ZH |
  | ------------------ | --------- | --------- | --------- | --------- |
  | LayoutXLM-Base     |   86.72   |   **90.88**   |   86.24   |   66.01   |
  | ERNIE-LayoutX-Base | **89.31** | 90.29 | **88.58** | **69.57** |

- Evaluation Methods

  - All the above tasks do the Hyper Parameter searching based on Grid Search method. The evaluation step interval of FUNSD and XFUND-ZH are both 100, metric is F1-Score. The evaluation step interval of RVL-CDIP is 2000, metric is Accuracy. The evaluation step interval of DocVQA-ZH is 10000, metric is [ANLS](https://arxiv.org/pdf/1907.00490.pdf),

  - Hyper Parameters search ranges

    | Hyper Parameters  |  FUNSD  | RVL-CDIP (sampled)  | XFUND-ZH | DocVQA-ZH |
    | ----------------- | ------- | -------- | -------- | --------- |
    | learning_rate     | 5e-6, 1e-5, 2e-5, 5e-5 | 5e-6, 1e-5, 2e-5, 5e-5 | 5e-6, 1e-5, 2e-5, 5e-5 | 5e-6, 1e-5, 2e-5, 5e-5 |
    | batch_size        | 1, 2, 4 |  8, 16, 24   | 1, 2, 4 |   8, 16, 24  |
    | warmup_ratio      |     -   | 0, 0.05, 0.1 |    -    | 0, 0.05, 0.1 |

    The strategy of ``lr_scheduler_type`` for FUNSD and XFUND is constant, so warmup_ratio is excluded.

  - ``max_steps`` is applied for the fine-tuning on both FUNSD and XFUND-ZH, 10000 steps and 20000 steps respectively; ``num_train_epochs`` is set to 6 and 20 for DocVQA-ZH and RVL-CDIP respectively.

- Best Hyper Parameter

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

#### 4.1 Key Information Extraction

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

#### 4.2 Document Question Answering

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

#### 4.3 Document Image Classification

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

- Export the model fine-tuned on DocVQA-ZH

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

We provide the deploy example on Key Information Extraction, Document Question Answering and Document Image Classification, please follow the [ERNIE-Layout Python Deploy Guide](./deploy/python/README.md)


<a name="References"></a>

## References

- [ERNIE-Layout: Layout-Knowledge Enhanced Multi-modal Pre-training for Document Understanding](http://arxiv.org/abs/2210.06155)

- [ICDAR 2019 Competition on Scene Text Visual Question Answering](https://arxiv.org/pdf/1907.00490.pdf)

- [XFUND dataset](https://github.com/doc-analysis/XFUND)

- [FUNSD dataset](https://guillaumejaume.github.io/FUNSD/)

- [RVL-CDIP dataset](https://adamharley.com/rvl-cdip/)

- [Competition of Insurance Document Visual Cognition Question Answering](http://ailab.aiwin.org.cn/competitions/49)
