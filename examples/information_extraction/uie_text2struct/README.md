# UIE

- Code for [`Unified Structure Generation for Universal Information Extraction`](https://arxiv.org/pdf/2203.12277.pdf)

## Requirements

General

- Python (verified on 3.8)
- CUDA (verified on 11.1/10.1)
Python Packages
CUDA 10.1
``` bash
conda create -n uie python=3.8
python -m pip install paddlepaddle-gpu==2.2.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install -r requirements.txt
```

CUDA 11.1
``` bash
conda create -n uie python=3.8
python -m pip install paddlepaddle-gpu==2.2.2.post111 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install -r requirements.txt
```

## Quick Start

### Data Format

Details of preprocessing see `dataset_processing/`.
Data folder contains four files：

```text
data/text2spotasoc/absa/14lap
├── entity.schema       # Entity Types for converting SEL to Record
├── relation.schema     # Relation Types for converting SEL to Record
├── event.schema        # Event Types for converting SEL to Record
├── record.schema       # Spot/Asoc Type for constructing SSI
├── test.json
├── train.json
└── val.json
```

train/val/test.json are data files, and each line is a JSON instance.
Each JSON instance contains `text` and `record` fields, in which `text` is plain text, and `record` is the SEL representation of the extraction structure.
Details definition see [DATASETS.md](docs/DATASETS.md).
```text
{
  "text": "Great laptop that offers many great features !",
  "record": "<extra_id_0> <extra_id_0> opinion <extra_id_5> great <extra_id_1> <extra_id_0> aspect <extra_id_5> features <extra_id_0> positive <extra_id_5> great <extra_id_1> <extra_id_1> <extra_id_1>"
}
```

Note:
- Use the extra character of T5 as the structure indicators, such as `<extra_id_0>`, `<extra_id_1>`, `<extra_id_5>`.

| Token  | Role |
| ------------- | ------------- |
| <extra_id_0>  | Start of Label Name |
| <extra_id_1>  | End of Label Name   |
| <extra_id_2>  | Start of Input Text |
| <extra_id_5>  | Start of Text Span  |
| <extra_id_6>  | NULL span for Rejection |

- `record.schema` is the record schema file for building SSI.
It contains three lines: the first line is spot name list, the second line is asoc name list. And the third line is spot-to-asoc dictionary (do not use in code, can be ignored).

  ```text
  ["aspect", "opinion"]
  ["neutral", "positive", "negative"]
  {"aspect": ["neutral", "positive", "negative"], "opinion": []}
  ```

### Pretrained Models
You can find the pre-trained models as following:
- [uie-base-en]()
- [uie-large-en]()

### Model Training

Training scripts as follows:

- `run_seq2struct.py`: Python code entry
- `run_seq2struct.bash`: Model training and evaluating process script.

You can change arguments in the `run_seq2struct.bash` to conduct different experiments:
- data_folder=data/absa/14lap
- model_name=pd_models/uie-base-en
- metric_for_best_model=offset-rel-strict-F1
- map_config=config/offset_map/closest_offset_en.yaml

Trained models are saved in the `output_dir` specified by `run_seq2struct.bash`.

Simple Training Command
```
bash run_seq2struct.bash
```

Progress logs
```
...
2022-04-21 18:41:22,517 - __main__ - INFO - Meta Sample Negative: -1, Ordered SSI: False
2022-04-21 18:41:22,556 - __main__ - INFO - Meta Sample Negative: -1, Ordered SSI: True
2022-04-21 18:41:22,568 - __main__ - INFO - ********** Running training **********
2022-04-21 18:41:22,568 - __main__ - INFO -   Num examples = 906
2022-04-21 18:41:22,568 - __main__ - INFO -   Num Epochs = 50
2022-04-21 18:41:22,568 - __main__ - INFO -   Device train batch size = 16
2022-04-21 18:41:22,568 - __main__ - INFO -   Device eval  batch size = 128
2022-04-21 18:41:22,568 - __main__ - INFO -   Total  train batch size (w. accumulation) = 16
2022-04-21 18:41:22,568 - __main__ - INFO -   Gradient Accumulation steps = 1
2022-04-21 18:41:22,569 - __main__ - INFO -   Total optimization steps = 2850
...
```

Final Result
```
...
2022-04-21 19:22:42,982 - __main__ - INFO -   offset-rel-strict-P = 65.66265060240963
2022-04-21 19:22:42,982 - __main__ - INFO -   offset-rel-strict-R = 60.22099447513812
2022-04-21 19:22:42,982 - __main__ - INFO -   offset-rel-strict-F1 = 62.824207492795395
...
```

| Metric      | Definition |
| ----------- | ----------- |
| ent-(P/R/F1)      | Micro-F1 of Entity (Entity Type, Entity Span) |
| rel-strict-(P/R/F1)   | Micro-F1 of Relation Strict (Relation Type, Arg1 Span, Arg1 Type, Arg2 Span, Arg2 Type) |
| rel-boundary-(P/R/F1)   | Micro-F1 of Relation Boundary (Relation Type, Arg1 Span, Arg2 Span) |
| evt-trigger-(P/R/F1)   | Micro-F1 of Event Trigger (Event Type, Trigger Span) |
| evt-role-(P/R/F1)   | Micro-F1 of Relation Boundary (Event Type, Arg Role, Arg Span) |

- `offset`: offset-bsaed match
- `string`: string-bsaed match
- `match_mode=set`: evaluation with removing duplicate extracted results (usually for string evaluation, such as distant supervised dataset NYT)
- `match_mode=normal`: each gold only can be matched once

### Model Evaluation

We have uploaded several checkpoints of our experiments in the paper, you can evaluate the performance of each checkpoint by run `inference.py` for each checkpoint.

```
python inference.py \
    --data data/text2spotasoc/absa/14lap \
    --model pd_models/absa_14lap \
    --batch_size 64
```

| Model      | Metric | Score |
| ----------- | ----------- | ----------- |
| [ent_ace04ent]() | test_offset-ent-F1 | 86.86 |
| [ent_ace05ent]() | test_offset-ent-F1 | 85.88 |
| [ent_conll03]() | test_offset-ent-F1 | 92.97 |
| [rel_ace05-rel]() | test_offset-rel-strict-F1 | 66.16 |
| [rel_conll04_large]() | test_offset-rel-strict-F1 | 74.96 |
| [rel_scierc_large]() | test_offset-rel-strict-F1 | 36.98 |
| [rel_nyt]() | test_string-rel-boundary-F1 (set) | 93.52 |
| [evt_ace05evt]() | test_offset-evt-trigger-F1 | 73.97 |
| [evt_ace05evt]() | test_offset-evt-role-F1 | 55.75 |
| [evt_casie]() | test_offset-evt-trigger-F1 | 69.96 |
| [evt_casie]() | test_offset-evt-role-F1 | 61.25 |
| [absa_14lap]() | test_offset-rel-strict-F1 | 65.24 |
| [absa_14res]() | test_offset-rel-strict-F1 | 74.58 |
| [absa_15res]() | test_offset-rel-strict-F1 | 68.30 |
| [absa_16res]() | test_offset-rel-strict-F1 | 76.49 |
| [absa_14lap_base]() | test_offset-rel-strict-F1 | 63.95 |
| [absa_14res_base]() | test_offset-rel-strict-F1 | 73.62 |
| [absa_15res_base]() | test_offset-rel-strict-F1 | 64.67 |
| [absa_16res_base]() | test_offset-rel-strict-F1 | 73.22 |
| [rel_nyt_base]() | test_string-rel-boundary-F1 (set) | 92.45 |

### Data Collator

**_Sampling Strategy_** and **_Rejection Mechanism_** can be adopted in the training process.

- `uie/seq2seq/data_collator/meta_data_collator.py` class _DataCollatorForMetaSeq2Seq_ is for collating data, class _DynamicSSIGenerator_ is for prompt sampling
- `run_seq2struct.py` class _DataTrainingArguments_ contains related parameters

Related parameters in parse_args() are briefly introduced here:

- About **_Sampling Strategy_**
``` text
    - max_prefix_length       Maximum length of prompt
    - record_schema           record schema read from record.schema
    - meta_negative           number of negative schema
    - meta_positive_rate      rate of positive spot
    - ordered_prompt          Whether to sort the spot prompt and asoc prompt or not, default is random
```

- About **_Rejection Mechanism_**
``` text
  - spot_noise              The noise rate of null spot
  - asoc_noise              The noise rate of null asoc
```

## Citation

If this code helps you, please cite this paper:

Yaojie Lu, Qing Liu, Dai Dai, Xinyan Xiao, Hongyu Lin, Xianpei Han, Le Sun, Hua Wu.
Unified Structure Generation for Universal Information Extraction.
Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics.

```
@misc{lu-etal-2022-uie,
  author = {Lu, Yaojie and
    Liu, Qing and
    Dai, Dai and
    Xiao, Xinyan and
    Lin, Hongyu and
    Han, Xianpei and
    Sun, Le and
    Wu, Hua},
  title = {Unified Structure Generation for Universal Information Extraction},
  url = {https://arxiv.org/abs/2203.12277},
  year = {2022},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
}
```
