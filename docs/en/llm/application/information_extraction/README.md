# Universal Information Extraction Model PP-UIE

 **Table of Contents**

- [1. Model Introduction](#模型简介)
- [2. Getting Started](#开箱即用)
  - [2.1 Entity Extraction](#实体抽取)
  - [2.2 Relation Extraction](#关系抽取)
  - [2.3 Model Selection](#模型选择)
  - [2.4 More Configurations](#更多配置)
- [3. Training Customization](#训练定制)
  - [3.1 Code Structure](#代码结构)
  - [3.2 Data Annotation](#数据标注)
  - [3.3 Model Fine-tuning](#模型微调)
  - [3.4 One-Click Prediction with Custom Models](#定制模型一键预测)
  - [3.5 Experimental Metrics](#实验指标)

<a name="模型简介"></a>

## 1. Model Introduction

The Universal Information Extraction Model (PP-UIE) is a general-purpose information extraction model developed by the PaddleNLP team based on open-source models and high-quality datasets. Drawing from Baidu's UIE modeling approach, PaddleNLP has trained and open-sourced a large model for Chinese and English universal information extraction. It supports unified training for information extraction tasks including Named Entity Recognition (NER), Relation Extraction (RE), and Event Extraction (EE). The model is available in four versions (0.5B, 1.5B, 7B, and 14B) to accommodate different information extraction scenarios. It demonstrates significant improvements in ACC and F1 metrics across multiple datasets (including Boson, CLUENER, CCIR2021, etc.) compared to other general information extraction models.

<a name="开箱即用"></a>

## 2. Getting Started

```paddlenlp.Taskflow``` provides universal information extraction capabilities, supporting extraction of multiple information types including but not limited to named entity recognition (e.g., person names, locations, organizations), relations (e.g., movie directors, song release dates), and events (e.g., traffic accidents, earthquakes). Users can define extraction targets using natural language, enabling zero-shot extraction of corresponding information from input text. **Ready out-of-the-box to meet diverse information extraction needs.**

<a name="实体抽取"></a>

#### 2.1 Entity Extraction

Named Entity Recognition (NER) identifies entities with specific meanings in text. In open-domain information extraction, there are no restrictions on entity types - users can define custom categories.

- Example: Extracting entities of types "Time", "Player", and "Event Name". The schema is constructed as:

  ```text
  ['Time', 'Player', 'Event Name']
  ```

  Usage example:

  ```python
  from paddlenlp import Taskflow

  schema = ['Time', 'Player', 'Event Name']
  ie = Taskflow('information_extraction', schema=schema)
  ie("2月8日，北京冬奥会自由式滑雪女子大跳台决赛中，中国选手谷爱凌以188.25分获得金牌！")
  ```

  Output:
  ```python
  {'Time': [{'text': '2月8日', 'start': 0, 'end': 4}],
   'Player': [{'text': '谷爱凌', 'start': 39, 'end': 42}],
   'Event Name': [{'text': '北京冬奥会自由式滑雪女子大跳台决赛', 'start': 6, 'end': 31}]}
  ```

<a name="关系抽取"></a>

#### 2.2 Relation Extraction

Relation Extraction (RE) identifies semantic relationships between entities in text. Users can define relation types and entity roles.

- Example: Extracting relations of type "Singer" and "Album". The schema is:

  ```text
  {
    'Singer': [
      'Song',
      'Album'
    ]
  }
  ```

  Usage example:

  ```python
  schema = {'Singer': ['Song', 'Album']}
  ie.set_schema(schema)
  ie("周杰伦的《最伟大的作品》是音乐杰作，收录于他的最新专辑《最伟大的作品》中。")
  ```

  Output:
  ```python
  {'Singer': [{'text': '周杰伦', 'start': 0, 'end': 3},
    {'relations': {'Song': [{'text': '最伟大的作品', 'start': 5, 'end': 11}],
                   'Album': [{'text': '最伟大的作品', 'start': 33, 'end': 39}]}}]}
  ```

[//]: # (Continue translating subsequent sections following the same pattern...)
```python
    from pprint import pprint
    from paddlenlp import Taskflow

    schema = ['Time', 'Athlete', 'Event Name'] # Define the schema for entity extraction
    ie = Taskflow('information_extraction',
                  schema= ['Time', 'Athlete', 'Event Name'],
                  schema_lang="zh",
                  batch_size=1,
                  model='paddlenlp/PP-UIE-0.5B',
                  precision='float16')
    pprint(ie("On the morning of February 8, during the Beijing Winter Olympics freestyle skiing women's big air final, Chinese athlete Gu Ailing won the gold medal with 188.25 points!"))
    # Output
    [{'Time': [{'text': 'On the morning of February 8'}],
      'Event Name': [{'text': 'Beijing Winter Olympics freestyle skiing women's big air final'}],
      'Athlete': [{'text': 'Gu Ailing'}]}]
    ```

<a name="关系抽取"></a>

#### 2.2 Relation Extraction

  Relation Extraction (RE) refers to identifying entities and extracting semantic relationships between entities from text to obtain triplet information, i.e., <subject, predicate, object>.

  - For example, using "Competition Name" as the extraction subject, and extracting relationship types as "Organizer", "Host", and "Time", the schema is constructed as follows:

    ```text
    {
      'Competition Name': [
        'Organizer',
        'Host',
        'Time'
      ]
    }
    ```

    Example call:

    ```python
    schema = {'Competition Name': ['Organizer', 'Host', 'Time']} # Define the schema for relation extraction
    ie.set_schema(schema) # Reset schema
    pprint(ie('The 2022 Language and Intelligence Technology Competition was jointly organized by the Chinese Information Processing Society of China and the China Computer Federation, hosted by Baidu, the Evaluation Work Committee of the Chinese Information Processing Society of China, and the Natural Language Processing Committee of the China Computer Federation. It has been held for 4 consecutive years and has become one of the most popular Chinese NLP events globally.'))
    # Output
    [{'Competition Name': [{'relations': {'Organizer': [{'text': 'Chinese Information Processing Society of China,China Computer Federation'}],
                          'Time': [{'text': '2022'}],
                          'Host': [{'text': 'Baidu,Evaluation Work Committee of Chinese Information Processing Society of China,Natural Language Processing Committee of China Computer Federation'}]},
            'text': 'Language and Intelligence Technology Competition'}]}]
    ```
<a name="Model Selection"></a>

#### 2.3 Model Selection

- Multiple model options to meet accuracy and speed requirements

  | Model | Architecture | Language |
  | :---: | :--------: | :--------: |
  | `paddlenlp/PP-UIE-0.5B` | 24-layers, 896-hidden, 14-heads | Chinese, English |
  | `paddlenlp/PP-UIE-1.5B` | 28-layers, 1536-hidden, 12-heads | Chinese, English |
  | `paddlenlp/PP-UIE-7B` | 28-layers, 3584-hidden, 28-heads | Chinese, English |
  | `paddlenlp/PP-UIE-14B` | 48-layers, 5120-hidden, 40-heads | Chinese, English |

<a name="More Configurations"></a>

#### 2.4 More Configurations

```python
>>> from paddlenlp import Taskflow

>>> ie = Taskflow('information_extraction',
                  schema = {'Competition Name': ['Organizer', 'Host', 'Date']},
                  schema_lang="zh",
                  batch_size=1,
                  model='paddlenlp/PP-UIE-0.5B',
                  precision='float16')
```

* `schema`: Defines the extraction targets, refer to the usage examples of different tasks in the out-of-box section.
* `schema_lang`: Sets the language of schema, default is `zh`, options include `zh` and `en`. Different schema construction patterns exist between Chinese and English, thus requiring language specification.
* `batch_size`: Batch processing size, adjust according to hardware configuration, default is 1.
* `model`: Select model for the task, options include `paddlenlp/PP-UIE-0.5B`, `paddlenlp/PP-UIE-1.5B`, `paddlenlp/PP-UIE-7B`, `paddlenlp/PP-UIE-14B`.
* `precision`: Select model precision, default is `float16`, options include `float16`, `bfloat16` and `float32`. If choosing `float16` for GPU environments, ensure:
  - Proper installation of NVIDIA drivers and base software with **CUDA ≥11.2, cuDNN ≥8.1.1**
  - GPU CUDA Compute Capability >7.0 (e.g., V100, T4, A10, A100, GTX 20/30 series)
- `bfloat16` selection requires corresponding hardware/software support but provides acceleration for large models and batch processing, especially effective when combined with mixed precision.
** includes NVIDIA A100 and H800 GPUs, while requiring CUDA >=11.2, cuDNN >=8.1.1 and other software environments. For details on CUDA Compute Capability and precision support, please refer to NVIDIA documentation: [GPU Hardware and Supported Precision Matrix](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix).

Additionally, you can quickly call the model and perform inference using the following code:
```python
from paddlenlp.transformers import AutoModelForCausalLM
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.generation import GenerationConfig
from paddlenlp.trl import llm_utils

model_id = "paddlenlp/PP-UIE-0.5B"

model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention=False)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
generation_config = GenerationConfig.from_pretrained(model_id)


template = """
You are a reading comprehension expert. Please extract entities from the given sentence and question. Note that if entities exist, they must appear verbatim in the original sentence. Output the exact original text of corresponding entities without modification. If no entity can be extracted, output "No corresponding entity".
 **Sentence Start**
 {sentence}
 **Sentence End**
 **Question Start**
 {prompt}
 **Question End**
 **Answer Start**
 """

sentences = [
    "On February 12, at the Harbin Asian Winter Games women's singles short program figure skating competition, Chinese athlete Zhu Yi was the first to take the stage and delivered an excellent performance, scoring 62.90 points, setting a new personal best for her short program career.",
    "On February 12, at the Harbin Asian Winter Games cross-country skiing men's 4×7.5km relay final, the Chinese team composed of Li Minglin, Ciren Zhandui, Baolin, and Wang Qiang won the gold medal.",
    "On February 13, at the Harbin Asian Winter Games biathlon women's 4×6km relay competition, the Chinese team consisting of Tang Jialin, Wen Ying, Chu Yuanmeng, and Meng Fanqi claimed the gold medal.",
    "According to official measurement from China Earthquake Networks: A magnitude 3.5 earthquake occurred in Fengqing County, Lincang City, Yunnan Province (24.34°N, 99.98°E) on May 16 at 06:08, with a focal depth of 10 kilometers.",
    "The song 'Farewell' is included in Sun Yao's album 'Love Story'.",
]

prompts = [
    "time, athlete, event name",
    "time, athlete, event name",
    "time, athlete, event name",
    "earthquake magnitude, time, epicenter location, focal depth",
    "song title, artist, album",
]

inputs = [template.format(sentence=sentence, prompt=prompt) for sentence, prompt in zip(sentences, prompts)]
inputs = [tokenizer.apply_chat_template(sentence, tokenize=False) for sentence in inputs]
input_features = tokenizer(
    inputs,
    max_length=512,
    return_position_ids=False,
    truncation=True,
    truncation_side="left",
    padding=True,
    return_tensors="pd",
    add_special_tokens=False,
)

outputs = model.generate(
    **input_features,
    max_new_tokens=200,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=llm_utils.get_eos_token_id(tokenizer, generation_config),
    pad_token_id=tokenizer.pad_token_id,
    decode_strategy="greedy_search",
    temperature=1.0,
    top_k=1,
    top_p=1.0,
    repetition_penalty=1.0,
)


def get_clean_entity(text):
    ind1 = text.find("\n **Answer End**\n\n")
    if ind1 != -1:
        pred = text[:ind1]
    else:
        pred = text
    return pred


results = tokenizer.batch_decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
results = [get_clean_entity(result) for result in results]

for sentence, prompt, result in zip(sentences, prompts, results):
    print("-" * 50)
    print(f"Sentence: {sentence}")
    print(f"Prompt: {prompt}")
    print(f"Result: {result}")
```
<a name="Training Customization"></a>

## 3. Training Customization

For simple extraction tasks, ```paddlenlp.Taskflow``` can be used directly for zero-shot extraction. For specialized scenarios, we recommend using light customization (fine-tuning the model with a small amount of annotated data) to further improve performance. The following example of "reimbursement work order information extraction" demonstrates how to fine-tune the PP-UIE model with just a few dozen training samples.

<a name="Code Structure"></a>

#### 3.1 Code Structure

```shell
.
├── utils.py          # Data processing utilities
├── doccano.py        # Data annotation script
├── doccano.md        # Data annotation documentation
└── README.md
```

<a name="Data Annotation"></a>

#### 3.2 Data Annotation

We recommend using the [doccano](https://github.com/doccano/doccano) annotation platform. This example also integrates the pipeline from annotation to training - after exporting data from doccano, you can easily convert it to the required format for model input using the [doccano.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/application/information_extraction/doccano.py) script, achieving seamless transition. For detailed annotation guidelines, please refer to the [doccano Annotation Guide](doccano.md).

Original data example:

```text
深大到双龙28块钱4月24号交通费
```

The extraction schema is:

```python
schema = ['Departure place', 'Destination', 'Cost', 'Time']
```

Annotation steps:

1. Create a "Sequence Labeling" project in doccano.
2. Define entity labels: In this case, we need to define labels for `Departure place`, `Destination`, `Cost`, and `Time`.
3. Start annotating data with these labels. Below shows a doccano annotation example:

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167336891-afef1ad5-8777-456d-805b-9c65d9014b80.png height=100 hspace='10'/>
</div>

4. After annotation completion, export the data from doccano, rename it to ``doccano_ext.json``, and place it in the ``./data`` directory.

5. We provide a pre-annotated file [doccano_ext.json](https://bj.bcebos.com/paddlenlp/datasets/uie/doccano_ext.json) which can be directly downloaded and placed in the `./data` directory. Execute the following script for data conversion, which will generate train/validation/test files in the `./data` directory.

```shell
python doccano.py \
    --doccano_file ./data/doccano_ext.json \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --schema_lang ch
```

Configurable parameters:

- ``doccano_file``
Data annotation files exported from doccano.
- ``save_dir``: Directory to save training data, defaults to ``data`` directory.
- ``negative_ratio``: Maximum negative example ratio. This parameter only applies to extraction tasks. Proper negative example construction can improve model performance. The number of negative examples relates to actual label count. Maximum negative examples = negative_ratio * positive examples count.
- ``splits``: Proportions for splitting dataset into training, validation and test sets. Default [0.8, 0.1, 0.1] means dividing data into 8:1:1 ratio for train/dev/test sets respectively.
- ``task_type``: Task type selection. Currently only supports Information Extraction (`ie`).
- ``is_shuffle``: Whether to shuffle the dataset. Default is False.
- ``seed``: Random seed. Default is 1000.
- ``schema_lang``: Language selection for schema. Options: `ch` (Chinese) and `en` (English). Default is `ch`. For English datasets, please select `en`.

Notes:
- By default, the doccano.py script will split data into train/dev/test sets according to specified proportions
- Each execution of doccano.py script will overwrite existing files with same name
- During model training phase, we recommend constructing some negative examples to improve model performance. This functionality is built-in during data conversion phase. The ratio of automatically constructed negative samples can be controlled via `negative_ratio`; Number of negative samples = negative_ratio * number of positive samples.
- For files exported from doccano, each data entry is assumed to be correctly annotated manually by default.

<a name="模型微调"></a>

#### 3.3 Model Fine-tuning

We recommend using [Large Model Fine-tuning](../../docs/finetune.md) for model refinement. Simply input the model, dataset, etc. to efficiently perform fine-tuning and model compression tasks. This supports one-click multi-GPU training, mixed precision training, gradient accumulation, checkpoint restarting, logging display, and encapsulates common training configurations like optimizers and learning rate schedulers.

Use the following command to fine-tune the model using `paddlenlp/PP-UIE-0.5B` as the pre-trained model, and save the fine-tuned model to the specified path.

For GPU environments, you can specify the gpus parameter for multi-GPU training:

```shell
# Return to PaddleNLP/llm directory
python -u  -m paddle.distributed.launch --gpus "0,1" run_finetune.py ./config/qwen/sft_argument.json
```

Reference configuration for `sft_argument.json`:

```json
{
  "model_name_or_path": "paddlenlp/PP-UIE-0.5B",
  "data_path": "./data",
  "output_dir": "./checkpoints",
  "per_device_train_batch_size": 8,
  "gradient_accumulation_steps": 4,
  "num_train_epochs": 10,
  "learning_rate": 3e-5,
  "warmup_ratio": 0.1,
  "weight_decay": 0.01,
  "logging_steps": 10,
  "save_strategy": "epoch"
}
```
```shell
{
    "model_name_or_path": "paddlenlp/PP-UIE-0.5B",
    "dataset_name_or_path": "./application/information_extraction/data",
    "output_dir": "./checkpoints/ie_ckpts",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "per_device_eval_batch_size": 1,
    "eval_accumulation_steps":8,
    "num_train_epochs": 3,
    "learning_rate": 3e-05,
    "warmup_steps": 30,
    "logging_steps": 1,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "src_length": 1024,
    "max_length": 2048,
    "fp16": true,
    "fp16_opt_level": "O2",
    "do_train": true,
    "do_eval": true,
    "disable_tqdm": true,
    "load_best_model_at_end": true,
    "eval_with_do_generation": false,
    "metric_for_best_model": "accuracy",
    "recompute": false,
    "save_total_limit": 1,
    "tensor_parallel_degree": 1,
    "pipeline_parallel_degree": 1,
    "sharding": "stage2",
    "zero_padding": false,
    "unified_checkpoint": true,
    "use_flash_attention": false
  }
```
For more configuration details of `sft_arguments.json`, please refer to [LLM Fine-Tuning](../../docs/finetune.md)

<a name="One-Click Prediction with Custom Model"></a>

#### 3.4 One-Click Prediction with Custom Model

Using PaddleNLP's high-performance predictor for rapid inference:
- Built-in full-process fused operator strategy
- Supports Weight Only INT8 and INT4 inference, enabling weight, activation, and Cache KV quantization for INT8 and FP8 inference
- Supports both dynamic graph and static graph inference modes

Before inference, it is recommended to compile and install PaddleNLP's high-performance custom inference operators for large models. These high-performance operators can significantly improve inference speed for large models. Detailed installation instructions can be found in the [Large Model High-Performance Inference Operator Installation Guide](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/csrc/README.md)

After installation, high-performance inference can be performed according to the following instructions.
```shell
# Under the PaddleNLP/llm directory
python predict/predictor.py \
    --model_name_or_path ./checkpoints/ie_ckpts \
    --dtype float16 \
    --data_file ./application/information_extraction/data/test.json \
    --output_file ./output.json \
    --src_length 512 \
    --max_length 1024 \
    --batch_size 4 \
    --inference_model 1 \
    --quant_type weight_only_int8
```

Configuration parameter descriptions:

- ``model_name_or_path``: Required. Pretrained model name or local model path for warm-starting the model and tokenizer. Default is None.
- ``src_length``: Maximum token length of model input context. Default is 1024.
- ``max_length``: Maximum token length of model input (context + generated content). Default is 2048.
- ``inference_model``: Whether to use Inference Model for inference. Default is False. The Inference Model incorporates dynamic insertion and full-cycle operator fusion strategies, delivering better performance when enabled. **If PaddleNLP's high-performance custom inference operators for large models are not compiled and installed, this must be set to False**.
- ``quant_type``: Whether to use quantized inference. Default is None. Optional values include weight_only_int8, weight_only_int4, a8w8 and a8w8_fp8. **If PaddleNLP's high-performance custom inference operators for large models are not compiled and installed, this must remain None**.

More about `predictor.py`
