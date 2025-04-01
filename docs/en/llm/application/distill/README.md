# Three-Step Rapid Distillation of DeepSeek R1

## 1. High-Throughput, Low-Cost PaddlePaddle DeepSeek-R1 Inference Service Deployment

PaddlePaddle's next-generation framework 3.0 comprehensively upgrades large model inference capabilities. Through high-expandable intermediate representation (PIR), it deeply optimizes model compression, inference computing, service deployment, and multi-hardware inference from all aspects, supporting numerous open-source large models for high-performance inference, and achieves outstanding performance on DeepSeek V3/R1. PaddlePaddle framework 3.0 supports full-featured FP8 inference for DeepSeek V3/R1 and its series of distilled models, while providing INT8 quantization capabilities, breaking through the limitations of Hopper architecture. Additionally, it introduces 4-bit quantized inference, enabling single-machine deployment that reduces costs while significantly improving system throughput by nearly double, providing more efficient and economical deployment solutions.

For performance optimization, we implement multi-level pipeline orchestration for MLA operators and fine-grained register/shared memory allocation optimization, achieving up to 23% performance improvement compared to FlashMLA. Combining FP8 matrix computation tuning and dynamic quantization operator optimization based on PaddlePaddle's DeepSeek R1 FP8 inference, single-machine output exceeds 1000 tokens per second; when using 4-bit single-machine deployment, output reaches over 2000 tokens per second! Inference performance significantly leads other open-source solutions. Moreover, it supports MTP speculative decoding, breaking through large-batch inference acceleration - while maintaining decoding speed, throughput increases by 144%; with similar throughput, decoding speed improves by 42%. For long-sequence Prefill phase, through attention computation dynamic quantization, first-token inference speed improves by 37%.

<p align="center">
  <img src="https://github.com/user-attachments/assets/84a90f79-6fb7-434d-857e-cbd964558f02" align="middle"  width="500" />
</p>

DeepSeek-R1 Single-Machine WINT4 Inference: Taking 1 H800/A800 machine as example, deploy single-machine 4-bit quantized inference service.
  * Set variable model_name to declare the model to download. For specific supported static graph models, please refer to documentation.
  * Set model storage path MODEL_PATH, default mounted to /models path in container (ensure write permission to MODEL_PATH)
```shell
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name="deepseek-ai/DeepSeek-R1/weight_only_int4"
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models \
-e "model_name=${model_name}" \
-e "MP_NUM=8" \
-e "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.1 /bin/bash \
-c -ex 'start_server $model_name && tail -f /dev/null'&& docker logs -f $(docker ps -lq)
```
After starting the command, the model will automatically begin downloading and deploying the service. Users can verify if the model is successfully deployed via curl requests. For more details about DeepSeek-R1 deployment and request parameters, please refer to DeepSeek Deployment.
```shell
curl ${ip}:9965/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
      "model":"default",
      "text":"Hello, how are you?"
  }'
```

**For detailed deployment commands, please refer to [DeepSeek Deployment](../../docs/predict/deepseek.md).**

## 2. Comprehensive PaddlePaddle Data Distillation - Training - Evaluation Solution

We divide the data distillation solution into three parts: data distillation, model training, and model evaluation. Relevant code has been uploaded to PaddleNLP.

## Environment Preparation

### Install PaddleNLP Components
```shell
pip install --upgrade paddlenlp==3.0.0b4
```

Alternatively, install the latest develop branch code with:

```shell
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

### Install High-Performance Operator Components
Please refer to the [documentation](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/csrc) to install high-performance operator components.

For Python 3.10 users, precompiled versions can be installed with:
```shell
pip install --pre --upgrade paddlenlp_ops -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

### Install Data Distillation Dependencies

```shell
pip install -r requirements.txt
```

### 2.1. Data Distillation

We will perform data distillation using the Chinese version of the GSM8K dataset, then convert it to JSONL format.
```python
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from copy import deepcopy

from datasets import load_dataset

# Convert data for distillation
# GSM8K
dataset = load_dataset("meta-math/GSM8K_zh")["train"]
dataset.to_json("data/gsm8k_zh/GSM8K_zh.jsonl", force_ascii=False)
```
```shell
python distill_data.py \
    --input_file "./data/gsm8k_zh/GSM8K_zh.jsonl" \
    --output_dir "./data/GSM8K_distilled_zh" \
    --prompt_key "question_zh" \
    --response_key "deepseek_r1_response_zh" \
    --reasoning_key "deepseek_r1_reasoning_zh" \
    --prompt_suffix "\n请一步一步地推理，并将你的最终答案放在\boxed{}中。" \
    --base_urls "http://192.168.0.1:9965/v1,http://192.168.0.2:9965/v1" \
    --model deepseek-r1:671b \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens 32768 \
    --concurrency 16
```

Detailed explanations of the invocation parameters are as follows:
```text
--input_file: This parameter is used to specify the path of the input JSONL file. For example, in previous cases, we used the meta-math_gsm8k_zh.jsonl file as the input source.
--output_dir: This parameter is used to set the output directory, which will be used to save the distilled JSONL files. This directory also supports checkpointing for distillation, meaning that if the process is interrupted, you can rerun the script to resume distillation without starting over.
--prompt_key: This parameter specifies the field name in the input JSONL file used as the prompt. In the GSM8K dataset, we used the content of the question_zh field as the question text.
--response_key: This parameter specifies the field name in the output JSONL file used as the response. For example, we selected deepseek_r1_response_zh as the field for output response text.
--reasoning_key: This parameter specifies the field name in the output JSONL file used for reasoning. In the example, we used deepseek_r1_reasoning_zh as the field for output reasoning text.
--prompt_suffix: This parameter allows you to append additional content after the prompt text. For DeepSeek R1 processing math problems, we recommend appending "\nPlease reason step by step, and put your final answer within \boxed{}." to the prompt to align with DeepSeek R1 best practices.
--base_urls: This parameter specifies service endpoints. You can input multiple addresses separated by commas for load balancing. If the service is locally deployed, set this to http://127.0.0.1:8XXX/v1.
--api_keys: This parameter provides service API keys. You can input multiple keys separated by commas, corresponding to the base_urls addresses. No need to set this parameter for local deployments.
--model: This parameter specifies the target model name for distillation, e.g., deepseek-r1:671b.
--temperature: This parameter sets the generation temperature to control the randomness of the output.
--top_p: This parameter sets the top-p sampling threshold to control the diversity of the generated content.
--max_tokens: This parameter specifies the maximum token limit for generated content to ensure reasonable output length.
--concurrency: This parameter sets the number of concurrent requests to the API. You can adjust this to control parallel processing.

```

During execution, you will see a progress bar like the following, indicating distillation progress. Three files will be generated in the output_dir: the distilled dataset, API request logs, and current distillation status.

```text
[deepseek-r1:7b] Data Distilling Progress:   0%|▊           | 36/8792 [05:57<15:29:58,  6.37s/it]

tree ./data/GSM8K_distilled_zh
./data/GSM8K_distilled_zh
├── distilled-GSM8K_zh.jsonl
├── distilled-GSM8K_zh.log
└── distilled-GSM8K_zh.status

0 directories, 3 files
```

Since we used 8-way concurrency, the final distilled dataset may be out of order. To restore the original dataset order, you can use the following code:
```python
from datasets import load_dataset, Dataset

# Load the JSON format dataset using the datasets library
ds = load_dataset("json", data_files="./data/GSM8K_distilled_zh/distilled-GSM8K_zh.jsonl", split="train")

# Convert the Dataset object to a Pandas DataFrame
df = ds.to_pandas()

# Sort the DataFrame by line number
sorted_df = df.sort_values(by='line_num')

# Convert the sorted DataFrame back to a Dataset
sorted_ds = Dataset.from_pandas(sorted_df)

# Save the sorted Dataset to JSON format
sorted_ds.to_json("./data/GSM8K_distilled_zh/sorted-distilled-GSM8K_zh.jsonl", force_ascii=False)
```

Finally, we will obtain a dataset with the following new fields: `deepseek_r1_reasoning`, `deepseek_r1_response`, `deepseek_r1_completion_tokens`, and `deepseek_r1_prompt_tokens`, corresponding to the reasoning process and response from DeepSeek R1 respectively.
Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.

#### 72

**Answer**
\boxed{72}
Next, we can perform further distillation training on the distilled data. The dataset used here is [PaddlePaddle/GSM8K_distilled_zh](https://huggingface.co/datasets/PaddlePaddle/GSM8K_distilled_zh). Users can modify the input and output data to be the distilled data themselves. The format supported by PaddleNLP is as follows:

* src : str, List(str), the input instruction (instruction) or prompt (prompt) for the model, specifying the task the model should perform.
* tgt : str, List(str), the output of the model.

Sample data:
```json
{"src": "Give three tips for staying healthy.", "tgt": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."}
......
```
```python
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from copy import deepcopy

from datasets import load_dataset

# PaddlePaddle/GSM8K_distilled_zh
dataset = load_dataset("PaddlePaddle/GSM8K_distilled_zh")
dataset["train"].to_json("data/gsm8k_distilled_zh/GSM8K_distilled_zh-train.json", force_ascii=False)
dataset["test"].to_json("data/gsm8k_distilled_zh/GSM8K_distilled_zh-test.json", force_ascii=False)

# Generate data for SFT
def process_data_zh(example):
    src = example.get("question_zh", "")
    content = example.get("deepseek_r1_response_zh", "")
    reasoning_content = example.get("deepseek_r1_reasoning_zh", "")
    tgt = reasoning_content + content
    return {"src": src, "tgt": tgt}

def process_data_en(example):
    src = example.get("question", "")
    content = example.get("deepseek_r1_response", "")
    reasoning_content = example.get("deepseek_r1_reasoning", "")
    tgt = reasoning_content + content
    return {"src": src, "tgt": tgt}

# Construct Chinese SFT dataset
paddlenlp_datatset = deepcopy(dataset)
paddlenlp_datatset["train"] = paddlenlp_datatset["train"].map(
    process_data_zh, remove_columns=paddlenlp_datatset["train"].column_names
)
paddlenlp_datatset["test"] = paddlenlp_datatset["test"].map(
    process_data_zh, remove_columns=paddlenlp_datatset["test"].column_names
)
paddlenlp_datatset["train"].to_json("data/gsm8k_distilled_zh_sft/train.json", force_ascii=False)
paddlenlp_datatset["test"].to_json("data/gsm8k_distilled_zh_sft/dev.json", force_ascii=False)

# Construct English SFT dataset
paddlenlp_datatset = deepcopy(dataset)
paddlenlp_datatset["train"] = paddlenlp_datatset["train"].map(
    process_data_en, remove_columns=paddlenlp_datatset["train"].column_names
)
paddlenlp_datatset["test"] = paddlenlp_datatset["test"].map(
    process_data_en, remove_columns=paddlenlp_datatset["test"].column_names
)
paddlenlp_datatset["train"].to_json("data/gsm8k_distilled_en_sft/train.json", force_ascii=False)
paddlenlp_datatset["test"].to_json("data/gsm8k_distilled_en_sft/dev.json", force_ascii=False)
```
Natalia sold 48 clips in April. In May, she sold half of April's amount, which is 48 ÷ 2 = 24 clips. Therefore, the total number of clips sold in April and May is 48 + 24 = 72.

\boxed{72}
### 2.2. Model Training
By fine-tuning the distilled model, the model can acquire reasoning capabilities. The distilled data requires long-text training capabilities. PaddleNLP has optimized fine-tuning performance to the extreme, supporting 128K long-context training.

The Paddle framework leverages its unique FlashMask high-performance variable-length attention masking technology, combined with Zero Padding data flow optimization strategy in PaddleNLP, achieving significant memory reduction. This innovation enables fine-tuning code to seamlessly scale from 8K to unprecedented 128K long-text training, with training efficiency improved by 1.8x compared to LLama-Factory.

Here we employ full-parameter supervised fine-tuning (SFT) algorithm to precisely adjust small model parameters. This process is highly efficient, requiring only input models and datasets to complete fine-tuning and model compression. Additionally, we provide one-click multi-GPU training, mixed-precision training, gradient accumulation, checkpointing and logging. We also encapsulate common configurations like optimizer selection and learning rate scheduling. For more fine-tuning algorithms, please refer to the large model fine-tuning configuration guide.

Use the following command to fine-tune the model with `Qwen/Qwen2.5-Math-7B` as pretrained model, saving the fine-tuned model to specified path. When using GPU environment, specify gpus parameter for multi-GPU training. Training configuration refers to `sft_argument.json`.

Supervised Fine-tuning (SFT)
```shell
python -u -m paddle.distributed.launch \
    --devices "0,1,2,3,4,5,6,7" \
    ../../run_finetune.py \
    sft_argument.json
```

The `sft_argument.json` configuration file is as follows. For detailed parameter explanations, please refer to [documentation](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/finetune.md):
```json
{
  "model_name_or_path": "Qwen/Qwen2.5-Math-7B",
  "dataset_name_or_path": "./data/gsm8k_distilled_en_sft",
  "output_dir": "./checkpoints/Qwen2.5-Math-7B/gsm8k_distilled/",
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "per_device_eval_batch_size": 1,
  "eval_accumulation_steps": 16,
  "num_train_epochs": 3,
  "learning_rate": 3e-05,
  "warmup_steps": 30,
  "logging_steps": 1,
  "evaluation_strategy": "no",
  "save_strategy": "epoch",
  "src_length": 8192,
  "max_length": 16384,
  "bf16": true,
  "fp16_opt_level": "O2",
  "do_train": true,
  "do_eval": false,
  "eval_steps": 10,
  "disable_tqdm": true,
  "load_best_model_at_end": false,
  "eval_with_do_generation": false,
  "metric_for_best_model": "accuracy",
  "recompute": true,
  "save_total_limit": 1,
  "tensor_parallel_degree": 1,
  "pipeline_parallel_degree": 1,
  "sharding": "stage2",
  "zero_padding": true,
  "unified_checkpoint": true,
  "use_flash_attention": true,
  "flash_mask": true
}
```

### 2.3. Model Evaluation
We provide evaluation scripts on GSM8K to facilitate comparison of the model's mathematical reasoning capabilities after fine-tuning. For detailed implementation, please refer to the evaluation scripts.
```shell
python -u -m paddle.distributed.launch \
    --devices 0,1,2,3 \
    distill_eval.py \
    --eval_file ./data/gsm8k_distilled_zh/GSM8K_distilled_zh-test.json \
    --eval_question_key question_zh \
    --eval_answer_key answer_only \
    --eval_prompt "\nPlease reason step by step, and put your final answer within \boxed{}." \
    --model_name_or_path "./checkpoints/Qwen/Qwen2.5-Math-7B/gsm8k_distilled/" \
    --inference_model true \
    --dtype bfloat16 \
    --batch_size 32 \
    --use_flash_attention true \
    --src_length 1024 \
    --max_length 3072 \
    --total_max_length 4096 \
    --decode_strategy greedy_search \
    --temperature 0.95 \
    --top_p 0.6 \
    --data_file ./data/gsm8k_distilled_zh/GSM8K_distilled_zh-test.json \
    --eval_results results-gsm8k/Qwen/Qwen2.5-Math-7B/output_zh.json
```
On the GSM8K (0-shot) benchmark, the Qwen2.5-Math-7B model (83.82%) distilled by PaddleNLP demonstrates significant performance improvement over the original model (68.21%), achieving an accuracy boost of approximately 15.61 percentage points.

| Model Name                           | GSM8K(en, 0-shot) |
|--------------------------------------|:-----------------:|
| Qwen2.5-Math-7B                      |       68.2        |
| DeepSeek-R1-Distill-Qwen-7B          |       77.3        |
| Qwen2.5-Math-7B (PaddleNLP Distill)  |       83.8        |

## 3. Efficient and Fast Paddle Model Local Deployment Solution

After model evaluation, users can deploy the model for downstream tasks. PaddleNLP provides high-performance dynamic graph deployment (user-friendly) and service-oriented deployment (efficient and stable) solutions to accommodate different scenarios. Through testing in L20 environments, our service-oriented deployment solution demonstrates outstanding inference performance, significantly surpassing other open-source solutions.

|         Distill Model         | Framework | Precision | Concurrency | Avg First Token Latency (s) | Avg Full Sentence Latency (s) | QPS  | OTPS   | Improvement |
|:-----------------------------:|:---------:|:---------:|:-----------:|:--------------------------:|:-----------------------------:|:----:|:------:|:-----------:|
| DeepSeek-R1-Distill-Qwen-32B  | vLLM-0.7.3 | W8A8      |      64     |            1.29            |             23.71             | 2.63 | 496.76 |             |
|                               | paddlenlp | W8A8      |      64     |            3.76            |             18.81             | 3.32 | 626.22 |     26%     |
| DeepSeek-R1-Distill-Qwen-14B  | vLLM-0.7.3 | W8A8      |     256     |            1.05            |             40.39             | 6.02 | 1131.94|             |
|                               | paddlenlp | W8A8      |     256     |            4.77            |             18.72             | 12.37| 1873.14|     65%     |
|  DeepSeek-R1-Distill-Qwen-7B  | vLLM-0.7.3 | W8A8      |     256     |            0.53            |             19.82             | 12.27| 2310.39|             |
|                               | paddlenlp | W8A8      |     256     |            0.45            |             10.46             | 22.23| 3842.89|     66%     |
| DeepSeek-R1-Distill-Qwen-1.5B | vLLM-0.7.3 | W8A8      |     256     |            0.23            |              8.34             | 28.11| 5259.05|             |
|                               | paddlenlp | W8A8      |     256     |            2.92            |              6.70             | 36.51| 6884.9 |     31%     |

Here we summarize the deployment process. Users can refer to the [tutorial](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/general_model_inference.md) for hands-on practice:
- **Model Dynamic-to-Static Conversion**: Convert dynamic graph models to static graph models for efficient inference deployment.
- **Model Service-oriented Deployment**: Deploy static graph models as services for convenient invocation.
