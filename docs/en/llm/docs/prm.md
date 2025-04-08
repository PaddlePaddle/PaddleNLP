# PaddlePaddle Large Model Suite PRM Documentation
## 1. Algorithm Overview
The Process Reward Model (PRM) provides reward signals for each reasoning step and the final answer, addressing the issue where Final Answer Incorrect but Process Correct (FP) cases might be mistakenly awarded positive rewards in Outcome Reward Models (ORM).

Implementation-wise, PRM employs next token prediction with binary classification at the end of each step to determine correctness.

## 2. Quick Start
Here we demonstrate PRM implementation using **Llama 3** as an example.
### 2.1 Environment Setup
- PaddlePaddle 3.0-beta
- PaddleNLP develop

Clone the repository to begin:

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git
# pip install ./PaddleNLP (use develop version)
cd PaddleNLP/llm
# Proceed to working directory
```

### 2.2 Data Preparation
We support PRM data in JSON format with each entry containing:
- `src`: `str, List(str)`, user dialogue content
- `tgt`: `str, List(str)`, system response content
- `responses`: `List(str)`, reasoning step responses
- `labels`: `List(str)`, corresponding labels (one positive and one negative marker) matching `responses` length

Sample data:

[
    {
        "src": [
            "Tony has $87. He needs to buy some cheese, which costs $7 a pound and a pound of beef that costs $5 a pound. After buying the beef and his cheese, he has $
```json
{
    "type": "prm",
    "samples": [
        {
            "query": "At the market, 7 pounds of beef costs $7, and 5 pounds of beef costs $5. He spent $87 in total. He bought the same weight of each. How many pounds of cheese did he buy?",
            "response": "Step 1: He bought 7 / 5 = <<7/5=1.4>>1.4 pounds of beef.\nStep 2: He spent 7 + 5 = <<7+5=12>>12 on beef and cheese.\nStep 3: So, he spent 12 - 87 = 75.\nStep 4: That means he bought 87 - 75 = <<87-75=12>>12 pounds of cheese. The answer is: 12"
        },
        ...
    ]
},
...
]

For testing convenience, we preprocess the [Math-Shepherd](https://huggingface.co/datasets/peiyi9979/Math-Shepherd) dataset into the corresponding format. Usage instructions:

```bash
wget https://bj.bcebos.com/paddlenlp/datasets/examples/math-shepherd.tar.gz
tar -zxvf math-shepherd.tar.gz
```

### 2.3 PRM Training

```bash
# PRM startup command reference
python -u -m paddle.distributed.launch --gpus "0,1,2,3" ./llm/alignment/rm/flashmask/run_reward.py ./llm/config/mistral/prm_flashmask_argument.json
```

## 3. PRM Parameter Specifications

### Training Arguments
- `output_dir`: Directory for saving related files including models, checkpoints, tokenizer files, evaluation results, etc. Default: `"./checkpoints/dpo_ckpts"`;
- `per_device_train_batch_size`: Training batch size per device. Default: `1`;
- `gradient_accumulation_steps`: Number of gradient accumulation steps. Default: `8`, meaning parameters are updated every `8` steps;
- `per_device_eval_batch_size`: Evaluation batch size per device. Default: `1`;
- `num_train_epochs`: Number of training epochs. Default: `1`;
- `max_steps`: Maximum number of training steps. Default: `100`;
- `learning_rate`: Initial learning rate for optimizer. Default: `1e-06`;
- `warmup_steps`: Number of warmup steps. Default: 0. When warmup_steps>0, it overrides warmup_ratio. Default: `10`;
- `logging_steps`: Interval steps for logging. Default: `1`;
```
- `evaluation_strategy`: Evaluation strategy. "no": no evaluation during training; "steps": evaluate at every eval_steps; "epoch": evaluate at end of each epoch.
- `save_strategy`: Saving strategy. "no": no saving during training; "steps": save at every eval_steps; "epoch": save at end of each epoch.
- `eval_steps`: Evaluation interval in steps, default is `100`.
- `save_steps`: Model saving interval in steps, default is `500`.
- `bf16`: Whether to enable BF16 training. Enabling BF16 can accelerate training. Default is `True`.
- `fp16_opt_level`: Can be set to O1 or O2. At O1 level, operators in whitelist use float16/bfloat16 computation while blacklisted operators use float32. At O2 level, model parameters are converted to float16/bfloat16. Operators will use float16/bfloat16 only if all floating-point inputs are float16/bfloat16; otherwise, float32 is used. Default is O1. Default is `"O2"`.
- `do_train`: Whether to enable training. Default is `True`.
- `do_eval`: Whether to enable evaluation. Default is `True`.
- `load_best_model_at_end`: Whether to load best model at training end. Default is `True`.
- `tensor_parallel_degree`: This parameter indicates the number of splits for a transformer layer. This method increases communication overhead but saves memory. Recommended tensor_parallel_degree<=8, prefer intra-machine communication.
- `sharding_parallel_degree`: Data parallelism degree for sharded parameter slicing.
- `sharding`: Whether to use Sharding data parallelism. Default is `stage1`.
- `recompute`: Recompute strategy. Currently supports full strategy. Enabling can reduce memory to allow larger batch sizes. Full recompute slows training by ~30%.
- `recompute_granularity`: Recompute granularity. Can be `full`, `full_attn` or `core_attn`.
- `unified_checkpoint`: Whether to use unified checkpoint. Default is `True`.

### PRM Parameters
- `process_reward`: Whether to enable PRM training. `True` for PRM, `False` for ORM.

### Data Parameters (DataArgument)
- `train_dataset_path`: Path to training dataset.
- `dev_dataset_path`: Path to validation dataset.
- `max_seq_len`: Maximum sequence length for input. Default is `4096`.
- `max_prompt_len`: Maximum prompt length for input. Default is `2048`.
- `autotuner_benchmark`: Whether to enable autotuner benchmark. Default is `False`.
- `benchmark`: Whether to enable benchmarking. Default is `False`.
- `zero_padding`
### Parameters (DataArgument)
 - `zero_padding`: Whether to use zero padding, default is `True`;
 - `greedy_zero_padding`: Whether to use greedy zero padding, enabling this helps reduce padding ratio, default is `False`;
 - `lazy`: Whether to return `MapDataset` or `IterDataset`. `True` represents `IterDataset`, `False` represents `MapDataset`. Recommended to enable lazy mode for large datasets. Note that shuffle is not performed when lazy is True.

### Model Parameters (ModelArgument)
 - `model_name_or_path`: Name of the pre-trained model to use or local path to the model. Supported model weights vary by model directory;
 - `tokenizer_name_or_path`: Name or path of the pre-trained tokenizer, if different from the model;
 - `use_flash_attention`: Whether the model uses FlashAttention, default is `False`;
 - `recompute_granularity`: Granularity of recomputation, default is `"full"`;
 - `flash_mask`: Whether to use FlashMask, requires FlashAttention to be enabled;
 - `virtual_pp_degree`: Virtual pipeline parallelism degree, default is `1`;

### PRM Parameters
 - `placeholder_token`: Placeholder token used for PRM scoring at the end of each inference step. Should have length 1 in the current model's tokenizer, default is `ки` (requires dataset configuration);
 - `reward_tokens`: Tokens identifying PRM scoring. Should be a comma-separated string of two tokens, first being positive marker and second being negative marker, default is `"+,-"` (requires dataset configuration).
