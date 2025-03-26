# PaddlePaddle Large Model Fine-Tuning Documentation

## 1. PaddlePaddle Fine-Tuning Features
Large Model Fine-Tuning (Supervised Fine-Tuning, SFT) is a crucial component of Large Language Models (LLMs). Its primary objectives are to enable models to follow instructions and generate expected responses, effectively enhance the performance of general models in specific domains and application scenarios, and better meet personalized applications of large models. This method is used to improve and customize pre-trained large language models.

- **Easy-to-Use Parallel Strategies**: Supports pure Data Parallelism, Sharding Parallelism, Tensor Parallelism, Pipeline Parallelism, and Sequence Parallelism.
- **Multiple Precision Training**: Full-parameter fine-tuning with 16/32-bit, LoRA fine-tuning with 4/8/16-bit, and mixed-precision quantized LoRA.
- **Extreme Performance Optimization**: FlashAttention-2, FlashMask, Greedy Zero Padding.
- **Advanced Fine-Tuning Strategies**: LoRA+, PiSSA, rsLoRA, NEFTune, VeRA, MoRA, ReFT, MoSLoRA.

For more algorithm details, please refer to [PaddlePaddle Large Model Algorithm Documentation](algorithm_overview.md).

## 2. Introduction to Large Model Fine-Tuning

Here we introduce commonly used SFT techniques:
<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/4556e9f0-d855-418f-914f-bcecccce6dba">
</div>
<div align="center">
    <font size ="1">
    Principles of Large Model Fine-Tuning
    </font>
</div>

- **Full-Parameter Fine-Tuning**: The most common SFT technique, retraining all parameters of the pre-trained model on instruction datasets. This method typically delivers the best results but requires substantial computational resources.

- **LoRA**: Low-Rank Adaptation is the most widely used Parameter-Efficient Fine-Tuning (PEFT) technique. Instead of retraining the entire model, it freezes original weights and introduces low-rank matrices to each target linear layer. This reduces the number of trainable parameters by over 99%, significantly decreasing memory usage and training time.

- **QLoRA**: Quantization-Aware Low-Rank Adaptation reduces memory usage by up to 33% compared to standard LoRA, making it particularly useful in GPU memory-constrained scenarios. QLoRA typically takes about 20% more time than regular LoRA, but its significant memory savings make it the only viable option when GPU memory is limited.

## 3. Quick Start

Next, we will use **Llama 3** as an example to demonstrate how to perform full-parameter SFT and LoRA fine-tuning using a unified script.

### 3.1 Environment Preparation

- PaddlePaddle 3.0-beta
- PaddleNLP 3.0.0b3
- PaddleSlim develop

git clone the code to your local machine, and you're ready to start.
```bash
    git clone https://github.com/PaddlePaddle/PaddleNLP.git
    # pip install ./PaddleNLP using develop version
    cd PaddleNLP/llm
    # enter running directory
```

### 3.2 Fine-tuning Data Preparation

For user convenience in testing, we provide an example dataset [Advertisement Generation Dataset](https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz). Users can also follow the dataset format to create their own datasets for fine-tuning. The supported data format requires each line to contain a dictionary with the following fields:

- `src`: `str, List(str)`, the model's input instruction (instruction), prompt, or the task the model should perform.
- `tgt`: `str, List(str)`, the model's output.

Sample data:
```
{"src": "type#dress*color#blue*style#fresh*pattern#bow", "tgt": "The dress features three-dimensional bow decorations with blue stripes, creating a full and layered silhouette while adding a touch of sweetness. This design highlights the girl's fresh and charming appearance."}
...
```

### 3.3 Full-Parameter Fine-tuning

```
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_finetune.py ./config/llama/sft_argument.json
```

Notes:
1. Setting both `zero_padding` and `greedy_zero_padding` to True helps improve training efficiency. It's recommended to set `per_device_train_batch_size` to 1, control batch size via `gradient_accumulation_steps`, and adjust `max_length` appropriately.
2. Set `use_flash_attention` to True to enable FlashAttention. With FlashAttention enabled, set `flash_mask` to True to enable FlashMask.
3. The SFT API supports 4D parallel strategy. Adjust via `tensor_parallel_degree`, `pipeline_parallel_degree`, `sharding`, and `sharding_parallel_degree`.

### 3.4 PEFT
#### 3.4.1 LoRA/QLoRA

```bash
python -u -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py --config ./config/llama/sft_argument.json --lora ./config/llama/lora_argument.json
# For QLoRA:
python -u -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py --config ./config/llama/sft_argument.json --q_lora ./config/llama/lora_argument.json
```
```bash
# Single-GPU LoRA
python  run_finetune.py ./config/llama/lora_argument.json

# Single-GPU QLoRA
python  run_finetune.py ./config/llama/qlora_argument.json

# Multi-GPU LoRA
python  -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7"  run_finetune.py ./config/llama/lora_argument.json

# Multi-GPU QLoRA
python  -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7"  run_finetune.py ./config/llama/qlora_argument.json
```

**Note:**
1. Setting both `zero_padding` and `greedy_zero_padding` to True improves training efficiency. It is recommended to set `per_device_train_batch_size` to 1, use `gradient_accumulation_steps` to control batch size, and adjust `max_length` appropriately.
2. The LoRA strategy is applied to all Linear layers by default.
3. The backbone model can be quantized to low bits by setting `weight_quantize_algo`, e.g., 'weight_only_int4', 'weight_only_int8', 'nf4' or 'fp4'. Refer to the fine-tuning parameter description for details.
4. Set `use_flash_attention` to True to enable FlashAttention. When FlashAttention is enabled, set `flash_mask` to True to enable FlashMask.
5. The LoRA API supports 4D parallel strategy. Adjust the parallel training strategy by controlling `tensor_parallel_degree`, `pipeline_parallel_degree`, `sharding`, and `sharding_parallel_degree`, enabling **LoRA fine-tuning of hundred-billion parameter models on single machines**.
6. Supports algorithms like rsLoRA, LoRa+, PiSSA, and MosLoRA (currently not supporting tensor model parallelism) through parameters `rslora`, `lora_plus_scale`, `pissa`, `lora_use_mixer`, `use_mora`, etc.

To facilitate subsequent **compression** and **static graph inference**, we provide a LoRA parameter merging script that integrates LoRA parameters into the backbone model and saves the corresponding weights.
```bash
python merge_lora_params.py \
    --model_name_or_path ./base_model \
    --lora_path ./checkpoints/lora_ckpts \
    --output_path ./checkpoints/lora_merge \
    --device "gpu" \
    --safe_serialization True
```

<summary>&emsp; Script Parameter Description</summary><div>

- `lora_path`: Path to LoRA parameters and configuration for initializing LoRA parameters, default is None.
`model_name_or_path`: Required, path to the backbone model parameters, default None.
- `merge_model_path`: Required, path to save merged parameters, default None.
- `device`: Running environment, default gpu.
- `safe_serialization`: Whether to save as safetensor format, default True.
</div>


#### 3.4.2 Prefix Tuning
```
# Single-GPU Prefix Tuning
python run_finetune.py ./config/llama/pt_argument.json

# Multi-GPU Prefix Tuning
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_finetune.py ./config/llama/pt_argument.json
```

#### 3.4.3 VeRA
```
# Single-GPU VeRA
python run_finetune.py ./config/llama/vera_argument.json

# Multi-GPU VeRA (tensor model parallelism not currently supported)
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_finetune.py ./config/llama/vera_argument.json
```

For subsequent **compression** and **static graph inference**, we provide a VeRA parameter merging script to integrate VeRA parameters into the backbone model and save corresponding weights.
```
python merge_vera_params.py \
    --model_name_or_path ./base_model \
    --vera_path ./checkpoints/vera_ckpts \
    --merge_vera_model_path ./checkpoints/vera_merge \
    --device "gpu" \
    --safe_serialization True
```

<summary>&emsp; Script Parameter Description</summary><div>

- `vera_path`: Path to VeRA parameters and configuration for initialization, default None.
- `model_name_or_path`: Required, path to backbone model parameters, default None.
- `merge_vera_model_path`: Required, path to save merged parameters, default None.
- `device`: Running environment, default gpu.
</div>

#### 3.4.4 LoKr
```
# Single-GPU LoKr
python run_finetune.py ./config/llama/lokr_argument.json

# Multi-GPU LoKr (tensor model parallelism not currently supported)
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_finetune.py ./config/llama/lokr_argument.json
```

For subsequent **compression** and **static graph inference**, we provide a LoKr parameter merging script to integrate LoKr parameters into the backbone model and save corresponding weights.
```python merge_lokr_params.py \
    --model_name_or_path ./base_model \
    --lokr_path ./checkpoints/lokr_ckpts \
    --merge_lokr_model_path ./checkpoints/lokr_merge \
    --device "gpu" \
    --safe_serialization True
```

<summary>&emsp; Script Parameters</summary><div>

- `lokr_path`: Path to LoKr parameters and configurations for initialization, defaults to None.
- `model_name_or_path`: Required, path to backbone model parameters, defaults to None.
- `merge_lokr_model_path`: Required, path to save merged parameters, defaults to None.
- `device`: Execution environment, defaults to gpu.
</div>

#### 3.4.4 ReFT
```
# Single-GPU ReFT
python  run_finetune.py ./config/llama/reft_argument.json

# Multi-GPU ReFT (Tensor model parallelism not currently supported)
python  -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7"  run_finetune.py ./config/llama/reft_argument.json
```
ReFT currently only supports dynamic graph prediction. Prediction script:
```
python ./predict/reft_predictor.py \
    --model_name_or_path ./base_model \
    --reft_path ./checkpoints/lokr_ckpts \
    --output_file output.json \
    --batch_size 1 \
    --data_file "./data/dev.json"
    --max_length 4096
```

<summary>&emsp; Script Parameters</summary><div>

- `reft_path`: Path to ReFT parameters and configurations for initialization.
- `model_name_or_path`: Path to backbone model parameters.
- `batch_size`: Batch size. Larger values increase GPU memory usage, smaller values reduce GPU memory usage.
- `data_file`: JSON file for inference, defaults to None. Example data:
    ```json
    {"tgt":"", "src": "Write a 300-word novel outline about Li Bai time-traveling to modern times and becoming a corporate clerk"}
    {"tgt":"", "src": "Create a list of 5 questions for interviewing a sci-fi writer"}
    ```
- `output_file`: File to save inference results.
- `src_length`: Maximum token length for model input context.
- `max_length`: Maximum token length for model input (context + generated content).
</div>

## 4. Fine-tuning Parameters Introduction
<summary>&emsp; Model Parameters (ModelArgument) </summary><div>

- `model_name_or_path`: Pretrained model name or local path, used to warm-start the model and tokenizer, defaults to None. For **supported model weights** of each model, please refer to the respective model directories.
- `use_flash_attention`: Whether to use FlashAttention, defaults to False.
- `flash_mask`: Whether to use FlashMask, defaults to False. Please enable FlashAttention first.
- `lora`: Whether to enable LoRA fine-tuning strategy, defaults to False.
- `lora_path`: Path for LoRA parameters and configuration to initialize LoRA parameters, defaults to None.
- `lora_rank`: Rank value in LoRA algorithm, defaults to 8.
- `rslora`: Whether to use rsLoRA algorithm.
- `lora_plus_scale`: Whether to use LoRA+, setting the learning rate ratio between B and A.
- `neftune`: Whether to use [NEFT](https://arxiv.org/abs/2310.05914) for fine-tuning, defaults to False.
- `neftune_noise_alpha`: NEFT alpha parameter, defaults to 5.0.
- `vera`: Whether to enable [VeRA](https://arxiv.org/abs/2310.11454) fine-tuning strategy, defaults to False.
- `vera_rank`: Rank value in VeRA algorithm, defaults to 8.
- `lokr`: Whether to enable [LoKr](https://arxiv.org/abs/2309.14859) fine-tuning strategy, defaults to False.
- `lokr_rank`: Rank value in LoKr algorithm, defaults to 8.
- `use_long_sequence_strategies`: Whether to use long sequence extension strategies, defaults to False.
- `reft`: Whether to enable [ReFT](https://arxiv.org/abs/2404.03592) fine-tuning strategy, defaults to False.
- `use_mora`: Whether to enable [MoRA](https://arxiv.org/abs/2405.12130) fine-tuning strategy, defaults to False.
- `lora_use_mixer`: Whether to enable [MosLoRA](https://arxiv.org/abs/2406.11909) strategy, defaults to False.
- `pissa`: Whether to enable [PiSSA](https://arxiv.org/abs/2404.02948) strategy, defaults to False.
- `strategy_type`: Type of long sequence extension strategy, defaults to None.
- `strategy_name`: Specific name of long sequence extension strategy, defaults to None.
- `rope_scaling_factor`: Scaling factor when applying RoPE extension strategy.
</div>

<summary>&emsp; Data Parameters (DataArgument)</summary><div>

- `dataset_name_or_path`: Local dataset directory or built-in dataset name, defaults to None. The script automatically handles single-file and multi-file scenarios, searching for `dataset_name_or_path/train.json` or `dataset_name_or_path/train/*.json` as training files, and `dataset_name_or_path/dev.json` or...
`dataset_name_or_path/dev/*.json` as the validation set file.
- `zero_padding`: Whether to use Zero Padding dataflow (reduces padding redundancy computation, significantly improves effective token computation efficiency), defaults to False. When `eval_with_do_generation` is set to True, evaluation process does not support Zero Padding dataflow.
- `greedy_zero_padding`: Greedy Zero Padding dataflow, defaults to False. Please enable this based on setting `zero_padding` to True.
- `src_length`: Maximum token length of model input context, defaults to 1024.
- `max_length`: Maximum token length of model input (context + generated content), defaults to 2048. When `zero_padding` is set to True, it also serves as the maximum input length for Zero Padding dataflow model training. It is recommended to set this to the maximum allowed input length of the model, with `per_device_train_batch_size` set to 1 and using `gradient_accumulation_steps` to control batch size.
- `lazy`: Set to False to use `MapDataset`, set to True to use `IterDataset`, defaults to False. For large datasets, it is recommended to set to True. `IterDataset` avoids loading all data into memory at once. Note: Requires setting `max_steps` and setting `evaluation_strategy` and `save_strategy` to `steps`.
- `autoregressive`: Whether to use autoregressive generation (i.e., training data is unsupervised), defaults to False.
- `use_pose_convert`: Whether to use PoSE algorithm for data processing, defaults to False.

</div>

<summary>&emsp; Generation Parameters (GenerateArgument)</summary><div>

Note: The following parameters only take effect when `eval_with_do_generation` is True and model.generate() is called.

- `top_k`: Number of highest probability tokens to keep for top-k filtering in "sampling" strategy. Defaults to 1, equivalent to greedy strategy.
- `top_p`: Cumulative probability for top-p filtering in "sampling" strategy. Defaults to 1.0, indicating no effect.
</div>

<summary>&emsp; Training Parameters (TrainingArguments)</summary><div>

The following only introduces some commonly used parameters in TrainingArguments. For details, please refer to [TrainingArguments Documentation](https://paddlenlp.readthedocs.io/zh/latest/trainer.html).

- `output_dir`: Directory for saving related files, mainly including model-related files, checkpoints during training, tokenizer-related files, and evaluation result files. Defaults to None.
- `per_device_train_batch_size`: Batch size for training set, corresponding to micro batch size. Defaults to 8. This parameter needs to be set according to the specific dataset. Larger values require higher GPU memory and increase training costs, while smaller values reduce GPU memory usage and speed up training.
- `gradient_accumulation_steps`: The number of steps for gradient accumulation. As the name suggests, this parameter accumulates gradients over multiple steps before performing a single parameter update, with a default value of 1. This is equivalent to multiplying the original training batch size by `gradient_accumulation_steps`.
- `per_device_eval_batch_size`: The evaluation batch size for the validation set, corresponding to micro batch size, with a default of 8. Larger values consume more GPU memory, while smaller values reduce memory usage.
- `num_train_epochs`: The number of training epochs, with a default of 3.
- `learning_rate`: The initial learning rate for the optimizer, with a default of 5e-5.
- `warmup_steps`: The number of warmup steps, defaulting to 0. When `warmup_steps` > 0, it overrides the `warmup_ratio` setting.
- `evaluation_strategy`: The evaluation strategy, defaulting to "no". Options: "no" (no evaluation during training), "steps" (evaluate every `eval_steps`), "epoch" (evaluate at each epoch end).
- `save_strategy`: The model saving strategy, defaulting to "no". Options: "no" (no saving during training), "steps" (save every `eval_steps`), "epoch" (save at each epoch end).
- `fp16`: Whether to enable FP16 training to accelerate training, defaulting to False.
- `bf16`: Whether to enable BF16 training to accelerate training, defaulting to False.
- `fp16_opt_level`: Can be set to O1 or O2. At O1 level, whitelisted ops use float16/bfloat16 while blacklisted ops use float32. At O2 level, model parameters are converted to float16/bfloat16, and ops use float16/bfloat16 only if all floating-point inputs are float16/bfloat16; otherwise, float32 is used. Default is O1.
- `do_train`: Whether to enable training, defaulting to False.
- `do_eval`: Whether to enable evaluation, defaulting to False.
- `recompute`: Whether to enable recomputation (currently supports full strategy). Enabling this can reduce GPU memory usage to allow larger batch sizes, defaulting to False.
- `refined_recompute`: Fine-grained recomputation that balances memory and performance by precisely controlling recomputed components. Currently only supports `llama` series and `qwen` series models. For detailed usage, refer to the [TrainingArguments documentation](#这里).
- `tensor_parallel_degree`: The degree of tensor parallelism, indicating the number of splits for a transformer layer. Note: This method increases communication overhead. Recommended value ≤8, preferably using intra-machine communication. Default is -1 (disabled).
- `pipeline_parallel_degree`: The degree of pipeline parallelism. (E.g., if set to 4 for a 12-layer model, each pipeline stage contains 3 layers.) Default is -1 (disabled).
- `sharding_parallel_degree`: Indicates the Sharding parallelism size for grouped parameter sharding. Default value is 1, meaning group parameter sharding is not enabled.
- `sharding`: Whether to use Paddle's Sharding data parallelism. Supports sharding `stage1`, `stage2` or `stage3`. Note that `stage2` and `stage3` can be combined with `offload`.
- `optim`: Default is `adamw`, supports `adamw`, `adamw_mini`.
</div>



<summary>&emsp; Representation Fine-Tuning (ReFT) Parameters (ReftArgument) </summary><div>

- `model_name_or_path`: Pre-trained model name or local model path for warm-starting the model and tokenizer. Default is None. Supported model weights are detailed in each model's documentation.
- `layers`: Which layers of the model to intervene. Default is all, meaning intervening in all layers.
- `position`: Which positions of tokens to intervene. Default is f7, indicating intervention in the first 7 tokens.
- `intervention_type`: Type of intervention network. Default is LoReftIntervention.
- `rank`: Low-rank dimension of the intervention network. Default is 8.
- `act_fn`: Activation function in the intervention network. Default is linear.
- `add_bias`: Whether to add bias in the intervention network. Default is False.
- `dropout`: Dropout rate in the intervention network. Default is 0.00.
</div>
