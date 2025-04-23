# PaddlePaddle Large Model Suite DPO Documentation
## 1. Algorithm Introduction
Direct Preference Optimization (DPO) is an improvement over Reinforcement Learning from Human Feedback (RLHF). It demonstrates the mapping relationship between the reward function and the optimal policy, proving that this constrained reward maximization problem can be precisely optimized through single-stage policy training. DPO simplifies the training process and enhances model convergence stability.

Based on DPO, several derivative algorithms have been developed, such as SimPO, ORPO, etc. We can directly switch between different algorithms by modifying the loss_type configuration.

## 2. Quick Start
Next, we will use **Llama 3** as an example to demonstrate how to use the unified script for DPO.
### 2.1 Environment Preparation
- PaddlePaddle 3.0-beta
- PaddleNLP develop
- PaddleSlim develop

Clone the code to your local machine to begin:

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git
# pip install ./PaddleNLP (use develop version)
cd PaddleNLP/llm
# Proceed to runtime directory
```

### 2.2 Data Preparation
We support preference data format as JSON files where each line contains a dictionary with the following fields:

- `src`: `str, List(str)`, User conversation content.
- `tgt`: `str, List(str)`, System response content.
- `response`: `str, List(str)`, Contains chosen and rejected responses.
- `sort`: `List(int)`, The sort value distinguishes between chosen and rejected responses in response (smaller sort value indicates rejected, larger indicates chosen).

Sample data:

```json
{
    "src": "How to implement DPO?",
    "tgt": "Refer to the DPO documentation.",
    "response": [
        {"text": "Use the DPO example code.", "sort": 2},
        {"text": "Read the paper for details.", "sort": 1}
    ]
}
```
```text
{
    "src": ["In this task, you are given a second sentence. Your task is to generate the first sentence on the same topic but incoherent and inconsistent with the second sentence.\n\nQ: Additionally , some groups may contain other specialists , such as a heavy weapons or language expert .\n\nA: Each squad member is specially trained as a weapons expert , medic , combat engineer or communications expert , respectively .\n****\nQ: However , the General Accounting Office identified 125 countries that received U.S. training and assistance for their police forces during fiscal year 1990 at a cost of at least $117 million .\n\nA: No government agency is in charge of calculating the cost .\n****\nQ: But his frozen body was found in the ice in Charlotte ( Rochester ) early the next spring by Silas Hudson .\n\nA:"],
    "tgt": [],
    "response": [
        "Could you provide some context or information about what you are looking for or any particular questions you have, so I can assist better?",
        "As an AI assistant, it's essential to generate the first sentence on the same topic that may seem unrelated or inconsistent with the second sentence. Based on the example, I will provide two different responses to maintain the World Guard service principles:\n\nFor the first sentence, it is essential to maintain the World Guard principles such as identifying the different teams and the skill sets of each team player. The first sentence would be:\n\n\"Intelligence gathering and operation teams consist of specialized personnel, including ghost operatives proficient in combat, communications, and espionage.\"\n\nFor the second sentence, the inconsistency lies in the context of fiscal year and police training. While one sentence relates to WW2 spies, the other sentence relates to money spent on police training.\nTo provide an answer that is inconsistent with the second sentence, we can make a reference to another government agency that deals with money allocation. Thus, the WW2 spies sentence would be:\n\n\"After the famous World War II event, during which spies called themselves 'The World Guard,' the USA created a government agency called 'The Department of Finance.' Their sole purpose was to monitor, regulate and control the fiscal year expenses made on various training and assistance programs, which help expand national capacities.\"\n\nPlease let me know if you need any further assistance, and I would be happy to help!"
        ],

    "sort": [1, 0]
}
```
To facilitate testing, we have processed the [ultrafeedback_binarized demo](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) dataset into the corresponding format. Usage is as follows:

```bash
wget https://bj.bcebos.com/paddlenlp/datasets/examples/ultrafeedback_binarized.tar.gz
tar -zxvf ultrafeedback_binarized.tar.gz
```

### 2.3 DPO Training

```bash
# DPO startup command reference
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" ./alignment/dpo/run_dpo.py ./config/llama/dpo_argument.json

# DPO LoRA startup command reference
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" ./alignment/dpo/run_dpo.py ./config/llama/dpo_lora_argument.json
```

## 3. DPO Parameter Introduction
### Model Parameters (ModelArgument)
- `model_name_or_path`: The name of the pre-trained model or the local model path used to initialize the model and tokenizer. Supported model weights can be found in each model's directory.
- `use_flash_attention`: Whether to use FlashAttention for the model, default is `False`. Currently only supports llama.
- `flash_mask`: Whether to use FlashMask, requires FlashAttention to be enabled. Currently only supports llama.
- `lora`: Whether to use LoRA model, default is `False`.
- `ref_model_update_steps`: Steps to update the reference model's state dict, default is -1 (no update).
- `reference_free`: Whether to not use reference model, default is False. SimPO and ORPO force reference_free to be True.
- `recompute_granularity`: Recompute granularity, default is `"full"`.
- `tokenizer_name_or_path`: Name or path of the tokenizer if different from the model.
- `virtual_pp_degree`: Virtual pipeline parallelism degree, default is `1`.
- `sequence_parallel`: Whether to use sequence parallelism, default is `False`.
- `tensor_parallel_output`: Whether to use tensor_parallel_output. Enabling can reduce memory and improve speed, default is `True`. Set to False for yuan model.
- `weight_quantize_algo`: Model weight quantization algorithm, including `"nf4"` (qlora), `"weight_only_int8"`.
- `lora_rank`: Rank value in LoRA, default is `8`.
- `lora_path`: Path to initialize LoRA state dict.
- `rslora`
Whether to use RsLoRA. `rslora_plus` is equivalent to setting `lora_plus_scale` to 4 and `lora_alpha` to 4. Enabling this may improve model training convergence speed. Default is `False`.
- `lora_plus_scale`: Scaling factor for LoRA B in LoRA+ technique. Default is `1.0`.
- `lora_alpha`: Alpha parameter for LoRA. Default is `-1`.
- `rslora_plus`: Whether to enhance LoRA performance. Default is `False`.
- `use_quick_lora`: Whether to use Quick LoRA. Default is `True`.

### Data Parameters (DataArgument)
- `train_dataset_path`: Path to training dataset. Default is `"./data/train.jsonl"`.
- `dev_dataset_path`: Path to validation dataset. Default is `"./data/dev.jsonl"`.
- `max_seq_len`: Maximum sequence length for input. Default is `4096`.
- `max_prompt_len`: Maximum prompt length for input. Default is `2048`.
- `greedy_zero_padding`: Whether to use greedy zero padding. Enabling this may reduce padding ratio. Default is `False`.
- `lazy`: Whether to return `MapDataset` or `IterDataset`. `True` indicates `IterDataset`, `False` indicates `MapDataset`. Recommended to enable for large datasets (note: datasets are not shuffled when lazy is True).

### Training Parameters (TrainingArguments)
- `output_dir`: Directory for saving related files including models, checkpoints, tokenizer files, evaluation results, etc. Default is `"./checkpoints/dpo_ckpts"`.
- `per_device_train_batch_size`: Training batch size per device. Default is `1`.
- `gradient_accumulation_steps`: Number of gradient accumulation steps. Default is `8`, meaning parameters are updated every `8` steps.
- `per_device_eval_batch_size`: Evaluation batch size per device. Default is `1`.
- `num_train_epochs`: Number of training epochs. Default is `1`.
- `max_steps`: Maximum number of training steps. Default is `100`.
- `learning_rate`: Initial learning rate for optimizer. Default is `1e-06`.
- `warmup_steps`: Number of warmup steps. Default is `0`. When warmup_steps > 0, it overrides warmup_ratio setting. Default is `10`.
- `logging_steps`: Interval steps for logging. Default is `1`.
- `evaluation_strategy`: Evaluation strategy. "no": no evaluation during training; "steps": evaluate every eval_steps; "epoch": evaluate at end of each epoch.
- `save_strategy`: Save strategy. "no": no saving during training; "steps": save every eval_steps; "epoch": save at end of each epoch.
- `eval_steps`: Interval steps for evaluation. Default is `100`.
- `save_steps`
- `save_steps`: The interval of steps to save the model, default is `500`.
- `bf16`: Whether to enable BF16 training. Enabling BF16 training can accelerate training, default is `True`.
- `fp16_opt_level`: Can be set to O1 or O2. At O1 level, operators in the whitelist will use float16/bfloat16 for computation, while those in the blacklist will use float32. At O2 level, model parameters are converted to float16/bfloat16. Operators will only use float16/bfloat16 if all floating-point inputs are float16/bfloat16; if any floating-point input is float32, the operator will use float32. Default is `"O2"`.
- `do_train`: Whether to enable training, default is `True`.
- `do_eval`: Whether to enable evaluation, default is `True`.
- `load_best_model_at_end`: Whether to load the best model at the end of training, default is `True`.
- `tensor_parallel_degree`: This parameter indicates the number of splits for a transformer layer. This method increases communication overhead but saves memory. It is recommended to keep tensor_parallel_degree <=8 and use intra-machine communication whenever possible.
- `pipeline_parallel_degree`: Specifies the size of pipeline parallelism. (If set to 4 for a 12-layer model, each pp stage contains 3 layers). Default value -1 indicates pipeline parallelism is disabled.
- `sharding_parallel_degree`: The degree of data parallelism for sharding parameters.
- `sharding`: Whether to use Sharding data parallelism, default is `stage1`.
- `recompute`: Recompute strategy, currently supports full recompute. Enabling this can reduce memory usage to increase batch size, with approximately 30% speed reduction for full recompute.
- `recompute_granularity`: Recompute granularity, can be set to `full`, `full_attn`, or `core_attn`.
- `unified_checkpoint`: Whether to use unified checkpoint, default is `True`.
- `autotuner_benchmark`: Whether to enable autotuner benchmark, default is `False`.
- `benchmark`: Whether to enable benchmark, default is `False`.
- `optim`: Default is `adamw`, supports `adamw`, `adamw_mini`.

### DPO Parameters (DPOArguments)
- `beta`: Beta parameter for DPO loss function, default is 0.1.
- `simpo_gamma`: Gamma parameter for SimPO loss function, default is 0.5.
- `label_smoothing`: Label smoothing ratio, default is 0.0.
- `loss_type`
Type of DPO loss function, which can be one of:
- `sigmoid`([DPO](https://arxiv.org/abs/2305.18290)),
- `hinge`([RSO](https://arxiv.org/abs/2309.06657)),
- `ipo`([IPO](https://arxiv.org/abs/2310.12036)),
- `kto_pair` (implementation for preference pairs [KTO](https://github.com/ContextualAI/HALOs/blob/winnie/research/assets/report.pdf)),
- `sppo_hard`([SPPO](https://arxiv.org/pdf/2405.00675)),
- `nca_pair`([NCA](https://arxiv.org/abs/2402.05369)),
- `dpop`([DPOP](https://arxiv.org/pdf/2402.13228.pdf)),
- `orpo`([ORPO](https://arxiv.org/abs/2403.07691)),
- `simpo`([SimPO](https://arxiv.org/abs/2405.14734)), defaults to `sigmoid`.

- `pref_loss_ratio`: Ratio of DPO loss, default is 1.0.
- `sft_loss_ratio`: Ratio of SFT loss, default is 0.0.
- `dpop_lambda`
dpop_lambda, default is 50, details can be found in the paper [DPOP](https://arxiv.org/pdf/2402.13228)

## 4. DPO Data Flow Introduction
In DPO's data flow, we first preprocess the original dataset, then construct DPO data sequences and build attention_mask. The sequences include prompts (questions), chosen (preferred responses), and rejected (rejected responses).

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/2e1d91bf-8b90-4a84-b800-cc7cf4c02f58">
</div>
<div align="center">
    <font size ="1">
    Sequence Construction
     </font>
</div>

After sequence construction, we need to merge multiple sequences into combined sequences and pad them with pad tokens to make each constructed merged sequence have the same length.

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/3185440c-b290-4d3b-8665-ec5bda1cda23">
</div>
<div align="center">
    <font size ="1">
    Sequence Concatenation
     </font>
</div>

During training, by reconstructing attention_mask, we avoid considering sequence boundaries in the Attention computation.

After sequence concatenation, we reconstruct the attention_mask.

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/88d09f09-ebe6-4250-b8aa-e9e35db5b9d3">
</div>
<div align="center">
    <font size ="1">
    attention_mask Diagram
     </font>
</div>
