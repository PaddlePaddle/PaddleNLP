# ERNIE-3.5-SE

## 1. Model Overview

We implement a Parallel Transformer architecture with parallel computation of Attention and FFN layers. By fusing linear layer computations required for both Attention and FFN into unified operators, this design reduces kernel invocation and communication overhead, thereby enhancing training efficiency. We observe that the first FFN layer and last Attention layer contribute minimally, thus adopt a "Head-Truncated and Tail-Removed" strategy to shift computational resources from lower FFN layers to upper layers. This approach maintains equivalent performance to standard Transformer architectures at same FLOPs, while achieving better training speed and throughput.

<table>
<tr>
 <td><img src="https://github.com/PaddlePaddle/PaddleNLP/assets/16911935/89ca3093-4039-44c7-abce-4a47de6af1f6" height="300"> </td>
 <td><img src="https://github.com/PaddlePaddle/PaddleNLP/assets/16911935/3c89a72d-34b8-4711-b13e-d31063fc92d3" height="300"> </td>
</tr>
<tr>
 <td> Parallel Transformer </td>
 <td> Head-Truncated and Tail-Removed Strategy </td>
</tr>
</table>

* Rope Embedding + [Randomized Positional Encoding](https://aclanthology.org/2023.acl-short.161): We employ Rotary Position Embedding (RoPE) while retaining linear layer biases for better extrapolation capability. To enhance long-context extrapolation, we adopt random interval sampling of position IDs, enabling the model to handle longer sequences than trained.

<img src="https://github.com/PaddlePaddle/PaddleNLP/assets/20554008/423622c1-aed9-4ea9-83b0-d5d3efbaf35b" title="Random Positional Encoding" height="300">

* Sequence Length Warmup: Improves model convergence efficiency through dynamic adjustment of sequence lengths during initial training phases.

## 2. Pretraining

For pretraining data preparation, refer to [this document](../../tools/preprocess/docs/OpenWebText2.md).

For user convenience, we provide 100k preprocessed training samples:
```shell
wget https://bj.bcebos.com/paddlenlp/models/transformers/ernie/data/ernie_openwebtext_100k_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/ernie/data/ernie_openwebtext_100k_idx.npz
```

Organize all downloaded files into a single directory for training:

```shell
mkdir data
mv ernie_openwebtext_100k_ids.npy ./data
mv ernie_openwebtext_100k_idx.npz ./data
```
By using the following script, you can start the pre-training of ernie-3.5-se-3b, or directly refer to run_trainer_stage2.sh.
```shell
task_name="ernie35_hybrid"
python -u -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name""_log" \
    run_pretrain.py \
    --model_type "ernie" \
    --model_name_or_path "baidu/ernie-3.5-se-3b" \
    --tokenizer_name_or_path "ernie-tokenizer" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --split 949,50,1 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --use_flash_attention 1 \
    --use_fused_ln 1 \
    --bf16 \
    --fp16_opt_level "O2"  \
    --scale_loss 512 \
    --learning_rate 0.0003 \
    --min_learning_rate 0.00003 \
    --lr_scheduler_type "cosine" \
    --max_steps 300000 \
    --save_steps 200 \
    --adam_beta2 0.95 \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --max_grad_norm 1.0 \
    --logging_steps 2 \
    --dataloader_num_workers 0 \
    --sharding "stage2" \
    --sharding_parallel_degree 8 \
    --eval_steps 200 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --continue_training 0\
    --recompute 1 \
    --do_train \
    --do_eval \
    --save_total_limit 10 \
    --device "gpu"
```

Notes:
1. Requires paddle develop version for training. Need to install missing packages like `pip install fast_dataindex visualdl==2.5.3`
2. `use_flash_attention` needs to be enabled on A100 machines, otherwise the loss may be abnormal (quickly dropping to 0.00x, which is unusually low). It's recommended to use cuda11.8 environment.
3. `continue_training` indicates loading training from an existing pre-trained model. Set to 0 if you need to start training from scratch.
4. `use_flash_attention` requires corresponding version of paddle and flash_attn package.
## 3. Fine-tuning

### SFT
```shell
python -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    finetune_generation.py \
    --output_dir "output_sft/$task_name" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --model_name_or_path <PATH_TO_CKPT> \
    --task_name squad \
    --num_train_epochs 2 \
    --learning_rate 3e-5 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --src_length 1024 \
    --tgt_length 1024 \
    --bf16 \
    --fp16_opt_level O2 \
    --do_train \
    --do_eval \
    --disable_tqdm True \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --eval_with_do_generation False \
    --recompute \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --sharding "stage2" \
    --sharding_parallel_degree 8
```

### LoRA

The `use_fused_ln` requires installing custom operators from [this directory](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/model_zoo/gpt-3/external_ops) via `python setup.py install`. If operators are still not found after installation, additional PYTHONPATH configuration is required.

5. The current script is for sharding version. Users requiring 4D parallel training (data, sharding, tensor, and pipeline parallelism) can adjust relevant parameters accordingly.
```shell
python finetune_generation.py \
    --output_dir ./checkpoints/ \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --model_name_or_path <PATH_TO_CKPT> \
    --task_name squad \
    --num_train_epochs 2 \
    --learning_rate 3e-4 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --src_length 1024 \
    --tgt_length 1024 \
    --bf16 \
    --fp16_opt_level O2 \
    --do_train \
    --do_eval \
    --disable_tqdm True \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --eval_with_do_generation False \
    --recompute \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --lora True \
    --lora_rank 8
```

Parameter explanations:

- `model_name_or_path`: Name of built-in pretrained model or directory containing the model.
- `num_train_epochs`: Total number of training epochs to perform (if not an integer, will perform the percentage of the last epoch before stopping training).
- `max_steps`: Total number of training steps for the model.
- `learning_rate`: Learning rate for parameter updates.
- `warmup_steps`: Number of steps for learning rate warmup.
- `eval_steps`: Interval steps for model evaluation.
- `logging_steps`: Interval steps for printing training logs.
- `save_steps`: Interval steps for saving model parameters.
- `save_total_limit`: Number of model checkpoints to retain.
- `output_dir`: Directory to save model parameters.
- `src_length`: Maximum input length for context, default is 128.
- `tgt_length`: Maximum length for generated text, default is 160.
- `gradient_accumulation_steps`: Number of steps for gradient accumulation of model parameters, which can be used to effectively increase batch size. Actual batch_size = per_device_train_batch_size * gradient_accumulation_steps.
- `bf16`: Use bfloat16 precision for model training and inference.
- `fp16_opt_level`: bfloat16 precision training mode, `O2` indicates pure bfloat16 training.
- `recompute`: Use recomputation strategy to save GPU memory during training.
`do_train`: whether to train the model.
- `do_eval`: whether to evaluate the model.
- `tensor_parallel_degree`: number of model parallelism.
- `eval_with_do_generation`: whether to call model.generate during evaluation, default is False.
- `lora`: whether to use LoRA technique.
- `merge_weights`: whether to merge weights of original model and LoRA model.
- `lora_rank`: rank value in LoRA algorithm, default is 8.
- `lora_path`: path for LoRA parameters and configuration, used to initialize LoRA parameters.
- `task_name`: built-in dataset task name
- `data_name`: built-in dataset name, must define dataset name together with dataset task name
- `dataset_path`: path to custom dataset.


## 4. Dynamic Graph Prediction

```shell
python predict_generation.py \
    --model_name_or_path <PATH_TO_CKPT> \
    --tokenizer_name_or_path ernie-tokenizer
```
