# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# models
model_name_or_path = "meta-llama/Meta-Llama-3-8B"
tokenizer_name_or_path = "meta-llama/Meta-Llama-3-8B"

# data
checkpoint_dirs = {
    "input_dir": "./data",
    "output_dir": "./checkpoints/pretrain_ckpts",
    "unified_checkpoint": True,
    "save_total_limit": 2,
}

training_contronl = {
    "do_train": True,
    "do_eval": True,
    "do_predict": True,
    "disable_tqdm": True,
    "recompute": False,
    "distributed_dataloader": 1,
    "recompute_granularity": "full",
}


training_args = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "per_device_eval_batch_size": 2,
    "tensor_parallel_degree": 2,
    "pipeline_parallel_degree": 1,
    "sharding": "stage2",
    "virtual_pp_degree": 1,
    "sequence_parallel": 0,
    "max_seq_length": 4096,
    "learning_rate": 3e-05,
    "min_learning_rate": 3e-06,
    "warmup_steps": 30,
    "logging_steps": 1,
    "max_steps": 10000,
    "save_steps": 5000,
    "eval_steps": 1000,
    "weight_decay": 0.01,
    "warmup_ratio": 0.01,
    "max_grad_norm": 1.0,
    "dataloader_num_workers": 1,
    "continue_training": 0,
}
accelerate = {
    "use_flash_attention": True,
    "use_fused_rms_norm": True,
    "use_fused_rope": True,
    "bf16": True,
    "fp16_opt_level": "O2",
}
