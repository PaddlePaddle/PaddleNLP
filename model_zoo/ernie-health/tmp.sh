# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

python -u -m paddle.distributed.launch    \
--gpus 4,5,6,7      run_pretrain_trainer.py     \
--input_dir "./data"     \
--output_dir "output/$task_name"     \
--max_seq_length 512     \
--gradient_accumulation_steps 1    \
--per_device_train_batch_size 2     \
--learning_rate 0.001     \
--max_steps 1000000     \
--save_steps 50000     \
--weight_decay 0.01     \
--warmup_ratio 0.01     \
--max_grad_norm 1.0     \
--logging_steps 20     \
--dataloader_num_workers 2     \
--device "gpu"    \
--fp16  false    \
--fp16_opt_level "O1"      \
--do_train     \
--disable_tqdm True    \
--save_total_limit 10 \
--overwrite_output_dir
