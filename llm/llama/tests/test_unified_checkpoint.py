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


from test_parallel_dygraph_dataparallel import TestMultipleGpus

pretrain_arguments = {
    "model_name_or_path": "facebook/llama-7b",
    "tokenizer_name_or_path": "facebook/llama-7b",
    "input_dir": "./data",
    "output_dir": "./checkpoints/qwen_pretrain_ckpts",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "per_device_eval_batch_size": 16,
    "tensor_parallel_degree": 1,
    "pipeline_parallel_degree": 4,
    "sharding": "stage1",
    "virtual_pp_degree": 1,
    "sequence_parallel": 0,
    "use_flash_attention": "false",
    "use_fused_rms_norm": "false",
    "max_seq_length": 1024,
    "learning_rate": 3e-05,
    "min_learning_rate": 3e-06,
    "warmup_steps": 30,
    "logging_steps": 1,
    "max_steps": 10000,
    "save_steps": 5000,
    "eval_steps": 1000,
    "weight_decay": 0.01,
    "fp16": "true",
    "fp16_opt_level": "O2",
    "warmup_ratio": 0.01,
    "max_grad_norm": 1.0,
    "dataloader_num_workers": 0,
    "continue_training": 0,
    "do_train": "true",
    "do_eval": "true",
    "do_predict": "true",
    "disable_tqdm": "true",
    "recompute": 1,
    "distributed_dataloader": 0,
    "recompute_granularity": "full",
    "save_total_limit": 2,
}


class TestGLMTensorParallel(TestMultipleGpus):
    def testPaddleTensorParallelGLM(self):

        self.run_8gpu("run_pretrain.py", **pretrain_arguments)
