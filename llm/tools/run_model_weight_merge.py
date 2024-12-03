# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# from llm.argument import (
#     DataArgument,
#     GenerateArgument,
#     ModelArgument,
#     QuantArgument,
#     TrainingArguments,
# )
# from paddlenlp.transformers import (
#     AutoConfig,
#     AutoModelForCausalLM,
#     AutoModelForCausalLMPipe,
#     AutoTokenizer,
#     Llama3Tokenizer,
#     LlamaForCausalLM,
#     LlamaForCausalLMPipe,
#     LlamaTokenizer,
#     Qwen2ForCausalLM,
#     Qwen2ForCausalLMPipe,
#     register_sequence_parallel_allreduce_hooks,
# )


from paddlenlp.mergekit import MergeConfig, MergeModel

# from paddlenlp.trainer import PdArgumentParser, get_last_checkpoint

merge_config = MergeConfig(
    merge_type="linear",
    linear_ratio=0.5,
    n_process=2,
    dtype="bfloat16",
    device="cpu",
    merge_preifx="model",
)
mergekit = MergeModel(merge_config)
# model_class = AutoModelForCausalLM

# print(merge_config)
# paddle.set_device("cpu")
# parser = PdArgumentParser((GenerateArgument, QuantArgument, ModelArgument, DataArgument, TrainingArguments))
# if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
#     gen_args, quant_args, model_args, data_args, training_args = parser.parse_json_file_and_cmd_lines()
# else:
#     gen_args, quant_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# if training_args.fp16_opt_level == "O2":
#     if training_args.fp16:
#         dtype = "float16"
#     elif training_args.bf16:
#         dtype = "bfloat16"
#     else:
#         raise ValueError("Please specific dtype: --fp16 or --bf16")
# else:
#     dtype = "float32"

# model_config = AutoConfig.from_pretrained(
#     model_args.model_name_or_path,
#     dtype=dtype,
#     from_aistudio=model_args.from_aistudio,
#     # quantization_config=quantization_config,
# )

# model = model_class.from_pretrained(
#     model_args.model_name_or_path,
#     config=model_config,
#     from_aistudio=model_args.from_aistudio,
# )
model_path0 = "/root/paddlejob/workspace/env_run/output/huxinye/model/cnn1"
model_path1 = "/root/paddlejob/workspace/env_run/output/huxinye/model/cnn2"
output_path = "/root/paddlejob/workspace/env_run/output/huxinye/model/cnn_merge"


mergekit.merge_model(model_path0, model_path1, output_path)
merge_config.save_pretrained(output_path)
