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

real_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARD=8
BS=1
GAS=16
LR=1e-6
STEP=10
MODEL=""
export SHELLS_PATH=""
export CODES_PATH=""
# 当前目录
OUTPUT_DIR=""
dataset_name_or_path=""

log_dir=${real_script_dir}/sft_log
# 读取命令行参数
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -n|--num_cards)
          CARD="$2"
          shift
          shift
          ;;
    -b|--per_device_train_batch_size)
          BS="$2"
          shift
          shift
          ;;
    -g|--gradient_accumulation_steps)
          GAS="$2"
          shift
          shift
          ;;
    -s|--max_steps)
          STEP="$2"
          shift
          shift
          ;;
    -l|--learning_rate)
          LR="$2"
          shift
          shift
          ;;
    -m|--model_name_or_path)
          MODEL="$2"
          shift
          shift
          ;;
    -o|--output_model_path)
          OUTPUT_DIR="$2"
          shift
          shift
          ;;
    --dataset_name_or_path)
          dataset_name_or_path="$2"
          shift
          shift
          ;;
    *)
          echo "Unknown option: $1"
          exit 1
          ;;
  esac
done


set -x
if [[ ! -f "./npu_env.sh" ]]; then
    echo "do not find npu_env.sh, please cp test_tools/ascend910_shell/npu_env.sh in run path"
    exit 2
fi

source ./npu_env.sh

echo "train_add_params : $train_add_params"
# 默认train_add_params为： --fp16 true --continue_training 1  ，具体任务可以按需修改，直接在这下面export train_add_params=xxx
export FLAGS_npu_storage_format=1
export HCCL_INTRA_PCIE_EHABLE=0
export HCCL_INTRA_ROCE_ENABLE=1
#unset PADDLE_TRAINER_ENDPOINTS
#unset DISTRIBUTED_TRAINER_ENDPOINTS
export GLOG_v=0
export FLAGS_NPU_MC2=1
export MC2_Recompute=1
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export MULTI_STREAM_MEMORY_REUSE=1
export PYTHONPATH=${CODES_PATH}:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/atb/1.0.RC1/atb/lib/:${LD_LIBRARY_PATH}
gpu_num=$(npu-smi info -l | grep NPU | wc -l)
gpus=`seq -s ',' 0 $((gpu_num-1))`
export ASCEND_RT_VISIBLE_DEVICES=${gpus}


export nnodes=${PADDLE_TRAINERS_NUM}
export ips=${PADDLE_TRAINERS}
export host=${POD_IP}
export master_ip=${MASTER_IP}


# 统一kill相关进程
ps aux | grep "run_finetune.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps aux | grep run_pretrain.py | grep -v grep | awk '{print $2}' | xargs -r kill -9


pushd ${CODES_PATH}

python -u  -m paddle.distributed.launch \
    --ips "${ips}" \
    --host "${host}" \
    --nnodes "${nnodes}" \
    --master "${master_ip}" \
    --devices "${gpus}" \
    --log_dir "${log_dir}" \
    ../../run_finetune.py \
    --device "npu" \
    --model_name_or_path "${MODEL}" \
    --dataset_name_or_path "${dataset_name_or_path}" \
    --output_dir "${OUTPUT_DIR}" \
    --logging_dir  "./sft_logs" \
    --per_device_train_batch_size ${BS} \
    --gradient_accumulation_steps ${GAS} \
    --per_device_eval_batch_size ${BS} \
    --eval_accumulation_steps 1 \
    --max_steps ${STEP} \
    --decay_steps 2000 \
    --learning_rate ${LR} \
    --warmup_steps 2 \
    --save_steps 1000 \
    --logging_steps 1 \
    --evaluation_strategy no \
    --src_length 512 \
    --max_length 1024 \
    --fp16_opt_level "O2" \
    --do_train true \
    --disable_tqdm true \
    --eval_with_do_generation false \
    --metric_for_best_model "accuracy" \
    --recompute false \
    --tensor_parallel_degree 8 \
    --pipeline_parallel_degree 1 \
    --tensor_parallel_output true \
    --zero_padding 0 \
    --amp_master_grad true \
    --fuse_attention_qkv true \
    --fuse_attention_ffn true \
    --sequence_parallel 1 \
    --use_flash_attention 0 \
    --use_fused_rope 1 \
    --use_fused_rms_norm 1 \
    --pad_to_multiple_of 1024 \
    --sharding_parallel_degree 2 \
    --sharding "stage1" \
    --sharding_parallel_config "enable_stage1_tensor_fusion enable_stage1_overlap" \
    --skip_memory_metrics 0 \
    $train_add_params