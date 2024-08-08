set -x
temp_dir=./tmp
checkpoint_path=${1-"checkpoints/qwen_ckpt_quant_pt_1000/checkpoint-500-ori"}
output=${2-"ckpt"}
jobs_num=${3-4}
worker_num=${4-4}
bit_cnt=${5-8}
quant_stage=${6-3}
shard_num=${7-4}

mkdir -p ${temp_dir}

jobs_num_per_worker=$((jobs_num / worker_num))
echo $jobs_num_per_worker

pids=""
for dev_id in $(seq 0 $((worker_num - 1))); do
    global_dev_id=dev_id
    start_idx=$((dev_id * jobs_num_per_worker))
    end_idx=$(((dev_id + 1) * jobs_num_per_worker))
    python test_faiss_gpu.py \
        --checkpoint_path ${checkpoint_path} \
        --output ${output} \
        --quant_bits_opt ${bit_cnt} \
        --start_idx ${start_idx} \
        --quant_stage ${quant_stage} \
        --fusion_quant \
        --shard_num ${shard_num} \
        --end_idx ${end_idx} &> ${temp_dir}/workerlog.${dev_id} & 
    pids="$pids $!"
done

wait $pids
