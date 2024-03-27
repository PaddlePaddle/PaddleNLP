WORLD_SIZE=8
GBS=32
MP=2
PP=4
VPP=1
SD=$(($WORLD_SIZE / ($MP * $PP)))

MODEL_TYPE="gpt-13b"

export log_dir=log_${MODEL_TYPE}
export fused_softmax_with_triangular=True
export USE_FAST_LN=True
#export GLOG_vmodule=process_group_nccl=3

CONFIG_FILE="ppfleetx/configs/nlp/gpt/pretrain_gpt_13B_dp8.yaml"

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

rank=0 #$PADDLE_TRAINER_ID
#master="xx.xx.xx.xx" #`cat /root/paddlejob/workspace/hostfile | head -n 1 | awk '{print $1}'`
master=$(cat /root/paddlejob/workspace/hostfile | head -n 1 | awk '{print $1}')
nnodes=$(($WORLD_SIZE / 8))
distributed_args="--master=${master}:36677 --nnodes ${nnodes} --rank $rank --run_mode=collective"
OUTPUT_FILENAME=${MODEL_TYPE}_gbs${GBS}_mp${MP}pp${PP}sd${SD}_vpp${VPP}.20240318

rm -rf ${log_dir}

python -m paddle.distributed.launch \
    --log_dir log_gpt175b --devices "0,1,2,3,4,5,6,7" ${distributed_args} ${autoconfig_args} \
    ./tools/train.py \
    -c ${CONFIG_FILE} \
    -o Engine.mix_precision.enable=True \
    -o Engine.mix_precision.use_main_grad=True \
    -o Engine.mix_precision.dtype="bfloat16" \
    -o Engine.mix_precision.level="O2" \
    -o Engine.logging_freq="1" \
    -o Engine.max_steps=20 \
    -o Engine.eval_freq=10010 \
    -o Engine.save_load.save_steps=500000 \
    -o Data.Train.dataset.input_dir=./data \
    -o Model.max_position_embeddings=4096 \
    -o Data.Eval.dataset.max_seq_len=4096 \
    -o Data.Train.dataset.max_seq_len=4096 \
    -o Distributed.mp_degree=$MP \
    -o Distributed.pp_degree=$PP \
    -o Distributed.sharding.sharding_degree=${SD} \
    -o Model.virtual_pp_degree=${VPP} \
    -o Global.micro_batch_size=1 \
    -o Global.global_batch_size=${GBS} \
    -o Global.seed=1234 \
    -o Model.use_recompute=False \
    -o Model.use_flash_attn=True \
    -o Data.Eval.dataset.input_dir=./data \
    -o Model.sequence_parallel=True \
    -o Model.fused_linear=True \
    -o Optimizer.lr.decay_steps=30000 2>&1 | tee log_${OUTPUT_FILENAME}.txt

#    -o Model.recompute_granularity='full' \
