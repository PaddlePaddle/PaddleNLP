#wget https://paddlenlp.bj.bcebos.com/models/transformers/gpt2/train.data.json_ids.npz
#mkdir data
#mv train.data.json_ids.npz data

export DATA_DIR=./data
export PYTHONPATH=$PYTHONPATH:../../../../

rm -rf dp2_pp2_mp2
export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
log_dir=dp1_pp1_mp1_test
rm -rf output/
rm -rf $log_dir

python -m paddle.distributed.launch --log_dir $log_dir --gpus "6,7" run_pretrain.py \
    --model_type gpt \
    --model_name_or_path gpt2-small-en \
    --input_dir "./data"\
    --output_dir "output"\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 500000\
    --save_steps 100000\
    --decay_steps 320000\
    --eval_freq 1000\
    --warmup_rate 0.01\
    --micro_batch_size 8\
    --local_batch_size 8\
    --dp_degree 1\
    --mp_degree 1\
    --pp_degree 2\
    --use_amp True\
    --scale_loss 32768\
    --device gpu


#rm -rf $log_dir
#python -m paddle.distributed.launch --log_dir $log_dir --gpus "0,1,2,3,4,5,6,7" run_pretrain.py \
#   --model_type gpt2 \
#   --model_name_or_path gpt2-small-en \
#   --input_dir "./data"\
#   --output_dir "output"\
#   --weight_decay 0.01\
#   --grad_clip 0\
#   --max_steps 50000\
#   --save_steps 100000\
#   --decay_steps 320000\
#   --warmup_rate 0.01\
#   --batch_size 8\
#   --device gpu
#


#   --use_amp True \
#   --dp_degree 2 \
#   --pp_degree 2 \
#   --mp_degree 2 \
#   --micro_batch_size 2
#

#nsys profile --stats=true -t cuda python -m paddle.distributed.launch --log_dir dp2_pp1_mp4 --gpus "0,1,2,3,4,5,6,7" run_pretrain.py \
#   --model_type gpt2 \
#   --model_name_or_path gpt2-small-en \
#   --input_dir "./data"\
#   --output_dir "output"\
#   --weight_decay 0.01\
#   --grad_clip 1.0\
#   --max_steps 8\
#   --save_steps 100000\
#   --decay_steps 320000\
#   --warmup_rate 0.01\
#   --batch_size 8\
#   --device gpu \
#   --use_amp False \
#   --dp_degree 8 \
#   --pp_degree 1 \
#   --mp_degree 1 \
#   --micro_batch_size 2


