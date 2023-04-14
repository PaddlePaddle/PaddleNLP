export USE_FAST_LN=1 # 需要先在https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/gpt-3/external_ops目录下执行python setup.py install
export USE_LINEAR_WITH_GRAD_ADD=1

export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/lib64/:$LD_LIBRARY_PATH
#source unset_paddle_env.sh

export NCCL_DEBUG=INFO

rm -rf log

nohup python -m paddle.distributed.launch \
    --master=127.0.0.1:8888 \
    --nnodes=1 \
    --log_dir log \
    tools/train.py \
    -c ppfleetx/configs/nlp/gpt/pretrain_gpt_13B_dp8.yaml \
    -o Distributed.dp_degree=1 \
    -o Distributed.mp_degree=4 \
    -o Distributed.pp_degree=2 \
    -o Global.seed=1234 \
    -o Global.global_batch_size=8 \
    -o Global.local_batch_size=8 \
    -o Global.micro_batch_size=1 \
    -o Engine.max_steps=30 \
    -o Engine.mix_precision.enable=True \
    -o Engine.mix_precision.dtype='bfloat16' \
    -o Engine.mix_precision.level='O2' \
    -o Engine.mix_precision.use_main_grad=True \
    -o Engine.save_load.save_steps=300 \
    -o Data.Train.dataset.input_dir='./enwiki_text/' \
    -o Data.Eval.dataset.input_dir='./enwiki_text/' \
    -o Data.Train.dataset.split='[98, 2, 0]' \
    -o Data.Eval.dataset.split='[98, 2, 0]' \
    -o Data.Train.dataset.max_seq_len=32000 \
    -o Data.Eval.dataset.max_seq_len=32000 \
    -o Optimizer.weight_decay=0.1 \
    -o Optimizer.beta1=0.9 \
    -o Optimizer.beta2=0.95 \
    -o Optimizer.epsilon=1.0e-8 \
    -o Optimizer.lr.min_lr=6.0e-6 \
    -o Optimizer.lr.max_lr=6.0e-5 \
    -o Optimizer.lr.warmup_rate=0.001 \
    -o Optimizer.lr.decay_steps=90000 \
    -o Model.max_position_embeddings=32000 \
    -o Model.hidden_dropout_prob=0.1 \
    -o Model.attention_probs_dropout_prob=0.1 \
    -o Model.use_flash_attn=True \
    -o Model.sequence_parallel=True \
    -o Model.initializer_range=0.006 \
    -o Model.fused_linear=True \
    -o Model.use_recompute=False \
    -o Model.recompute_granularity='core_attn' \
    -o Model.virtual_pp_degree=1 \
    -o Model.hidden_size=4096 \
    -o Model.num_attention_heads=64 \
    -o Model.num_layers=48 >run.log 2>&1 &

