export CUDA_VISIBLE_DEVICES=5
# export FLAGS_prim_all=true
# export FLAGS_prim_forward=true
export FLAGS_enable_new_ir_api=True
export PYTHONPATH=/home/ssd2/xiongkun/Paddle/build/python
python train.py \
    -c /home/ssd3/xiongkun/PaddleNLP/model_zoo/gpt-3/ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml \
    -o Global.micro_batch_size=1 \
    -o Global.local_batch_size=1 