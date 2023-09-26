export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

python -m paddle.distributed.launch \
    --gpus "0" \
     export_model.py \
    --model_name_or_path facebook/llama-13b \
    --output_path ./llama13b-inference_model_fp16_mp1 \
    --dtype float16 \
    --inference_model \
    --block_attn