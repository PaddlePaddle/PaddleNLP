#!/bin/bash
set -x

export FLAGS_call_stack_level=2
export CUDA_VISIBLE_DEVICES=0,1

python -m paddle.distributed.launch \
        inference.py --model_type gpt \
        --model_path ../../static/inference_model_pp1mp2/
