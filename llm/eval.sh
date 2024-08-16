#CUDA_VISIBLE_DEVICES=0 python evaluate.py --base_model ../../../checkpoint-bk/winogrande_full/checkpoint-350/ --dataset ./winogrande/dev.json --model quant_winogrande
export PYTHONPATH=../:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
python evaluate.py --base_model  ../../../checkpoint/hellaswag/checkpoint-1000-bk --dataset ./hellaswag/dev.json --model hellaswag_quant_mw
