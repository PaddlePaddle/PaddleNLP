# cd external_ops && python setup.py install && cd -

export USE_FAST_LN=1
export USE_LINEAR_WITH_GRAD_ADD=1

python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7"     ./tools/auto_export.py     -c ./ppfleetx/configs/nlp/gpt/auto/generation_gpt_175B_mp8.yaml

python -m paddle.distributed.launch  projects/gpt/inference.py --mp_degree 8 --model_dir output