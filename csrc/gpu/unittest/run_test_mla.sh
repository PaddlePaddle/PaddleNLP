export FLAGS_mla_use_tensorcore=1
# export FLAGS_cascade_attention_max_partition_size=8888
export FLAGS_mla_dec_chunk_size=-1
export CUDA_VISIBLE_DEVICES=7

# python test_absorb_mla.py

ncu=`which ncu`

${ncu} --section regex:'^(?!Nvlink)' -o test_128_2 -f --import-source on --cache-control=all --clock-control=base -k regex:MLAWithKVCacheKernel --print-source=cuda,sass --page source python test_absorb_mla.py