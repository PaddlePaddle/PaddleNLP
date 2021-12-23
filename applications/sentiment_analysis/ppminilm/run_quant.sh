export CUDA_VISIBLE_DEVICES=1

python  quant_post.py \
        --base_model_path "./checkpoints/ppminilm" \
        --static_model_dir "./checkpoints/static" \
        --quant_model_dir "./checkpoints/quant" \
        --algorithm "avg" \
        --dev_path "./data/dev_cls.txt" \
        --label_path "./data/label_cls.dict" \
        --batch_size 4 \
        --max_seq_len 256 \
        --save_model_filename "infer.pdmodel" \
        --save_params_filename "infer.pdiparams" \
        --input_model_filename "infer.pdmodel" \
        --input_param_filename "infer.pdiparams"

