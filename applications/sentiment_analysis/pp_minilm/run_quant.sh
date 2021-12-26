export CUDA_VISIBLE_DEVICES=0

python  quant_post.py \
        --base_model_path "../checkpoints/pp_minilm" \
        --static_model_dir "../checkpoints/pp_checkpoints/static" \
        --quant_model_dir "../checkpoints/pp_checkpoints/quant" \
        --algorithm "avg" \
        --dev_path "../data/cls_data/dev.txt" \
        --label_path "../data/cls_data/label.dict" \
        --batch_size 4 \
        --max_seq_len 256 \
        --save_model_filename "infer.pdmodel" \
        --save_params_filename "infer.pdiparams" \
        --input_model_filename "infer.pdmodel" \
        --input_param_filename "infer.pdiparams"

