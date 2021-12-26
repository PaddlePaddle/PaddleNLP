export CUDA_VISIBLE_DEVICES=0

model_type=$1

if [ ! $model_type ]; then
echo "Please enter the correct export model type, for example: sh run_export extraction"
elif [ $model_type = extraction ]; then 
python  export_model.py \
        --model_type "extraction" \
        --model_path "./checkpoints/ext_checkpoints/best.pdparams" \
        --save_path "./checkpoints/ext_checkpoints/static/infer" 

elif [ $model_type = classification ]; then
python  export_model.py \
        --model_type "classification" \
        --model_path "./checkpoints/cls_checkpoints/best.pdparams" \
        --save_path "./checkpoints/cls_checkpoints/static/infer" 
        
elif [ $model_type = pp_minilm ]; then
python  export_model.py \
        --model_type "pp_minilm" \
        --base_model_name "ppminilm-6l-768h" \
        --model_path "./checkpoints/pp_checkpoints/best.pdparams" \
        --save_path "./checkpoints/pp_checkpoints/static/infer" 
else
echo "Three model_types are supported:  [extraction, classification, pp_minilm]"
fi
