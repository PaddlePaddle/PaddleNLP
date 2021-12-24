export CUDA_VISIBLE_DEVICES=1

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
        
elif [ $model_type = speedup ]; then
python  export_model.py \
        --model_type "speedup" \
        --base_model_name_or_path "./checkpoints/ppminilm" \
        --model_path "./checkpoints/sp_checkpoints/best.pdparams" \
        --save_path "./checkpoints/sp_checkpoints/static/infer" 
else
echo "Three model_types are supported:  [extraction, classification, speedup]"
fi
