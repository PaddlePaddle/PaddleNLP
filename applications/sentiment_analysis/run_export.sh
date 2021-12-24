export CUDA_VISIBLE_DEVICES=1

model_type=$1

if [ ! $model_type ]; then
echo "Please enter the correct export model type, for example: sh run_export extraction"
elif [ $model_type = extraction ]; then 
python  export_model.py \
        --model_type "extraction" \
        --model_path "./extraction/checkpoints/best_ext.pdparams" \
        --save_path "./extraction/checkpoints/static/infer" 

elif [ $model_type = classification ]; then
python  export_model.py \
        --model_type "classification" \
        --model_path "./classification/checkpoints/best_cls.pdparams" \
        --save_path "./classification/checkpoints/static/infer" 
        
elif [ $model_type = speedup ]; then
python  export_model.py \
        --model_type "speedup" \
        --base_model_name_or_path "./speedup/checkpoints/ppminilm" \
        --model_path "./speedup/checkpoints/best_mini.pdparams" \
        --save_path "./speedup/checkpoints/static/infer" 
else
echo "Three model_types are supported:  [extraction, classification, speedup]"
fi
