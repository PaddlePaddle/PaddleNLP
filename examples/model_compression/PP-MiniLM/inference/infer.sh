echo 原来的模型
python  infer.py  --task_name tnews --model_path  tnews/float  --use_trt  --collect_shape
python  infer.py  --task_name tnews --model_path  tnews/float   --use_trt
python  infer.py  --task_name tnews --model_path  tnews/float   --use_trt
python  infer.py  --task_name tnews --model_path  tnews/float   --use_trt
python  infer.py  --task_name tnews --model_path  tnews/float   --use_trt
python  infer.py  --task_name tnews --model_path  tnews/float   --use_trt


echo 裁剪后
python infer.py --task_name tnews --model_path ofa_models/TNEWS/0.75/sub_static/float  --use_trt --collect_shape
python infer.py --task_name tnews --model_path ofa_models/TNEWS/0.75/sub_static/float  --use_trt
python infer.py --task_name tnews --model_path ofa_models/TNEWS/0.75/sub_static/float  --use_trt
python infer.py --task_name tnews --model_path ofa_models/TNEWS/0.75/sub_static/float  --use_trt
python infer.py --task_name tnews --model_path ofa_models/TNEWS/0.75/sub_static/float  --use_trt
python infer.py --task_name tnews --model_path ofa_models/TNEWS/0.75/sub_static/float  --use_trt


echo int8推理
python  infer.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt  --collect_shape
python  infer.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt
python  infer.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt
python  infer.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt
python  infer.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt
python  infer.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt

