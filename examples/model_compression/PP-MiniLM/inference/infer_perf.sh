task = tnews
echo Inference of orgin FP32 model
python  infer_perf.py  --task_name ${task} --model_path  tnews/float  --use_trt  --collect_shape
python  infer_perf.py  --task_name ${task} --model_path  tnews/float   --use_trt
python  infer_perf.py  --task_name ${task} --model_path  tnews/float   --use_trt
python  infer_perf.py  --task_name ${task} --model_path  tnews/float   --use_trt
python  infer_perf.py  --task_name ${task} --model_path  tnews/float   --use_trt
python  infer_perf.py  --task_name ${task} --model_path  tnews/float   --use_trt


echo After OFA
python infer_perf.py --task_name ${task} --model_path ofa_models/TNEWS/0.75/sub_static/float  --use_trt --collect_shape
python infer_perf.py --task_name ${task} --model_path ofa_models/TNEWS/0.75/sub_static/float  --use_trt
python infer_perf.py --task_name ${task} --model_path ofa_models/TNEWS/0.75/sub_static/float  --use_trt
python infer_perf.py --task_name ${task} --model_path ofa_models/TNEWS/0.75/sub_static/float  --use_trt
python infer_perf.py --task_name ${task} --model_path ofa_models/TNEWS/0.75/sub_static/float  --use_trt
python infer_perf.py --task_name ${task} --model_path ofa_models/TNEWS/0.75/sub_static/float  --use_trt


echo After quantization
python  infer_perf.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt  --collect_shape
python  infer_perf.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt
python  infer_perf.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt
python  infer_perf.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt
python  infer_perf.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt
python  infer_perf.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt

