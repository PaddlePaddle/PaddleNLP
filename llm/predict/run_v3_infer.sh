export PYTHONPATH=/opt/output/work_dir/paddle-deepseek/final/PaddleNLP:$PYTHONPATH



export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,pypi.tuna.tsinghua.edu.cn,paddle-ci.gz.bcebos.com 

export PPNLP_HOME="/opt/output/ppnlp_home"

export XPU_PADDLE_FC_FLOAT=1

rm -rf log
export GLOG_v=0
# export CUDA_VISIBLE_DEVICES=7
export XPUAPI_DEBUG=0xa1

# 前期可基于lite模型验证
# MODEL_TAG=deepseek-ai/DeepSeek-V2-Lite-Chat
# MODEL_TAG=deepseek-ai/DeepSeek-V2-Chat
# MODEL_TAG=deepseek-ai/DeepSeek-V3
# MODEL_TAG=deepseek-ai/DeepSeek-R1
# MODEL_TAG=/opt/output/work_dir/paddle-deepseek/deepseek-v3-6-layers/DeepSeek-V3-tiny
MODEL_TAG=/opt/output/work_dir/paddle-deepseek/final/PaddleNLP/llm/predict/deepseek-v3-inference

# QUANT_MODE=
# QUANT_MODE=weight_only_int8
# QUANT_MODE=weight_only_int4

MOE_QUANT_MODE=weight_only_int8


# python wht_predictor.py \
# python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" wht_predictor.py \
# python -u  -m paddle.distributed.launch --devices "0,1" wht_predictor.py \
# python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" wht_predictor.py \
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" predictor.py \
  --model_name_or_path ${MODEL_TAG} \
  --dtype bfloat16 \
  --device xpu \
  --mode static \
  --inference_model 1 \
  --block_attn 1 \
  --total_max_length 4096 \
  --batch_size 1 \
  --temperature 0.7 \
  --top_p 0.95 \
  --moe_quant_type ${MOE_QUANT_MODE} \
  --mla_use_matrix_absorption 1 \
  # --quant_type ${QUANT_MODE} \