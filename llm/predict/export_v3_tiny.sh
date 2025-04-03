export PYTHONPATH=/opt/output/work_dir/paddle-deepseek/final/PaddleNLP:$PYTHONPATH

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,pypi.tuna.tsinghua.edu.cn,paddle-ci.gz.bcebos.com 

export PPNLP_HOME="/opt/output/ppnlp_home"

# export GLOG_v=6

# 前期可基于lite模型验证
MODEL_TAG=/opt/output/ppnlp_home/models/deepseek-ai/DeepSeek-V3-Tiny/DeepSeek-V3-tiny

# QUANT_MODE=
# QUANT_MODE=weight_only_int8
# QUANT_MODE=weight_only_int4
# export CUDA_VISIBLE_DEVICES=0
# export XPUAPI_DEBUG=0xa1

MOE_QUANT_MODE=weight_only_int8

python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" export_model.py \
    --model_name_or_path ${MODEL_TAG} \
    --device xpu \
    --inference_model 1 \
    --block_attn 1 \
    --output_path ./deepseek-v3-tiny-inference \
    --dtype bfloat16 \
    --moe_quant_type ${MOE_QUANT_MODE} \
    --mla_use_matrix_absorption 1 \
