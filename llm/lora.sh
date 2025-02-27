export PYTHONPATH=/ssd1/mayongqiang/PaddleNLP/:$PYTHONPATH
# export PATH="$HOME/.cargo/bin:$PATH"
# export PATH=/root/anaconda3/bin/:$PATH

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,pypi.tuna.tsinghua.edu.cn,paddle-ci.gz.bcebos.com 

export PPNLP_HOME="/ssd1/mayongqiang/ppnlp_home/"

# 前期可基于lite模型验证
MODEL_TAG=deepseek-ai/DeepSeek-V2-Lite-Chat
# MODEL_TAG=deepseek-ai/DeepSeek-V2-Chat
# MODEL_TAG=deepseek-ai/DeepSeek-V3
# MODEL_TAG=deepseek-ai/DeepSeek-R1

# QUANT_MODE=
QUANT_MODE=weight_only_int8
# QUANT_MODE=weight_only_int4

# python run_finetune.py ./devices/dcu/llama/lora_argument.json 
# python run_pretrain.py ./config/llama/lora_argument.json --continue_training False
python run_finetune.py ./config/deepseek-v3/lora_argument.json
