#!/bin/bash

# Environment Configuration
# -------------------------
# Set default values for service ports and runtime parameters
export HEALTH_HTTP_PORT=${HEALTH_HTTP_PORT:-"8110"}          # Health check endpoint port
export METRICS_HTTP_PORT=${METRICS_HTTP_PORT:-"8722"}        # Metrics monitoring port
export SERVICE_GRPC_PORT=${SERVICE_GRPC_PORT:-"8811"}        # gRPC service port
export SERVICE_HTTP_PORT=${SERVICE_HTTP_PORT:-"9965"}        # HTTP service port
export INTER_PROC_PORT=${INTER_PROC_PORT:-"8813"}            # INFER QUEUE port
export DTYPE=${DTYPE:-"bfloat16"}                            # Data type for computation
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}     # Comma-separated GPU device IDs
export MAX_SEQ_LEN=${MAX_SEQ_LEN:-"8192"}                    # Maximum sequence length
export MAX_DEC_LEN=${MAX_DEC_LEN:-"8192"}                    # Maximum decoding length
export MP_NUM=${MP_NUM:-"1"}                                 # Model parallelism degree

# Deployment Configuration
# ------------------------
docker_image=${docker_image:-"ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v1.0"}
model_path=${model_path:-~/llm_models/}  # Local model path (will be mounted to container)
model_name=${model_name:-"DeepSeek-R1/DeepSeek-R1-Distill-Llama-8B/weight_only_int8"}      # need to download model name，
tag="3.0-beta4"

# Model Preparation
# -----------------
# Verify model existence or download from remote
if [ -d "${model_path}" ]; then
    echo "Model directory exists at ${model_path}, skipping download."
else
    echo "Downloading model: ${model_name}..."
    python download_model.py --url https://paddlenlp.bj.bcebos.com/models/static/${model_name}/${tag} --dir ${model_path} --model_name ${model_name}
fi

# Container Deployment
# --------------------
# Start Docker container with GPU support and proper isolation
docker run --gpus all \
    --privileged \
    --cap-add=SYS_PTRACE \
    --network=host \
    --shm-size=5G \
    -v "${model_path}":/models/ \
    -e "HEALTH_HTTP_PORT=${HEALTH_HTTP_PORT}" \
    -e "METRICS_HTTP_PORT=${METRICS_HTTP_PORT}" \
    -e "SERVICE_GRPC_PORT=${SERVICE_GRPC_PORT}" \
    -e "SERVICE_HTTP_PORT=${SERVICE_HTTP_PORT}" \
    -e "INTER_PROC_PORT=${INTER_PROC_PORT}" \
    -e "MP_NUM=${MP_NUM}" \
    -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
    -e "DTYPE=${DTYPE}" \
    -e "MAX_SEQ_LEN=${MAX_SEQ_LEN}" \
    -e "MAX_DEC_LEN=${MAX_DEC_LEN}" \
    -dit "${docker_image}" bash -c -x '
    # Container initialization script
    cd /opt/output/Serving
    bash start_server.sh  # Start serving process
    '