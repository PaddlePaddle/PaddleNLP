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


# Download files from file_list.txt
download_files() {
    # Get file list
    FILE_LIST_URL="${base_url}/file_list.txt"
    if ! wget -q "$FILE_LIST_URL" -O /tmp/file_list.txt; then
        echo "Failed to download file list: $FILE_LIST_URL,  please check your model_name $model_name."
        exit 1
    fi

    # Read and clean file list
    mapfile -t FILES < <(grep -v '^[[:space:]]*$' /tmp/file_list.txt | sed 's/^[ \t]*//;s/[ \t]*$//')
    if [[ ${#FILES[@]} -eq 0 ]]; then
        echo "No files found in file list"
        exit 1
    fi

    echo "Found ${#FILES[@]} files to download"

    # Create save directory
    mkdir -p "$model_path"

    # Download each file
    for FILE in "${FILES[@]}"; do
        FILE_URL="${base_url}/${FILE}"
        if ! wget -q --show-progress -P "$model_path" "$FILE_URL"; then
            echo "Failed to download: $FILE, please check your model_name $model_name."
            # Remove potentially corrupted file
            rm -f "${model_path}/${FILE}"
            exit 1
        fi
        echo "Save path: $model_path/$FILE"
    done

    # Cleanup temporary file
    rm /tmp/file_list.txt
}



# Model Preparation
# -----------------
# Verify model existence or download from remote
if [ -d "${model_path}" ]; then
    echo "Model directory exists at ${model_path}, skipping download."
else
    echo "Downloading model: ${model_name}..."
    base_url=https://paddlenlp.bj.bcebos.com/models/static/${model_name}/${tag}
    download_files
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