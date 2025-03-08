#!/bin/bash

MODEL_NAME=${model_name}
TAG=${tag}
MODEL_DIR=${MODEL_DIR:-/models}
SPECULATE_MODEL_PATH=${SPECULATE_MODEL_PATH}
POD_0_IP=${POD_0_IP:-"127.0.0.1"}
HOST_IP=$(hostname -I | awk '{print $1}')


if [ -z "$MODEL_NAME" ]; then
    echo "Error: Model Dir is empty"
    exit 1
fi

SUPPORTED_PATTERNS=(
    ".*Qwen.*"
    ".+Llama.+"
    ".+Mixtral.+"
    ".+DeepSeek.+"
)

MATCHED=false
for PATTERN in "${SUPPORTED_PATTERNS[@]}"; do
    if [[ "$MODEL_NAME" =~ $PATTERN ]]; then
        MATCHED=true
        break
    fi
done

if [ "$MATCHED" = false ]; then
    echo "Error: $MODEL_NAME is not in the supported list. Supported models: Qwen, Llama, Mixtral, DeepSeek."
    echo "Please check the model name from this document: https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md"
    exit 1
fi

echo "Start downloading model: $MODEL_NAME"
BASE_URL="https://paddlenlp.bj.bcebos.com/models/static/$TAG/$MODEL_NAME"

if [ "$MP_NNODE" -eq 1 ]; then
    TEMP_FILE="model"
elif [ "$POD_0_IP" == "$HOST_IP" ]; then
    TEMP_FILE="node1"
else
    TEMP_FILE="node2"
fi

MODEL_URL="$BASE_URL/$TEMP_FILE"

if [ -z "$SPECULATE_MODEL_PATH" ]; then
    echo "Downloading from $MODEL_URL to $MODEL_DIR"
    download_model -u $MODEL_URL -d $MODEL_DIR 
elif [ -n "$SPECULATE_MODEL_PATH" ]; then  
    echo "Downloading from $MODEL_URL to $MODEL_DIR"
    download_model -u $MODEL_URL -d $MODEL_DIR 
    MTP_URL="$BASE_URL/mtp"
    echo "Downloading from $MTP_URL to $SPECULATE_MODEL_PATH"
    download_model -u $MTP_URL -d $SPECULATE_MODEL_PATH 
fi

if [ $? -ne 0 ]; then
    echo "Error: Download failed!"
    exit 1
fi

echo "Download completed successfully."
