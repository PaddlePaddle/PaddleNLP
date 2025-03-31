# Speculative Decoding Tutorial

Speculative decoding is an algorithm that accelerates inference by speculatively generating multiple tokens in one pass followed by verification and acceptance. PaddleNLP provides a simple and efficient speculative decoding workflow. Below are usage instructions for various speculative decoding algorithms in PaddleNLP.

## Efficient Speculative Decoding Framework

Traditional methods for Draft Token verification and MTP (Eagle/Draft Model) inference typically expand batch_size by multiples of the Draft Token count. Our efficient attention mechanism maintains the original batch size while reducing computation, as shown in the following diagram:

![Framework Diagram](https://github.com/user-attachments/assets/fd938c8c-fcd5-4ead-ae6e-f455af415e52)

Reference [Vllm batch_extension](https://docs.google.com/document/d/1T-JaS2T1NRfdP51qzqpyakoCXxSXTtORppiwaj5asxA/edit?tab=t.0#heading=h.kk7dq05lc6q8)

## Parameter Description

- `speculate_method`: Speculative decoding algorithm. Default is `None`. Valid options: `None`, `inference_with_reference`, `mtp`, `eagle`.
  - `None`: Regular autoregressive decoding.
  - `inference_with_reference`: Context-based speculative decoding. [Paper Link](https://arxiv.org/pdf/2304.04487).

- `speculate_max_draft_token_num`: Maximum number of draft tokens generated per iteration in speculative decoding. Default is 1, maximum supported is 5.

- `speculate_max_ngram_size`: Maximum window size for n-gram matching of draft tokens. Default is `1`. In `inference_with_reference` algorithm, draft tokens are first matched from prompt using n-gram window sliding. Window size and input-output overlap jointly determine the draft token generation overhead, thus affecting the acceleration performance.

- `speculate_verify_window` (temporarily deprecated): Verification strategy defaults to TopP + window verification. Window size is 2 by default. For details, refer to [Speculative Decoding Tutorial](./speculative_decoding.md).

- `speculate_max_candidate_len` (temporarily deprecated): Maximum candidate tokens generated. Verification is performed by comparing candidate tokens with draft tokens (only effective in TopP + window verification). Default is 5.

- `draft_model_name_or_path`: Path to Draft Model in `MTP` or `EAGLE` mode.

- `draft_model_quant_type`: Inference quantization precision for Draft Model in `MTP` or `EAGLE` mode. Refer to `--quant_type`.

- `return_full_hidden_states`: In `MTP` or `EAGLE`...
## Whether to return all hidden states, defaults to `False`.

Others:
1. The maximum batch_size currently supported by speculative decoding is 128

## Inference with reference

This algorithm matches draft tokens from prompts using n-gram windows, suitable for scenarios with significant input-output overlap such as code editing, document querying, etc. For more information, refer to the [paper](https://arxiv.org/pdf/2304.04487).

### Usage commands

```shell
# Dynamic graph model inference command reference
python predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --speculate_method inference_with_reference --speculate_max_draft_token_num 5 --speculate_max_ngram_size 2
```
## Multi-Token Prediction (MTP)

Paper: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf
![MTP](https://github.com/user-attachments/assets/6cdc3d94-7bab-4c0f-991c-875ae24816a6)

Supports inference for DeepSeek-V3/R1 & MTP. For detailed principles, please refer to the paper.

Key features:
1. During Base Model validation, optimized Attention is used to generate all Draft Token logits in one pass, eliminating the need for batch_extension to increase batch_size. This addresses the issue of speculative decoding failing to accelerate inference performance under large batches.
2. In the MTP inference phase, unified handling of Draft Token acceptance from previous rounds (e.g., full rejection, partial acceptance, full acceptance) is maintained, requiring only the original batch size to process all input requests at once.
3. Using a decoupled framework where Base Model and MTP load weights separately, supporting multiple decoding methods post-export.

### Supported Quantized Precisions for MTP

| Base Model | Deployment Machine | Base Model Quantization Type | MTP Quantization Type |
| --- | --- | --- | --- |
| DeepSeek-R1 | TP8 | a8w8_fp8_wint4 | a8w8_fp8 |
| DeepSeek-R1 | TP8 | weight_only_int4 | weight_only_int8 |
| DeepSeek-R1 | TP16(2*TP8) | a8w8_fp8 | a8w8_fp8 |
| DeepSeek-R1 | TP16(2*TP8) | weight_only_int8 | weight_only_int8 |

Supports mixed-precision inference combinations for DeepSeek-R1 and MTP. Deployment can be done via containers or scripts. Below are partial examples:

### Method 1: One-click Container Deployment
**DeepSeek-R1(a8w8_fp8_wint4) + MTP(a8w8_fp8), Single-machine TP8 Deployment**

```bash
# FP8 example
docker run --gpus all --shm-size 10g -v /path/to/R1-AWQ:/opt/tritonrt/model_repository/llm/1/ -v /path/to/MTP-FP8:/opt/tritonrt/model_repository/mtp/1/ -p 8080:8080 -p 8081:8081 -p 8082:8082 --ulimit memlock=-1 --ulimit stack=67108864 registry.cn-hangzhou.aliyuncs.com/trt-inference-server/triton_23.09_tp8:latest
```

### Inference Command Example
```bash
python3 -m lmdeploy.mtp.pipeline \
    --model-path /path/to/DeepSeek-R1 \
    --mtp-path /path/to/MTP \
    --tp 8 \
    --session-len 4096 \
    --max-batch-size 16 \
    --quantization a8w8_fp8_wint4 \
    --mtp-quant a8w8_fp8 \
    --cache-max-entry-count 0.5 \
    --num-tokens 8
```

**Parameters:**
- `--tp`: Tensor parallel configuration
- `--quantization`: Base model quantization type
- `--mtp-quant`: MTP model quantization type
- `--num-tokens`: Number of tokens generated per forward pass (8 for MTP, 1 for regular decoding)
- Other parameters are consistent with regular LMDeploy inference
```shell
export MODEL_PATH=${MODEL_PATH:-/PATH_TO_MODEL/}
export MODEL_MTP_PATH=${MODEL_PATH:-/PATH_TO_MTP/}
export model_name="deepseek-ai/DeepSeek-R1-MTP/a8w8_fp8_wint4"

docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -v $MODEL_MTP_PATH:/models-mtp -v /ssd2/paddle_example:/work -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.1 /bin/bash \
-c -ex 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export MP_NUM=8 && export SPECULATE_MODEL_QUANT_TYPE="a8w8_fp8" && export SPECULATE_METHOD="mtp" && export SPECULATE_MODEL_PATH="/models-mtp" && export SPECULATE_MAX_DRAFT_TOKEN_NUM=1 && export BLOCK_BS=32 && export BLOCK_RATIO=0.25 && export BATCH_SIZE="128" && start_server $model_name && tail -f /dev/null'
```

**DeepSeek-R1(weight_only_int8) + MTP(weight_only_int8), Dual-machine TP16 Configuration**

1. One-command Container Launch for MTP Inference Service

```shell
# Ensure mutual ping between 2 machine nodes
# First node (master)
ping 192.168.0.1
# Second node (slave)
ping 192.168.0.2
model_name=${model_name:-"deepseek-ai/DeepSeek-R1-MTP-2nodes/weight_only_int8"}
export POD_0_IP=master_ip
export POD_IPS=master_ip,slave_ip # This environment variable must be consistent on both machines
# Default service port, modify via export if conflicts occur
export SERVICE_HTTP_PORT=${PUSH_MODE_HTTP_PORT:-${SERVICE_HTTP_PORT:-"9965"}}
# Note: SPECULATE_MODEL_QUANT_TYPE should match the MTP-supported quantization precision table
export SPECULATE_MODEL_QUANT_TYPE="weight_only_int8"
# /PATH_TO_MODEL # Model mount path
# /PATH_TO_MTP # MTP mount path
```
```shell
# node1
export model_name=${model_name:-"deepseek-ai/DeepSeek-R1-MTP-2nodes/weight_only_int8"}
export MODEL_PATH=${MODEL_PATH:-/PATH_TO_MODEL/}
export MODEL_MTP_PATH=${MODEL_PATH:-/PATH_TO_MTP/}
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -v $MODEL_MTP_PATH:/models-mtp -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.1 /bin/bash \
-c -ex 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export MP_NUM=16 && export MP_NNODE=2 && export POD_0_IP=192.168.0.1 && export POD_IPS=192.168.0.1,192.168.0.2 && export SPECULATE_MODEL_QUANT_TYPE="weight_only_int8" && export SPECULATE_METHOD="mtp" && export SPECULATE_MODEL_PATH="/models-mtp" && export SPECULATE_MAX_DRAFT_TOKEN_NUM=1 && start_server $model_name && tail -f /dev/null'

# node2
export model_name=${model_name:-"deepseek-ai/DeepSeek-R1-MTP-2nodes/weight_only_int8"}
export MODEL_PATH=${MODEL_PATH:-/PATH_TO_MODEL/}
export MODEL_MTP_PATH=${MODEL_PATH:-/PATH_TO_MTP/}
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -v $MODEL_MTP_PATH:/models-mtp -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.1 /bin/bash \
-c -ex 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export MP_NUM=16 && export MP_NNODE=2 && export POD_0_IP=192.168.0.1 && export POD_IPS=192.168.0.1,192.168.0.2 &&export SPECULATE_MODEL_QUANT_TYPE="weight_only_int8" && export SPECULATE_METHOD="mtp" && export SPECULATE_MODEL_PATH="/models-mtp" && export SPECULATE_MAX_DRAFT_TOKEN_NUM=1 && start_server $model_name  && tail -f /dev/null'
```
**Service Request**

curl Request
```shell
curl ${ip}:9965/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
      "model":"default",
      "text":"Hello, how are you?"
  }'
```
OpenAI Request
```python
import openai
client = openai.Client(base_url=f"http://127.0.0.1:9965/v1/chat/completions", api_key="EMPTY_API_KEY")
# Non-streaming response
response = client.completions.create(
    model="default",
    prompt="Hello, how are you?",
  max_tokens=50,
  stream=False,
)

print(response)
print("\n")

# Streaming response
response = client.completions.create(
    model="default",
    prompt="Hello, how are you?",
  max_tokens=100,
  stream=True,
)

for chunk in response:
  if chunk.choices[0] is not None:
    print(chunk.choices[0].text, end='')
print("\n")
```

### Method 2: Script-based Inference Testing
#### DeepSeek_R1 Dynamic Graph + MTP Dynamic Graph, Dual-machine TP16
1. DeepSeek_R1 uses weight_only_int8, MTP uses weight_only_int8

```shell
For dual-machine inference, ensure nodes can ping each other:
# First node (master)
ping 192.168.0.1
# Second node (slave)
ping 192.168.0.2
```

node1 and node2 use identical scripts
```shell
$ cat run_dynamic_mtp.sh

export MODEL_TAG=deepseek-ai/DeepSeek-R1
export DRAFT_MODEL_TAG=deepseek-ai/DeepSeek-R1-MTP
export QUANT_MODE=weight_only_int8
export DRAFT_MODEL_QUANT_MODE=weight_only_int8
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=2048
# Operator acceleration strategy
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m paddle.distributed.launch --ips "192.168.0.1,192.168.0.2"\
  --gpus ${CUDA_VISIBLE_DEVICES} \
  predictor.py \
  --model_name_or_path ${MODEL_TAG} \
  --dtype bfloat16 \
  --mode dynamic \
  --inference_model 1 \
  --append_attn 1 \
  --total_max_length ${TOTAL_MAX_LENGTH} \
  --quant_type ${QUANT_MODE} \
  --max_length ${MAX_DEC_LEN} \
  --speculate_method mtp \
  --draft_model_name_or_path ${DRAFT_MODEL_TAG} \
  --draft_model_quant_type ${DRAFT_MODEL_QUANT_MODE} \
  --speculate_max_draft_token_num 1 \
  --return_full_hidden_states 1 \
  --mla_use_matrix_absorption 1

$ bash run_dynamic_mtp.sh
```

2. DeepSeek_R1 uses FP8, MTP uses FP8

Scripts for node1 and node2 are identical
```shell
$ cat run_dynamic_mtp.sh

export MODEL_TAG=deepseek-ai/DeepSeek-R1-FP8
export DRAFT_MODEL_TAG=deepseek-ai/DeepSeek-R1-MTP-FP8
export QUANT_MODE=a8w8_fp8
export DRAFT_MODEL_QUANT_MODE=a8w8_fp8
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=2048
# Operator acceleration strategy
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m paddle.distributed.launch --ips "192.168.0.1,192.168.0.2"\
  --gpus ${CUDA_VISIBLE_DEVICES} \
  predictor.py \
  --model_name_or_path ${MODEL_TAG} \
  --dtype bfloat16 \
  --mode dynamic \
  --inference_model 1 \
  --append_attn 1 \
  --total_max_length ${TOTAL_MAX_LENGTH} \
  --quant_type ${QUANT_MODE} \
  --max_length ${MAX_DEC_LEN} \
  --speculate_method mtp \
  --draft_model_name_or_path ${DRAFT_MODEL_TAG} \
  --draft_model_quant_type ${DRAFT_MODEL_QUANT_MODE} \
  --speculate_max_draft_token_num 1 \
  --return_full_hidden_states 1 \
  --mla_use_matrix_absorption 1

$ bash run_dynamic_mtp.sh
```

#### DeepSeek_R1 Dynamic Graph + MTP Dynamic Graph, Single-Node TP8
1. DeepSeek_R1 uses weight_only_int8, MTP uses weight_only_int8
```shell
$ cat run_dynamic_mtp.sh

export MODEL_TAG=deepseek-ai/DeepSeek-R1
export DRAFT_MODEL_TAG=deepseek-ai/DeepSeek-R1-MTP
export QUANT_MODE=weight_only_int4
export DRAFT_MODEL_QUANT_MODE=weight_only_int8
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=2048
# Operator acceleration strategy
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m paddle.distributed.launch \
  --gpus ${CUDA_VISIBLE_DEVICES} \
  predictor.py \
  --model_name_or_path ${MODEL_TAG} \
  --dtype bfloat16 \
  --mode dynamic \
  --inference_model 1 \
  --append_attn 1 \
  --total_max_length ${TOTAL_MAX_LENGTH} \
  --quant_type ${QUANT_MODE} \
  --max_length ${MAX_DEC_LEN} \
  --speculate_method mtp \
  --draft_model_name_or_path ${DRAFT_MODEL_TAG} \
  --draft_model_quant_type ${DRAFT_MODEL_QUANT_MODE} \
  --speculate_max_draft_token_num 1 \
  --return_full_hidden_states 1 \
  --mla_use_matrix_absorption 1

$ bash run_dynamic_mtp.sh
```

#### [**Recommended**] Base Model Static Graph + MTP Dynamic Graph, Dual-machine TP16
> Notes:
1. The speculative decoding exported model supports all methods, so set speculate_method to the default inference_with_reference here.
2. The static graph model can be exported from DeepSeek-R1, or directly download the pre-uploaded model

```shell
# To launch dual-machine inference, ensure the two machine nodes can ping each other
# First node (master)
ping 192.168.0.1
# Second node (slave)
ping 192.168.0.2
```

1. DeepSeek-R1 uses weight_only_int8, MTP uses weight_only_int8

The scripts for node1 and node2 are identical
```shell
# Export Script
$ cat export.sh

export MODEL_TAG=deepseek-ai/DeepSeek-R1
export OUTPUT_PATH=/path/to/exported_model
export QUANT_MODE=weight_only_int8
export TOTAL_MAX_LENGTH=8192
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m paddle.distributed.launch --ips "192.168.0.1,192.168.0.2"\
  --gpus ${CUDA_VISIBLE_DEVICES} \
  export_model.py \
  --model_name_or_path ${MODEL_TAG} \
  --output_path ${OUTPUT_PATH} \
  --dtype bfloat16 \
  --inference_model 1 \
  --append_attn 1 \
  --block_attn 1 \
  --total_max_length ${TOTAL_MAX_LENGTH} \
  --quant_type ${QUANT_MODE} \
  --speculate_method inference_with_reference \
  --return_full_hidden_states 1 \
  --mla_use_matrix_absorption 1

$ bash export.sh
```

node1 and node2 scripts are identical
```shell
# Inference Script
$ cat run_mtp_infer.sh

export OUTPUT_PATH=/path/to/exported_model
export DRAFT_MODEL_TAG=deepseek-ai/DeepSeek-R1-MTP
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=2048
export QUANT_MODE=weight_only_int8
export DRAFT_MODEL_QUANT_MODE=weight_only_int8
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m paddle.distributed.launch --ips "192.168.0.1,192.168.0.2"\
  --gpus ${CUDA_VISIBLE_DEVICES} \
  predictor.py \
  --model_name_or_path ${OUTPUT_PATH} \
  --dtype bfloat16 \
  --mode static \
  --inference_model 1 \
  --append_attn 1 \
  --total_max_length ${TOTAL_MAX_LENGTH} \
  --quant_type ${QUANT_MODE} \
  --max_length 1024 \
  --speculate_method mtp \
  --draft_model_name_or_path ${DRAFT_MODEL_TAG} \
  --draft_model_quant_type ${DRAFT_MODEL_QUANT_MODE} \
  --speculate_max_draft_token_num 1 \
  --return_full_hidden_states 1 \
  --mla_use_matrix_absorption 1

$ bash run_mtp_infer.sh
```

2. DeepSeek-R1 uses FP8, MTP uses FP8, dual-machine TP16

node1 and node2 have identical scripts
```
```shell
# Export script
$ cat export.sh

export MODEL_TAG=deepseek-ai/DeepSeek-R1-FP8
export OUTPUT_PATH=/path/to/exported_model
export QUANT_MODE=a8w8_fp8
export TOTAL_MAX_LENGTH=8192
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m paddle.distributed.launch --ips "192.168.0.1,192.168.0.2"\
  --gpus ${CUDA_VISIBLE_DEVICES} \
  export_model.py \
  --model_name_or_path ${MODEL_TAG} \
  --output_path ${OUTPUT_PATH} \
  --dtype bfloat16 \
  --inference_model 1 \
  --append_attn 1 \
  --total_max_length ${TOTAL_MAX_LENGTH} \
  --quant_type ${QUANT_MODE} \
  --speculate_method inference_with_reference \
  --return_full_hidden_states 1 \
  --mla_use_matrix_absorption 1

$ bash export.sh
```

node1 and node2 scripts are the same
```shell
# Inference Script
$ cat run_mtp_infer.sh

export OUTPUT_PATH=/path/to/exported_model
export DRAFT_MODEL_TAG=deepseek-ai/DeepSeek-R1-MTP-FP8
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=2048
export QUANT_MODE=a8w8_fp8
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m paddle.distributed.launch --ips "192.168.0.1,192.168.0.2"\
  --gpus ${CUDA_VISIBLE_DEVICES} \
  predictor.py \
  --model_name_or_path ${OUTPUT_PATH} \
  --dtype bfloat16 \
  --mode static \
  --inference_model 1 \
  --append_attn 1 \
  --total_max_length ${TOTAL_MAX_LENGTH} \
  --quant_type ${QUANT_MODE} \
  --max_length ${MAX_DEC_LEN} \
  --speculate_method mtp \
  --draft_model_name_or_path ${DRAFT_MODEL_TAG} \
  --draft_model_quant_type ${QUANT_MODE} \
  --speculate_max_draft_token_num 1 \
  --return_full_hidden_states 1 \
  --mla_use_matrix_absorption 1

$ bash run_mtp_infer.sh
```
