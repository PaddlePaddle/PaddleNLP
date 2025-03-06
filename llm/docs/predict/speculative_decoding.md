# 投机解码教程

投机解码是一个通过投机性地一次性猜测多个 token 然后进行验证和接收的算法，通过投机解码可以极大地减小推理时延。PaddleNLP 提供了简单、高效的投机解码推理流程。下面提供 PaddleNLP 中各种投机解码算法的使用说明。

## Inference with reference

该算法通过 n-gram 窗口从 prompt 中匹配 draft tokens，适合输入和输出有很大 overlap 的场景如代码编辑、文档查询等，更多信息查看查看[论文地址](https://arxiv.org/pdf/2304.04487)。

### 使用命令

```shell
# 动态图模型推理命令参考
python predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --speculate_method inference_with_reference --speculate_max_draft_token_num 5 --speculate_max_ngram_size 2
```

**Note:**

1. 该算法目前只支持 llama 系列模型。
2. 投机解码同时支持量化推理，具体命令参考[推理示例](./inference.md)，将 speculate_method 等投机解码参数加上即可。

## Multi-Token Prediction(MTP)

Paper：https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf
![MTP](https://github.com/user-attachments/assets/6cdc3d94-7bab-4c0f-991c-875ae24816a6)

支持 DeepSeek-V3/R1 & MTP 的推理功能，具体原理请查阅论文。

特性功能：
1. Base Model 验证阶段使用优化的 Attention，可一次性产出所有 Draft Token 的 logits，无需使用 batch_extension。
2. 在 Base model 接受部分或所有 Draft Token 的情况下，MTP 均可通过一次 Forward 完成对所有 Token 的计算。
3. 使用分离框架，即 Base Model 和 MTP 分别加载权重。

### 使用命令
**DeepSeek-V3/R1 + MTP, 两机 TP16 划分**

1. 一键容器启动 Base Model 静态图 + MTP 动态图推理服务

```shell
需要保证2机器节点可以互相ping通
# 第一个节点(master)
ping 192.168.0.1
# 第二个节点(slave)
ping 192.168.0.2
model_name=${model_name:-"deepseek-ai/DeepSeek-R1-2nodes/weight_only_int8"}
export POD_0_IP=master_ip
export POD_IPS=master_ip,slave_ip # 该环境变量在2机上都需保持一致
# 服务化默认启动端口，如果冲突可以通过export进行修改
export SERVICE_HTTP_PORT=${PUSH_MODE_HTTP_PORT:-${SERVICE_HTTP_PORT:-"9965"}}
# 开启 MTP
export SPECULATE_METHOD="mtp"
# /PATH_TO_MODEL # 模型挂载路径
# /PATH_TO_MTP # MTP 挂载路径
```

```shell
# node1
export model_name=${model_name:-"deepseek-ai/DeepSeek-R1-2nodes/a8w8_fp8"}
export MODEL_PATH=${MODEL_PATH:-/PATH_TO_MODEL/}
export MODEL_MTP_PATH=${MODEL_PATH:-/PATH_TO_MTP/}
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -v $MODEL_MTP_PATH:/models-mtp -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v1.0 /bin/bash \
-c -ex 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export MP_NUM=16 && export MP_NNODE=2 && export POD_0_IP=192.168.0.1 && export POD_IPS=192.168.0.1,192.168.0.2 && export SPECULATE_MODEL_QUANT_TYPE="weight_only_int8" && export SPECULATE_METHOD="mtp" && export SPECULATE_MODEL_PATH="/models-mtp" && export SPECULATE_MAX_DRAFT_TOKEN_NUM=1 && start_server $model_name && tail -f /dev/null'

# node2
export model_name=${model_name:-"deepseek-ai/DeepSeek-R1-2nodes/a8w8_fp8"}
export MODEL_PATH=${MODEL_PATH:-/PATH_TO_MODEL/}
export MODEL_MTP_PATH=${MODEL_PATH:-/PATH_TO_MTP/}
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -v $MODEL_MTP_PATH:/models-mtp -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v1.0 /bin/bash \
-c -ex 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export MP_NUM=16 && export MP_NNODE=2 && export POD_0_IP=192.168.0.1 && export POD_IPS=192.168.0.1,192.168.0.2 &&export SPECULATE_MODEL_QUANT_TYPE="weight_only_int8" && export SPECULATE_METHOD="mtp" && export SPECULATE_MODEL_PATH="/models-mtp" && export SPECULATE_MAX_DRAFT_TOKEN_NUM=1 && start_server $model_name  && tail -f /dev/null'
```

### 请求服务化

curl 请求
```shell
curl ${ip}:9965/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
      "model":"default",
      "text":"Hello, how are you?"
  }'
```
OpenAI 请求
```python
import openai
client = openai.Client(base_url=f"http://127.0.0.1:9965/v1/chat/completions", api_key="EMPTY_API_KEY")
# 非流式返回
response = client.completions.create(
    model="default",
    prompt="Hello, how are you?",
  max_tokens=50,
  stream=False,
)

print(response)
print("\n")

# 流式返回
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

2. Base Model 动态图 + MTP 动态图

```shell
export MODEL_TAG=deepseek-ai/DeepSeek-V3
export DRAFT_MODEL_TAG=deepseek-ai/DeepSeek-V3-MTP
export QUANT_MODE=weight_only_int8
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=2048
# 算子加速策略
export FLAGS_mla_use_tensorcore=1
export FLAGS_cascade_attention_max_partition_size=${TOTAL_MAX_LENGTH}
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
mpirun python -m paddle.distributed.launch \
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
  --speculate_max_draft_token_num 1 \
  --speculate_max_ngram_size 3 \
  --return_full_hidden_states 1 \
  --mla_use_matrix_absorption 1
```

3.【**推荐**】Base Model 静态图 + MTP 动态图

Base Model 静态图导出

> 注：投机解码导出支持所有方法，因此这里 speculate_method 设为默认的 inference_with_reference 即可

```shell
export MODEL_TAG=deepseek-ai/DeepSeek-V3
export OUTPUT_PATH=/path/to/exported_model
export QUANT_MODE=weight_only_int8
export TOTAL_MAX_LENGTH=8192
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
mpirun python -m paddle.distributed.launch \
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
```

推理脚本

```shell
export OUTPUT_PATH=/path/to/exported_model
export DRAFT_MODEL_TAG=deepseek-ai/DeepSeek-V3-MTP
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=2048
export QUANT_MODE=weight_only_int8
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
mpirun python -m paddle.distributed.launch \
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
  --speculate_max_draft_token_num 1 \
  --speculate_max_ngram_size 3 \
  --return_full_hidden_states 1 \
  --mla_use_matrix_absorption 1
```

4. R1-FP8 Model 动态图 + MTP 动态图

```shell
export MODEL_TAG=deepseek-ai/DeepSeek-R1-FP8
export DRAFT_MODEL_TAG=deepseek-ai/DeepSeek-R1-MTP-FP8
export QUANT_MODE=a8w8_fp8
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=2048
# 算子加速策略
export FLAGS_mla_use_tensorcore=1
export FLAGS_cascade_attention_max_partition_size=${TOTAL_MAX_LENGTH}
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
mpirun python -m paddle.distributed.launch \
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
  --speculate_max_draft_token_num 1 \
  --speculate_max_ngram_size 3 \
  --return_full_hidden_states 1 \
  --mla_use_matrix_absorption 1
```

5.【**推荐**】R1 Model 静态图 + MTP 动态图

R1 Model 静态图导出

> 注：投机解码导出支持所有方法，因此这里 speculate_method 设为默认的 inference_with_reference 即可

```shell
export MODEL_TAG=deepseek-ai/DeepSeek-R1-FP8
export OUTPUT_PATH=/path/to/exported_model
export QUANT_MODE=a8w8_fp8
export TOTAL_MAX_LENGTH=8192
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
mpirun python -m paddle.distributed.launch \
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
```

推理脚本

```shell
export OUTPUT_PATH=/path/to/exported_model
export DRAFT_MODEL_TAG=deepseek-ai/DeepSeek-R1-MTP-FP8
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=2048
export QUANT_MODE=a8w8_fp8
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
mpirun python -m paddle.distributed.launch \
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
  --draft_model_quant_type ${QUANT_MODE} \
  --speculate_max_draft_token_num 1 \
  --speculate_max_ngram_size 3 \
  --return_full_hidden_states 1 \
  --mla_use_matrix_absorption 1
```