# DeepSeek

本文档展示了如何在 PaddleNLP 中构建和运行[DeepSeek](https://www.deepseek.com/) 系列大模型。

## 模型介绍

* DeepSeek 系列大模型是由深度求索（DeepSeek Inc.）研发的高效开源语言模型，专注提升模型推理效率与多场景应用能力。

* [DeepSeek V3](https://www.deepseek.com/): 2024年12月，DeepSeek-V3 首个版本上线并同步开源，DeepSeek-V3 为 MoE 模型，671B 参数，激活 37B。
* [DeepSeek R1](https://www.deepseek.com/): 2025年1月，深度求索发布 DeepSeek-R1，并同步开源模型权重。
* [DeepSeek R1 Distill Model](https://www.deepseek.com/): 2025年1月，深度求索在开源 R1 模型的同时，通过 DeepSeek-R1 的输出，蒸馏了6个小模型并开源，分别是 Qwen1.5B、7B、14B、32B 以及 Llama8B、70B。

## 已验证的模型（CKPT）

|Model|
|:-|
|deepseek-ai/DeepSeek-V2-Chat|
|deepseek-ai/DeepSeek-V2-Lite-Chat|
|deepseek-ai/DeepSeek-V3|
|deepseek-ai/DeepSeek-R1|
|deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B|
|deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|
|deepseek-ai/DeepSeek-R1-Distill-Qwen-14B|
|deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|
|deepseek-ai/DeepSeek-R1-Distill-Llama-8B|
|deepseek-ai/DeepSeek-R1-Distill-Llama-70B|

## 预制的静态图

|模型名称|精度|MTP|节点数|静态图下载 model_name|
|:------|:-:|:-:|:-:|:-:|
| deepseek-ai/DeepSeek-R1  |weight_only_int4|否|1| deepseek-ai/DeepSeek-R1/weight_only_int4 |
| deepseek-ai/DeepSeek-R1  |weight_only_int4|是|1| deepseek-ai/DeepSeek-R1-MTP/weight-only-int4 |
| deepseek-ai/DeepSeek-R1  |weight_only_int8|否|2| deepseek-ai/DeepSeek-R1-2nodes/weight_only_int8 |
| deepseek-ai/DeepSeek-R1  |weight_only_int8|是|2| deepseek-ai/DeepSeek-R1-MTP-2nodes/weight-only-int8 |
| deepseek-ai/DeepSeek-R1  |a8w8_fp8|否|2| deepseek-ai/DeepSeek-R1-2nodes/a8w8_fp8|
| deepseek-ai/DeepSeek-R1  |a8w8_fp8|是|2| deepseek-ai/DeepSeek-R1-MTP-2nodes/a8w8_fp8|
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B |weight_only_int8|-|-| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/weight_only_int8 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   |weight_only_int8|-|-|deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/weight_only_int8 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B  |weight_only_int8|-|-|deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/weight_only_int8 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  |weight_only_int8|-|-|deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/weight_only_int8 |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B  |weight_only_int8|-|-|deepseek-ai/DeepSeek-R1-Distill-Llama-8B/weight_only_int8 |
| deepseek-ai/DeepSeek-R1-Distill-Llama-70B |weight_only_int8|-|-|deepseek-ai/DeepSeek-R1-Distill-Llama-70B/weight_only_int8 |


## 一键启动推理服务

### deepseek-ai/DeepSeek-R1

单机 WINT4-TP8 推理

```shell
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v /PATH_TO_MODEL/:/models \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v1.0 /bin/bash \
-c -ex 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export MP_NUM=8 &&  model_name=${model_name:-"deepseek-ai/DeepSeek-R1/weight_only_int4"} && cd /opt/output/Serving && bash start_server.sh $model_name && tail -f /dev/null'\
&& docker exec -it $(docker ps -lq) sh -c "while [ ! -f /opt/output/Serving/log/workerlog.0 ]; do sleep 1; done; tail -f /opt/output/Serving/log/workerlog.0"
```

两机 WINT8-TP16 推理

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
# /PATH_TO_MODEL # 模型挂载路径
```

```shell
# node1
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v /PATH_TO_MODEL/:/models \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v1.0 /bin/bash \
-c -ex 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export MP_NUM=16 && export MP_NNODE=2 && export POD_0_IP=192.168.0.1 && export POD_IPS=192.168.0.1,192.168.0.2 && model_name=${model_name:-"deepseek-ai/DeepSeek-R1-2nodes/weight_only_int8"} && cd /opt/output/Serving && bash start_server.sh $model_name && tail -f /dev/null'\
&& docker exec -it $(docker ps -lq) sh -c "while [ ! -f /opt/output/Serving/log/workerlog.0 ]; do sleep 1; done; tail -f /opt/output/Serving/log/workerlog.0"

# node2
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v /PATH_TO_MODEL/:/models -dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v1.0 /bin/bash \
-c -ex 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export MP_NUM=16 && export MP_NNODE=2 && export POD_0_IP=192.168.0.1 && export POD_IPS=192.168.0.1,192.168.0.2 && model_name=${model_name:-"deepseek-ai/DeepSeek-R1-2nodes/weight_only_int8"} && cd /opt/output/Serving && bash start_server.sh $model_name && tail -f /dev/null'\
&& docker exec -it $(docker ps -lq) sh -c "while [ ! -f /opt/output/Serving/log/workerlog.0 ]; do sleep 1; done; tail -f /opt/output/Serving/log/workerlog.0"
```

两机 FP8-TP16 推理

```shell
# node1
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v /PATH_TO_MODEL/:/models \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v1.0 /bin/bash \
-c -ex 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export MP_NUM=16 && export MP_NNODE=2 && export POD_0_IP=192.168.0.1 && export POD_IPS=192.168.0.1,192.168.0.2 && model_name=${model_name:-"deepseek-ai/DeepSeek-R1-2nodes/a8w8_fp8"} && cd /opt/output/Serving && bash start_server.sh $model_name && tail -f /dev/null'\
&& docker exec -it $(docker ps -lq) sh -c "while [ ! -f /opt/output/Serving/log/workerlog.0 ]; do sleep 1; done; tail -f /opt/output/Serving/log/workerlog.0"

# node2
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v /PATH_TO_MODEL/:/models \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v1.0 /bin/bash \
-c -ex 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export MP_NUM=16 && export MP_NNODE=2 && export POD_0_IP=192.168.0.1 && export POD_IPS=192.168.0.1,192.168.0.2 && model_name=${model_name:-"deepseek-ai/DeepSeek-R1-2nodes/a8w8_fp8"} && cd /opt/output/Serving && bash start_server.sh $model_name && tail -f /dev/null'\
&& docker exec -it $(docker ps -lq) sh -c "while [ ! -f /opt/output/Serving/log/workerlog.0 ]; do sleep 1; done; tail -f /opt/output/Serving/log/workerlog.0"
```

开启 MTP 模式，参考 [投机解码部分](./speculative_decoding.md)。

### deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

```shell
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v /PATH_TO_MODEL/:/models \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v1.0 /bin/bash \
-c -ex 'model_name=${model_name:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/weight_only_int8"} && cd /opt/output/Serving && bash start_server.sh $model_name && tail -f /dev/null'\
&& docker exec -it $(docker ps -lq) sh -c "while [ ! -f /opt/output/Serving/log/workerlog.0 ]; do sleep 1; done; tail -f /opt/output/Serving/log/workerlog.0"
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

## 模型本地推理

### deepseek-ai/DeepSeek-R1
单机 WINT4-TP8 推理

```shell
# 动态图推理
export MODEL_TAG=deepseek-ai/DeepSeek-R1
export QUANT_MODE=weight_only_int4
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=4096
export FLAGS_mla_use_tensorcore=1 # only support Hopper, Amper shoule be 0
export FLAGS_cascade_attention_max_partition_size=${TOTAL_MAX_LENGTH}
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
  --mla_use_matrix_absorption 1


# 动转静导出模型
export MODEL_TAG=deepseek-ai/DeepSeek-R1
export OUTPUT_PATH=/path/to/exported_model
export QUANT_MODE=weight_only_int4
export TOTAL_MAX_LENGTH=8192
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m paddle.distributed.launch \
  --gpus ${CUDA_VISIBLE_DEVICES} \
  export_model.py \
  --model_name_or_path ${MODEL_TAG} \
  --output_path ${OUTPUT_PATH} \
  --dtype bfloat16 \
  --inference_model 1 \
  --append_attn 1 \
  --total_max_length ${TOTAL_MAX_LENGTH} \
  --quant_type ${QUANT_MODE} \
  --mla_use_matrix_absorption 1


# 静态图推理
export OUTPUT_PATH=/path/to/exported_model
export QUANT_MODE=weight_only_int4
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=4096
export FLAGS_mla_use_tensorcore=1 # only support Hopper, Amper shoule be 0
export FLAGS_cascade_attention_max_partition_size=${TOTAL_MAX_LENGTH}
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m paddle.distributed.launch \
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
  --mla_use_matrix_absorption 1
```

两机 WINT8-TP16 推理
```shell
# 动态图推理
export MODEL_TAG=deepseek-ai/DeepSeek-R1
export QUANT_MODE=weight_only_int8
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=4096
export FLAGS_mla_use_tensorcore=1 # only support Hopper, Amper shoule be 0
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
  --mla_use_matrix_absorption 1


# 动转静导出模型
export MODEL_TAG=deepseek-ai/DeepSeek-R1
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
  --mla_use_matrix_absorption 1


# 静态图推理
export OUTPUT_PATH=/path/to/exported_model
export QUANT_MODE=weight_only_int8
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=4096
export FLAGS_mla_use_tensorcore=1 # only support Hopper, Amper shoule be 0
export FLAGS_cascade_attention_max_partition_size=${TOTAL_MAX_LENGTH}
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
  --max_length ${MAX_DEC_LEN} \
  --mla_use_matrix_absorption 1
```

两机 FP8-TP16 推理
```shell
# 动态图推理
export MODEL_TAG=deepseek-ai/DeepSeek-R1-FP8
export QUANT_MODE=a8w8_fp8
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=4096
export FLAGS_mla_use_tensorcore=1 # only support Hopper, Amper shoule be 0
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
  --mla_use_matrix_absorption 1 \
  --weight_block_size 128 128


# 动转静导出模型
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
  --mla_use_matrix_absorption 1 \
  --weight_block_size 128 128


# 静态图推理
export OUTPUT_PATH=/path/to/exported_model
export QUANT_MODE=a8w8_fp8
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=4096
export FLAGS_mla_use_tensorcore=1 # only support Hopper, Amper shoule be 0
export FLAGS_cascade_attention_max_partition_size=${TOTAL_MAX_LENGTH}
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
  --max_length ${MAX_DEC_LEN} \
  --mla_use_matrix_absorption 1 \
  --weight_block_size 128 128
```

### deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

单机单卡 WINT8 推理

```shell
# 动态图推理
python predictor.py --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type weight_only_int8


# 动转静导出模型
python export_model.py --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type weight_only_int8


# 静态图推理
python predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1
```