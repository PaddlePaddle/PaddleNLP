# 投机解码教程

投机解码是一个通过投机性地一次性猜测多个 token 然后进行验证和接收的算法，通过投机解码可以极大地减小推理时延。PaddleNLP 提供了简单、高效的投机解码推理流程。下面提供 PaddleNLP 中各种投机解码算法的使用说明。
## 高效投机解码框架
传统的 Draft Token 验证以及 MTP(Eagle/Draft Model) 推理方法，batch_size 一般会膨胀 Draft Token 数量的倍数，我们高效的注意力机制则可以保持原始批次大小，减少计算量，如下图所示：
![框架图](https://github.com/user-attachments/assets/fd938c8c-fcd5-4ead-ae6e-f455af415e52)
参考 [Vllm batch_extension](https://docs.google.com/document/d/1T-JaS2T1NRfdP51qzqpyakoCXxSXTtORppiwaj5asxA/edit?tab=t.0#heading=h.kk7dq05lc6q8)
## 参数说明
- `speculate_method`: 推理解码算法，默认值为`None`，可选的数值有`None`、`inference_with_reference`、 `mtp`、 `eagle`。为`None`时为正常自回归解码，为`inference_with_reference`时为基于上下文的投机解码[论文地址](https://arxiv.org/pdf/2304.04487)。

- `speculate_max_draft_token_num`: 投机解码算法中每轮产生的最大 draft tokens 数目，默认值为 1，最大支持 5。

- `speculate_max_ngram_size`: n-gram 匹配 draft tokens 时的最大窗口大小，默认值为`1`。inference_with_reference 算法中会先从 prompt 中使用 ngram 窗口滑动匹配 draft tokens，窗口大小和输入输出重叠程度共同决定了产生 draft tokens 的开销从而影响 inference_with_reference 算法的加速效果。

- `speculate_verify_window`(暂时废弃): 投机解码 verify 策略默认采用 TopP + window verify 中的 window 大小，默认值为`2`。更多有关 TopP + window verify 的详细介绍参考[投机解码教程](./speculative_decoding.md)。

- `speculate_max_candidate_len`(暂时废弃): 产生的最大候选 tokens 数目，根据候选 tokens 与 draft tokens 比较来进行 verify(仅在 TopP + window verify 时生效)，默认值为`5`。

- `draft_model_name_or_path`: 在`MTP`或者`EAGLE`模式下，`Draft Model`的路径。

- `draft_model_quant_type`: 在`MTP`或者`EAGLE`模式下，`Draft Model`的推理量化精度，参考`--quant_type`。

- `return_full_hidden_states`: 在`MTP`或者`EAGLE`模式下，是否返回全部的隐藏层状态，默认为`False`。

Others：
1. 投机解码目前支持的最大 batch_size 为 128
## Inference with reference

该算法通过 n-gram 窗口从 prompt 中匹配 draft tokens，适合输入和输出有很大 overlap 的场景如代码编辑、文档查询等，更多信息查看查看[论文地址](https://arxiv.org/pdf/2304.04487)。

### 使用命令

```shell
# 动态图模型推理命令参考
python predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --speculate_method inference_with_reference --speculate_max_draft_token_num 5 --speculate_max_ngram_size 2
```

**Note:**

1. 该算法目前只支持 llama/DeepSeek-V3/R1 系列模型。
2. 投机解码同时支持量化推理，具体命令参考[推理示例](./inference.md)，将 speculate_method 等投机解码参数加上即可。

## Multi-Token Prediction(MTP)

Paper：https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf
![MTP](https://github.com/user-attachments/assets/6cdc3d94-7bab-4c0f-991c-875ae24816a6)

支持 DeepSeek-V3/R1 & MTP 的推理功能，具体原理请查阅论文。

特色功能：
1. Base Model 验证阶段使用优化的 Attention，可一次性产出所有 Draft Token 的 logits，无需使用 batch_extension 扩大 batch_size，解决了大批次下投机解码无法加速推理性能的问题。
2. 在 MTP 推理阶段，统一对上轮 Draft Token 接受情况的处理，例如全部拒绝、接受部分、接受全部，同样只需保持原始的批次大小，一次性处理所有输入请求。
3. 使用分离框架，即 Base Model 和 MTP 分别加载权重，导出后兼容多种解码方式。

### MTP 支持量化精度

| 基础模型 | 部署机器 | 基础模型量化类型 | MTP 量化类型 |
| --- | --- | --- | --- |
| DeepSeek-R1 | TP8 | a8w8_fp8_wint4 | a8w8_fp8 |
| DeepSeek-R1 | TP8 | weight_only_int4 | weight_only_int8 |
| DeepSeek-R1 | TP16(2*TP8) | a8w8_fp8 | a8w8_fp8 |
| DeepSeek-R1 | TP16(2*TP8) | weight_only_int8 |weight_only_int8 |

 支持 DeepSeek-R1 与 MTP 的多种推理混合精度，可通过容器部署或脚本的方式进行推理，以下为部分示例

### 方法一：使用容器一键部署
**DeepSeek-R1(a8w8_fp8_wint4) + MTP(a8w8_fp8), 单机 TP8 部署**

```shell
export MODEL_PATH=${MODEL_PATH:-/PATH_TO_MODEL/}
export MODEL_MTP_PATH=${MODEL_PATH:-/PATH_TO_MTP/}
export model_name="deepseek-ai/DeepSeek-R1-MTP/a8w8_fp8_wint4"

docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -v $MODEL_MTP_PATH:/models-mtp -v /ssd2/paddle_example:/work -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.3 /bin/bash \
-c -ex 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export MP_NUM=8 && export SPECULATE_MODEL_QUANT_TYPE="a8w8_fp8" && export SPECULATE_METHOD="mtp" && export SPECULATE_MODEL_PATH="/models-mtp" && export SPECULATE_MAX_DRAFT_TOKEN_NUM=1 && export BLOCK_BS=32 && export BLOCK_RATIO=0.25 && export BATCH_SIZE="128" && start_server $model_name && tail -f /dev/null'
```
**DeepSeek-R1(weight_only_int8) + MTP(weight_only_int8), 双机 TP16 划分**

1. 一键容器启动  MTP 推理服务

```shell
需要保证2机器节点可以互相ping通
# 第一个节点(master)
ping 192.168.0.1
# 第二个节点(slave)
ping 192.168.0.2
model_name=${model_name:-"deepseek-ai/DeepSeek-R1-MTP-2nodes/weight_only_int8"}
export POD_0_IP=master_ip
export POD_IPS=master_ip,slave_ip # 该环境变量在2机上都需保持一致
# 服务化默认启动端口，如果冲突可以通过export进行修改
export SERVICE_HTTP_PORT=${PUSH_MODE_HTTP_PORT:-${SERVICE_HTTP_PORT:-"9965"}}
#注意 SPECULATE_MODEL_QUANT_TYPE应与MTP支持量化精度表格一致
export SPECULATE_MODEL_QUANT_TYPE="weight_only_int8"
# /PATH_TO_MODEL # 模型挂载路径
# /PATH_TO_MTP # MTP 挂载路径
```

```shell
# node1
export model_name=${model_name:-"deepseek-ai/DeepSeek-R1-MTP-2nodes/weight_only_int8"}
export MODEL_PATH=${MODEL_PATH:-/PATH_TO_MODEL/}
export MODEL_MTP_PATH=${MODEL_PATH:-/PATH_TO_MTP/}
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -v $MODEL_MTP_PATH:/models-mtp -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.3 /bin/bash \
-c -ex 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export MP_NUM=16 && export MP_NNODE=2 && export POD_0_IP=192.168.0.1 && export POD_IPS=192.168.0.1,192.168.0.2 && export SPECULATE_MODEL_QUANT_TYPE="weight_only_int8" && export SPECULATE_METHOD="mtp" && export SPECULATE_MODEL_PATH="/models-mtp" && export SPECULATE_MAX_DRAFT_TOKEN_NUM=1 && start_server $model_name && tail -f /dev/null'

# node2
export model_name=${model_name:-"deepseek-ai/DeepSeek-R1-MTP-2nodes/weight_only_int8"}
export MODEL_PATH=${MODEL_PATH:-/PATH_TO_MODEL/}
export MODEL_MTP_PATH=${MODEL_PATH:-/PATH_TO_MTP/}
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -v $MODEL_MTP_PATH:/models-mtp -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.3 /bin/bash \
-c -ex 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export MP_NUM=16 && export MP_NNODE=2 && export POD_0_IP=192.168.0.1 && export POD_IPS=192.168.0.1,192.168.0.2 &&export SPECULATE_MODEL_QUANT_TYPE="weight_only_int8" && export SPECULATE_METHOD="mtp" && export SPECULATE_MODEL_PATH="/models-mtp" && export SPECULATE_MAX_DRAFT_TOKEN_NUM=1 && start_server $model_name  && tail -f /dev/null'
```

**请求服务化**

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

### 方法二：使用脚本推理测试
#### DeepSeek_R1 动态图 + MTP 动态图, 双机 TP16
1. DeepSeek_R1 使用 weight_only_int8, MTP 使用 weight_only_int8

```shell
启动2机推理 需要保证2机器节点可以互相ping通
# 第一个节点(master)
ping 192.168.0.1
# 第二个节点(slave)
ping 192.168.0.2
```

node1 和 node2 脚本相同
```shell
$ cat run_dynamic_mtp.sh

export MODEL_TAG=deepseek-ai/DeepSeek-R1
export DRAFT_MODEL_TAG=deepseek-ai/DeepSeek-R1-MTP
export QUANT_MODE=weight_only_int8
export DRAFT_MODEL_QUANT_MODE=weight_only_int8
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=2048
# 算子加速策略
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

2. DeepSeek_R1 使用 FP8, MTP 使用 FP8

node1 和 node2 脚本相同
```shell
$ cat run_dynamic_mtp.sh

export MODEL_TAG=deepseek-ai/DeepSeek-R1-FP8
export DRAFT_MODEL_TAG=deepseek-ai/DeepSeek-R1-MTP-FP8
export QUANT_MODE=a8w8_fp8
export DRAFT_MODEL_QUANT_MODE=a8w8_fp8
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=2048
# 算子加速策略
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
#### DeepSeek_R1 动态图 + MTP 动态图, 单机 TP8
1. DeepSeek_R1 使用 weight_only_int8, MTP 使用 weight_only_int8
```shell
$ cat run_dynamic_mtp.sh

export MODEL_TAG=deepseek-ai/DeepSeek-R1
export DRAFT_MODEL_TAG=deepseek-ai/DeepSeek-R1-MTP
export QUANT_MODE=weight_only_int4
export DRAFT_MODEL_QUANT_MODE=weight_only_int8
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=2048
# 算子加速策略
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

#### 【**推荐**】Base Model 静态图 + MTP 动态图, 双机 TP16
> 注：
1.投机解码导出的模型支持所有方法，因此这里 speculate_method 设为默认的 inference_with_reference 即可.
2.静态图模型可从 DeepSeek-R1 导出，或直接下载已上传模型

```shell
启动2机推理 需要保证2机器节点可以互相ping通
# 第一个节点(master)
ping 192.168.0.1
# 第二个节点(slave)
ping 192.168.0.2
```

1. DeepSeek-R1 使用 weight_only_int8，MTP 使用 weight_only_int8

node1 和 node2 脚本相同
```shell
# 导出脚本
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

node1 和 node2 脚本相同
```shell
# 推理脚本
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

2. DeepSeek-R1 使用 FP8, MTP 使用 FP8, 双机 TP16

node1 和 node2 脚本相同
```shell
# 导出脚本
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

node1 和 node2 脚本相同
```shell
# 推理脚本
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
