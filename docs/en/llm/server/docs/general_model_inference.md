# Large Model Service Deployment - Quick Start Tutorial

*This deployment tool is designed for server-side large model service deployment based on NVIDIA Triton framework. It provides service interfaces supporting gRPC and HTTP protocols, along with streaming token output capabilities. The underlying inference engine supports acceleration optimizations such as continuous batching, weight only int8, and post-training quantization (PTQ), delivering an easy-to-use and high-performance deployment experience.*

## Quick Start

Deploy using precompiled images, **using PaddlePaddle static graph models**. This section uses an a100/v100 machine to run meta-llama/Meta-Llama-3-8B-Instruct bf16 inference as an example. Other models need to be exported as **static graph model format** according to requirements. For more models, please refer to [LLaMA](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/llama.md), [Qwen](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/qwen.md), [DeepSeek](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/deepseek.md), [Mixtral](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/mixtral.md). For more detailed model inference and quantization tutorials, please refer to [Large Model Inference Tutorial](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/inference.md)

### Supported Images

|cuda version| Supported Arch | Image Address | Supported Devices |
|:------|:-:|:-:|:-:|
| cuda11.8 | 70 75 80 86 |ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda118-cudnn8-v2.1 |V100, T4, A100, A30, A10 |
| cuda12.4 | 80 86 89 90 |ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.1 |A100, A30, A10, L20, H20, H800 |

### Static Graph Quick Deployment

This method only supports one-click service launch for models listed in [One-Click Supported Models](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md).

To avoid long download times for large models, we provide automatic download scripts directly (refer to [Service Deployment Process](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/deploy_usage_tutorial.md)), supporting model download before service launch. After entering the container, perform static graph download according to single/multi-node models.

`MODEL_PATH` specifies the storage path for model download, which can be customized.
`model_name` specifies the name of the model to download. For a complete list of supported models, please refer to the [documentation](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md).

Note:
1. Ensure shm-size >=5, otherwise service startup may fail
2. Please verify the required environment and hardware specifications for the model before deployment. Refer to the [documentation](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md)

**A100 Deployment Example**
```shell
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name=${model_name:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B/weight_only_int8"}
docker run  -i --rm  --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.1 /bin/bash \
-c -ex 'start_server $model_name && tail -f /dev/null'
```

**V100 Deployment Example**

```shell
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name=${model_name:-"meta-llama/Meta-Llama-3-8B-Instruct-Block-Attn/float16"}
docker run  -i --rm  --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda118-cudnn8-v2.1 /bin/bash \
-c -ex 'start_server $model_name && tail -f /dev/null'
```

### Service Testing
```
curl 127.0.0.1:9965/v1/chat/completions \
  -H'Content-Type: application/json' \
  -d'{"text": "hello, llm"}'
```
### User Manual Export for Static Graph Deployment

For models that do not support one-click export, users need to manually export them for serving inference. Please refer to the following content for service-oriented deployment.

#### Model Export

High-performance deployment requires exporting the dynamic graph model to static inference format. The export commands for A100/V100 machines are as follows:

> MODEL_PATH # Path to save static graph model
> --dtype # Select export precision
> --append_attn # Only supported on sm>=80 devices
> --block_attn # Supported on sm<80 devices. If inference fails with append_attn, replace with block_attn
> [sm 对应 GPU 型号查询](https://developer.nvidia.com/cuda-gpus)

**A100 Deployment Example**
```shell
export MODEL_PATH=${MODEL_PATH:-$PWD}
docker run -i --rm --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH/:/models -dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.1 /bin/bash \
-c -ex 'cd /opt/source/PaddleNLP && export PYTHONPATH=$PWD:$PYTHONPATH && cd llm && python3 predict/export_model.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --output_path /models --dtype bfloat16 --inference_model 1 --append_attn 1' \
&& docker logs -f $(docker ps -lq)
```

**V100 Deployment Example**
⚠️ V100 only supports float16 due to hardware instruction limitations
```shell
export MODEL_PATH=${MODEL_PATH:-$PWD}
docker run -i --rm --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH/:/models -dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda118-cudnn8-v2.1 /bin/bash \
-c -ex 'cd /opt/source/PaddleNLP && export PYTHONPATH=$PWD:$PYTHONPATH && cd llm && python3 predict/export_model.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --output_path /models --dtype float16 --inference_model 1 --block_attn' \
&& docker logs -f $(docker ps -lq)
```

### Inference Service
For detailed deployment steps and parameter explanations, please refer to the [Documentation](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/deploy_usage_tutorial.md)

```shell
export docker_img=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.1

export MODEL_PATH=${MODEL_PATH:-$PWD}
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH/:/models -dit $docker_img /bin/bash \
-c -ex 'start_server && tail -f /dev/null'
```

**More Documentation**

- For detailed instructions on deployment tools, please refer to the [Service Deployment Process](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/deploy_usage_tutorial.md)
- For static graph supported models, please refer to [Static Graph Model Support](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md)
