# 大模型服务化部署-快速开始教程

*该部署工具是基于英伟达 Triton 框架专为服务器场景的大模型服务化部署而设计。它提供了支持 gRPC、HTTP 协议的服务接口，以及流式 Token 输出能力。底层推理引擎支持连续批处理、weight only int8、后训练量化（PTQ）等加速优化策略，为用户带来易用且高性能的部署体验。*

## 快速开始

  基于预编译镜像部署，**使用飞桨静态图模型部署**。本节以a100/v100机器跑 meta-llama/Meta-Llama-3-8B-Instruct bf16 推理为例子。其他模型需按照要求导出为**静态图模型格式**。更多模型请参考[LLaMA](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/llama.md)、[Qwen](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/qwen.md)、[DeepSeek](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/deepseek.md)、[Mixtral](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/mixtral.md), 更细致的模型推理、量化教程可以参考[大模型推理教程](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/inference.md)


### 支持镜像

|cuda版本| 支持硬件架构|镜像地址|支持的典型设备|
|:------|:-:|:-:|:-:|
| cuda11.8 | 70 75 80 86 |ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda118-cudnn8-v2.3 |V100，T4，A100，A30，A10 |
| cuda12.4 | 80 86 89 90 |ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.3 |A100，A30，A10，L20，H20，H800 |

 ### 静态图快速部署

该方法仅支持[可一键跑通的模型列表](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md)中的模型进行一键启动推理服务

`MODEL_PATH` 为指定模型下载的存储路径，可自行指定
`model_name` 为指定下载模型名称，具体支持模型可查看[文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md)

Note:
1. 请保证 shm-size >= 5，不然可能会导致服务启动失败
2. 部署前请确认模型所需要的环境和硬件，请参考[文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md)
   

**A100部署示例**
```shell
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name=${model_name:-"meta-llama/Meta-Llama-3-8B-Instruct-Append-Attn/bfloat16"}
docker run  -i --rm  --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.3 /bin/bash \
-c -ex 'start_server $model_name && tail -f /dev/null'
```


**V100部署示例**

```shell
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name=${model_name:-"meta-llama/Meta-Llama-3-8B-Instruct-Block-Attn/float16"}
docker run  -i --rm  --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -e "model_name=${model_name}" \ 
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda118-cudnn8-v2.3 /bin/bash \
-c -ex 'start_server $model_name && tail -f /dev/null'
```

### 服务测试
```
curl 127.0.0.1:9965/v1/chat/completions \
  -H'Content-Type: application/json' \
  -d'{"text": "hello, llm"}'
```
### 用户手动导出静态图部署

不支持一键导出的模型暂时仅支持用户自行导出进行服务化推理,可参考以下内容进行推理服务化部署

#### 模型导出

高性能部署需要先将动态图模型，导出为静态图推理格式，针对A100/V100机器的导出命令如下：  

> MODEL_PATH #静态图模型存放地址  
> --dtype #可选择导出精度  
> --append_attn #仅sm>=80的机器支持  
> --block_attn #支持sm<80的机器导出，如果append_attn无法推理可直接替换成block_attn  
>[sm对应GPU型号查询](https://developer.nvidia.com/cuda-gpus)  

**A100部署示例**
```shell
export MODEL_PATH=${MODEL_PATH:-$PWD}
docker run  -i --rm  --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH/:/models -dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.3 /bin/bash \
-c -ex 'cd /opt/source/PaddleNLP &&export PYTHONPATH=$PWD:$PYTHONPATH && cd llm && python3 predict/export_model.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --output_path /models --dtype bfloat16 --inference_model 1 --append_attn 1'\
&& docker logs -f $(docker ps -lq)
```

**V100部署示例**
 ⚠️ v100由于硬件指令限制，仅支持float16  
```shell
export MODEL_PATH=${MODEL_PATH:-$PWD}
docker run  -i --rm  --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH/:/models -dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda118-cudnn8-v2.3 /bin/bash \
-c -ex 'cd /opt/source/PaddleNLP &&export PYTHONPATH=$PWD:$PYTHONPATH&& cd llm && python3 predict/export_model.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --output_path /models --dtype float16 --inference_model 1 --block_attn'\
&& docker logs -f $(docker ps -lq)
```

### 服务化推理
具体的部署细节以及参数说明可以查看[文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/deploy_usage_tutorial.md)


```shell
export docker_img=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.3

export MODEL_PATH=${MODEL_PATH:-$PWD}
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH/:/models -dit $docker_img /bin/bash \
-c -ex 'start_server && tail -f /dev/null'
```


**更多文档**

- 部署工具详细说明请查看[服务化部署流程](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/deploy_usage_tutorial.md)
- 静态图支持模型请查看[静态图模型下载支持](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md)

## License

遵循 [Apache-2.0开源协议](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/LICENSE) 。
