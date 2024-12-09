
<h1 align="center"><b><em>大模型服务化部署</em></b></h1>

*该部署工具是基于英伟达Triton框架专为服务器场景的大模型服务化部署而设计。它提供了支持gRPC、HTTP协议的服务接口，以及流式Token输出能力。底层推理引擎支持连续批处理、weight only int8、后训练量化（PTQ）等加速优化策略，为用户带来易用且高性能的部署体验。*

# 快速开始

  基于预编译镜像部署，本节以 Meta-Llama-3-8B-Instruct-A8W8C8 为例，更多模型请参考[LLaMA](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/llama.md)、[Qwen](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/qwen.md)、[Mixtral](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/mixtral.md), 更细致的模型推理、量化教程可以参考[大模型推理教程](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/inference.md)：

  ```
    # 下载模型
    wget https://paddle-qa.bj.bcebos.com/inference_model/Meta-Llama-3-8B-Instruct-A8W8C8.tar
    mkdir Llama-3-8B-A8W8C8 && tar -xf Meta-Llama-3-8B-Instruct-A8W8C8.tar -C Llama-3-8B-A8W8C8

    # 挂载模型文件
    export MODEL_PATH=${PWD}/Llama-3-8B-A8W8C8

    docker run --gpus all --shm-size 5G --network=host --privileged --cap-add=SYS_PTRACE \
    -v ${MODEL_PATH}:/models/ \
    -dit registry.baidubce.com/paddlepaddle/fastdeploy:llm-serving-cuda123-cudnn9-v1.2 \
    bash -c 'export USE_CACHE_KV_INT8=1 && cd /opt/output/Serving && bash start_server.sh; exec bash'
  ```

  等待服务启动成功（服务初次启动大概需要40s），可以通过以下命令测试：

  ```
    curl 127.0.0.1:9965/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"text": "hello, llm"}'
  ```

Note:
1. 请保证 shm-size >= 5，不然可能会导致服务启动失败

更多关于该部署工具的使用方法，请查看[服务化部署流程](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/deploy_usage_tutorial.md)

# License

遵循 [Apache-2.0开源协议](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/LICENSE) 。
