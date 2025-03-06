# 模型跑通服务化

本文档展示了如何使用docker 在 PaddleNLP 中一键跑通大模型推理。[可一键跑通的模型列表](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md)

#快速开始
基于预编译镜像部署，使用飞桨静态图模型部署。本节以a100/v100机器跑Llama 3推理为例。其他模型需按照要求导出为静态图模型格式。 更细致的模型推理、量化教程可以参考[大模型推理教程](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/inference.md)：

## 一键启动推理服务(推荐)

该方法仅支持[可一键跑通的模型列表](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md)中的模型进行一键启动推理服务

本节以 meta-llama/Meta-Llama-3-8B-Instruct bf16 推理为例子

```shell
export MAX_SEQ_LEN=4096  # 以上模型支持MAX_SEQ_LEN=4096 其他模型可能会不同,可查看config.json 进行修改
export DTYPE="bfloat16"  # 默认是bfloat16 v100 需要替换成   export DTYPE="float16"
PATH_TO_MODEL  # 静态图模型存放路径
```
a100
```shell
docker run  -i --rm  --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v /PATH_TO_MODEL/:/models -dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v1.0 /bin/bash \
-c -ex 'model_name=${model_name:-"meta-llama/Meta-Llama-3-8B-Instruct-Append-Attn/bfloat16"} && cd /opt/output/Serving && export MAX_SEQ_LEN=4096 && bash start_server.sh && tail -f /dev/null'\
&& docker exec -it $(docker ps -lq) sh -c "while [ ! -f /opt/output/Serving/log/workerlog.0 ]; do sleep 1; done; tail -f /opt/output/Serving/log/workerlog.0"
```
v100
```shell
docker run  -i --rm  --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v /PATH_TO_MODEL/:/models -dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda118-cudnn8-v1.0 /bin/bash \
-c -ex 'model_name=${model_name:-"meta-llama/Meta-Llama-3-8B-Instruct-Append-Attn/bfloat16"} && cd /opt/output/Serving && export MAX_SEQ_LEN=4096 && export DTYPE="float16" && bash start_server.sh && tail -f /dev/null'\
&& docker exec -it $(docker ps -lq) sh -c "while [ ! -f /opt/output/Serving/log/workerlog.0 ]; do sleep 1; done; tail -f /opt/output/Serving/log/workerlog.0"
```


## 用户手动导出

不支持一键导出的模型暂时仅支持用户自行导出进行服务化推理,可参考以下内容进行推理服务化部署

### 模型导出
```shell
PATH_TO_MODEL  # 静态图模型存放路径
--dtype 可选择导出精度
--append_attn 仅sm>=80的机器支持
--block_attn 支持sm<80的机器导出 如果append_attn无法推理可直接替换成 block_attn
```
高性能部署需要先将动态图模型，导出为静态图推理格式，针对A100，V100机器的导出命令如下：

a100
```shell
docker run  -i --rm  --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v /PATH_TO_MODEL/:/models -dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v1.0 /bin/bash \
-c -ex 'cd /opt/source/PaddleNLP &&export PYTHONPATH=$PWD:$PYTHONPATH && cd llm && python3 predict/export_model.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --output_path /models --dtype bfloat16 --inference_model 1 --append_attn 1'\
&& docker logs -f $(docker ps -lq)
```

v100
```shell
docker run  -i --rm  --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v /PATH_TO_MODEL/:/models -dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda118-cudnn8-v1.0 /bin/bash \
-c -ex 'cd /opt/source/PaddleNLP/llm &&export PYTHONPATH=$PWD:$PYTHONPATH&& python3 predict/export_model.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --output_path /models --dtype float16 --inference_model 1 --block_attn'\
&& docker logs -f $(docker ps -lq)
```

### 服务化推理
a100
```shell
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v /PATH_TO_MODEL/:/models -dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v1.0 /bin/bash \
-c -ex 'cd /opt/output/Serving && export MAX_SEQ_LEN=4096 &&bash start_server.sh && tail -f /dev/null'\
&& docker exec -it $(docker ps -lq) sh -c "while [ ! -f /opt/output/Serving/log/workerlog.0 ]; do sleep 1; done; tail -f /opt/output/Serving/log/workerlog.0"
```

v100
```shell
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v /PATH_TO_MODEL/:/models -dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda118-cudnn8-v1.0 /bin/bash \
-c -ex 'cd /opt/output/Serving && export MAX_SEQ_LEN=4096 &&export DTYPE="float16" && bash start_server.sh && tail -f /dev/null'\
&& docker exec -it $(docker ps -lq) sh -c "while [ ! -f /opt/output/Serving/log/workerlog.0 ]; do sleep 1; done; tail -f /opt/output/Serving/log/workerlog.0"
```

## 服务化测试

```shell
观察到如下日志后可以
analysis_predictor.cc:1158] ======= pir optimization completed =======
可以ctrl+c 进行服务化测试
```

```shell
curl 127.0.0.1:9965/v1/chat/completions \
  -H'Content-Type: application/json' \
  -d'{"text": "hello, llm"}'
```
## 镜像

|cuda版本| 支持硬件架构|镜像地址|支持的典型设备|
|:------|:-:|:-:|:-:|
| cuda11.8 | 70 75 80 86 |ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda118-cudnn8-v1.0 |V100，T4，A100，A30，A10 |
| cuda12.4 | 80 86 89 90 |ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v1.0 |A100，A30，A10
L20，H20，H100 |