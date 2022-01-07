# PaddleNLP Benchmark 测试脚本

目前我们为用户提供了GPT模型的Benchmark性能测试脚本。

启动测试脚本的方法如下：

```script
ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7"
docker pull ${ImageName}

run_cmd="set -xe;
        cd /workspace ;
        bash -x tests/benchmark/run_all.sh static"

nvidia-docker run --name test_paddle_gpt -i  \
    --net=host \
    --shm-size=1g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"
```
