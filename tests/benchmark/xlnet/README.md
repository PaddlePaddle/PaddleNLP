# XLNet Benchmark 测试脚本

目前我们为用户提供了XLNet模型的Benchmark性能测试脚本。

启动测试脚本的方法如下：

1. 创建docker容器
```script
ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7"
docker pull ${ImageName}
nvidia-docker run -it --name=test_paddle_xlnet --net=host --shm-size=1g -v $PWD:/workspace ${ImageName} /bin/bash
cd /workspace
```

2. 克隆PaddleNLP项目
```script
git clone https://github.com/PaddlePaddle/PaddleNLP.git
```

3. 进入xlnet测试目录，执行脚本
```script
cd PaddleNLP

# 不打开profile选项，执行
bash tests/benchmark/xlnet/run_all.sh

# 打开profile选项，执行
bash tests/benchmark/xlnet/run_all.sh on
```
