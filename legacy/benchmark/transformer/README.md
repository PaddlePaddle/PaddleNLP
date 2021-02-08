# Transformer Benchmark with Fleet API

### 静态图
如果是需要单机多卡训练，则使用下面的命令进行训练：
``` shell
cd static/
export CUDA_VISIBLE_DEVICES=0
python3 train.py
```

### 动态图
如果使用单机单卡进行训练可以使用如下命令：
``` shell
cd dygraph/
export CUDA_VISIBLE_DEVICES=0
python3 train.py
```

如果使用单机多卡进行训练可以使用如下命令：
``` shell
cd dygraph/
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --selected_gpus=0,1,2,3,4,5,6,7 train.py
```
