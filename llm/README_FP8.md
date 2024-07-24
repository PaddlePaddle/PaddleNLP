# llama2 FP8量化推理

## 环境准备

- PaddlePaddle develop

- PaddleNLP  develop

## PTQ 量化

需要借助 PaddleSlim 生成量化后的 act scale

```shell
pip install paddleslim
cd Paddle/llm
wget https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz
tar xvf AdvertiseGen.tar.gz
```

运行量化脚本，此时会自动下载`meta-llama/Llama-2-13b-chat`模型，默认路径一般在`/root/.paddlenlp/models/meta-llama/Llama-2-13b-chat/` ，量化后的模型文件在`./checkpoints/llama_ptq_ckpts`中，其中`act_scales.json`为我们所需要的 act scale

```shell
python  run_finetune.py ./config/llama/ptq_argument.json
```

 将`./checkpoints/llama_ptq_ckpts/act_scales.json`拷贝到原始模型文件夹`/root/.paddlenlp/models/meta-llama/Llama-2-13b-chat/`中

## FP8量化

运行 model_convert_fp8.sh，将模型权重量化为 fp8数据类型

```shell
sh ./predict/model_convert_fp8.sh /root/.paddlenlp/models/meta-llama/Llama-2-13b-chat/
```

## 推理

### 动态图推理

```shell
sh ./predict/run_dynamic_infer.sh /root/.paddlenlp/models/meta-llama/Llama-2-13b-chat/
```

### 静态图推理

动转静

```shell
sh ./predict/run_export.sh /root/.paddlenlp/models/meta-llama/Llama-2-13b-chat/
```

静态图推理

```shell
sh ./predict/run_static_infer.sh /root/.paddlenlp/models/meta-llama/Llama-2-13b-chat/
```
