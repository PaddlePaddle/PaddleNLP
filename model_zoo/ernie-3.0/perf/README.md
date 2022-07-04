# 性能测试

本目录提供了 GPU 和 CPU 下性能测试的脚本。

脚本默认不使用 FasterTokenizer 对推理加速，可通过传入参数`--faster_tokenizer`来使用 FasterTokenizer 加速推理，当使用 FasterTokenizer 时，需要先安装 faster_tokenizer 包，并通过设置环境变量 `OMP_NUM_THREADS` 来设置切词的线程数，例如：

```shell
pip install faster_tokenizer
export OMP_NUM_THREADS=4
```

性能测试需要静态图模型 ${MODEL_PATH}，包含 `*.pdmodel` 和 `*.pdiparams` 文件。动态图转静态图可参考[预训练模型导出脚本](../../bert/export_model.py)，传入动态图模型的路径 ${DYGRAPH_PATH}，以及导出的静态图路径 ${MODEL_PATH}。

```shell
python export_model.py \
    --model_path ${DYGRAPH_PATH} \
    --output_path ${MODEL_PATH} \
```

这里以 CLUE 中的 IFLYTEK 数据集为例，因为 IFLYTEK 是长文本数据集（平均长度是 289.17），能够适应性能测试（最大序列长度为 128）的配置。

## GPU

### 环境依赖

GPU 下的推理可以使用 TensorRT 加速，可以安装[包含 TensorRT 预测库的 PaddlePaddle](https://www.paddlepaddle.org.cn/inference/user_guides/download_lib.html)。需要注意的是，量化模型的预测必须安装包含 TensorRT 预测库的 PaddlePaddle。

### 启动测试

推理精度默认 PF32，针对 FP16 或者 INT8 的推理需要传入对应参数 `--fp16` 或者 `--int8`。还可以传入 `batch_size` 等参数。如：

```shell
MODEL_NAME_OR_PATH=ernie-3.0-medium-zh
bs=32
# Step1: collect shape info
python clue_infer.py \
    --perf \
    --use_trt \
    --device gpu \
    --use_inference \
    --collect_shape \
    --batch_size $bs \
    --faster_tokenizer \
    --model_path ${MODEL_PATH} \
    --model_name_or_path ${MODEL_NAME_OR_PATH}

# Step2(Option1): precision FP32
python clue_infer.py \
    --device gpu \
    --use_inference \
    --use_trt  \
    --batch_size $bs \
    --faster_tokenizer \
    --model_path ${MODEL_PATH} \
    --model_name_or_path ${MODEL_NAME_OR_PATH}

# Step2(Option2):precision FP16
python clue_infer.py \
    --perf \
    --device gpu  \
    --use_trt \
    --fp16  \
    --batch_size $bs \
    --use_inference \
    --faster_tokenizer \
    --model_path ${MODEL_PATH} \
    --model_name_or_path ${MODEL_NAME_OR_PATH}
```

## CPU

### 环境依赖

CPU 预测依赖 onnxruntime、paddle2onnx

```shell
python -m pip install onnxruntime paddle2onnx
```

### 启动测试

CPU 下的性能测试需要指定参数线程数`--num_threads`、`batch_size`等参数。

```shell
MODEL_NAME_OR_PATH=ernie-3.0-medium-zh
bs=32
# Option1: thread num: 16
python clue_infer.py \
    --device cpu \
    --num_threads 1 \
    --batch_size $bs \
    --faster_tokenizer \
    --model_path ${MODEL_PATH} \
    --model_name_or_path ${MODEL_NAME_OR_PATH}

# Option2: thread num: 8
python clue_infer.py \
    --device cpu \
    --num_threads 8 \
    --batch_size $bs \
    --faster_tokenizer \
    --model_path ${MODEL_PATH} \
    --model_name_or_path ${MODEL_NAME_OR_PATH}
```
