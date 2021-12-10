# FasterGeneration Performance

以下性能数据为非加速版generate方法和FasterGeneration对比数据。

- **测试设备:** Tesla V100-SXM2-16GB
- **Batch Size:** 4
- **Max Length:** 32

## 性能数据
***

CUDA 10.1, cudnn 7, gcc 82

torch version 1.10.0+cu102, transformers version 4.12.5

**BART:**

| Model Size | Decode Strategy| FasterGeneration(FP32)<br>(ms) | FasterGeneration(FP16)<br>(ms) | HF generate<br>(ms) | Speed Up Rate<br>(Faster32/HF) | Speed Up Rate<br>(Faster16/HF) |
|-----|----|---|---|---|---|---|
|num_layers = 6<br>num_attention_heads = 12<br>hidden_size = 768<br>(bart-base)|top_k = 1|37.53|34.01|136.89|3.65|4.02
| |top_k = 4    |39.33|34.98|146.89|3.73|4.2 |
| |top_k = 8    |42.35|34.77|136.80|3.23|3.93|
| |top_k = 16   |40.95|35.45|148.45|3.63|4.19|
| |top_p = 0.4  |45.83|33.32|184.36|4.02|5.53|
| |num_beams = 4|44.72|37.51|242.73|5.43|6.47|
| |num_beams = 8|61.56|40.27|273.93|4.45|6.8 |
| |num_beams = 16|82.05|46.68|433.51|5.28|9.29|
|num_layers = 12<br>num_attention_heads = 16<br>hidden_size = 1024<br>(bart-large)|top_k = 1|55.03|45.44|199.27|3.62|4.39|
| |top_k = 4|70.12|56.81|220.96|3.15|3.89|
| |top_k = 8|69.96|57.73|201.06|2.87|3.48|
| |top_k = 16|69.16|59.62|223.73|3.23|3.75|
| |top_p = 0.4|73.49|61.43|275.86|3.75|4.49|
| |num_beams = 4|66.44|50.71|277.61|4.18|5.47|
| |num_beams = 8|135.30|85.75|314.78|2.33|3.67|
| |num_beams = 16|168.01|100.22|441.95|2.63|4.41|

**GPT:**

| Model Size | Decode Strategy| FasterGeneration(FP32)<br>(ms) | FasterGeneration(FP16)<br>(ms) | HF generate<br>(ms) | Speed Up Rate<br>(Faster32/HF) | Speed Up Rate<br>(Faster16/HF) |
|-----|----|---|---|---|---|---|
|num_layers = 12<br>num_attention_heads = 12<br>hidden_size = 768<br>(gpt2)|top_k = 1|69.29|59.20|363.93|5.25|6.15|
| |top_k = 4|68.07|60.92|391.02|5.74|6.42|
| |top_k = 8|69.16|60.45|401.18|5.80|6.64|
| |top_k = 16|73.59|62.40|401.55|5.46|6.44|
| |top_p = 0.4|95.61|76.26|429.63|4.49|5.63|
|num_layers = 24<br>num_attention_heads = 16<br>hidden_size = 1024<br>(gpt2-medium)|top_k = 1|127.04|95.13|726.83|5.72|7.64|
| |top_k = 4|126.74|93.95|694.53|5.48|7.39|
| |top_k = 8|128.11|94.07|743.63|5.80|7.91|
| |top_k = 16|126.78|95.00|732.96|5.78|7.72|
| |top_p = 0.4|143.36|105.40|756.12|5.27|7.17|
|num_layers = 36<br>num_attention_heads = 20<br>hidden_size = 1280<br>(gpt2-large)top_k = 1|236.80|200.37|1057.94|4.47|5.28|
| |top_k = 4|236.69|201.95|1075.17|4.54|5.32|
| |top_k = 8|237.04|202.00|1084.60|4.58|5.37|
| |top_k = 16|235.01|201.79|1110.75|4.73|5.5|
| |top_p = 0.4|270.31|205.84|1111.16|4.11|5.4|

***

CUDA 11.2, cudnn 8, gcc 82

torch version 1.10.0+cu113, transformers version 4.12.5

**BART:**

| Model Size | Decode Strategy| FasterGeneration(FP32)<br>(ms) | FasterGeneration(FP16)<br>(ms) | HF generate<br>(ms) | Speed Up Rate<br>(Faster32/HF) | Speed Up Rate<br>(Faster16/HF) |
|-----|----|---|---|---|---|---|
|num_layers = 6<br>num_attention_heads = 12<br>hidden_size = 768<br>(bart-base)|top_k = 1|30.08|27.95|166.90|5.55|5.97
| |top_k = 4    |30.82|30.01|184.58|5.99|6.15 |
| |top_k = 8    |32.06|31.05|183.44|5.72|5.91|
| |top_k = 16   |32.66|32.35|187.14|5.73|5.78|
| |top_p = 0.4  |37.99|30.25|208.33|5.48|6.89|
| |num_beams = 4|45.99|37.51|285.01|5.43|7.6|
| |num_beams = 8|50.12|37.82|316.56|6.32|8.37|
| |num_beams = 16|67.66|40.98|467.76|6.91|11.41|
|num_layers = 12<br>num_attention_heads = 16<br>hidden_size = 1024<br>(bart-large)|top_k = 1|50.23|39.08|222.59|4.43|5.7|
| |top_k = 4|60.59|48.32|307.76|5.08|6.37|
| |top_k = 8|59.67|49.65|310.49|5.20|6.25|
| |top_k = 16|59.15|52.68|333.75|5.64|6.34|
| |top_p = 0.4|61.36|50.83|340.74|5.55|6.7|
| |num_beams = 4|65.60|53.24|336.28|5.12|6.32|
| |num_beams = 8|76.20|54.13|396.62|5.20|7.33|
| |num_beams = 16|102.04|61.11|531.92|5.21|8.7|

**GPT:**

| Model Size | Decode Strategy| FasterGeneration(FP32)<br>(ms) | FasterGeneration(FP16)<br>(ms) | HF generate<br>(ms) | Speed Up Rate<br>(Faster32/HF) | Speed Up Rate<br>(Faster16/HF) |
|-----|----|---|---|---|---|---|
|num_layers = 12<br>num_attention_heads = 12<br>hidden_size = 768<br>(gpt2)|top_k = 1|49.75|40.15|483.02|9.71|12.03|
| |top_k = 4|49.70|41.69|496.63|9.99|11.91|
| |top_k = 8|51.81|40.81|485.77|9.38|11.9|
| |top_k = 16|50.36|42.88|488.38|9.70|11.39|
| |top_p = 0.4|68.30|53.58|544.53|7.97|10.16|
|num_layers = 24<br>num_attention_heads = 16<br>hidden_size = 1024<br>(gpt2-medium)|top_k = 1|109.86|76.88|936.02|8.52|12.18|
| |top_k = 4|109.69|78.70|943.71|8.60|11.99|
| |top_k = 8|109.70|78.39|963.73|8.79|12.29|
| |top_k = 16|111.18|79.05|945.27|8.50|11.96|
| |top_p = 0.4|127.54|89.76|999.28|7.83|11.13|
|num_layers = 36<br>num_attention_heads = 20<br>hidden_size = 1280<br>(gpt2-large)|top_k = 1|205.92|142.85|1368.78|6.65|9.58|
| |top_k = 4|205.43|140.40|1374.83|6.69|9.79|
| |top_k = 8|205.62|139.47|1406.42|6.84|10.08|
| |top_k = 16|205.16|139.77|1392.37|6.79|9.96|
| |top_p = 0.4|221.06|152.35|1452.07|6.57|9.53|


## 测试方法

运行如下命令即可bart性能测试：

```sh
bash run_perf_bart.sh
```

运行如下命令即可启动gpt性能测试：

```sh
bash run_perf_gpt.sh
```

运行以上命令后，脚本会自动使用不同的模型参数进行性能测试，结果如下图所示：

```sh
...
[2021-12-10 08:11:37,255] [   DEBUG] - skipping 'FasterTransformer' extension (up-to-date) build
Namespace(decode_strategy='sampling', max_length=32, model_name_or_path='bart-base', num_beams=1, top_k=1, top_p=1.0, use_fp16_decoding=False)
Faster FP32 cost: 40.13654176145792
PD cost: 511.413540635258
HF cost: 138.49875444546342
Speed up Faster FP32/PD: 12.741843671403577
Speed up Faster FP32/HF: 3.4506897796177394
...
...
[2021-12-10 08:13:42,858] [   DEBUG] - skipping 'FasterTransformer' extension (up-to-date) build
Namespace(decode_strategy='sampling', max_length=32, model_name_or_path='bart-base', num_beams=1, top_k=1, top_p=1.0, use_fp16_decoding=True)
Faster FP16 cost: 34.004870522767305
...
```
可以看到，对于每组参数，脚本会先输出FP32和竞品的测试对比，再单独输出FP16的性能数据。

**NOTE:** 根据测试环境和机器状态的不同，以上性能测试脚本的结果可能与表中结果有所出入。
