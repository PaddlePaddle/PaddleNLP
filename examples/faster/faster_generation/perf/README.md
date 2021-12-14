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
|num_layers = 6<br>num_attention_heads = 12<br>hidden_size = 768<br>(bart-base)|top_k = 1|31.1|27.4|139.46|4.48|5.09
| |top_k = 4    |32.13|29.06|149.81|4.66|5.16|
| |top_k = 8    |31.7|28.36|154.3|4.87|5.44|
| |top_k = 16   |32.93|28.66|145.85|4.43|5.09|
| |top_p = 0.4  |33.35|29.01|173.18|5.19|5.97|
| |num_beams = 4|47.55|38.02|252.71|5.31|6.65|
| |num_beams = 8|52.19|41.39|282.3|5.41|6.82|
| |num_beams = 16|67.18|45.82|441.59|6.57|9.64|
|num_layers = 12<br>num_attention_heads = 16<br>hidden_size = 1024<br>(bart-large)|top_k = 1|45.8|37.43|173.08|3.78|4.62|
| |top_k = 4|51.11|48.28|246.27|4.82|5.1|
| |top_k = 8|61.61|50.67|246.19|4.0|4.86|
| |top_k = 16|63.81|48.33|272.93|4.28|5.65|
| |top_p = 0.4|63.0|50.05|288.76|4.58|5.77|
| |num_beams = 4|65.54|48.58|273.84|4.18|5.64|
| |num_beams = 8|75.68|52.59|340.86|4.5|6.48|
| |num_beams = 16|102.87|62.25|477.97|4.65|7.68|

**GPT:**

| Model Size | Decode Strategy| FasterGeneration(FP32)<br>(ms) | FasterGeneration(FP16)<br>(ms) | HF generate<br>(ms) | Speed Up Rate<br>(Faster32/HF) | Speed Up Rate<br>(Faster16/HF) |
|-----|----|---|---|---|---|---|
|num_layers = 12<br>num_attention_heads = 12<br>hidden_size = 768<br>(gpt2)|top_k = 1|50.84|40.37|399.58|7.86|9.9|
| |top_k = 4|50.38|38.81|419.55|8.33|10.81|
| |top_k = 8|51.23|36.78|411.7|8.04|11.19|
| |top_k = 16|51.03|38.76|408.36|8.0|10.54|
| |top_p = 0.4|68.55|48.04|489.45|7.14|10.19|
|num_layers = 24<br>num_attention_heads = 16<br>hidden_size = 1024<br>(gpt2-medium)|top_k = 1|111.37|79.73|753.11|6.76|9.45|
| |top_k = 4|110.53|80.48|767.48|6.94|9.54|
| |top_k = 8|109.87|78.92|754.99|6.87|9.57|
| |top_k = 16|110.61|85.26|764.16|6.91|8.96|
| |top_p = 0.4|127.51|87.72|830.24|6.51|9.46|
|num_layers = 36<br>num_attention_heads = 20<br>hidden_size = 1280<br>(gpt2-large)|top_k = 1|203.76|142.85|1108.26|5.44|7.76|
| |top_k = 4|204.18|139.49|1230.63|6.03|8.82|
| |top_k = 8|204.22|139.14|1238.96|6.07|8.9|
| |top_k = 16|204.11|140.04|1148.05|5.62|8.2|
| |top_p = 0.4|222.12|150.68|1248.75|5.62|8.29|


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
