# FasterGeneration Performance

以下性能数据为非加速版generate方法和FasterGeneration对比数据。

- **测试设备:** Tesla V100-SXM2-16GB
- **Batch Size:** 4
- **Max Length:** 32
- **精度:** FP16

表格

## 测试方法

运行如下命令即可bart性能测试：

```sh
python bart_perf.py \
    --model_name_or_path=bart-base \
    --decode_strategy=sampling \
    --num_beams=4 \
    --top_k=16 \
    --top_p=1.0 \
    --max_length=32 \
```

运行如下命令即可启动gpt性能测试：

```sh
python gpt_perf.py \
    --model_name_or_path=gpt2-en \
    --decode_strategy=sampling \
    --top_k=1 \
    --top_p=1.0 \
    --max_length=32 \
```

其中参数释义如下：
- `model_name_or_path` 指定测试使用的模型参数。其中bart可以在`bart-base`和`bart-large`中选择，gpt可以在`gpt2-en`、`gpt2-medium-en`和`gpt2-large-en`中选择。
- `decode_strategy` 表示预测解码时采取的策略，可选"sampling"、"greedy_search"和"beam_search"之一。**注意GPT当前不支持beam_search**
- `top_k` 表示采用"sampling"解码策略时，token的概率按从大到小排序，生成的token只从前`top_k`个中进行采样。
- `top_p` 表示采用"sampling"解码策略时，token的概率按从大到小排序，生成的token只从概率累加到`top_p`的前某几个中进行采样。
- `max_length` 表示预测生成的句子的最大长度。

**NOTE:** 根据测试环境和机器状态的不同，以上性能测试脚本的结果可能与表中结果有所出入。
