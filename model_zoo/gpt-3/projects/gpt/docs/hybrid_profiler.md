# Profiler

本文档主要包括在 GPT 中开启 Profiler 并分析调试分析结果的方法，在模型开发中使用 Profiler 分析工具的方法请参考[教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/performance_improving/profiling_model.html)和[API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/profiler/Profiler_cn.html)。

## 参数配置

使用 Profiler 功能需要在任务配置文件中添加 Profiler 配置信息并确保字段为 `enable: True` 以开启分析器。

完整的可配置参数如下所示，可以根据使用场景调整配置。

```
Profiler:
  enable: True
  scheduler: [1, 5]
  profiler_log: log_path
  detailed: True
  record_shapes: True
  profile_memory: True
  summary:
    overview: True
    device: True
    model: True
    dist: True
    kernel: True
    op: True
    mem: True
    memcpy: True
```

其中参数说明：

| **参数名**                      | **参数释义**               |  **默认值** |
|------------------------------|------------------------|------------------------|
|  enable |   是否开启 Profiler | False |
|  scheduler  | 定义分析区间，如 [1, 5] 记录 step 1 到 step 4 的分析数据 | None |
|  profiler_log  | 日志文件目录 |   profiler_log |
|  detailed  | 是否显示详细信息 |   False |
|  record_shapes  |   是否记录 tensor shape 相关信息 | True |
|  profile_memory |   是否统计 memory 相关信息 | True |

其中，当 detailed=True 时会打印所有 summary 表格数据，当 detailed=False 时用户可以根据以下说明定制需要展示的表格信息。

| **参数名**                      | **参数释义**               |  **默认值** |
|------------------------------|------------------------|------------------------|
|  summary.overview | 显示每种类型的 Event 时间消耗 |  True |
|  summary.device | 显示 CPU 和 GPU 的平均利用率信息 |  False |
|  summary.model  | 显示模型 dataloader、forward、backward、optimization 时间消耗 |  True |
|  summary.dist  | 显示计算、通信以及重叠时间 |  False |
|  summary.kernel  | 显示 GPU 执行的 kernel 信息 |  True |
|  summary.op  | 显示框架中算子 (op) 的执行信息 |  True |
|  summary.mem  | 显示内存/显存占用统计信息 |  False |
|  summary.memcpy  | 显示框架中调用内存操作所花费的时间 | False |

## 运行分析

本节以 gpt混合并行 为例，首先进入目录，

```
cd PaddleNLP/model_zoo/gpt-3
```


修改`ppfleetx/configs/nlp/gpt/pretrain_gpt_base.yaml` 中 Profiler.enable 为 True, 同时可以根据上节说明调整相关配置，或者使用命令行参数覆盖，例如可以使用以下命令运行程序，
```
python -m paddle.distributed.launch \
    ./tools/train.py -c \
    ./ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_dp8.yaml -o Profiler.enable=True

```

> 在使用 Profiler 工具进行性能分析时，建议减少 train 的步数，获得分析数据即可停止训练。

## 结果分析

在训练结束后会有以下数据：

* 根据配置信息在控制台打印 summary 表格
* 在配置的 `profiler_log` 目录保存 profiler json 文件

这里保存的 json 文件可以通过如下两种方式查看：

* 在 chrome 浏览器中打开 chrome://tracing/，然后打开 json 文件查看
* 根据控制台信息安装并启动 `visualdl --logdir log_path` 然后根据提示在浏览器中**性能分析**模块查看

具体的信息含义解释以及分析方法请参考[文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/performance_improving/profiling_model.html)。

> 在使用 visualdl 时，如果 log 文件数据较大，启动会比较耗时，请耐心等待。

## 附录

控制台打印的 summary 信息示例如下所示。

**Overview Summary**
```
---------------------------------------------Overview Summary---------------------------------------------
Time unit: ms
-------------------------  -------------------------  -------------------------  -------------------------
Event Type                 Calls                      CPU Time                   Ratio (%)
-------------------------  -------------------------  -------------------------  -------------------------
ProfileStep                4                          18591.04                   100.00
  CudaRuntime              87527                      8555.11                    46.02
  Operator                 21912                      1883.11                    10.13
  UserDefined              13116                      1841.33                    9.90
  OperatorInner            33668                      1018.39                    5.48
  Forward                  8                          731.46                     3.93
  Backward                 4                          671.82                     3.61
  Optimization             4                          315.91                     1.70
  Dataloader               4                          1.37                       0.01
-------------------------  -------------------------  -------------------------  -------------------------
                           Calls                      GPU Time                   Ratio (%)
-------------------------  -------------------------  -------------------------  -------------------------
  Kernel                   16092                      4924.90                    26.49
  Memcpy                   4278                       3617.26                    19.46
  Memset                   780                        2.31                       0.01
  Communication            192                        2363.13                    12.71
-------------------------  -------------------------  -------------------------  -------------------------
```

**Model Summary**

```
-----------------------------------------------------Model Summary-----------------------------------------------------
Time unit: ms
---------------  ------  -----------------------------------------------  ---------------------------------------------
Name             Calls   CPU Total / Avg / Max / Min / Ratio(%)           GPU Total / Avg / Max / Min / Ratio(%)
---------------  ------  -----------------------------------------------  ---------------------------------------------
ProfileStep      4       18591.04 / 4647.76 / 14114.47 / 757.27 / 100.00  4924.90 / 1231.22 / 2853.61 / 682.04 / 100.00
  Dataloader     4       1.37 / 0.34 / 0.85 / 0.16 / 0.01                 0.00 / 0.00 / 0.00 / 0.00 / 0.00
  Forward        8       731.46 / 91.43 / 133.28 / 49.03 / 3.93           714.83 / 89.35 / 174.91 / 4.72 / 14.51
  Backward       4       671.82 / 167.96 / 168.29 / 167.52 / 3.61         1701.53 / 425.38 / 426.97 / 424.10 / 34.55
  Optimization   4       315.91 / 78.98 / 89.07 / 73.78 / 1.70            108.27 / 27.07 / 27.09 / 27.06 / 2.20
  Others         -       16870.48 / - / - / - / 90.75                     2400.27 / - / - / - / 48.74
---------------  ------  -----------------------------------------------  ---------------------------------------------
```

**Operator Summary**

```
----------------------------------------------------------------Operator Summary-----------------------------------------------------------------
Time unit: ms
----------------------------------------------------  ------  -----------------------------------------  ----------------------------------------
Name                                                  Calls   CPU Total / Avg / Max / Min / Ratio(%)     GPU Total / Avg / Max / Min / Ratio(%)
----------------------------------------------------  ------  -----------------------------------------  ----------------------------------------
-----------------------------------------------------------Thread: All threads merged------------------------------------------------------------
GradNodePyLayer_RecomputeFunction_backward            96      663.37 / 6.91 / 17.17 / 4.01 / 18.56       1629.87 / 16.98 / 17.41 / 16.69 / 26.98
  TransformerDecoderLayer                             96      262.68 / 2.74 / 5.91 / 1.90 / 39.60        661.18 / 6.89 / 7.11 / 6.73 / 40.57
  backward                                            96      318.62 / 3.32 / 10.57 / 1.31 / 48.03       968.69 / 10.09 / 10.31 / 9.91 / 59.43
matmul dygraph                                        2312    200.13 / 0.09 / 1.61 / 0.04 / 5.60         1487.76 / 0.64 / 9.81 / 0.22 / 24.63
  matmul infer_meta                                   964     1.42 / 0.00 / 0.01 / 0.00 / 0.71           0.00 / 0.00 / 0.00 / 0.00 / 0.00
  matmul compute                                      964     71.38 / 0.07 / 1.59 / 0.03 / 35.67         644.02 / 0.67 / 9.81 / 0.22 / 43.29
    MEMSET                                            192     - / - / - / - / -                          0.42 / 0.00 / 0.00 / 0.00 / 0.07
    volta_fp16_s884gemm_fp16_128x128_ldg8_f2f_nn      384     - / - / - / - / -                          199.35 / 0.52 / 0.83 / 0.22 / 30.95
    volta_fp16_s884gemm_fp16_256x128_ldg8_f2f_nn      384     - / - / - / - / -                          263.96 / 0.69 / 0.79 / 0.59 / 40.99
    volta_h884gemm_64x128_ldg8_nn                     192     - / - / - / - / -                          141.13 / 0.74 / 0.92 / 0.61 / 21.91
    void cutlass::Kernel<cutlass_70_tensorop_f16_...  4       - / - / - / - / -                          39.15 / 9.79 / 9.81 / 9.78 / 6.08
  matmul node_creation                                676     2.05 / 0.00 / 0.03 / 0.00 / 1.02           0.00 / 0.00 / 0.00 / 0.00 / 0.00
...
```

**Kernel Summary**
```
---------------------------------------------------------------Kernel Summary---------------------------------------------------------------
Time unit: ms
------------------------------------------------------------------------------------------  ------  ----------------------------------------
Name                                                                                        Calls   GPU Total / Avg / Max / Min / Ratio(%)
------------------------------------------------------------------------------------------  ------  ----------------------------------------
ncclKernel_AllReduce_RING_LL_Sum_half(ncclWorkElem)                                         96      2360.57 / 24.59 / 2202.54 / 0.46 / 47.93
volta_fp16_s884gemm_fp16_256x128_ldg8_f2f_nn                                                384     263.96 / 0.69 / 0.79 / 0.59 / 5.36
volta_fp16_s884gemm_fp16_128x128_ldg8_f2f_stages_32x1_tn                                    384     241.74 / 0.63 / 0.84 / 0.22 / 4.91
void paddle::operators::VectorizedRandomGenerator<phi::dtype::float16, unsigned char>       580     209.08 / 0.36 / 0.97 / 0.06 / 4.25
volta_h884gemm_64x128_ldg8_nn                                                               288     203.89 / 0.71 / 0.92 / 0.57 / 4.14
volta_fp16_s884gemm_fp16_128x128_ldg8_f2f_nn                                                384     199.35 / 0.52 / 0.83 / 0.22 / 4.05
volta_h884gemm_256x64_ldg8_tn                                                               288     149.52 / 0.52 / 0.54 / 0.45 / 3.04
void phi::funcs::VectorizedBroadcastKernel<phi::dtype::float16, phi::dtype::float16, ph...  1352    123.12 / 0.09 / 0.40 / 0.05 / 2.50
void paddle::operators::SoftmaxMaskFuseUpperTriangleGPUKernel<phi::dtype::float16, 10>      192     122.37 / 0.64 / 0.66 / 0.60 / 2.48
void cutlass::Kernel<cutlass_70_tensorop_f16_s884gemm_f16_256x128_nt_align8>                100     103.07 / 1.03 / 8.08 / 0.73 / 2.09
void phi::funcs::VectorizedElementwiseKernel<phi::dtype::float16, paddle::operators::Cu...  292     90.80 / 0.31 / 0.83 / 0.06 / 1.84
volta_h884gemm_64x128_ldg8_nt                                                               192     79.76 / 0.42 / 0.43 / 0.40 / 1.62
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eige...  576     75.36 / 0.13 / 0.20 / 0.07 / 1.53
...
```
