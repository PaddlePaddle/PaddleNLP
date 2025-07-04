# 概述
`Galvatron`是一个为`Transformer`模型设计的自动并行系统，可在特定集群环境和模型配置下搜索最优并行策略，以实现高效计算。

# 系统架构
`Galvatron`架构由三个核心模块组成。

- 性能分析器
    - 硬件性能分析器:测量设备间并行策略的关键算子带宽。
    - 模型性能分析器:分析不同模型组件的计算模式和内存需求。
    - 运行时性能分析器:监控模型单次迭代的运行时间和显存占用，其已集成于`AutoTrainer`类中。

- 负载估算器
根据性能分析器的数据，在特定集群和模型配置下，对给定并行策略进行运行时间和显存开销的建模，估算结果与实际运行时间的误差较小（A100集群下测试误差通常在5%以内，如下图所示）。
![](./imgs/cost_model.png)

- 策略搜索引擎
生成所有可行的并行策略，并基于负载估算器的准确性，针对特定批量大小和累积步数，快速高效地求解最优并行策略。

# 使用流程
`./llm/auto_parallel/galvatron-llama`提供了`llama-7b`的完整示例。
- 分析硬件性能
```
    cd ./llm/auto_parallel/galvatron-llama
    bash scripts/profile_hardware.sh
```
- 分析模型性能
执行以下命令生成时间与显存性能配置文件：
```
    bash scripts/profile_computation.sh
    bash scripts/profile_memory.sh
```
- 验证成本模型
在`check_cost_model.sh`中配置性能配置文件路径、并行策略、批量大小和累积步数，执行以下命令预测执行时间和显存开销。为验证建模准确性，可在`train_dist_random.sh`中修改相同配置并执行以获取真实性能数据。

```
    bash scripts/check_cost_model.sh
```

- 搜索最优并行策略
调用以下命令为特定批量大小列表搜索最优并行策略，结果将保存至`configs/optimal_solution.json`：
```
    bash scripts/search_dist.sh
```
