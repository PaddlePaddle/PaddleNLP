

### `Galvatron`使用文档

### 使用`Galvatron`进行性能分析

使用 `Galvatron `的第一步是对硬件环境和模型计算时间进行性能分析。`Galvatron` 会自动将分析结果保存到配置文件中。

#### 分析硬件环境性能

（1）首先，`cd ./llm/auto_parallel/galvatron-llama-submit`，对于`scripts/profile_hardware.sh`，根据实际环境修改`launch`环境变量与`PROFILE_HARDWARE_ARGS`配置。

（2）随后，`bash scripts/profile_hardware.sh `运行，将生成五个脚本，分别为`scripts/profile_allreduce.sh`，`scripts/profile_p2p.sh`，`scripts/profile_allreduce_sp.sh`，`scripts/profile_all2all.sh`，`scripts/profile_overlap.sh`。

（3）随后依次运行上述脚本，将在`configs`下生成对应的`json`文件

#### 分析模型性能

（1）首先，`cd ./llm/auto_parallel/galvatron-llama-submit`，对于`scripts/profile_memory.sh`与`scripts/profile_computaion.sh`，修改`MODEL_ARGS`，以替换为实际的模型配置与序列长度

（2）随后，选择并修改`scripts/profile_memory.sh`的`MODEL_PROFILER_ARGS`。其共有两种`profile_mode`，分别为`static`与`sequence`。对于长序列，我们选择后者，并适当修改``profile_max_seq_length`。对于短序列，将`profile_fixed_seq_length_list`设定为确定的序列长度即可。最后，运行`bash scripts/profile_memory.sh`，将在`configs`中生成显存配置

（3）进而，选择并修改`scripts/profile_computation.sh`的`MODEL_PROFILER_ARGS`。其共有两种`profile_mode`，分别为`batch`与`sequence`。对于长序列，我们选择后者。对于短序列，将`profile_fixed_seq_length_list`设定为确定的序列长度即可。最后，运行`bash scripts/profile_computation.sh`，将在`configs`中生成时间配置

#### 策略搜索

给定集群和内存预算，`Galvatron` 搜索引擎将自动生成最优并行策略。优化后的并行策略将以 `JSON `文件形式保存在 `configs` 中用于训练。

- 在`search_dist.sh`中，修改`ProfileDataParserArgs`
  - 其中，`time_profile_mode`与`memory_profile_mode`与分析模型性能时保持一致
  - `hidden_size_list`、`layer_num_list`、`seqlen_list`修改为实际的配置
  - 其余所有config的path修改为实际的path
- 在`search_dist.sh`中，修改`SearchEngineArgs`
  - `max_tp_size`与`max_pp_size`分别表示期望的`tp_size`与`pp_size`的上限
  - `memory_upper_limit`表示单卡的显存上限，单位为`GB`
- 最后，运行`bash scripts/search_dist.sh`，将在`configs`中生成搜索出的最优策略，最终运行`bash  train_qwen_fine_grained.sh`即可根据最优策略运行。



### `Galvatron`安装注意事项

在paddlenlp的根目录下`pip install -e .`后，进入`cd ./paddlenlp/experimental/galvatron/search_engine`目录，运行

```
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) dp_core.cpp -o galvatron_dp_core$(python3-config --extension-suffix)
```

，将在本目录下生成`galvatron_dp_core.cpython-39-x86_64-linux-gnu.so`

