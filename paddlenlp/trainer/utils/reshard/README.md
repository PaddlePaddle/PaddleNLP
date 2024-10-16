# 模型热启自适应reshard功能

## 背景
大模型多卡分布式的训练保存的Checkpoint通常是根据张量并行、FSDP、流水线并行进行切分保存的。
在恢复模型训练时，可能会面临的分布式策略改变、训练节点数发生变化的情况。比如探索不同的混合并行策略，或切换训练任务（pretrain -> sft、rlhf）。在这种情况下，用户往往需要离线手动对Checkpoint进行处理，比较繁琐。
为了解决上述的问题，解放生产力，我们开发了模型热启自适应reshard功能。

## 功能

模型热启自适应 reshard 功能支持：
    在同一个模型在其他配置不变的情况下, 任意切换4D分布式并行策略（MP、PP、SHARDING、DP）后，依然可以自适应分布式并行策略，恢复原来的模型状态量。这里的模型状态量不仅包括权重本身，还包括learning rate scheduler 的状态量以及优化器的状态量。

## 使用方法

    在训练时，通过设置--load_sharded_model true --save_sharded_model true 开启reshard功能。
    load_sharded_model 表示加载的checkpoint 是否是可以自适应reshard的checkpoint，save_sharded_model 表示是否保存成可以自适应reshard的checkpoint。

    如果要支持 pp reshard, 还需要在脚本里，额外注册两个函数：
        regitser_extract_layer_name_func(extract_layer_name_func)
        register_index_layer_func(index_layer_func)
    其中，extract_layer_name_func 是一个函数，用于从参数名提取出层名；index_layer_func 是一个函数，用于从层名提取出层的 id (标识第几层)。

    该功能的具体使用方法可以参考 https://github.com/PaddlePaddle/PaddleNLP/pull/7629 中的demo。重点关注 llm/llama/run_pretrain.py 和 llm/llama/run_sharding_v2.sh
    这两个脚本。
