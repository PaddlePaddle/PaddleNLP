# DeepseekV3 自动并行训练

## 1. 模型组网介绍

- 动静统一自动并行组网可以在 [modeling_auto.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/deepseek_v3/modeling_auto.py) 中找到。当前，主要支持预训练，包括动态图和动转静训练，未来计划扩展以支持 SFT 等流程。

## 2. 预训练准备

1. **安装 Paddle**：请安装最新的 Paddle，建议使用 nightly 版本。您可以访问 [Paddle 官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html) 进行安装。

2. **下载和准备数据**：
   下载预处理好的数据，并将其解压到 `./data` 目录下：
   ```shell
   # llama 模型数据下载
   wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
   wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx

   mkdir data
   mv llama_openwebtext_100k.bin ./data
   mv llama_openwebtext_100k.idx ./data
   ```

3. **安装自定义算子**：
   ```shell
   # 编译自定义算子，可选
   cd ../../../slm/model_zoo/gpt-3/external_ops/
   python3 setup.py install
   cd -
   ```

## 3. 预训练

当前 deepseek-v3 完整支持张量并行、流水线并行、数据并行、专家并行等并行策略，但尚未支持 DeepEp、DualPipe 等高级优化策略。完整的 v3 模型参数规模为671B，使用 2048 卡并采用 16 路 PP 和 128 路 DP 的并行策略进行训练。受限于硬件资源，下述脚本中将模型训练规模进行了缩减，以便在单机 8 卡上进行展示。脚本中将模型的隐藏层数由 61 层缩减到 2 层、Moe 专家由 256 个缩减到 16 个，dense 层数由 3 层缩减到 0 层。

- **调整并行策略**：用户可以通过修改 `run_pretrain_auto.sh` 脚本中的 `pipeline_parallel_degree`、`tensor_parallel_degree`、`sharding_parallel_degree` 参数来调整流水线并行、张量并行、专家并行的卡数。

- **调整模型规模**：用户可以通过修改 `run_pretrain_auto.sh` 脚本中的 `num_hidden_layers`、`n_routed_experts`、`first_k_dense_replace` 参数来控制隐藏层数、Moe 层内的 Moe 专家数、前置的 dense 隐藏层数。

- **动态图训练**：参考训练脚本 **run_pretrain_auto.sh**，运行 8 卡 dp8 的并行策略。
  ```shell
  bash run_pretrain_auto.sh
  ```
