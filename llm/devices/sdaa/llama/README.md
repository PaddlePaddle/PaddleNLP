## 🚣‍♂️ 使用 PaddleNLP 在太初 sdaa 下运行 Llama-2-13b-chat 模型 🚣

PaddleNLP 在太初 sdaa 上对 Llama-2-13b-chat 模型进行了深度适配和优化，实现了 sdaa device 推理入口和 GPU 的基本统一，仅需修改 device 即可完成推理任务的迁移。

## 🚀 快速开始 🚀

### 0. 机器准备。快速开始之前，您需要准备一台插有太初 T100加速卡的机器，要求如下：

| 芯片类型 | 驱动版本 |
| --- | --- |
| 太初 T100 | 1.3.0|


### 1. 环境准备：(这将花费您5～15min 时间)

#### 1.1 拉取镜像
```bash
# 注意此镜像包含预编译的飞桨安装包, TecoDriver, TecoToolKit等，可以一键运行paddlenlp模型
wget http://mirrors.tecorigin.com/repository/teco-3rd-repo/custom_device/ubuntu22.04/x86_64/1.3.0/paddle_sdaa_1.3.0_llm_infer.tar
docker load < paddle_sdaa_1.3.0_llm_infer.tar
```

#### 1.2 参考如下命令启动容器
```bash
docker run -itd --name="paddle-sdaa-dev" --net=host --privileged --cap-add SYS_PTRACE --cap-add SYS_ADMIN --shm-size 128g jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/paddle_sdaa:1.3.0-llm-infer /bin/bash
```

#### 1.3 下载 PaddleNLP 仓库代码，并安装依赖
```bash
# PaddleNLP是基于PaddlePaddle『飞桨』的自然语言处理和大语言模型(LLM)开发库，存放了基于『飞桨』框架实现的各种大模型，Llama-2-13b-chat模型也包含其中。为了便于您更好地使用PaddleNLP，您需要clone整个仓库。
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
export PYTHONPATH=/path/to/PaddleNLP:$PYTHONPATH
pip install -r requirements.txt
cd csrc/sdaa && python setup_sdaa.py install && cd ../../llm/devices/sdaa/llama
```
### 2. 推理：(这将花费您15~30min 时间)
#### 2.1 动态图分布式推理

执行如下命令进行推理：
```bash
bash dynamic_infer_llama_sdaa.sh
```
首次推理会自动下载权重，可以使用自动下载的权重，或者下载之后指定权重路径。成功运行后，可以查看到推理结果的生成。

样例将下载的权重 meta-llama/Llama-2-13b-chat 文件夹保存到/workspace/weights，示例如下：
```
[2024-12-10 15:42:51,992] [    INFO] - set state for layer 30
[2024-12-10 15:42:53,666] [    INFO] - set state for layer 31
[2024-12-10 15:42:55,202] [    INFO] - set state for layer 32
[2024-12-10 15:42:56,724] [    INFO] - set state for layer 33
[2024-12-10 15:42:58,314] [    INFO] - set state for layer 34
[2024-12-10 15:43:00,041] [    INFO] - set state for layer 35
[2024-12-10 15:43:01,515] [    INFO] - set state for layer 36
[2024-12-10 15:43:03,034] [    INFO] - set state for layer 37
[2024-12-10 15:43:04,746] [    INFO] - set state for layer 38
[2024-12-10 15:43:06,390] [    INFO] - set state for layer 39
[2024-12-10 15:43:08,682] [    INFO] - We are using <class 'paddlenlp.transformers.llama.configuration.LlamaConfig'> to load '/workspace/weights/meta-llama/Llama-2-13b-chat'.
[2024-12-10 15:43:08,682] [    INFO] - Loading configuration file /workspace/weights/meta-llama/Llama-2-13b-chat/config.json
[2024-12-10 15:43:08,683] [    INFO] - Loading configuration file /workspace/weights/meta-llama/Llama-2-13b-chat/generation_config.json
[2024-12-10 15:43:08,752] [    INFO] - Start predict
[2024-12-10 15:43:08,789] [    INFO] - We are using <class 'paddlenlp.transformers.llama.tokenizer.LlamaTokenizer'> to load '/workspace/weights/meta-llama/Llama-2-13b-chat'.
[2024-12-10 15:43:08,806] [    INFO] - Start read result message
[2024-12-10 15:43:08,806] [    INFO] - Current path is /workspace/paddlenlp/llm
[2024-12-10 15:43:29,178] [    INFO] - running spend 20.372194528579712
[2024-12-10 15:43:29,187] [    INFO] - Finish read result message
[2024-12-10 15:43:29,192] [    INFO] - End predict
***********Source**********
解释一下温故而知新
***********Target**********

***********Output**********
 "温故而知新" (wēn gù er zhī xīn) is a Chinese idiom that means "to understand the old in order to know the new." It is often used to convey the idea that one must have a deep understanding of the past and traditional ways of doing things in order to truly appreciate and understand new ideas and innovations.

The phrase is often used in the context of education, where students are encouraged to study the classics and learn from the past in order to gain a solid foundation for understanding new concepts and ideas. It is also used in business and technology, where companies may look to the past for inspiration and guidance as they develop new products and services.

In essence, "温故而知新" suggests that one cannot truly understand the new without first understanding the old, and that a deep appreciation for the past is essential for making progress and innovation.
```
#### 2.2 静态图分布式推理

##### 2.2.1 静态图导出

执行如下命令进行静态图导出，为静态图分布式推理做好准备：
```bash
bash static_export_llama_sdaa.sh
```
成功运行后，可以查看到模型导出的结果，样例如下：
```bash
[2024-12-10 15:30:28,991] [    INFO] - set state for layer 24
[2024-12-10 15:30:30,246] [    INFO] - set state for layer 25
[2024-12-10 15:30:31,586] [    INFO] - set state for layer 26
[2024-12-10 15:30:32,892] [    INFO] - set state for layer 27
[2024-12-10 15:30:34,228] [    INFO] - set state for layer 28
[2024-12-10 15:30:35,530] [    INFO] - set state for layer 29
[2024-12-10 15:30:36,925] [    INFO] - set state for layer 30
[2024-12-10 15:30:38,233] [    INFO] - set state for layer 31
[2024-12-10 15:30:39,635] [    INFO] - set state for layer 32
[2024-12-10 15:30:40,992] [    INFO] - set state for layer 33
[2024-12-10 15:30:42,375] [    INFO] - set state for layer 34
[2024-12-10 15:30:43,717] [    INFO] - set state for layer 35
[2024-12-10 15:30:45,076] [    INFO] - set state for layer 36
[2024-12-10 15:30:46,423] [    INFO] - set state for layer 37
[2024-12-10 15:30:47,827] [    INFO] - set state for layer 38
[2024-12-10 15:30:49,216] [    INFO] - set state for layer 39
[2024-12-10 15:30:51,136] [    INFO] - We are using <class 'paddlenlp.transformers.llama.configuration.LlamaConfig'> to load '/workspace/weights/meta-llama/Llama-2-13b-chat'.
[2024-12-10 15:30:51,136] [    INFO] - Loading configuration file /workspace/weights/meta-llama/Llama-2-13b-chat/config.json
[2024-12-10 15:30:51,137] [    INFO] - Loading configuration file /workspace/weights/meta-llama/Llama-2-13b-chat/generation_config.json
/root/miniconda3/envs/paddle_env/lib/python3.10/site-packages/paddle/jit/dy2static/program_translator.py:747: UserWarning: full_graph=False don't support input_spec arguments. It will not produce any effect.
You can set full_graph=True, then you can assign input spec.

  warnings.warn(
/root/miniconda3/envs/paddle_env/lib/python3.10/site-packages/paddle/jit/api.py:1106: UserWarning: What you save is a function, and `jit.save` will generate the name of the model file according to `path` you specify. When loading these files with `jit.load`, you get a `TranslatedLayer` whose inference result is the same as the inference result of the function you saved.
  warnings.warn(
I1210 15:30:58.707722 1174678 program_interpreter.cc:242] New Executor is Running.
[2024-12-10 15:31:10,381] [    INFO] - Configuration saved in ./output_dir/exported_model/llama2_13b_chat_wint8_block_size32/config.json
[2024-12-10 15:31:10,382] [    INFO] - Configuration saved in ./output_dir/exported_model/llama2_13b_chat_wint8_block_size32/generation_config.json
[2024-12-10 15:31:10,382] [    INFO] - tokenizer config file saved in ./output_dir/exported_model/llama2_13b_chat_wint8_block_size32/tokenizer_config.json
[2024-12-10 15:31:10,382] [    INFO] - Special tokens file saved in ./output_dir/exported_model/llama2_13b_chat_wint8_block_size32/special_tokens_map.json
[2024-12-10 15:31:10,383] [    INFO] - Chat-template config file saved in ./output_dir/exported_model/llama2_13b_chat_wint8_block_size32/chat_template.json
LAUNCH INFO 2024-12-10 15:31:12,346 Pod completed
LAUNCH INFO 2024-12-10 15:31:12,347 Exit code 0
```
##### 2.2.2 静态图分布式推理

执行如下命令进行静态图分布式推理：
```bash
bash static_infer_llama_sdaa.sh
```
成功运行后，可以查看到推理结果的生成，样例如下：
```bash
[2024-12-10 15:36:24,150] [    INFO] topology.py:370 - Total 4 data comm group(s) create successfully!
[2024-12-10 15:36:24,150] [    INFO] topology.py:370 - Total 1 model comm group(s) create successfully!
[2024-12-10 15:36:24,150] [    INFO] topology.py:370 - Total 4 sharding comm group(s) create successfully!
[2024-12-10 15:36:24,150] [    INFO] topology.py:290 - HybridParallelInfo: rank_id: 0, mp_degree: 4, sharding_degree: 1, pp_degree: 1, dp_degree: 1, sep_degree: 1, mp_group: [0, 1, 2, 3],  sharding_group: [0], pp_group: [0], dp_group: [0], sep:group: None, check/clip group: [0, 1, 2, 3]
[2024-12-10 15:36:24,152] [    INFO] - We are using <class 'paddlenlp.transformers.llama.tokenizer.LlamaTokenizer'> to load 'output_dir/exported_model/llama2_13b_chat_wint8_block_size32'.
[2024-12-10 15:36:24,164] [    INFO] - We are using <class 'paddlenlp.transformers.llama.configuration.LlamaConfig'> to load 'output_dir/exported_model/llama2_13b_chat_wint8_block_size32'.
[2024-12-10 15:36:24,164] [    INFO] - Loading configuration file output_dir/exported_model/llama2_13b_chat_wint8_block_size32/config.json
[2024-12-10 15:36:24,165] [    INFO] - We are using <class 'paddlenlp.transformers.llama.configuration.LlamaConfig'> to load 'output_dir/exported_model/llama2_13b_chat_wint8_block_size32'.
[2024-12-10 15:36:24,165] [    INFO] - Loading configuration file output_dir/exported_model/llama2_13b_chat_wint8_block_size32/config.json
[2024-12-10 15:36:24,198] [    INFO] - We are using <class 'paddlenlp.transformers.llama.configuration.LlamaConfig'> to load 'output_dir/exported_model/llama2_13b_chat_wint8_block_size32'.
[2024-12-10 15:36:24,198] [    INFO] - Loading configuration file output_dir/exported_model/llama2_13b_chat_wint8_block_size32/config.json
[2024-12-10 15:36:24,199] [    INFO] - Loading configuration file output_dir/exported_model/llama2_13b_chat_wint8_block_size32/generation_config.json
I1210 15:36:24.239424 1334951 analysis_predictor.cc:2142] MKLDNN is enabled
I1210 15:36:24.239473 1334951 analysis_predictor.cc:2167] CustomDevice is enabled
I1210 15:36:24.239486 1334951 analysis_predictor.cc:2210] Model is mixed precision type with float16, we will use a new PassStrategy. Note that only GPU/XPU backend is supported for now.
I1210 15:36:24.239490 1334951 analysis_predictor.cc:2259] Ir optimization is turned off, no ir pass will be executed.
--- Running analysis [ir_graph_build_pass]
I1210 15:36:24.260483 1334951 executor.cc:183] Old Executor is Running.
--- Running analysis [ir_analysis_pass]
--- Running analysis [ir_params_sync_among_devices_pass]
I1210 15:36:25.863914 1334951 ir_params_sync_among_devices_pass.cc:140] Sync params from CPU to sdaa:0
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [save_optimized_model_pass]
--- Running analysis [ir_graph_to_program_pass]
I1210 15:36:29.991195 1334951 analysis_predictor.cc:2348] ======= ir optimization completed =======
I1210 15:36:30.000306 1334951 gen_comm_id_helper.cc:212] Server listening on: 127.0.1.1:36942 successful.
I1210 15:36:30.088883 1334951 task_node.cc:43] Constructing TaskNode for DistModelInf. The TaskNode's id is: 0. And the TaskNode's max_run_time and max_slot_num will be set to 1.
LAUNCH INFO 2024-12-10 15:37:24,254 Pod completed
LAUNCH INFO 2024-12-10 15:37:24,254 Exit code 0
I1210 15:36:30.189157 1334951 server.cpp:1107] Server[paddle::distributed::MessageServiceImpl] is serving on port=36942.
I1210 15:36:30.189195 1334951 server.cpp:1110] Check out http://dmx-19:36942 in web browser.
I1210 15:36:30.189320 1334951 message_bus.cc:201] Message bus's listen port thread starts successful.
[2024-12-10 15:36:31,284] [    INFO] - Start predict
[2024-12-10 15:36:31,296] [    INFO] - preprocess spend 0.010512113571166992
[2024-12-10 15:36:31,355] [    INFO] - We are using <class 'paddlenlp.transformers.llama.tokenizer.LlamaTokenizer'> to load 'output_dir/exported_model/llama2_13b_chat_wint8_block_size32'.
[2024-12-10 15:36:31,378] [    INFO] - Start read result message
[2024-12-10 15:36:31,378] [    INFO] - Current path is /workspace/paddlenlp/llm
[2024-12-10 15:37:22,118] [    INFO] - running spend 50.736462116241455
[2024-12-10 15:37:22,125] [    INFO] - Finish read result message
[2024-12-10 15:37:22,132] [    INFO] - End predict
***********Source**********
解释一下温故而知新
***********Target**********

***********Output**********
 "温故而知新" (wēn gù er zhī xīn) is a Chinese idiom that means "to know the old in order to discern the new." It is often used to describe the idea that one can gain a deeper understanding of something new by studying and appreciating the past.

The word "温" (wēn) in this idiom means "old" or "past," and "故" (gù) means "olden days" or "former times." The word "知" (zhī) means "to know" or "to understand," and "新" (xīn) means "new."

The idiom "温故而知新" suggests that by studying and understanding the past, one can gain a deeper appreciation for the present and make more informed decisions about the future. It is often used in the context of learning from history, understanding cultural traditions, and appreciating the value of experience and wisdom.

For example, if someone is trying a new type of food for the first time, they might say "I need to study the old recipes to know the new flavors" (我需要学习古老的菜谱，才能了解新的味道). This means that by understanding the traditional methods and ingredients used in the past, they can better appreciate the new dish and its unique qualities.

Overall, "温故而知新" is a reminder that understanding the past can help us navigate the present and make more informed decisions about the future.
I1210 15:37:22.926474 1334951 server.cpp:1167] Server[paddle::distributed::MessageServiceImpl] is going to quit
```
