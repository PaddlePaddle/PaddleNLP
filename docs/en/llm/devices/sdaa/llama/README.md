# Running Llama-2-13b-chat Model on Tai Chu sdaa with PaddleNLP

PaddleNLP has deeply adapted and optimized the Llama-2-13b-chat model for Tai Chu sdaa, achieving basic unification of the inference entry between sdaa device and GPU. Only device modification is required to migrate inference tasks.

## 🚀 Quick Start 🚀

### 0. Machine Preparation. Before starting, you need to prepare a machine with Tai Chu T100 accelerator cards, with the following requirements:

| Chip Type | Driver Version |
| --- | --- |
| Tai Chu T100 | 1.3.0 |

### 1. Environment Setup: (This will take 5-15 minutes)

#### 1.1 Pull Docker Image
```bash
# Note: This image contains precompiled PaddlePaddle packages, TecoDriver, TecoToolKit, etc., enabling one-click execution of paddlenlp models
wget http://mirrors.tecorigin.com/repository/teco-3rd-repo/custom_device/ubuntu22.04/x86_64/1.3.0/paddle_sdaa_1.3.0_llm_infer.tar
docker load < paddle_sdaa_1.3.0_llm_infer.tar
```

#### 1.2 Start Container with Reference Command
```bash
docker run -itd --name="paddle-sdaa-dev" --net=host --privileged --cap-add SYS_PTRACE --cap-add SYS_ADMIN --shm-size 128g jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/paddle_sdaa:1.3.0-llm-infer /bin/bash
```

#### 1.3 Download PaddleNLP Repository Code and Install Dependencies
```bash
# PaddleNLP is an NLP and LLM development library based on PaddlePaddle, containing various large models implemented with PaddlePaddle framework, including Llama-2-13b-chat model. To facilitate your usage, you need to clone the entire repository.
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
export PYTHONPATH=/path/to/PaddleNLP:$PYTHONPATH
pip install -r requirements.txt
cd csrc/sdaa && python setup_sdaa.py install && cd ../../llm/devices/sdaa/llama
```

### 2. Inference: (This will take 15-30 minutes)
#### 2.1 Dynamic Graph Distributed Inference

Execute the following command for inference:
```bash
bash dynamic_infer_llama_sdaa.sh
```
The first run will automatically download weights. You can use auto-downloaded weights or specify weight path after download. Upon successful execution, generated inference results can be viewed.

The example will save downloaded weights meta-llama/Llama-2-13b-chat to /workspace/weights, as shown below:
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
Explain the meaning of "温故而知新"
***********Target**********

***********Output**********
 "温故而知新" (wēn gù er zhī xīn) is a Chinese idiom that means "to understand the old in order to know the new." It is often used to convey the idea that one must have a deep understanding of the past and traditional ways of doing things in order to truly appreciate and understand new ideas and innovations.

The phrase is often used in the context of education, where students are encouraged to study the classics and learn from the past in order to gain a solid foundation for understanding new concepts and ideas. It is also used in business and technology, where companies may look to the past for inspiration and guidance as they develop new products and services.

In essence, "温故而知新" suggests that one cannot truly understand the new without first understanding the old, and that a deep appreciation for the past is essential for making progress and innovation.
```
#### 2.2 Static Graph Distributed Inference

##### 2.2.1 Static Graph Export

Execute the following command to perform static graph export, preparing for static graph distributed inference:
```bash
bash static_export_llama_sdaa.sh
```

After successful execution, the model export results can be viewed. An example is shown below:
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
##### 2.2.2 Static Graph Distributed Inference

Execute the following command for static graph distributed inference:
```bash
bash static_infer_llama_sdaa.sh
```

After successful execution, the generated inference results can be viewed. An example output is shown below:
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
Explain "温故而知新"
***********Target**********

***********Output**********
 "温故而知新" (wēn gù ér zhī xīn) is a Chinese idiom that translates to "Gaining new insights by reviewing the old." This phrase originates from the Confucian classic "The Analects" (《论语》), specifically from Book II, Chapter 11. It embodies the concept of deriving fresh understanding through the study of past knowledge and experiences.

The idiom can be broken down as:
- 温 (wēn): To review or revisit
- 故 (gù): The old, past knowledge
- 知新 (zhī xīn): To acquire new knowledge

This principle emphasizes that:
1. Historical knowledge serves as foundation for innovation
2. Past experiences can inform present decisions
3. Continuous learning requires building upon existing understanding

Modern applications include:
- Technical fields: Reusing and improving upon existing codebases
- Academic research: Building upon previous studies to make new discoveries
- Business strategy: Analyzing past market trends to predict future patterns

For example, in software development, a programmer might say: "By refactoring legacy code (温故), we discovered more efficient algorithms (知新)." This demonstrates how revisiting existing implementations can lead to technical innovations.
I1210 15:37:22.926474 1334951 server.cpp:1167] Server[paddle::distributed::MessageServiceImpl] is going to quit
```
Certainly! Please provide the Chinese text you need translated, and I'll ensure the translation adheres to all your specified requirements while maintaining technical accuracy and format integrity.
