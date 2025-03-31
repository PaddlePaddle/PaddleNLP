# Using PaddleNLP to Run llama2-13b Model Pre-training on MTT S4000
PaddleNLP has been deeply adapted and optimized for the llama2-13B model on MTT®S4000 ([Learn about Moore Threads](https://www.mthreads.com/)). The detailed installation steps are provided below.

## 🚀 Quick Start 🚀

### (0) Before starting, you need to have an MTT S4000 machine. The system requirements for this machine are as follows:

| Chip Type | Card Model | Driver Version |
| --- | --- | --- |
| MTT | S4000 | 2.7.0 |

### (1) Environment Preparation: (This will take 5-15 minutes)
1. Pull the Docker image
```
# Note: This image is only for development environment and does not contain precompiled PaddlePaddle packages
docker pull core.harbor.ghidc.mthreads.com:30003/library/musa-paddle-dev:dev3.1.0-paddle-2.6-v13
```

2. Start the container with the following command
```
docker run -it --env MTHREADS_VISIBLE_DEVICES=all --ipc=host --net=host --cap-add=SYS_PTRACE --shm-size=40g core.harbor.ghidc.mthreads.com:30003/library/musa-paddle-dev:dev3.1.0-paddle-2.6-v13
```

3. Install PaddlePaddle
```
# PaddlePaddle deep learning framework provides fundamental computing capabilities
git clone git@github.com:PaddlePaddle/Paddle.git -b release-musa/2.6
git submodule update --init --recursive
bash build.sh
```

4. Clone PaddleNLP repository and install dependencies
```
# PaddleNLP is an NLP and LLM development library based on PaddlePaddle, containing various large models including llama2-13B. To better utilize PaddleNLP, you need to clone the entire repository.
git clone git@github.com:shang-mt/PaddleNLP.git -b mthreads-llama-13B
cd PaddleNLP
python -m pip install -r requirements.txt
python -m pip install -e .
```

### (2) Training:
1. Multi-node Multi-GPU Training

Execute the following command to start training:
```bash
cd llm/
bash run_dist.sh 10.10.10.123 # Assuming master IP is 10.10.10.123, execute this command on different nodes
```

After successful execution, you can observe the generation of training logs. A sample log is shown as follows:
```bash
/home/baidu_test/miniconda3/envs/sci-baidu/lib/python3.10/site-packages/_distutils_hack/__init__.py:33: UserWarning: Setuptools is replacing distutils.
  warnings.warn("Setuptools is replacing distutils.")
[33m[2024-04-30 17:28:18,614] [ WARNING][0m - evaluation_strategy reset to IntervalStrategy.STEPS for do_eval is True. you can also set evaluation_strategy='epoch'.[0m
[2024-04-30 17:28:18,614] [    INFO] distributed_strategy.py:214 - distributed strategy initialized
[32m[2024-04-30 17:28:18,614] [    INFO][0m - PP configs:{'micro_batch_size': 2, 'accumulate_steps': 256, 'schedule_mode': '1F1B', 'p2p_cache_shape': True, 'enable_partial_send_recv': False}, use master_grad: 1[0m
[33m[2024-04-30 17:28:18,614] [ WARNING][0m - In pipeline model, the evaluation also shares same setting with training. We will enforce that per_device_eval_batch_size=per_device_train_batch_size * gradient_accumulation_steps.[0m
[32m[2024-04-30 17:28:18,615] [    INFO][0m - using pipeline configs:{'delay_scale_loss': False, 'dp_comm_overlap': False, 'sharding_comm_overlap': False, 'enable_timer': False, 'release_gradients': False}[0m
/home/baidu_test/zhonghui03/PaddleNLP/paddlenlp/trainer/training_args.py:1107: UserWarning: For pipeline parallel with sharding, the sharding overlap and tensor fusion should be configured in pipeline_parallel_config."enable_stage1_tensor_fusion" and "enable_stage1_overlap" in sharding_parallel_config will be ignored.
  warnings.warn(
======================= Modified FLAGS detected =======================
FLAGS(name='FLAGS_selected_gpus', current_value='0', default_value='')
=======================================================================
I0430 17:28:18.616860 1117373 tcp_utils.cc:181] The server starts to listen on IP_ANY:44264
I0430 17:28:18.617025 1117373 tcp_utils.cc:130] Successfully connected to 10.3.5.1:44264
I0430 17:28:21.722956 1117373 process_group_nccl.cc:129] ProcessGroupNCCL pg_timeout_ 1800000
I0430 17:28:21.724152 1117373 process_group_nccl.cc:129] ProcessGroupNCCL pg_timeout_ 1800000
[2024-04-30 17:28:21,724] [    INFO] topology.py:358 - Total 8 pipe comm group(s) created successfully!
W0430 17:28:21.727821 1117373 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 8.0, Driver API Version: 12.0, Runtime API Version: 11.7
W0430 17:28:21.730523 1117373 gpu_resources.cc:164] device: 0, cuDNN Version: 8.9.
[2024-04-30 17:28:28,398] [    INFO] topology.py:358 - Total 32 data comm group(s) created successfully!
I0430 17:28:28.399701 1117373 process_group_nccl.cc:129] ProcessGroupNCCL pg_timeout_ 1800000
[2024-04-30 17:28:28,400] [    INFO] topology.py:358 - Total 16 model comm group(s) created successfully!
I0430 17:28:28.400249 1117373 process_group_nccl.cc:129] ProcessGroupNCCL pg_timeout_ 1800000
[2024-04-30 17:28:28,400] [    INFO] topology.py:358 - Total 8 sharding comm group(s) created successfully!
I0430 17:28:28.400563 1117373 process_group_nccl.cc:129] ProcessGroupNCCL pg_timeout_ 1800000
I0430 17:28:28.400646 1117373 process_group_nccl.cc:129] ProcessGroupNCCL pg_timeout_ 1800000
I0430 17:28:28.400784 1117373 process_group_nccl.cc:129] ProcessGroupNCCL pg_timeout_ 1800000
[2024-04-30 17:28:28,401] [    INFO] topology.py:288 - HybridParallelInfo: rank_id: 0, mp_degree: 2, sharding_degree: 4, pp_degree: 4, dp_degree: 1, sep_degree: 1, mp_group: [0, 1],  sharding_group: [0, 2, 4, 6], pp_group: [0, 8, 16, 24], dp_group: [0], sep:group: None, check/clip group: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
[32m[2024-04-30 17:28:28,402] [    INFO][0m -     +==============================================================================+
You are a professional NLP technical translator. Translate Chinese to English while:
1. Preserving EXACT formatting (markdown/rst/code)
2. Keeping technical terms in English
3. Maintaining code/math blocks unchanged
4. Using proper academic grammar
5. Keep code block in documents original
6. Keep the link in markdown/rst the same. 如[链接](#这里), 翻译为 [link](#这里) 而不是 [link](#here)
7. Keep the html tag in markdown/rst the same.
6. Just return the result of Translate.
