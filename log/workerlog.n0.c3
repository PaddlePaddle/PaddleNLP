/home/ldn/anaconda3/envs/paddle-test/lib/python3.10/site-packages/_distutils_hack/__init__.py:26: UserWarning: Setuptools is replacing distutils.
  warnings.warn("Setuptools is replacing distutils.")
I0805 08:54:45.237290 2294392 custom_operator.cc:1296] register pir custom op :write_int8_cache_kv
I0805 08:54:45.237401 2294392 custom_operator.cc:1296] register pir custom op :update_inputs
I0805 08:54:45.237453 2294392 custom_operator.cc:1296] register pir custom op :transpose_remove_padding
I0805 08:54:45.237474 2294392 custom_operator.cc:1296] register pir custom op :set_stop_value_multi_ends_v2
I0805 08:54:45.237504 2294392 custom_operator.cc:1296] register pir custom op :step_paddle
I0805 08:54:45.237582 2294392 custom_operator.cc:1296] register pir custom op :get_token_penalty_multi_scores
I0805 08:54:45.237602 2294392 custom_operator.cc:1296] register pir custom op :set_value_by_flags_and_idx_v2
I0805 08:54:45.237622 2294392 custom_operator.cc:1296] register pir custom op :dequant_int8
I0805 08:54:45.237640 2294392 custom_operator.cc:1296] register pir custom op :set_stop_value_multi_ends
I0805 08:54:45.237658 2294392 custom_operator.cc:1296] register pir custom op :encode_rotary_qk
I0805 08:54:45.237679 2294392 custom_operator.cc:1296] register pir custom op :get_output
I0805 08:54:45.237699 2294392 custom_operator.cc:1296] register pir custom op :fused_get_rotary_embedding
I0805 08:54:45.237720 2294392 custom_operator.cc:1296] register pir custom op :set_value_by_flags_and_idx
I0805 08:54:45.237735 2294392 custom_operator.cc:1296] register pir custom op :save_with_output
I0805 08:54:45.237753 2294392 custom_operator.cc:1296] register pir custom op :write_cache_kv
I0805 08:54:45.237769 2294392 custom_operator.cc:1296] register pir custom op :get_token_penalty_multi_scores_v2
I0805 08:54:45.237789 2294392 custom_operator.cc:1296] register pir custom op :get_padding_offset
I0805 08:54:45.237807 2294392 custom_operator.cc:1296] register pir custom op :quant_int8
I0805 08:54:45.237828 2294392 custom_operator.cc:1296] register pir custom op :rebuild_padding_v2
I0805 08:54:45.237844 2294392 custom_operator.cc:1296] register pir custom op :get_padding_offset_v2
I0805 08:54:45.237861 2294392 custom_operator.cc:1296] register pir custom op :flash_attn_bwd
I0805 08:54:45.237882 2294392 custom_operator.cc:1296] register pir custom op :qkv_transpose_split
I0805 08:54:45.237901 2294392 custom_operator.cc:1296] register pir custom op :rebuild_padding
I0805 08:54:45.237915 2294392 custom_operator.cc:1296] register pir custom op :save_output
[2024-08-05 08:54:46,485] [    INFO] distributed_strategy.py:214 - distributed strategy initialized
[32m[2024-08-05 08:54:46,486] [    INFO][0m - PP configs:{'micro_batch_size': 4, 'accumulate_steps': 8, 'schedule_mode': '1F1B', 'p2p_cache_shape': True, 'enable_partial_send_recv': True}, use master_grad: False[0m
[32m[2024-08-05 08:54:46,486] [    INFO][0m - using pipeline configs:{'delay_scale_loss': False, 'dp_comm_overlap': False, 'sharding_comm_overlap': False, 'enable_timer': False, 'release_gradients': False, 'overlap_p2p_comm': False, 'clear_every_step_cache': False, 'use_batch_p2p_comm': True}[0m
======================= Modified FLAGS detected =======================
FLAGS(name='FLAGS_cudnn_dir', current_value='/home/ldn/anaconda3/envs/paddle-test/lib/python3.10/site-packages/paddle/../nvidia/cudnn/lib', default_value='')
FLAGS(name='FLAGS_curand_dir', current_value='/home/ldn/anaconda3/envs/paddle-test/lib/python3.10/site-packages/paddle/../nvidia/curand/lib', default_value='')
FLAGS(name='FLAGS_cublas_dir', current_value='/home/ldn/anaconda3/envs/paddle-test/lib/python3.10/site-packages/paddle/../nvidia/cublas/lib', default_value='')
FLAGS(name='FLAGS_cudnn_deterministic', current_value=True, default_value=False)
FLAGS(name='FLAGS_embedding_deterministic', current_value=1, default_value=0)
FLAGS(name='FLAGS_cupti_dir', current_value='/home/ldn/anaconda3/envs/paddle-test/lib/python3.10/site-packages/paddle/../nvidia/cuda_cupti/lib', default_value='')
FLAGS(name='FLAGS_nccl_dir', current_value='/home/ldn/anaconda3/envs/paddle-test/lib/python3.10/site-packages/paddle/../nvidia/nccl/lib', default_value='')
FLAGS(name='FLAGS_selected_gpus', current_value='3', default_value='')
FLAGS(name='FLAGS_cusparse_dir', current_value='/home/ldn/anaconda3/envs/paddle-test/lib/python3.10/site-packages/paddle/../nvidia/cusparse/lib', default_value='')
FLAGS(name='FLAGS_nvidia_package_dir', current_value='/home/ldn/anaconda3/envs/paddle-test/lib/python3.10/site-packages/paddle/../nvidia', default_value='')
FLAGS(name='FLAGS_cusolver_dir', current_value='/home/ldn/anaconda3/envs/paddle-test/lib/python3.10/site-packages/paddle/../nvidia/cusolver/lib', default_value='')
=======================================================================
I0805 08:54:46.487257 2294392 tcp_utils.cc:130] Successfully connected to 127.0.0.1:46279
I0805 08:54:46.500957 2294392 process_group_nccl.cc:138] ProcessGroupNCCL pg_timeout_ 1800000
I0805 08:54:46.500975 2294392 process_group_nccl.cc:139] ProcessGroupNCCL nccl_comm_init_option_ 0
I0805 08:54:47.551985 2294392 process_group_nccl.cc:138] ProcessGroupNCCL pg_timeout_ 1800000
I0805 08:54:47.552028 2294392 process_group_nccl.cc:139] ProcessGroupNCCL nccl_comm_init_option_ 0
[2024-08-05 08:54:47,552] [    INFO] topology.py:357 - Total 4 pipe comm group(s) create successfully!
W0805 08:54:47.562469 2294392 gpu_resources.cc:119] Please NOTE: device: 3, GPU Compute Capability: 8.0, Driver API Version: 12.2, Runtime API Version: 12.2
W0805 08:54:47.563360 2294392 gpu_resources.cc:164] device: 3, cuDNN Version: 9.0.
W0805 08:54:47.563381 2294392 gpu_resources.cc:196] WARNING: device: 3. The installed Paddle is compiled with CUDA 12.3, but CUDA runtime version in your machine is 12.2, which may cause serious incompatible bug. Please recompile or reinstall Paddle with compatible CUDA version.
ubuntu:2294392:2294392 [3] NCCL INFO Bootstrap : Using eno1:10.3.242.26<0>
ubuntu:2294392:2294392 [3] NCCL INFO NET/Plugin : dlerror=libnccl-net.so: cannot open shared object file: No such file or directory No plugin found (libnccl-net.so), using internal implementation
ubuntu:2294392:2294392 [3] NCCL INFO cudaDriverVersion 12020
NCCL version 2.20.5+cuda12.4
ubuntu:2294392:2294664 [3] NCCL INFO NET/IB : No device found.
ubuntu:2294392:2294664 [3] NCCL INFO NET/Socket : Using [0]eno1:10.3.242.26<0> [1]br-8d7ae5405055:172.18.0.1<0>
ubuntu:2294392:2294664 [3] NCCL INFO Using non-device net plugin version 0
ubuntu:2294392:2294664 [3] NCCL INFO Using network Socket
ubuntu:2294392:2294664 [3] NCCL INFO comm 0x5648fa225d90 rank 0 nranks 2 cudaDev 3 nvmlDev 3 busId 57000 commId 0x5a6de7bd616de810 - Init START
ubuntu:2294392:2294664 [3] NCCL INFO Setting affinity for GPU 3 to ffffffff,00000000,ffffffff
ubuntu:2294392:2294664 [3] NCCL INFO comm 0x5648fa225d90 rank 0 nRanks 2 nNodes 1 localRanks 2 localRank 0 MNNVL 0
ubuntu:2294392:2294664 [3] NCCL INFO Channel 00/02 :    0   1
ubuntu:2294392:2294664 [3] NCCL INFO Channel 01/02 :    0   1
ubuntu:2294392:2294664 [3] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1
ubuntu:2294392:2294664 [3] NCCL INFO P2P Chunksize set to 131072
ubuntu:2294392:2294664 [3] NCCL INFO Channel 00 : 0[3] -> 1[7] via SHM/direct/direct
ubuntu:2294392:2294664 [3] NCCL INFO Channel 01 : 0[3] -> 1[7] via SHM/direct/direct
ubuntu:2294392:2294664 [3] NCCL INFO Connected all rings
ubuntu:2294392:2294664 [3] NCCL INFO Connected all trees
ubuntu:2294392:2294664 [3] NCCL INFO NCCL_ALGO set by environment to Tree
ubuntu:2294392:2294664 [3] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
ubuntu:2294392:2294664 [3] NCCL INFO 2 coll channels, 0 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
ubuntu:2294392:2294664 [3] NCCL INFO comm 0x5648fa225d90 rank 0 nranks 2 cudaDev 3 nvmlDev 3 busId 57000 commId 0x5a6de7bd616de810 - Init COMPLETE
[2024-08-05 08:54:49,121] [    INFO] topology.py:357 - Total 8 data comm group(s) create successfully!
I0805 08:54:49.121733 2294392 process_group_nccl.cc:138] ProcessGroupNCCL pg_timeout_ 1800000
I0805 08:54:49.121765 2294392 process_group_nccl.cc:139] ProcessGroupNCCL nccl_comm_init_option_ 0
[2024-08-05 08:54:49,121] [    INFO] topology.py:357 - Total 2 model comm group(s) create successfully!
[2024-08-05 08:54:49,122] [    INFO] topology.py:357 - Total 8 sharding comm group(s) create successfully!
I0805 08:54:49.122383 2294392 process_group_nccl.cc:138] ProcessGroupNCCL pg_timeout_ 1800000
I0805 08:54:49.122401 2294392 process_group_nccl.cc:139] ProcessGroupNCCL nccl_comm_init_option_ 0
I0805 08:54:49.122478 2294392 process_group_nccl.cc:138] ProcessGroupNCCL pg_timeout_ 1800000
I0805 08:54:49.122493 2294392 process_group_nccl.cc:139] ProcessGroupNCCL nccl_comm_init_option_ 0
[2024-08-05 08:54:49,122] [    INFO] topology.py:279 - HybridParallelInfo: rank_id: 3, mp_degree: 4, sharding_degree: 1, pp_degree: 2, dp_degree: 1, sep_degree: 1, mp_group: [0, 1, 2, 3],  sharding_group: [3], pp_group: [3, 7], dp_group: [3], sep:group: None, check/clip group: [0, 1, 2, 3, 4, 5, 6, 7]
[32m[2024-08-05 08:54:49,124] [    INFO][0m -     +==============================================================================+
    |                                                                              |
    |                         DistributedStrategy Overview                         |
    |                                                                              |
    +==============================================================================+
    |                        a_sync=True <-> a_sync_configs                        |
    +------------------------------------------------------------------------------+
    |                               k_steps                    -1                  |
    |                     max_merge_var_num                    1                   |
    |                       send_queue_size                    16                  |
    |               independent_recv_thread                  False                 |
    |         min_send_grad_num_before_recv                    1                   |
    |                      thread_pool_size                    1                   |
    |                       send_wait_times                    1                   |
    |               runtime_split_send_recv                  False                 |
    |                        launch_barrier                   True                 |
    |             heter_worker_device_guard                   cpu                  |
    |                        lr_decay_steps                    10                  |
    |                            use_ps_gpu                    0                   |
    +==============================================================================+
    |                    Environment Flags, Communication Flags                    |
    +------------------------------------------------------------------------------+
    |                                  mode                    1                   |
    |                               elastic                  False                 |
    |                                  auto                  False                 |
    |                   sync_nccl_allreduce                   True                 |
    |                         nccl_comm_num                    1                   |
    |            use_hierarchical_allreduce                  False                 |
    |   hierarchical_allreduce_inter_nranks                    1                   |
    |                       sync_batch_norm                  False                 |
    |                   fuse_all_reduce_ops                   True                 |
    |                  fuse_grad_size_in_MB                    32                  |
    |              fuse_grad_size_in_TFLOPS                   50.0                 |
    |               cudnn_exhaustive_search                  False                 |
    |             conv_workspace_size_limit                   512                  |
    |    cudnn_batchnorm_spatial_persistent                  False                 |
    |                        fp16_allreduce                  False                 |
    |               last_comm_group_size_MB                   1.0                  |
    |                find_unused_parameters                  False                 |
    |            without_graph_optimization                   True                 |
    |                 fuse_grad_size_in_num                    8                   |
    |                 calc_comm_same_stream                  False                 |
    |                                   asp                  False                 |
    |                       fuse_grad_merge                  False                 |
    |                             semi_auto                  False                 |
    |                            adam_d2sum                  False                 |
    |                           auto_search                  False                 |
    |                        heter_ccl_mode                  False                 |
    |                         is_fl_ps_mode                  False                 |
    |                      with_coordinator                  False                 |
    |                            split_data                   True                 |
    |                  downpour_table_param                    []                  |
    |                       fs_client_param                                        |
    |                    execution_strategy                                        |
    +==============================================================================+
    |                                Build Strategy                                |
    +------------------------------------------------------------------------------+
    |           enable_sequential_execution                  False                 |
    |              fuse_elewise_add_act_ops                  False                 |
    |                       fuse_bn_act_ops                  False                 |
    |              fuse_relu_depthwise_conv                  False                 |
    |                    fuse_broadcast_ops                  False                 |
    |                fuse_all_optimizer_ops                  False                 |
    |                        enable_inplace                  False                 |
    |     enable_backward_optimizer_op_deps                   True                 |
    |                 cache_runtime_context                  False                 |
    |                   fuse_bn_add_act_ops                   True                 |
    |                    enable_auto_fusion                  False                 |
    |                          enable_addto                  False                 |
    |                      fix_op_run_order                  False                 |
    |              allow_cuda_graph_capture                  False                 |
    |                       reduce_strategy                    0                   |
    |                    fuse_gemm_epilogue                  False                 |
    |                   debug_graphviz_path                                        |
    |                       fused_attention                  False                 |
    |                     fused_feedforward                  False                 |
    |            fuse_dot_product_attention                  False                 |
    |                          fuse_resunit                  False                 |
    +==============================================================================+
[0m
[32m[2024-08-05 08:54:49,125] [    INFO][0m - The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).[0m
[35m[2024-08-05 08:54:49,126] [   DEBUG][0m - ============================================================[0m
[35m[2024-08-05 08:54:49,126] [   DEBUG][0m -      Model Configuration Arguments      [0m
[35m[2024-08-05 08:54:49,126] [   DEBUG][0m - paddle commit id              : a842a0f40f6111fb0c2df218130d0560aa747bc8[0m
[35m[2024-08-05 08:54:49,126] [   DEBUG][0m - paddlenlp commit id           : 272a4b04992242b0a3ca250c6a4b2b5d5c660f66[0m
[35m[2024-08-05 08:54:49,127] [   DEBUG][0m - aistudio_repo_id              : None[0m
[35m[2024-08-05 08:54:49,127] [   DEBUG][0m - aistudio_repo_license         : Apache License 2.0[0m
[35m[2024-08-05 08:54:49,127] [   DEBUG][0m - aistudio_repo_private         : True[0m
[35m[2024-08-05 08:54:49,127] [   DEBUG][0m - aistudio_token                : None[0m
[35m[2024-08-05 08:54:49,127] [   DEBUG][0m - attention_probs_dropout_prob  : 0.1[0m
[35m[2024-08-05 08:54:49,128] [   DEBUG][0m - continue_training             : True[0m
[35m[2024-08-05 08:54:49,128] [   DEBUG][0m - flash_mask                    : False[0m
[35m[2024-08-05 08:54:49,128] [   DEBUG][0m - from_aistudio                 : False[0m
[35m[2024-08-05 08:54:49,128] [   DEBUG][0m - fuse_attention_ffn            : None[0m
[35m[2024-08-05 08:54:49,128] [   DEBUG][0m - fuse_attention_qkv            : None[0m
[35m[2024-08-05 08:54:49,128] [   DEBUG][0m - hidden_dropout_prob           : 0.1[0m
[35m[2024-08-05 08:54:49,129] [   DEBUG][0m - lora                          : True[0m
[35m[2024-08-05 08:54:49,129] [   DEBUG][0m - lora_path                     : None[0m
[35m[2024-08-05 08:54:49,129] [   DEBUG][0m - lora_plus_scale               : 1.0[0m
[35m[2024-08-05 08:54:49,129] [   DEBUG][0m - lora_rank                     : 8[0m
[35m[2024-08-05 08:54:49,129] [   DEBUG][0m - model_name_or_path            : __internal_testing__/unified-ckpt-llama-170m-for-peft[0m
[35m[2024-08-05 08:54:49,129] [   DEBUG][0m - neftune                       : False[0m
[35m[2024-08-05 08:54:49,130] [   DEBUG][0m - neftune_noise_alpha           : 5.0[0m
[35m[2024-08-05 08:54:49,130] [   DEBUG][0m - num_prefix_tokens             : 128[0m
[35m[2024-08-05 08:54:49,130] [   DEBUG][0m - pissa                         : False[0m
[35m[2024-08-05 08:54:49,130] [   DEBUG][0m - prefix_path                   : None[0m
[35m[2024-08-05 08:54:49,130] [   DEBUG][0m - prefix_tuning                 : False[0m
[35m[2024-08-05 08:54:49,130] [   DEBUG][0m - reft                          : False[0m
[35m[2024-08-05 08:54:49,131] [   DEBUG][0m - rslora                        : False[0m
[35m[2024-08-05 08:54:49,131] [   DEBUG][0m - save_to_aistudio              : False[0m
[35m[2024-08-05 08:54:49,131] [   DEBUG][0m - tokenizer_name_or_path        : None[0m
[35m[2024-08-05 08:54:49,131] [   DEBUG][0m - use_fast_layer_norm           : False[0m
[35m[2024-08-05 08:54:49,131] [   DEBUG][0m - use_quick_lora                : False[0m
[35m[2024-08-05 08:54:49,131] [   DEBUG][0m - vera                          : False[0m
[35m[2024-08-05 08:54:49,132] [   DEBUG][0m - vera_rank                     : 8[0m
[35m[2024-08-05 08:54:49,132] [   DEBUG][0m - weight_blocksize              : 64[0m
[35m[2024-08-05 08:54:49,132] [   DEBUG][0m - weight_double_quant           : False[0m
[35m[2024-08-05 08:54:49,132] [   DEBUG][0m - weight_double_quant_block_size: 256[0m
[35m[2024-08-05 08:54:49,132] [   DEBUG][0m - weight_quantize_algo          : None[0m
[35m[2024-08-05 08:54:49,132] [   DEBUG][0m - [0m
[35m[2024-08-05 08:54:49,133] [   DEBUG][0m - ============================================================[0m
[35m[2024-08-05 08:54:49,133] [   DEBUG][0m -       Data Configuration Arguments      [0m
[35m[2024-08-05 08:54:49,133] [   DEBUG][0m - paddle commit id              : a842a0f40f6111fb0c2df218130d0560aa747bc8[0m
[35m[2024-08-05 08:54:49,133] [   DEBUG][0m - paddlenlp commit id           : 272a4b04992242b0a3ca250c6a4b2b5d5c660f66[0m
[35m[2024-08-05 08:54:49,133] [   DEBUG][0m - chat_template                 : None[0m
[35m[2024-08-05 08:54:49,134] [   DEBUG][0m - dataset_name_or_path          : ./unified_checkpoint/peft_input/data/[0m
[35m[2024-08-05 08:54:49,134] [   DEBUG][0m - eval_with_do_generation       : False[0m
[35m[2024-08-05 08:54:49,134] [   DEBUG][0m - intokens                      : None[0m
[35m[2024-08-05 08:54:49,134] [   DEBUG][0m - lazy                          : False[0m
[35m[2024-08-05 08:54:49,134] [   DEBUG][0m - max_length                    : 2048[0m
[35m[2024-08-05 08:54:49,134] [   DEBUG][0m - pad_to_multiple_of            : None[0m
[35m[2024-08-05 08:54:49,135] [   DEBUG][0m - save_generation_output        : False[0m
[35m[2024-08-05 08:54:49,135] [   DEBUG][0m - src_length                    : 1024[0m
[35m[2024-08-05 08:54:49,135] [   DEBUG][0m - task_name                     : None[0m
[35m[2024-08-05 08:54:49,135] [   DEBUG][0m - task_name_or_path             : None[0m
[35m[2024-08-05 08:54:49,135] [   DEBUG][0m - zero_padding                  : False[0m
[35m[2024-08-05 08:54:49,135] [   DEBUG][0m - [0m
[35m[2024-08-05 08:54:49,136] [   DEBUG][0m - ============================================================[0m
[35m[2024-08-05 08:54:49,136] [   DEBUG][0m -      Quant Configuration Arguments      [0m
[35m[2024-08-05 08:54:49,136] [   DEBUG][0m - paddle commit id              : a842a0f40f6111fb0c2df218130d0560aa747bc8[0m
[35m[2024-08-05 08:54:49,136] [   DEBUG][0m - paddlenlp commit id           : 272a4b04992242b0a3ca250c6a4b2b5d5c660f66[0m
[35m[2024-08-05 08:54:49,136] [   DEBUG][0m - auto_clip                     : False[0m
[35m[2024-08-05 08:54:49,136] [   DEBUG][0m - autoclip_step                 : 8[0m
[35m[2024-08-05 08:54:49,137] [   DEBUG][0m - awq_step                      : 8[0m
[35m[2024-08-05 08:54:49,137] [   DEBUG][0m - do_awq                        : False[0m
[35m[2024-08-05 08:54:49,137] [   DEBUG][0m - do_gptq                       : False[0m
[35m[2024-08-05 08:54:49,137] [   DEBUG][0m - do_ptq                        : False[0m
[35m[2024-08-05 08:54:49,137] [   DEBUG][0m - do_qat                        : False[0m
[35m[2024-08-05 08:54:49,137] [   DEBUG][0m - gptq_step                     : 8[0m
[35m[2024-08-05 08:54:49,138] [   DEBUG][0m - ptq_step                      : 32[0m
[35m[2024-08-05 08:54:49,138] [   DEBUG][0m - quant_type                    : a8w8[0m
[35m[2024-08-05 08:54:49,138] [   DEBUG][0m - shift                         : False[0m
[35m[2024-08-05 08:54:49,138] [   DEBUG][0m - shift_all_linears             : False[0m
[35m[2024-08-05 08:54:49,138] [   DEBUG][0m - shift_sampler                 : ema[0m
[35m[2024-08-05 08:54:49,138] [   DEBUG][0m - shift_step                    : 32[0m
[35m[2024-08-05 08:54:49,139] [   DEBUG][0m - smooth                        : False[0m
[35m[2024-08-05 08:54:49,139] [   DEBUG][0m - smooth_all_linears            : False[0m
[35m[2024-08-05 08:54:49,139] [   DEBUG][0m - smooth_k_piece                : 3[0m
[35m[2024-08-05 08:54:49,139] [   DEBUG][0m - smooth_piecewise_search       : False[0m
[35m[2024-08-05 08:54:49,139] [   DEBUG][0m - smooth_sampler                : none[0m
[35m[2024-08-05 08:54:49,139] [   DEBUG][0m - smooth_search_piece           : False[0m
[35m[2024-08-05 08:54:49,140] [   DEBUG][0m - smooth_step                   : 32[0m
[35m[2024-08-05 08:54:49,140] [   DEBUG][0m - weight_quant_method           : abs_max_channel_wise[0m
[35m[2024-08-05 08:54:49,140] [   DEBUG][0m - [0m
[35m[2024-08-05 08:54:49,140] [   DEBUG][0m - ============================================================[0m
[35m[2024-08-05 08:54:49,140] [   DEBUG][0m -    Generation Configuration Arguments   [0m
[35m[2024-08-05 08:54:49,141] [   DEBUG][0m - paddle commit id              : a842a0f40f6111fb0c2df218130d0560aa747bc8[0m
[35m[2024-08-05 08:54:49,141] [   DEBUG][0m - paddlenlp commit id           : 272a4b04992242b0a3ca250c6a4b2b5d5c660f66[0m
[35m[2024-08-05 08:54:49,141] [   DEBUG][0m - top_k                         : 1[0m
[35m[2024-08-05 08:54:49,141] [   DEBUG][0m - top_p                         : 1.0[0m
[35m[2024-08-05 08:54:49,141] [   DEBUG][0m - [0m
[35m[2024-08-05 08:54:49,141] [   DEBUG][0m - ============================================================[0m
[35m[2024-08-05 08:54:49,142] [   DEBUG][0m -       Reft Configuration Arguments      [0m
[35m[2024-08-05 08:54:49,142] [   DEBUG][0m - paddle commit id              : a842a0f40f6111fb0c2df218130d0560aa747bc8[0m
[35m[2024-08-05 08:54:49,142] [   DEBUG][0m - paddlenlp commit id           : 272a4b04992242b0a3ca250c6a4b2b5d5c660f66[0m
[35m[2024-08-05 08:54:49,142] [   DEBUG][0m - act_fn                        : None[0m
[35m[2024-08-05 08:54:49,142] [   DEBUG][0m - add_bias                      : False[0m
[35m[2024-08-05 08:54:49,142] [   DEBUG][0m - dropout                       : 0.0[0m
[35m[2024-08-05 08:54:49,143] [   DEBUG][0m - intervention_type             : LoreftIntervention[0m
[35m[2024-08-05 08:54:49,143] [   DEBUG][0m - layers                        : all[0m
[35m[2024-08-05 08:54:49,143] [   DEBUG][0m - position                      : f7+l7[0m
[35m[2024-08-05 08:54:49,143] [   DEBUG][0m - rank                          : 8[0m
[35m[2024-08-05 08:54:49,143] [   DEBUG][0m - [0m
[33m[2024-08-05 08:54:49,144] [ WARNING][0m - Process rank: 3, device: gpu, world_size: 8, distributed training: True, 16-bits training: True[0m
[32m[2024-08-05 08:54:49,145] [    INFO][0m - We are using <class 'paddlenlp.transformers.llama.configuration.LlamaConfig'> to load '__internal_testing__/unified-ckpt-llama-170m-for-peft'.[0m
[32m[2024-08-05 08:54:49,145] [    INFO][0m - Loading configuration file /home/ldn/.paddlenlp/models/__internal_testing__/unified-ckpt-llama-170m-for-peft/config.json[0m
[32m[2024-08-05 08:54:49,150] [    INFO][0m - Final model config: LlamaConfig {
  "alibi": false,
  "architectures": [
    "LlamaForCausalLM"
  ],
  "bos_token_id": 1,
  "dtype": "float16",
  "eos_token_id": 2,
  "hidden_size": 1024,
  "immediate_clear_past_key_value": false,
  "initializer_range": 0.02,
  "intermediate_size": 2816,
  "long_sequence_init_args": {},
  "long_sequence_strategy_name": null,
  "long_sequence_strategy_type": null,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 8,
  "num_hidden_layers": 8,
  "num_key_value_heads": 8,
  "pad_token_id": 0,
  "paddlenlp_version": "3.0.0b0.post20240805",
  "pipeline_parallel_degree": 2,
  "recompute": true,
  "rms_norm_eps": 1e-06,
  "rope_scaling_factor": 1.0,
  "rope_scaling_type": null,
  "rope_theta": 10000.0,
  "seq_length": 2048,
  "tensor_parallel_degree": 4,
  "tensor_parallel_output": false,
  "tensor_parallel_rank": 3,
  "tie_word_embeddings": false,
  "use_fast_layer_norm": false,
  "use_flash_attention_for_generation": false,
  "use_last_token_for_generation": false,
  "use_long_sequence_strategies": false,
  "vocab_size": 32000
}
[0m
[32m[2024-08-05 08:54:49,151] [    INFO][0m - We are using <class 'paddlenlp.transformers.llama.modeling_pp.LlamaForCausalLMPipe'> to load '__internal_testing__/unified-ckpt-llama-170m-for-peft'.[0m
[32m[2024-08-05 08:54:49,151] [    INFO][0m - Loading weights file from cache at /home/ldn/.paddlenlp/models/__internal_testing__/unified-ckpt-llama-170m-for-peft/model_state.pdparams[0m
[32m[2024-08-05 08:54:49,466] [    INFO][0m - Starting to convert orignal state_dict to tensor parallel state_dict.[0m
[32m[2024-08-05 08:54:49,621] [    INFO][0m - Loaded weights file from disk, setting weights to model.[0m
[2024-08-05 08:54:49,622] [    INFO] pp_layers.py:607 - start segment network..
[2024-08-05 08:54:49,622] [    INFO] pp_layers.py:613 - segment with method: layer:LlamaDecoderLayer; result: 0, 5, 11
[2024-08-05 08:54:49,622] [    INFO] pp_layers.py:629 - stage=0, global_rank=3 ,layer_number=5
[2024-08-05 08:54:49,622] [    INFO] pp_layers.py:634 - 0: LlamaEmbeddingPipe
[2024-08-05 08:54:49,622] [    INFO] pp_layers.py:634 - 1: LlamaDecoderLayerPipe
[2024-08-05 08:54:49,622] [    INFO] pp_layers.py:634 - 2: LlamaDecoderLayerPipe
[2024-08-05 08:54:49,622] [    INFO] pp_layers.py:634 - 3: LlamaDecoderLayerPipe
[2024-08-05 08:54:49,622] [    INFO] pp_layers.py:634 - 4: LlamaDecoderLayerPipe
[2024-08-05 08:54:49,622] [    INFO] pp_layers.py:629 - stage=1, global_rank=3 ,layer_number=6
[2024-08-05 08:54:49,622] [    INFO] pp_layers.py:634 - 5: LlamaDecoderLayerPipe
[2024-08-05 08:54:49,622] [    INFO] pp_layers.py:634 - 6: LlamaDecoderLayerPipe
[2024-08-05 08:54:49,622] [    INFO] pp_layers.py:634 - 7: LlamaDecoderLayerPipe
[2024-08-05 08:54:49,622] [    INFO] pp_layers.py:634 - 8: LlamaDecoderLayerPipe
[2024-08-05 08:54:49,622] [    INFO] pp_layers.py:634 - 9: LlamaRMSNormPipe
[2024-08-05 08:54:49,622] [    INFO] pp_layers.py:634 - 10: LlamaLMHead
[2024-08-05 08:54:49,622] [    INFO] pp_layers.py:656 - loss: LlamaPretrainingCriterion
[2024-08-05 08:54:50,189] [    INFO] pp_layers.py:706 - flush 5 of layers into run_function
[33m[2024-08-05 08:54:50,384] [ WARNING][0m - Some weights of the model checkpoint at __internal_testing__/unified-ckpt-llama-170m-for-peft were not used when initializing LlamaForCausalLMPipe: ['llama.layers.4.input_layernorm.weight', 'llama.layers.4.mlp.down_proj.weight', 'llama.layers.4.mlp.gate_proj.weight', 'llama.layers.4.mlp.up_proj.weight', 'llama.layers.4.post_attention_layernorm.weight', 'llama.layers.4.self_attn.k_proj.weight', 'llama.layers.4.self_attn.o_proj.weight', 'llama.layers.4.self_attn.q_proj.weight', 'llama.layers.4.self_attn.v_proj.weight', 'llama.layers.5.input_layernorm.weight', 'llama.layers.5.mlp.down_proj.weight', 'llama.layers.5.mlp.gate_proj.weight', 'llama.layers.5.mlp.up_proj.weight', 'llama.layers.5.post_attention_layernorm.weight', 'llama.layers.5.self_attn.k_proj.weight', 'llama.layers.5.self_attn.o_proj.weight', 'llama.layers.5.self_attn.q_proj.weight', 'llama.layers.5.self_attn.v_proj.weight', 'llama.layers.6.input_layernorm.weight', 'llama.layers.6.mlp.down_proj.weight', 'llama.layers.6.mlp.gate_proj.weight', 'llama.layers.6.mlp.up_proj.weight', 'llama.layers.6.post_attention_layernorm.weight', 'llama.layers.6.self_attn.k_proj.weight', 'llama.layers.6.self_attn.o_proj.weight', 'llama.layers.6.self_attn.q_proj.weight', 'llama.layers.6.self_attn.v_proj.weight', 'llama.layers.7.input_layernorm.weight', 'llama.layers.7.mlp.down_proj.weight', 'llama.layers.7.mlp.gate_proj.weight', 'llama.layers.7.mlp.up_proj.weight', 'llama.layers.7.post_attention_layernorm.weight', 'llama.layers.7.self_attn.k_proj.weight', 'llama.layers.7.self_attn.o_proj.weight', 'llama.layers.7.self_attn.q_proj.weight', 'llama.layers.7.self_attn.v_proj.weight', 'llama.norm.weight', 'lm_head.weight']
- This IS expected if you are initializing LlamaForCausalLMPipe from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing LlamaForCausalLMPipe from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).[0m
[32m[2024-08-05 08:54:50,385] [    INFO][0m - All the weights of LlamaForCausalLMPipe were initialized from the model checkpoint at __internal_testing__/unified-ckpt-llama-170m-for-peft.
If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLMPipe for predictions without further training.[0m
[32m[2024-08-05 08:54:50,390] [    INFO][0m - We are using <class 'paddlenlp.transformers.llama.tokenizer.LlamaTokenizer'> to load '__internal_testing__/unified-ckpt-llama-170m-for-peft'.[0m
[32m[2024-08-05 08:54:51,499] [    INFO][0m - Mark only lora and trainable_module as trainable.[0m
[35m[2024-08-05 08:54:51,503] [   DEBUG][0m - Frozen parameters: 2.10e+07 || Trainable parameters:3.30e+05 || Total parameters:2.14e+07|| Trainable:1.54%[0m
[32m[2024-08-05 08:54:51,504] [    INFO][0m - The global seed is set to 42, local seed is set to 53 and random seed is set to 42.[0m
[32m[2024-08-05 08:54:51,531] [    INFO][0m - max_steps is given, it will override any value given in num_train_epochs[0m
[32m[2024-08-05 08:54:51,531] [    INFO][0m - Using half precision[0m
[35m[2024-08-05 08:54:51,549] [   DEBUG][0m - ============================================================[0m
[35m[2024-08-05 08:54:51,549] [   DEBUG][0m -     Training Configuration Arguments    [0m
[35m[2024-08-05 08:54:51,549] [   DEBUG][0m - paddle commit id              : a842a0f40f6111fb0c2df218130d0560aa747bc8[0m
[35m[2024-08-05 08:54:51,549] [   DEBUG][0m - paddlenlp commit id           : 272a4b04992242b0a3ca250c6a4b2b5d5c660f66[0m
[35m[2024-08-05 08:54:51,549] [   DEBUG][0m - _no_sync_in_gradient_accumulation: True[0m
[35m[2024-08-05 08:54:51,550] [   DEBUG][0m - adam_beta1                    : 0.9[0m
[35m[2024-08-05 08:54:51,550] [   DEBUG][0m - adam_beta2                    : 0.999[0m
[35m[2024-08-05 08:54:51,550] [   DEBUG][0m - adam_epsilon                  : 1e-08[0m
[35m[2024-08-05 08:54:51,550] [   DEBUG][0m - amp_custom_black_list         : None[0m
[35m[2024-08-05 08:54:51,550] [   DEBUG][0m - amp_custom_white_list         : None[0m
[35m[2024-08-05 08:54:51,550] [   DEBUG][0m - amp_master_grad               : False[0m
[35m[2024-08-05 08:54:51,550] [   DEBUG][0m - autotuner_benchmark           : False[0m
[35m[2024-08-05 08:54:51,550] [   DEBUG][0m - benchmark                     : False[0m
[35m[2024-08-05 08:54:51,550] [   DEBUG][0m - bf16                          : False[0m
[35m[2024-08-05 08:54:51,550] [   DEBUG][0m - bf16_full_eval                : False[0m
[35m[2024-08-05 08:54:51,550] [   DEBUG][0m - context_parallel_degree       : 1[0m
[35m[2024-08-05 08:54:51,550] [   DEBUG][0m - current_device                : gpu:3[0m
[35m[2024-08-05 08:54:51,550] [   DEBUG][0m - data_parallel_config          : [0m
[35m[2024-08-05 08:54:51,550] [   DEBUG][0m - data_parallel_degree          : 1[0m
[35m[2024-08-05 08:54:51,550] [   DEBUG][0m - data_parallel_rank            : 0[0m
[35m[2024-08-05 08:54:51,550] [   DEBUG][0m - dataloader_drop_last          : False[0m
[35m[2024-08-05 08:54:51,550] [   DEBUG][0m - dataloader_num_workers        : 0[0m
[35m[2024-08-05 08:54:51,551] [   DEBUG][0m - dataset_rank                  : 0[0m
[35m[2024-08-05 08:54:51,551] [   DEBUG][0m - dataset_world_size            : 1[0m
[35m[2024-08-05 08:54:51,551] [   DEBUG][0m - decay_steps                   : 0[0m
[35m[2024-08-05 08:54:51,551] [   DEBUG][0m - device                        : gpu[0m
[35m[2024-08-05 08:54:51,551] [   DEBUG][0m - disable_tqdm                  : True[0m
[35m[2024-08-05 08:54:51,551] [   DEBUG][0m - distributed_dataloader        : False[0m
[35m[2024-08-05 08:54:51,551] [   DEBUG][0m - do_eval                       : False[0m
[35m[2024-08-05 08:54:51,551] [   DEBUG][0m - do_export                     : False[0m
[35m[2024-08-05 08:54:51,551] [   DEBUG][0m - do_predict                    : False[0m
[35m[2024-08-05 08:54:51,551] [   DEBUG][0m - do_train                      : True[0m
[35m[2024-08-05 08:54:51,551] [   DEBUG][0m - enable_auto_parallel          : False[0m
[35m[2024-08-05 08:54:51,551] [   DEBUG][0m - eval_accumulation_steps       : 16[0m
[35m[2024-08-05 08:54:51,551] [   DEBUG][0m - eval_batch_size               : 8[0m
[35m[2024-08-05 08:54:51,551] [   DEBUG][0m - eval_steps                    : None[0m
[35m[2024-08-05 08:54:51,551] [   DEBUG][0m - evaluation_strategy           : IntervalStrategy.NO[0m
[35m[2024-08-05 08:54:51,551] [   DEBUG][0m - flatten_param_grads           : False[0m
[35m[2024-08-05 08:54:51,551] [   DEBUG][0m - force_reshard_pp              : False[0m
[35m[2024-08-05 08:54:51,552] [   DEBUG][0m - fp16                          : True[0m
[35m[2024-08-05 08:54:51,552] [   DEBUG][0m - fp16_full_eval                : False[0m
[35m[2024-08-05 08:54:51,552] [   DEBUG][0m - fp16_opt_level                : O2[0m
[35m[2024-08-05 08:54:51,552] [   DEBUG][0m - fuse_sequence_parallel_allreduce: False[0m
[35m[2024-08-05 08:54:51,552] [   DEBUG][0m - gradient_accumulation_steps   : 8[0m
[35m[2024-08-05 08:54:51,552] [   DEBUG][0m - greater_is_better             : None[0m
[35m[2024-08-05 08:54:51,552] [   DEBUG][0m - hybrid_parallel_topo_order    : pp_first[0m
[35m[2024-08-05 08:54:51,552] [   DEBUG][0m - ignore_data_skip              : False[0m
[35m[2024-08-05 08:54:51,552] [   DEBUG][0m - ignore_load_lr_and_optim      : False[0m
[35m[2024-08-05 08:54:51,552] [   DEBUG][0m - ignore_save_lr_and_optim      : False[0m
[35m[2024-08-05 08:54:51,552] [   DEBUG][0m - label_names                   : None[0m
[35m[2024-08-05 08:54:51,552] [   DEBUG][0m - lazy_data_processing          : True[0m
[35m[2024-08-05 08:54:51,552] [   DEBUG][0m - learning_rate                 : 0.0003[0m
[35m[2024-08-05 08:54:51,552] [   DEBUG][0m - load_best_model_at_end        : False[0m
[35m[2024-08-05 08:54:51,552] [   DEBUG][0m - load_sharded_model            : False[0m
[35m[2024-08-05 08:54:51,553] [   DEBUG][0m - local_process_index           : 3[0m
[35m[2024-08-05 08:54:51,553] [   DEBUG][0m - local_rank                    : 3[0m
[35m[2024-08-05 08:54:51,553] [   DEBUG][0m - log_level                     : -1[0m
[35m[2024-08-05 08:54:51,553] [   DEBUG][0m - log_level_replica             : -1[0m
[35m[2024-08-05 08:54:51,553] [   DEBUG][0m - log_on_each_node              : True[0m
[35m[2024-08-05 08:54:51,553] [   DEBUG][0m - logging_dir                   : ./unified_checkpoint/checkpoints/llama_lora_ckpts/runs/Aug05_08-54-46_ubuntu[0m
[35m[2024-08-05 08:54:51,553] [   DEBUG][0m - logging_first_step            : False[0m
[35m[2024-08-05 08:54:51,554] [   DEBUG][0m - logging_steps                 : 1[0m
[35m[2024-08-05 08:54:51,554] [   DEBUG][0m - logging_strategy              : IntervalStrategy.STEPS[0m
[35m[2024-08-05 08:54:51,554] [   DEBUG][0m - logical_process_index         : 3[0m
[35m[2024-08-05 08:54:51,554] [   DEBUG][0m - lr_end                        : 1e-07[0m
[35m[2024-08-05 08:54:51,554] [   DEBUG][0m - lr_scheduler_type             : SchedulerType.LINEAR[0m
[35m[2024-08-05 08:54:51,554] [   DEBUG][0m - max_evaluate_steps            : -1[0m
[35m[2024-08-05 08:54:51,555] [   DEBUG][0m - max_grad_norm                 : 1.0[0m
[35m[2024-08-05 08:54:51,555] [   DEBUG][0m - max_steps                     : 15[0m
[35m[2024-08-05 08:54:51,555] [   DEBUG][0m - metric_for_best_model         : None[0m
[35m[2024-08-05 08:54:51,555] [   DEBUG][0m - minimum_eval_times            : None[0m
[35m[2024-08-05 08:54:51,555] [   DEBUG][0m - no_cuda                       : False[0m
[35m[2024-08-05 08:54:51,556] [   DEBUG][0m - no_recompute_layers           : None[0m
[35m[2024-08-05 08:54:51,556] [   DEBUG][0m - num_cycles                    : 0.5[0m
[35m[2024-08-05 08:54:51,556] [   DEBUG][0m - num_train_epochs              : 1.0[0m
[35m[2024-08-05 08:54:51,556] [   DEBUG][0m - optim                         : OptimizerNames.ADAMW[0m
[35m[2024-08-05 08:54:51,556] [   DEBUG][0m - optimizer_name_suffix         : tp03_pp00[0m
[35m[2024-08-05 08:54:51,556] [   DEBUG][0m - output_dir                    : ./unified_checkpoint/checkpoints/llama_lora_ckpts[0m
[35m[2024-08-05 08:54:51,557] [   DEBUG][0m - overwrite_output_dir          : False[0m
[35m[2024-08-05 08:54:51,557] [   DEBUG][0m - past_index                    : -1[0m
[35m[2024-08-05 08:54:51,557] [   DEBUG][0m - per_device_eval_batch_size    : 8[0m
[35m[2024-08-05 08:54:51,557] [   DEBUG][0m - per_device_train_batch_size   : 4[0m
[35m[2024-08-05 08:54:51,557] [   DEBUG][0m - pipeline_parallel_config      : [0m
[35m[2024-08-05 08:54:51,557] [   DEBUG][0m - pipeline_parallel_degree      : 2[0m
[35m[2024-08-05 08:54:51,557] [   DEBUG][0m - pipeline_parallel_rank        : 0[0m
[35m[2024-08-05 08:54:51,557] [   DEBUG][0m - power                         : 1.0[0m
[35m[2024-08-05 08:54:51,557] [   DEBUG][0m - pp_recompute_interval         : 1[0m
[35m[2024-08-05 08:54:51,557] [   DEBUG][0m - prediction_loss_only          : False[0m
[35m[2024-08-05 08:54:51,557] [   DEBUG][0m - process_index                 : 3[0m
[35m[2024-08-05 08:54:51,557] [   DEBUG][0m - recompute                     : True[0m
[35m[2024-08-05 08:54:51,557] [   DEBUG][0m - recompute_granularity         : full[0m
[35m[2024-08-05 08:54:51,557] [   DEBUG][0m - recompute_use_reentrant       : False[0m
[35m[2024-08-05 08:54:51,557] [   DEBUG][0m - remove_unused_columns         : True[0m
[35m[2024-08-05 08:54:51,557] [   DEBUG][0m - report_to                     : ['visualdl'][0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - resume_from_checkpoint        : None[0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - run_name                      : ./unified_checkpoint/checkpoints/llama_lora_ckpts[0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - save_on_each_node             : False[0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - save_sharded_model            : False[0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - save_steps                    : 10[0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - save_strategy                 : IntervalStrategy.STEPS[0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - save_total_limit              : 1[0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - scale_loss                    : 32768[0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - seed                          : 42[0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - sep_parallel_degree           : 1[0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - sequence_parallel             : False[0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - sequence_parallel_config      : [0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - sharding                      : [][0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - sharding_degree               : -1[0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - sharding_parallel_config      : [0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - sharding_parallel_degree      : 1[0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - sharding_parallel_rank        : 0[0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - should_load_dataset           : True[0m
[35m[2024-08-05 08:54:51,558] [   DEBUG][0m - should_load_sharding_stage1_model: False[0m
[35m[2024-08-05 08:54:51,559] [   DEBUG][0m - should_log                    : False[0m
[35m[2024-08-05 08:54:51,559] [   DEBUG][0m - should_save                   : False[0m
[35m[2024-08-05 08:54:51,559] [   DEBUG][0m - should_save_model_state       : True[0m
[35m[2024-08-05 08:54:51,559] [   DEBUG][0m - should_save_sharding_stage1_model: False[0m
[35m[2024-08-05 08:54:51,559] [   DEBUG][0m - skip_memory_metrics           : True[0m
[35m[2024-08-05 08:54:51,559] [   DEBUG][0m - skip_profile_timer            : True[0m
[35m[2024-08-05 08:54:51,559] [   DEBUG][0m - tensor_parallel_config        : [0m
[35m[2024-08-05 08:54:51,559] [   DEBUG][0m - tensor_parallel_degree        : 4[0m
[35m[2024-08-05 08:54:51,559] [   DEBUG][0m - tensor_parallel_output        : False[0m
[35m[2024-08-05 08:54:51,559] [   DEBUG][0m - tensor_parallel_rank          : 3[0m
[35m[2024-08-05 08:54:51,559] [   DEBUG][0m - to_static                     : False[0m
[35m[2024-08-05 08:54:51,559] [   DEBUG][0m - train_batch_size              : 4[0m
[35m[2024-08-05 08:54:51,559] [   DEBUG][0m - unified_checkpoint            : True[0m
[35m[2024-08-05 08:54:51,559] [   DEBUG][0m - unified_checkpoint_config     : [''][0m
[35m[2024-08-05 08:54:51,559] [   DEBUG][0m - use_async_save                : False[0m
[35m[2024-08-05 08:54:51,559] [   DEBUG][0m - use_expert_parallel           : False[0m
[35m[2024-08-05 08:54:51,559] [   DEBUG][0m - use_flash_attention           : False[0m
[35m[2024-08-05 08:54:51,559] [   DEBUG][0m - use_fused_dropout_add         : False[0m
[35m[2024-08-05 08:54:51,560] [   DEBUG][0m - use_fused_linear              : False[0m
[35m[2024-08-05 08:54:51,560] [   DEBUG][0m - use_fused_rms_norm            : False[0m
[35m[2024-08-05 08:54:51,560] [   DEBUG][0m - use_fused_rope                : False[0m
[35m[2024-08-05 08:54:51,560] [   DEBUG][0m - use_hybrid_parallel           : True[0m
[35m[2024-08-05 08:54:51,560] [   DEBUG][0m - virtual_pp_degree             : 1[0m
[35m[2024-08-05 08:54:51,560] [   DEBUG][0m - wandb_api_key                 : None[0m
[35m[2024-08-05 08:54:51,560] [   DEBUG][0m - warmup_ratio                  : 0.0[0m
[35m[2024-08-05 08:54:51,560] [   DEBUG][0m - warmup_steps                  : 30[0m
[35m[2024-08-05 08:54:51,560] [   DEBUG][0m - weight_decay                  : 0.0[0m
[35m[2024-08-05 08:54:51,560] [   DEBUG][0m - weight_name_suffix            : tp03_pp00[0m
[35m[2024-08-05 08:54:51,560] [   DEBUG][0m - world_size                    : 8[0m
[35m[2024-08-05 08:54:51,560] [   DEBUG][0m - [0m
[32m[2024-08-05 08:54:51,560] [    INFO][0m - Starting training from resume_from_checkpoint : None[0m
ubuntu:2294392:2294772 [3] NCCL INFO Using non-device net plugin version 0
ubuntu:2294392:2294772 [3] NCCL INFO Using network Socket
ubuntu:2294392:2294772 [3] NCCL INFO comm 0x564904d61240 rank 3 nranks 8 cudaDev 3 nvmlDev 3 busId 57000 commId 0x123cc55294d9570 - Init START
ubuntu:2294392:2294772 [3] NCCL INFO Setting affinity for GPU 3 to ffffffff,00000000,ffffffff
ubuntu:2294392:2294772 [3] NCCL INFO NVLS multicast support is not available on dev 3
ubuntu:2294392:2294772 [3] NCCL INFO comm 0x564904d61240 rank 3 nRanks 8 nNodes 1 localRanks 8 localRank 3 MNNVL 0
ubuntu:2294392:2294772 [3] NCCL INFO Trees [0] 4/-1/-1->3->2 [1] 4/-1/-1->3->2
ubuntu:2294392:2294772 [3] NCCL INFO P2P Chunksize set to 524288
ubuntu:2294392:2294772 [3] NCCL INFO Channel 00/0 : 3[3] -> 1[1] via P2P/CUMEM
ubuntu:2294392:2294772 [3] NCCL INFO Channel 01/0 : 3[3] -> 1[1] via P2P/CUMEM
ubuntu:2294392:2294772 [3] NCCL INFO Connected all rings
ubuntu:2294392:2294772 [3] NCCL INFO Channel 00 : 3[3] -> 4[4] via SHM/direct/direct
ubuntu:2294392:2294772 [3] NCCL INFO Channel 01 : 3[3] -> 4[4] via SHM/direct/direct
ubuntu:2294392:2294772 [3] NCCL INFO Channel 00/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2294392:2294772 [3] NCCL INFO Channel 01/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2294392:2294772 [3] NCCL INFO Connected all trees
ubuntu:2294392:2294772 [3] NCCL INFO NCCL_ALGO set by environment to Tree
ubuntu:2294392:2294772 [3] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 512 | 512
ubuntu:2294392:2294772 [3] NCCL INFO 2 coll channels, 0 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
ubuntu:2294392:2294772 [3] NCCL INFO comm 0x564904d61240 rank 3 nranks 8 cudaDev 3 nvmlDev 3 busId 57000 commId 0x123cc55294d9570 - Init COMPLETE
[2024-08-05 08:54:53,804] [    INFO] pipeline_parallel.py:235 - dp_comm_overlap False;             sharding_comm_overlap False;             sharding_split_param False;
[2024-08-05 08:54:53,804] [    INFO] pipeline_parallel.py:303 - Pipeline Info -- num_stages: 2, stage_id: 0
[2024-08-05 08:54:53,804] [    INFO] pipeline_parallel.py:308 - start broadcast mp parameters
ubuntu:2294392:2294842 [3] NCCL INFO Using non-device net plugin version 0
ubuntu:2294392:2294842 [3] NCCL INFO Using network Socket
ubuntu:2294392:2294842 [3] NCCL INFO comm 0x564904d825b0 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 57000 commId 0x993e01a33492d757 - Init START
ubuntu:2294392:2294842 [3] NCCL INFO Setting affinity for GPU 3 to ffffffff,00000000,ffffffff
ubuntu:2294392:2294842 [3] NCCL INFO NVLS multicast support is not available on dev 3
ubuntu:2294392:2294842 [3] NCCL INFO comm 0x564904d825b0 rank 3 nRanks 4 nNodes 1 localRanks 4 localRank 3 MNNVL 0
ubuntu:2294392:2294842 [3] NCCL INFO Trees [0] 1/-1/-1->3->0 [1] 0/-1/-1->3->2 [2] -1/-1/-1->3->0 [3] 0/-1/-1->3->-1 [4] 1/-1/-1->3->0 [5] 0/-1/-1->3->2 [6] -1/-1/-1->3->0 [7] 0/-1/-1->3->-1
ubuntu:2294392:2294842 [3] NCCL INFO P2P Chunksize set to 524288
ubuntu:2294392:2294842 [3] NCCL INFO Channel 00/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2294392:2294842 [3] NCCL INFO Channel 01/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2294392:2294842 [3] NCCL INFO Channel 04/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2294392:2294842 [3] NCCL INFO Channel 05/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2294392:2294842 [3] NCCL INFO Channel 02/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2294392:2294842 [3] NCCL INFO Channel 03/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2294392:2294842 [3] NCCL INFO Channel 06/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2294392:2294842 [3] NCCL INFO Channel 07/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2294392:2294842 [3] NCCL INFO Connected all rings
ubuntu:2294392:2294842 [3] NCCL INFO Channel 02/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2294392:2294842 [3] NCCL INFO Channel 03/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2294392:2294842 [3] NCCL INFO Channel 06/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2294392:2294842 [3] NCCL INFO Channel 07/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2294392:2294842 [3] NCCL INFO Channel 00/0 : 3[3] -> 1[1] via P2P/CUMEM
ubuntu:2294392:2294842 [3] NCCL INFO Channel 04/0 : 3[3] -> 1[1] via P2P/CUMEM
ubuntu:2294392:2294842 [3] NCCL INFO Channel 01/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2294392:2294842 [3] NCCL INFO Channel 05/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2294392:2294842 [3] NCCL INFO Connected all trees
ubuntu:2294392:2294842 [3] NCCL INFO NCCL_ALGO set by environment to Tree
ubuntu:2294392:2294842 [3] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
ubuntu:2294392:2294842 [3] NCCL INFO 8 coll channels, 0 collnet channels, 0 nvls channels, 8 p2p channels, 2 p2p channels per peer
ubuntu:2294392:2294842 [3] NCCL INFO comm 0x564904d825b0 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 57000 commId 0x993e01a33492d757 - Init COMPLETE
[2024-08-05 08:54:55,222] [ WARNING] hybrid_parallel_optimizer.py:292 - While using ClipGradByGlobalNorm in TensorParallel, PipelineParallel or Sharding, the grad clip of original optimizer will be changed.
[32m[2024-08-05 08:54:55,224] [    INFO][0m - [timelog] checkpoint loading time: 0.00s (2024-08-05 08:54:55) [0m
[32m[2024-08-05 08:54:55,224] [    INFO][0m - ***** Running training *****[0m
[32m[2024-08-05 08:54:55,224] [    INFO][0m -   Num examples = 114,599[0m
[32m[2024-08-05 08:54:55,224] [    INFO][0m -   Num Epochs = 1[0m
[32m[2024-08-05 08:54:55,225] [    INFO][0m -   Instantaneous batch size per device = 4[0m
[32m[2024-08-05 08:54:55,225] [    INFO][0m -   Total train batch size (w. parallel, distributed & accumulation) = 32[0m
[32m[2024-08-05 08:54:55,225] [    INFO][0m -   Gradient Accumulation steps = 8[0m
[32m[2024-08-05 08:54:55,225] [    INFO][0m -   Total optimization steps = 15[0m
[32m[2024-08-05 08:54:55,225] [    INFO][0m -   Total num train samples = 480[0m
[35m[2024-08-05 08:54:55,227] [   DEBUG][0m -   Number of trainable parameters = 329,728 (per device)[0m
[35m[2024-08-05 08:54:55,239] [   DEBUG][0m -   Number of trainable parameters = 2,637,824 (all devices, roughly)[0m
W0805 08:54:56.670755 2294392 multiply_fwd_func.cc:75] got different data type, run type promotion automatically, this may cause data type been changed.
ubuntu:2294392:2294926 [3] NCCL INFO Using non-device net plugin version 0
ubuntu:2294392:2294926 [3] NCCL INFO Using network Socket
ubuntu:2294392:2294926 [3] NCCL INFO comm 0x5649151a72f0 rank 0 nranks 2 cudaDev 3 nvmlDev 3 busId 57000 commId 0xb11e1310e2afd9fc - Init START
ubuntu:2294392:2294926 [3] NCCL INFO Setting affinity for GPU 3 to ffffffff,00000000,ffffffff
ubuntu:2294392:2294926 [3] NCCL INFO comm 0x5649151a72f0 rank 0 nRanks 2 nNodes 1 localRanks 2 localRank 0 MNNVL 0
ubuntu:2294392:2294926 [3] NCCL INFO Channel 00/02 :    0   1
ubuntu:2294392:2294926 [3] NCCL INFO Channel 01/02 :    0   1
ubuntu:2294392:2294926 [3] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1
ubuntu:2294392:2294926 [3] NCCL INFO P2P Chunksize set to 131072
ubuntu:2294392:2294926 [3] NCCL INFO Channel 00 : 0[3] -> 1[7] via SHM/direct/direct
ubuntu:2294392:2294926 [3] NCCL INFO Channel 01 : 0[3] -> 1[7] via SHM/direct/direct
ubuntu:2294392:2294926 [3] NCCL INFO Connected all rings
ubuntu:2294392:2294926 [3] NCCL INFO Connected all trees
ubuntu:2294392:2294926 [3] NCCL INFO NCCL_ALGO set by environment to Tree
ubuntu:2294392:2294926 [3] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
ubuntu:2294392:2294926 [3] NCCL INFO 2 coll channels, 0 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
ubuntu:2294392:2294926 [3] NCCL INFO comm 0x5649151a72f0 rank 0 nranks 2 cudaDev 3 nvmlDev 3 busId 57000 commId 0xb11e1310e2afd9fc - Init COMPLETE
ubuntu:2294392:2294970 [3] NCCL INFO Channel 00 : 0[3] -> 1[7] via SHM/direct/direct
ubuntu:2294392:2294970 [3] NCCL INFO Channel 01 : 0[3] -> 1[7] via SHM/direct/direct
ubuntu:2294392:2294980 [3] NCCL INFO Channel 00 : 0[3] -> 1[7] via SHM/direct/direct
ubuntu:2294392:2294980 [3] NCCL INFO Channel 01 : 0[3] -> 1[7] via SHM/direct/direct
ubuntu:2294392:2295037 [3] NCCL INFO Using non-device net plugin version 0
ubuntu:2294392:2295037 [3] NCCL INFO Using network Socket
ubuntu:2294392:2295037 [3] NCCL INFO comm 0x56491533ab30 rank 3 nranks 8 cudaDev 3 nvmlDev 3 busId 57000 commId 0xd6b8ecfce2d966a - Init START
ubuntu:2294392:2295037 [3] NCCL INFO Setting affinity for GPU 3 to ffffffff,00000000,ffffffff
ubuntu:2294392:2295037 [3] NCCL INFO NVLS multicast support is not available on dev 3
ubuntu:2294392:2295037 [3] NCCL INFO comm 0x56491533ab30 rank 3 nRanks 8 nNodes 1 localRanks 8 localRank 3 MNNVL 0
ubuntu:2294392:2295037 [3] NCCL INFO Trees [0] 4/-1/-1->3->2 [1] 4/-1/-1->3->2
ubuntu:2294392:2295037 [3] NCCL INFO P2P Chunksize set to 524288
ubuntu:2294392:2295037 [3] NCCL INFO Channel 00/0 : 3[3] -> 1[1] via P2P/CUMEM
ubuntu:2294392:2295037 [3] NCCL INFO Channel 01/0 : 3[3] -> 1[1] via P2P/CUMEM
ubuntu:2294392:2295037 [3] NCCL INFO Connected all rings
ubuntu:2294392:2295037 [3] NCCL INFO Channel 00 : 3[3] -> 4[4] via SHM/direct/direct
ubuntu:2294392:2295037 [3] NCCL INFO Channel 01 : 3[3] -> 4[4] via SHM/direct/direct
ubuntu:2294392:2295037 [3] NCCL INFO Channel 00/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2294392:2295037 [3] NCCL INFO Channel 01/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2294392:2295037 [3] NCCL INFO Connected all trees
ubuntu:2294392:2295037 [3] NCCL INFO NCCL_ALGO set by environment to Tree
ubuntu:2294392:2295037 [3] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 512 | 512
ubuntu:2294392:2295037 [3] NCCL INFO 2 coll channels, 0 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
ubuntu:2294392:2295037 [3] NCCL INFO comm 0x56491533ab30 rank 3 nranks 8 cudaDev 3 nvmlDev 3 busId 57000 commId 0xd6b8ecfce2d966a - Init COMPLETE
[32m[2024-08-05 08:55:26,390] [    INFO][0m - Saving model checkpoint to ./unified_checkpoint/checkpoints/llama_lora_ckpts/checkpoint-10[0m
[32m[2024-08-05 08:55:26,591] [    INFO][0m - Unified model tensor parallel weights in shards[0m
ubuntu:2294392:2295659 [3] NCCL INFO Channel 04/1 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2294392:2295659 [3] NCCL INFO Channel 05/1 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2294392:2295662 [3] NCCL INFO Channel 02/1 : 3[3] -> 1[1] via P2P/CUMEM
ubuntu:2294392:2295662 [3] NCCL INFO Channel 03/1 : 3[3] -> 1[1] via P2P/CUMEM
ubuntu:2294392:2295668 [3] NCCL INFO Channel 06/1 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2294392:2295668 [3] NCCL INFO Channel 07/1 : 3[3] -> 2[2] via P2P/CUMEM
[32m[2024-08-05 08:55:26,817] [    INFO][0m - Unified checkpoint: generating sharded_index json files for model weight.[0m
[32m[2024-08-05 08:55:26,897] [    INFO][0m - Saving optimizer files.[0m
[32m[2024-08-05 08:55:27,401] [    INFO][0m - Unified optimizer tensor parallel in shards[0m
[32m[2024-08-05 08:55:27,643] [    INFO][0m - Unified master weight tensor parallel in shards[0m
[32m[2024-08-05 08:55:28,028] [    INFO][0m - Unified checkpoint: generating sharded_index json files for optimizer or master weight.[0m
[32m[2024-08-05 08:55:28,075] [    INFO][0m - Unified checkpoint: generating sharded_index json files for optimizer or master weight.[0m
[32m[2024-08-05 08:55:28,177] [    INFO][0m - [timelog] checkpoint saving time: 1.75s (2024-08-05 08:55:28) [0m
[32m[2024-08-05 08:55:41,401] [    INFO][0m - 
Training completed. 
[0m
[32m[2024-08-05 08:55:41,404] [    INFO][0m - Saving model checkpoint to ./unified_checkpoint/checkpoints/llama_lora_ckpts[0m
[32m[2024-08-05 08:55:41,609] [    INFO][0m - Unified model tensor parallel weights in shards[0m
[32m[2024-08-05 08:55:41,676] [    INFO][0m - Unified checkpoint: generating sharded_index json files for model weight.[0m
I0805 08:55:43.215900 2294392 process_group_nccl.cc:143] ProcessGroupNCCL destruct 
I0805 08:55:43.218775 2294392 process_group_nccl.cc:143] ProcessGroupNCCL destruct 
I0805 08:55:43.218847 2294392 process_group_nccl.cc:143] ProcessGroupNCCL destruct 
I0805 08:55:43.218868 2294392 process_group_nccl.cc:143] ProcessGroupNCCL destruct 
I0805 08:55:43.218883 2294392 process_group_nccl.cc:143] ProcessGroupNCCL destruct 
/home/ldn/anaconda3/envs/paddle-test/lib/python3.10/site-packages/_distutils_hack/__init__.py:26: UserWarning: Setuptools is replacing distutils.
  warnings.warn("Setuptools is replacing distutils.")
I0805 08:55:51.777807 2296433 custom_operator.cc:1296] register pir custom op :write_int8_cache_kv
I0805 08:55:51.777926 2296433 custom_operator.cc:1296] register pir custom op :update_inputs
I0805 08:55:51.777987 2296433 custom_operator.cc:1296] register pir custom op :transpose_remove_padding
I0805 08:55:51.778012 2296433 custom_operator.cc:1296] register pir custom op :set_stop_value_multi_ends_v2
I0805 08:55:51.778045 2296433 custom_operator.cc:1296] register pir custom op :step_paddle
I0805 08:55:51.778141 2296433 custom_operator.cc:1296] register pir custom op :get_token_penalty_multi_scores
I0805 08:55:51.778165 2296433 custom_operator.cc:1296] register pir custom op :set_value_by_flags_and_idx_v2
I0805 08:55:51.778190 2296433 custom_operator.cc:1296] register pir custom op :dequant_int8
I0805 08:55:51.778214 2296433 custom_operator.cc:1296] register pir custom op :set_stop_value_multi_ends
I0805 08:55:51.778236 2296433 custom_operator.cc:1296] register pir custom op :encode_rotary_qk
I0805 08:55:51.778264 2296433 custom_operator.cc:1296] register pir custom op :get_output
I0805 08:55:51.778288 2296433 custom_operator.cc:1296] register pir custom op :fused_get_rotary_embedding
I0805 08:55:51.778317 2296433 custom_operator.cc:1296] register pir custom op :set_value_by_flags_and_idx
I0805 08:55:51.778337 2296433 custom_operator.cc:1296] register pir custom op :save_with_output
I0805 08:55:51.778360 2296433 custom_operator.cc:1296] register pir custom op :write_cache_kv
I0805 08:55:51.778381 2296433 custom_operator.cc:1296] register pir custom op :get_token_penalty_multi_scores_v2
I0805 08:55:51.778406 2296433 custom_operator.cc:1296] register pir custom op :get_padding_offset
I0805 08:55:51.778427 2296433 custom_operator.cc:1296] register pir custom op :quant_int8
I0805 08:55:51.778453 2296433 custom_operator.cc:1296] register pir custom op :rebuild_padding_v2
I0805 08:55:51.778476 2296433 custom_operator.cc:1296] register pir custom op :get_padding_offset_v2
I0805 08:55:51.778497 2296433 custom_operator.cc:1296] register pir custom op :flash_attn_bwd
I0805 08:55:51.778524 2296433 custom_operator.cc:1296] register pir custom op :qkv_transpose_split
I0805 08:55:51.778548 2296433 custom_operator.cc:1296] register pir custom op :rebuild_padding
I0805 08:55:51.778565 2296433 custom_operator.cc:1296] register pir custom op :save_output
[2024-08-05 08:55:53,093] [    INFO] distributed_strategy.py:214 - distributed strategy initialized
[32m[2024-08-05 08:55:53,094] [    INFO][0m - PP configs:{'micro_batch_size': 4, 'accumulate_steps': 8, 'schedule_mode': '1F1B', 'p2p_cache_shape': True, 'enable_partial_send_recv': True}, use master_grad: False[0m
[32m[2024-08-05 08:55:53,094] [    INFO][0m - using pipeline configs:{'delay_scale_loss': False, 'dp_comm_overlap': False, 'sharding_comm_overlap': False, 'enable_timer': False, 'release_gradients': False, 'overlap_p2p_comm': False, 'clear_every_step_cache': False, 'use_batch_p2p_comm': True}[0m
======================= Modified FLAGS detected =======================
FLAGS(name='FLAGS_cudnn_dir', current_value='/home/ldn/anaconda3/envs/paddle-test/lib/python3.10/site-packages/paddle/../nvidia/cudnn/lib', default_value='')
FLAGS(name='FLAGS_cudnn_deterministic', current_value=True, default_value=False)
FLAGS(name='FLAGS_selected_gpus', current_value='3', default_value='')
FLAGS(name='FLAGS_cusparse_dir', current_value='/home/ldn/anaconda3/envs/paddle-test/lib/python3.10/site-packages/paddle/../nvidia/cusparse/lib', default_value='')
FLAGS(name='FLAGS_nccl_dir', current_value='/home/ldn/anaconda3/envs/paddle-test/lib/python3.10/site-packages/paddle/../nvidia/nccl/lib', default_value='')
FLAGS(name='FLAGS_curand_dir', current_value='/home/ldn/anaconda3/envs/paddle-test/lib/python3.10/site-packages/paddle/../nvidia/curand/lib', default_value='')
FLAGS(name='FLAGS_cusolver_dir', current_value='/home/ldn/anaconda3/envs/paddle-test/lib/python3.10/site-packages/paddle/../nvidia/cusolver/lib', default_value='')
FLAGS(name='FLAGS_nvidia_package_dir', current_value='/home/ldn/anaconda3/envs/paddle-test/lib/python3.10/site-packages/paddle/../nvidia', default_value='')
FLAGS(name='FLAGS_cublas_dir', current_value='/home/ldn/anaconda3/envs/paddle-test/lib/python3.10/site-packages/paddle/../nvidia/cublas/lib', default_value='')
FLAGS(name='FLAGS_cupti_dir', current_value='/home/ldn/anaconda3/envs/paddle-test/lib/python3.10/site-packages/paddle/../nvidia/cuda_cupti/lib', default_value='')
FLAGS(name='FLAGS_embedding_deterministic', current_value=1, default_value=0)
=======================================================================
I0805 08:55:53.095208 2296433 tcp_utils.cc:130] Successfully connected to 127.0.0.1:55271
I0805 08:55:53.168788 2296433 process_group_nccl.cc:138] ProcessGroupNCCL pg_timeout_ 1800000
I0805 08:55:53.168815 2296433 process_group_nccl.cc:139] ProcessGroupNCCL nccl_comm_init_option_ 0
I0805 08:55:53.775254 2296433 process_group_nccl.cc:138] ProcessGroupNCCL pg_timeout_ 1800000
I0805 08:55:53.775285 2296433 process_group_nccl.cc:139] ProcessGroupNCCL nccl_comm_init_option_ 0
[2024-08-05 08:55:53,775] [    INFO] topology.py:357 - Total 4 pipe comm group(s) create successfully!
W0805 08:55:53.781599 2296433 gpu_resources.cc:119] Please NOTE: device: 3, GPU Compute Capability: 8.0, Driver API Version: 12.2, Runtime API Version: 12.2
W0805 08:55:53.782143 2296433 gpu_resources.cc:164] device: 3, cuDNN Version: 9.0.
W0805 08:55:53.782159 2296433 gpu_resources.cc:196] WARNING: device: 3. The installed Paddle is compiled with CUDA 12.3, but CUDA runtime version in your machine is 12.2, which may cause serious incompatible bug. Please recompile or reinstall Paddle with compatible CUDA version.
ubuntu:2296433:2296433 [3] NCCL INFO Bootstrap : Using eno1:10.3.242.26<0>
ubuntu:2296433:2296433 [3] NCCL INFO NET/Plugin : dlerror=libnccl-net.so: cannot open shared object file: No such file or directory No plugin found (libnccl-net.so), using internal implementation
ubuntu:2296433:2296433 [3] NCCL INFO cudaDriverVersion 12020
NCCL version 2.20.5+cuda12.4
ubuntu:2296433:2296674 [3] NCCL INFO NET/IB : No device found.
ubuntu:2296433:2296674 [3] NCCL INFO NET/Socket : Using [0]eno1:10.3.242.26<0> [1]br-8d7ae5405055:172.18.0.1<0>
ubuntu:2296433:2296674 [3] NCCL INFO Using non-device net plugin version 0
ubuntu:2296433:2296674 [3] NCCL INFO Using network Socket
ubuntu:2296433:2296674 [3] NCCL INFO comm 0x560c19f6f0c0 rank 0 nranks 2 cudaDev 3 nvmlDev 3 busId 57000 commId 0x53e9b5a4c2d76cb0 - Init START
ubuntu:2296433:2296674 [3] NCCL INFO Setting affinity for GPU 3 to ffffffff,00000000,ffffffff
ubuntu:2296433:2296674 [3] NCCL INFO comm 0x560c19f6f0c0 rank 0 nRanks 2 nNodes 1 localRanks 2 localRank 0 MNNVL 0
ubuntu:2296433:2296674 [3] NCCL INFO Channel 00/02 :    0   1
ubuntu:2296433:2296674 [3] NCCL INFO Channel 01/02 :    0   1
ubuntu:2296433:2296674 [3] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1
ubuntu:2296433:2296674 [3] NCCL INFO P2P Chunksize set to 131072
ubuntu:2296433:2296674 [3] NCCL INFO Channel 00 : 0[3] -> 1[7] via SHM/direct/direct
ubuntu:2296433:2296674 [3] NCCL INFO Channel 01 : 0[3] -> 1[7] via SHM/direct/direct
ubuntu:2296433:2296674 [3] NCCL INFO Connected all rings
ubuntu:2296433:2296674 [3] NCCL INFO Connected all trees
ubuntu:2296433:2296674 [3] NCCL INFO NCCL_ALGO set by environment to Tree
ubuntu:2296433:2296674 [3] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
ubuntu:2296433:2296674 [3] NCCL INFO 2 coll channels, 0 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
ubuntu:2296433:2296674 [3] NCCL INFO comm 0x560c19f6f0c0 rank 0 nranks 2 cudaDev 3 nvmlDev 3 busId 57000 commId 0x53e9b5a4c2d76cb0 - Init COMPLETE
[2024-08-05 08:55:57,381] [    INFO] topology.py:357 - Total 8 data comm group(s) create successfully!
I0805 08:55:57.382566 2296433 process_group_nccl.cc:138] ProcessGroupNCCL pg_timeout_ 1800000
I0805 08:55:57.382624 2296433 process_group_nccl.cc:139] ProcessGroupNCCL nccl_comm_init_option_ 0
[2024-08-05 08:55:57,382] [    INFO] topology.py:357 - Total 2 model comm group(s) create successfully!
[2024-08-05 08:55:57,383] [    INFO] topology.py:357 - Total 8 sharding comm group(s) create successfully!
I0805 08:55:57.383256 2296433 process_group_nccl.cc:138] ProcessGroupNCCL pg_timeout_ 1800000
I0805 08:55:57.383275 2296433 process_group_nccl.cc:139] ProcessGroupNCCL nccl_comm_init_option_ 0
I0805 08:55:57.383352 2296433 process_group_nccl.cc:138] ProcessGroupNCCL pg_timeout_ 1800000
I0805 08:55:57.383365 2296433 process_group_nccl.cc:139] ProcessGroupNCCL nccl_comm_init_option_ 0
[2024-08-05 08:55:57,383] [    INFO] topology.py:279 - HybridParallelInfo: rank_id: 3, mp_degree: 4, sharding_degree: 1, pp_degree: 2, dp_degree: 1, sep_degree: 1, mp_group: [0, 1, 2, 3],  sharding_group: [3], pp_group: [3, 7], dp_group: [3], sep:group: None, check/clip group: [0, 1, 2, 3, 4, 5, 6, 7]
[32m[2024-08-05 08:55:57,385] [    INFO][0m -     +==============================================================================+
    |                                                                              |
    |                         DistributedStrategy Overview                         |
    |                                                                              |
    +==============================================================================+
    |                        a_sync=True <-> a_sync_configs                        |
    +------------------------------------------------------------------------------+
    |                               k_steps                    -1                  |
    |                     max_merge_var_num                    1                   |
    |                       send_queue_size                    16                  |
    |               independent_recv_thread                  False                 |
    |         min_send_grad_num_before_recv                    1                   |
    |                      thread_pool_size                    1                   |
    |                       send_wait_times                    1                   |
    |               runtime_split_send_recv                  False                 |
    |                        launch_barrier                   True                 |
    |             heter_worker_device_guard                   cpu                  |
    |                        lr_decay_steps                    10                  |
    |                            use_ps_gpu                    0                   |
    +==============================================================================+
    |                    Environment Flags, Communication Flags                    |
    +------------------------------------------------------------------------------+
    |                                  mode                    1                   |
    |                               elastic                  False                 |
    |                                  auto                  False                 |
    |                   sync_nccl_allreduce                   True                 |
    |                         nccl_comm_num                    1                   |
    |            use_hierarchical_allreduce                  False                 |
    |   hierarchical_allreduce_inter_nranks                    1                   |
    |                       sync_batch_norm                  False                 |
    |                   fuse_all_reduce_ops                   True                 |
    |                  fuse_grad_size_in_MB                    32                  |
    |              fuse_grad_size_in_TFLOPS                   50.0                 |
    |               cudnn_exhaustive_search                  False                 |
    |             conv_workspace_size_limit                   512                  |
    |    cudnn_batchnorm_spatial_persistent                  False                 |
    |                        fp16_allreduce                  False                 |
    |               last_comm_group_size_MB                   1.0                  |
    |                find_unused_parameters                  False                 |
    |            without_graph_optimization                   True                 |
    |                 fuse_grad_size_in_num                    8                   |
    |                 calc_comm_same_stream                  False                 |
    |                                   asp                  False                 |
    |                       fuse_grad_merge                  False                 |
    |                             semi_auto                  False                 |
    |                            adam_d2sum                  False                 |
    |                           auto_search                  False                 |
    |                        heter_ccl_mode                  False                 |
    |                         is_fl_ps_mode                  False                 |
    |                      with_coordinator                  False                 |
    |                            split_data                   True                 |
    |                  downpour_table_param                    []                  |
    |                       fs_client_param                                        |
    |                    execution_strategy                                        |
    +==============================================================================+
    |                                Build Strategy                                |
    +------------------------------------------------------------------------------+
    |           enable_sequential_execution                  False                 |
    |              fuse_elewise_add_act_ops                  False                 |
    |                       fuse_bn_act_ops                  False                 |
    |              fuse_relu_depthwise_conv                  False                 |
    |                    fuse_broadcast_ops                  False                 |
    |                fuse_all_optimizer_ops                  False                 |
    |                        enable_inplace                  False                 |
    |     enable_backward_optimizer_op_deps                   True                 |
    |                 cache_runtime_context                  False                 |
    |                   fuse_bn_add_act_ops                   True                 |
    |                    enable_auto_fusion                  False                 |
    |                          enable_addto                  False                 |
    |                      fix_op_run_order                  False                 |
    |              allow_cuda_graph_capture                  False                 |
    |                       reduce_strategy                    0                   |
    |                    fuse_gemm_epilogue                  False                 |
    |                   debug_graphviz_path                                        |
    |                       fused_attention                  False                 |
    |                     fused_feedforward                  False                 |
    |            fuse_dot_product_attention                  False                 |
    |                          fuse_resunit                  False                 |
    +==============================================================================+
[0m
[32m[2024-08-05 08:55:57,387] [    INFO][0m - The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).[0m
[35m[2024-08-05 08:55:57,387] [   DEBUG][0m - ============================================================[0m
[35m[2024-08-05 08:55:57,387] [   DEBUG][0m -      Model Configuration Arguments      [0m
[35m[2024-08-05 08:55:57,387] [   DEBUG][0m - paddle commit id              : a842a0f40f6111fb0c2df218130d0560aa747bc8[0m
[35m[2024-08-05 08:55:57,388] [   DEBUG][0m - paddlenlp commit id           : 272a4b04992242b0a3ca250c6a4b2b5d5c660f66[0m
[35m[2024-08-05 08:55:57,388] [   DEBUG][0m - aistudio_repo_id              : None[0m
[35m[2024-08-05 08:55:57,388] [   DEBUG][0m - aistudio_repo_license         : Apache License 2.0[0m
[35m[2024-08-05 08:55:57,388] [   DEBUG][0m - aistudio_repo_private         : True[0m
[35m[2024-08-05 08:55:57,388] [   DEBUG][0m - aistudio_token                : None[0m
[35m[2024-08-05 08:55:57,389] [   DEBUG][0m - attention_probs_dropout_prob  : 0.1[0m
[35m[2024-08-05 08:55:57,389] [   DEBUG][0m - continue_training             : True[0m
[35m[2024-08-05 08:55:57,389] [   DEBUG][0m - flash_mask                    : False[0m
[35m[2024-08-05 08:55:57,389] [   DEBUG][0m - from_aistudio                 : False[0m
[35m[2024-08-05 08:55:57,389] [   DEBUG][0m - fuse_attention_ffn            : None[0m
[35m[2024-08-05 08:55:57,390] [   DEBUG][0m - fuse_attention_qkv            : None[0m
[35m[2024-08-05 08:55:57,390] [   DEBUG][0m - hidden_dropout_prob           : 0.1[0m
[35m[2024-08-05 08:55:57,390] [   DEBUG][0m - lora                          : True[0m
[35m[2024-08-05 08:55:57,390] [   DEBUG][0m - lora_path                     : None[0m
[35m[2024-08-05 08:55:57,390] [   DEBUG][0m - lora_plus_scale               : 1.0[0m
[35m[2024-08-05 08:55:57,390] [   DEBUG][0m - lora_rank                     : 8[0m
[35m[2024-08-05 08:55:57,391] [   DEBUG][0m - model_name_or_path            : __internal_testing__/unified-ckpt-llama-170m-for-peft[0m
[35m[2024-08-05 08:55:57,391] [   DEBUG][0m - neftune                       : False[0m
[35m[2024-08-05 08:55:57,391] [   DEBUG][0m - neftune_noise_alpha           : 5.0[0m
[35m[2024-08-05 08:55:57,391] [   DEBUG][0m - num_prefix_tokens             : 128[0m
[35m[2024-08-05 08:55:57,391] [   DEBUG][0m - pissa                         : False[0m
[35m[2024-08-05 08:55:57,391] [   DEBUG][0m - prefix_path                   : None[0m
[35m[2024-08-05 08:55:57,392] [   DEBUG][0m - prefix_tuning                 : False[0m
[35m[2024-08-05 08:55:57,392] [   DEBUG][0m - reft                          : False[0m
[35m[2024-08-05 08:55:57,392] [   DEBUG][0m - rslora                        : False[0m
[35m[2024-08-05 08:55:57,392] [   DEBUG][0m - save_to_aistudio              : False[0m
[35m[2024-08-05 08:55:57,392] [   DEBUG][0m - tokenizer_name_or_path        : None[0m
[35m[2024-08-05 08:55:57,392] [   DEBUG][0m - use_fast_layer_norm           : False[0m
[35m[2024-08-05 08:55:57,393] [   DEBUG][0m - use_quick_lora                : False[0m
[35m[2024-08-05 08:55:57,393] [   DEBUG][0m - vera                          : False[0m
[35m[2024-08-05 08:55:57,393] [   DEBUG][0m - vera_rank                     : 8[0m
[35m[2024-08-05 08:55:57,393] [   DEBUG][0m - weight_blocksize              : 64[0m
[35m[2024-08-05 08:55:57,393] [   DEBUG][0m - weight_double_quant           : False[0m
[35m[2024-08-05 08:55:57,393] [   DEBUG][0m - weight_double_quant_block_size: 256[0m
[35m[2024-08-05 08:55:57,394] [   DEBUG][0m - weight_quantize_algo          : None[0m
[35m[2024-08-05 08:55:57,394] [   DEBUG][0m - [0m
[35m[2024-08-05 08:55:57,394] [   DEBUG][0m - ============================================================[0m
[35m[2024-08-05 08:55:57,394] [   DEBUG][0m -       Data Configuration Arguments      [0m
[35m[2024-08-05 08:55:57,394] [   DEBUG][0m - paddle commit id              : a842a0f40f6111fb0c2df218130d0560aa747bc8[0m
[35m[2024-08-05 08:55:57,394] [   DEBUG][0m - paddlenlp commit id           : 272a4b04992242b0a3ca250c6a4b2b5d5c660f66[0m
[35m[2024-08-05 08:55:57,395] [   DEBUG][0m - chat_template                 : None[0m
[35m[2024-08-05 08:55:57,395] [   DEBUG][0m - dataset_name_or_path          : ./unified_checkpoint/peft_input/data/[0m
[35m[2024-08-05 08:55:57,395] [   DEBUG][0m - eval_with_do_generation       : False[0m
[35m[2024-08-05 08:55:57,395] [   DEBUG][0m - intokens                      : None[0m
[35m[2024-08-05 08:55:57,395] [   DEBUG][0m - lazy                          : False[0m
[35m[2024-08-05 08:55:57,396] [   DEBUG][0m - max_length                    : 2048[0m
[35m[2024-08-05 08:55:57,396] [   DEBUG][0m - pad_to_multiple_of            : None[0m
[35m[2024-08-05 08:55:57,396] [   DEBUG][0m - save_generation_output        : False[0m
[35m[2024-08-05 08:55:57,396] [   DEBUG][0m - src_length                    : 1024[0m
[35m[2024-08-05 08:55:57,396] [   DEBUG][0m - task_name                     : None[0m
[35m[2024-08-05 08:55:57,396] [   DEBUG][0m - task_name_or_path             : None[0m
[35m[2024-08-05 08:55:57,397] [   DEBUG][0m - zero_padding                  : False[0m
[35m[2024-08-05 08:55:57,397] [   DEBUG][0m - [0m
[35m[2024-08-05 08:55:57,397] [   DEBUG][0m - ============================================================[0m
[35m[2024-08-05 08:55:57,397] [   DEBUG][0m -      Quant Configuration Arguments      [0m
[35m[2024-08-05 08:55:57,397] [   DEBUG][0m - paddle commit id              : a842a0f40f6111fb0c2df218130d0560aa747bc8[0m
[35m[2024-08-05 08:55:57,397] [   DEBUG][0m - paddlenlp commit id           : 272a4b04992242b0a3ca250c6a4b2b5d5c660f66[0m
[35m[2024-08-05 08:55:57,398] [   DEBUG][0m - auto_clip                     : False[0m
[35m[2024-08-05 08:55:57,398] [   DEBUG][0m - autoclip_step                 : 8[0m
[35m[2024-08-05 08:55:57,398] [   DEBUG][0m - awq_step                      : 8[0m
[35m[2024-08-05 08:55:57,398] [   DEBUG][0m - do_awq                        : False[0m
[35m[2024-08-05 08:55:57,398] [   DEBUG][0m - do_gptq                       : False[0m
[35m[2024-08-05 08:55:57,398] [   DEBUG][0m - do_ptq                        : False[0m
[35m[2024-08-05 08:55:57,399] [   DEBUG][0m - do_qat                        : False[0m
[35m[2024-08-05 08:55:57,399] [   DEBUG][0m - gptq_step                     : 8[0m
[35m[2024-08-05 08:55:57,399] [   DEBUG][0m - ptq_step                      : 32[0m
[35m[2024-08-05 08:55:57,399] [   DEBUG][0m - quant_type                    : a8w8[0m
[35m[2024-08-05 08:55:57,399] [   DEBUG][0m - shift                         : False[0m
[35m[2024-08-05 08:55:57,399] [   DEBUG][0m - shift_all_linears             : False[0m
[35m[2024-08-05 08:55:57,400] [   DEBUG][0m - shift_sampler                 : ema[0m
[35m[2024-08-05 08:55:57,400] [   DEBUG][0m - shift_step                    : 32[0m
[35m[2024-08-05 08:55:57,400] [   DEBUG][0m - smooth                        : False[0m
[35m[2024-08-05 08:55:57,400] [   DEBUG][0m - smooth_all_linears            : False[0m
[35m[2024-08-05 08:55:57,400] [   DEBUG][0m - smooth_k_piece                : 3[0m
[35m[2024-08-05 08:55:57,400] [   DEBUG][0m - smooth_piecewise_search       : False[0m
[35m[2024-08-05 08:55:57,401] [   DEBUG][0m - smooth_sampler                : none[0m
[35m[2024-08-05 08:55:57,401] [   DEBUG][0m - smooth_search_piece           : False[0m
[35m[2024-08-05 08:55:57,401] [   DEBUG][0m - smooth_step                   : 32[0m
[35m[2024-08-05 08:55:57,401] [   DEBUG][0m - weight_quant_method           : abs_max_channel_wise[0m
[35m[2024-08-05 08:55:57,401] [   DEBUG][0m - [0m
[35m[2024-08-05 08:55:57,401] [   DEBUG][0m - ============================================================[0m
[35m[2024-08-05 08:55:57,402] [   DEBUG][0m -    Generation Configuration Arguments   [0m
[35m[2024-08-05 08:55:57,402] [   DEBUG][0m - paddle commit id              : a842a0f40f6111fb0c2df218130d0560aa747bc8[0m
[35m[2024-08-05 08:55:57,402] [   DEBUG][0m - paddlenlp commit id           : 272a4b04992242b0a3ca250c6a4b2b5d5c660f66[0m
[35m[2024-08-05 08:55:57,402] [   DEBUG][0m - top_k                         : 1[0m
[35m[2024-08-05 08:55:57,402] [   DEBUG][0m - top_p                         : 1.0[0m
[35m[2024-08-05 08:55:57,403] [   DEBUG][0m - [0m
[35m[2024-08-05 08:55:57,403] [   DEBUG][0m - ============================================================[0m
[35m[2024-08-05 08:55:57,403] [   DEBUG][0m -       Reft Configuration Arguments      [0m
[35m[2024-08-05 08:55:57,403] [   DEBUG][0m - paddle commit id              : a842a0f40f6111fb0c2df218130d0560aa747bc8[0m
[35m[2024-08-05 08:55:57,403] [   DEBUG][0m - paddlenlp commit id           : 272a4b04992242b0a3ca250c6a4b2b5d5c660f66[0m
[35m[2024-08-05 08:55:57,404] [   DEBUG][0m - act_fn                        : None[0m
[35m[2024-08-05 08:55:57,404] [   DEBUG][0m - add_bias                      : False[0m
[35m[2024-08-05 08:55:57,404] [   DEBUG][0m - dropout                       : 0.0[0m
[35m[2024-08-05 08:55:57,404] [   DEBUG][0m - intervention_type             : LoreftIntervention[0m
[35m[2024-08-05 08:55:57,404] [   DEBUG][0m - layers                        : all[0m
[35m[2024-08-05 08:55:57,404] [   DEBUG][0m - position                      : f7+l7[0m
[35m[2024-08-05 08:55:57,405] [   DEBUG][0m - rank                          : 8[0m
[35m[2024-08-05 08:55:57,405] [   DEBUG][0m - [0m
[33m[2024-08-05 08:55:57,405] [ WARNING][0m - Process rank: 3, device: gpu, world_size: 8, distributed training: True, 16-bits training: True[0m
[32m[2024-08-05 08:55:57,406] [    INFO][0m - Checkpoint detected, resuming training at ./unified_checkpoint/checkpoints/llama_lora_ckpts/checkpoint-10. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch.[0m
[32m[2024-08-05 08:55:57,407] [    INFO][0m - We are using <class 'paddlenlp.transformers.llama.configuration.LlamaConfig'> to load '__internal_testing__/unified-ckpt-llama-170m-for-peft'.[0m
[32m[2024-08-05 08:55:57,407] [    INFO][0m - Loading configuration file /home/ldn/.paddlenlp/models/__internal_testing__/unified-ckpt-llama-170m-for-peft/config.json[0m
[32m[2024-08-05 08:55:57,412] [    INFO][0m - Final model config: LlamaConfig {
  "alibi": false,
  "architectures": [
    "LlamaForCausalLM"
  ],
  "bos_token_id": 1,
  "dtype": "float16",
  "eos_token_id": 2,
  "hidden_size": 1024,
  "immediate_clear_past_key_value": false,
  "initializer_range": 0.02,
  "intermediate_size": 2816,
  "long_sequence_init_args": {},
  "long_sequence_strategy_name": null,
  "long_sequence_strategy_type": null,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 8,
  "num_hidden_layers": 8,
  "num_key_value_heads": 8,
  "pad_token_id": 0,
  "paddlenlp_version": "3.0.0b0.post20240805",
  "pipeline_parallel_degree": 2,
  "recompute": true,
  "rms_norm_eps": 1e-06,
  "rope_scaling_factor": 1.0,
  "rope_scaling_type": null,
  "rope_theta": 10000.0,
  "seq_length": 2048,
  "tensor_parallel_degree": 4,
  "tensor_parallel_output": false,
  "tensor_parallel_rank": 3,
  "tie_word_embeddings": false,
  "use_fast_layer_norm": false,
  "use_flash_attention_for_generation": false,
  "use_last_token_for_generation": false,
  "use_long_sequence_strategies": false,
  "vocab_size": 32000
}
[0m
[32m[2024-08-05 08:55:57,413] [    INFO][0m - We are using <class 'paddlenlp.transformers.llama.modeling_pp.LlamaForCausalLMPipe'> to load '__internal_testing__/unified-ckpt-llama-170m-for-peft'.[0m
[32m[2024-08-05 08:55:57,413] [    INFO][0m - Loading weights file from cache at /home/ldn/.paddlenlp/models/__internal_testing__/unified-ckpt-llama-170m-for-peft/model_state.pdparams[0m
[32m[2024-08-05 08:55:57,694] [    INFO][0m - Starting to convert orignal state_dict to tensor parallel state_dict.[0m
[32m[2024-08-05 08:55:57,824] [    INFO][0m - Loaded weights file from disk, setting weights to model.[0m
[2024-08-05 08:55:57,825] [    INFO] pp_layers.py:607 - start segment network..
[2024-08-05 08:55:57,825] [    INFO] pp_layers.py:613 - segment with method: layer:LlamaDecoderLayer; result: 0, 5, 11
[2024-08-05 08:55:57,825] [    INFO] pp_layers.py:629 - stage=0, global_rank=3 ,layer_number=5
[2024-08-05 08:55:57,825] [    INFO] pp_layers.py:634 - 0: LlamaEmbeddingPipe
[2024-08-05 08:55:57,825] [    INFO] pp_layers.py:634 - 1: LlamaDecoderLayerPipe
[2024-08-05 08:55:57,825] [    INFO] pp_layers.py:634 - 2: LlamaDecoderLayerPipe
[2024-08-05 08:55:57,825] [    INFO] pp_layers.py:634 - 3: LlamaDecoderLayerPipe
[2024-08-05 08:55:57,825] [    INFO] pp_layers.py:634 - 4: LlamaDecoderLayerPipe
[2024-08-05 08:55:57,825] [    INFO] pp_layers.py:629 - stage=1, global_rank=3 ,layer_number=6
[2024-08-05 08:55:57,825] [    INFO] pp_layers.py:634 - 5: LlamaDecoderLayerPipe
[2024-08-05 08:55:57,825] [    INFO] pp_layers.py:634 - 6: LlamaDecoderLayerPipe
[2024-08-05 08:55:57,825] [    INFO] pp_layers.py:634 - 7: LlamaDecoderLayerPipe
[2024-08-05 08:55:57,825] [    INFO] pp_layers.py:634 - 8: LlamaDecoderLayerPipe
[2024-08-05 08:55:57,825] [    INFO] pp_layers.py:634 - 9: LlamaRMSNormPipe
[2024-08-05 08:55:57,825] [    INFO] pp_layers.py:634 - 10: LlamaLMHead
[2024-08-05 08:55:57,825] [    INFO] pp_layers.py:656 - loss: LlamaPretrainingCriterion
[2024-08-05 08:55:58,265] [    INFO] pp_layers.py:706 - flush 5 of layers into run_function
[33m[2024-08-05 08:55:58,439] [ WARNING][0m - Some weights of the model checkpoint at __internal_testing__/unified-ckpt-llama-170m-for-peft were not used when initializing LlamaForCausalLMPipe: ['llama.layers.4.input_layernorm.weight', 'llama.layers.4.mlp.down_proj.weight', 'llama.layers.4.mlp.gate_proj.weight', 'llama.layers.4.mlp.up_proj.weight', 'llama.layers.4.post_attention_layernorm.weight', 'llama.layers.4.self_attn.k_proj.weight', 'llama.layers.4.self_attn.o_proj.weight', 'llama.layers.4.self_attn.q_proj.weight', 'llama.layers.4.self_attn.v_proj.weight', 'llama.layers.5.input_layernorm.weight', 'llama.layers.5.mlp.down_proj.weight', 'llama.layers.5.mlp.gate_proj.weight', 'llama.layers.5.mlp.up_proj.weight', 'llama.layers.5.post_attention_layernorm.weight', 'llama.layers.5.self_attn.k_proj.weight', 'llama.layers.5.self_attn.o_proj.weight', 'llama.layers.5.self_attn.q_proj.weight', 'llama.layers.5.self_attn.v_proj.weight', 'llama.layers.6.input_layernorm.weight', 'llama.layers.6.mlp.down_proj.weight', 'llama.layers.6.mlp.gate_proj.weight', 'llama.layers.6.mlp.up_proj.weight', 'llama.layers.6.post_attention_layernorm.weight', 'llama.layers.6.self_attn.k_proj.weight', 'llama.layers.6.self_attn.o_proj.weight', 'llama.layers.6.self_attn.q_proj.weight', 'llama.layers.6.self_attn.v_proj.weight', 'llama.layers.7.input_layernorm.weight', 'llama.layers.7.mlp.down_proj.weight', 'llama.layers.7.mlp.gate_proj.weight', 'llama.layers.7.mlp.up_proj.weight', 'llama.layers.7.post_attention_layernorm.weight', 'llama.layers.7.self_attn.k_proj.weight', 'llama.layers.7.self_attn.o_proj.weight', 'llama.layers.7.self_attn.q_proj.weight', 'llama.layers.7.self_attn.v_proj.weight', 'llama.norm.weight', 'lm_head.weight']
- This IS expected if you are initializing LlamaForCausalLMPipe from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing LlamaForCausalLMPipe from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).[0m
[32m[2024-08-05 08:55:58,439] [    INFO][0m - All the weights of LlamaForCausalLMPipe were initialized from the model checkpoint at __internal_testing__/unified-ckpt-llama-170m-for-peft.
If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLMPipe for predictions without further training.[0m
[32m[2024-08-05 08:55:58,444] [    INFO][0m - We are using <class 'paddlenlp.transformers.llama.tokenizer.LlamaTokenizer'> to load '__internal_testing__/unified-ckpt-llama-170m-for-peft'.[0m
[32m[2024-08-05 08:55:59,554] [    INFO][0m - Mark only lora and trainable_module as trainable.[0m
[35m[2024-08-05 08:55:59,557] [   DEBUG][0m - Frozen parameters: 2.10e+07 || Trainable parameters:3.30e+05 || Total parameters:2.14e+07|| Trainable:1.54%[0m
[32m[2024-08-05 08:55:59,559] [    INFO][0m - The global seed is set to 42, local seed is set to 53 and random seed is set to 42.[0m
[32m[2024-08-05 08:55:59,591] [    INFO][0m - max_steps is given, it will override any value given in num_train_epochs[0m
[32m[2024-08-05 08:55:59,591] [    INFO][0m - Using half precision[0m
[35m[2024-08-05 08:55:59,608] [   DEBUG][0m - ============================================================[0m
[35m[2024-08-05 08:55:59,608] [   DEBUG][0m -     Training Configuration Arguments    [0m
[35m[2024-08-05 08:55:59,608] [   DEBUG][0m - paddle commit id              : a842a0f40f6111fb0c2df218130d0560aa747bc8[0m
[35m[2024-08-05 08:55:59,608] [   DEBUG][0m - paddlenlp commit id           : 272a4b04992242b0a3ca250c6a4b2b5d5c660f66[0m
[35m[2024-08-05 08:55:59,608] [   DEBUG][0m - _no_sync_in_gradient_accumulation: True[0m
[35m[2024-08-05 08:55:59,608] [   DEBUG][0m - adam_beta1                    : 0.9[0m
[35m[2024-08-05 08:55:59,608] [   DEBUG][0m - adam_beta2                    : 0.999[0m
[35m[2024-08-05 08:55:59,608] [   DEBUG][0m - adam_epsilon                  : 1e-08[0m
[35m[2024-08-05 08:55:59,608] [   DEBUG][0m - amp_custom_black_list         : None[0m
[35m[2024-08-05 08:55:59,608] [   DEBUG][0m - amp_custom_white_list         : None[0m
[35m[2024-08-05 08:55:59,609] [   DEBUG][0m - amp_master_grad               : False[0m
[35m[2024-08-05 08:55:59,609] [   DEBUG][0m - autotuner_benchmark           : False[0m
[35m[2024-08-05 08:55:59,609] [   DEBUG][0m - benchmark                     : False[0m
[35m[2024-08-05 08:55:59,609] [   DEBUG][0m - bf16                          : False[0m
[35m[2024-08-05 08:55:59,609] [   DEBUG][0m - bf16_full_eval                : False[0m
[35m[2024-08-05 08:55:59,609] [   DEBUG][0m - context_parallel_degree       : 1[0m
[35m[2024-08-05 08:55:59,609] [   DEBUG][0m - current_device                : gpu:3[0m
[35m[2024-08-05 08:55:59,609] [   DEBUG][0m - data_parallel_config          : [0m
[35m[2024-08-05 08:55:59,609] [   DEBUG][0m - data_parallel_degree          : 1[0m
[35m[2024-08-05 08:55:59,609] [   DEBUG][0m - data_parallel_rank            : 0[0m
[35m[2024-08-05 08:55:59,609] [   DEBUG][0m - dataloader_drop_last          : False[0m
[35m[2024-08-05 08:55:59,609] [   DEBUG][0m - dataloader_num_workers        : 0[0m
[35m[2024-08-05 08:55:59,609] [   DEBUG][0m - dataset_rank                  : 0[0m
[35m[2024-08-05 08:55:59,609] [   DEBUG][0m - dataset_world_size            : 1[0m
[35m[2024-08-05 08:55:59,609] [   DEBUG][0m - decay_steps                   : 0[0m
[35m[2024-08-05 08:55:59,609] [   DEBUG][0m - device                        : gpu[0m
[35m[2024-08-05 08:55:59,609] [   DEBUG][0m - disable_tqdm                  : True[0m
[35m[2024-08-05 08:55:59,610] [   DEBUG][0m - distributed_dataloader        : False[0m
[35m[2024-08-05 08:55:59,610] [   DEBUG][0m - do_eval                       : False[0m
[35m[2024-08-05 08:55:59,610] [   DEBUG][0m - do_export                     : False[0m
[35m[2024-08-05 08:55:59,610] [   DEBUG][0m - do_predict                    : False[0m
[35m[2024-08-05 08:55:59,610] [   DEBUG][0m - do_train                      : True[0m
[35m[2024-08-05 08:55:59,610] [   DEBUG][0m - enable_auto_parallel          : False[0m
[35m[2024-08-05 08:55:59,610] [   DEBUG][0m - eval_accumulation_steps       : 16[0m
[35m[2024-08-05 08:55:59,610] [   DEBUG][0m - eval_batch_size               : 8[0m
[35m[2024-08-05 08:55:59,610] [   DEBUG][0m - eval_steps                    : None[0m
[35m[2024-08-05 08:55:59,610] [   DEBUG][0m - evaluation_strategy           : IntervalStrategy.NO[0m
[35m[2024-08-05 08:55:59,610] [   DEBUG][0m - flatten_param_grads           : False[0m
[35m[2024-08-05 08:55:59,610] [   DEBUG][0m - force_reshard_pp              : False[0m
[35m[2024-08-05 08:55:59,610] [   DEBUG][0m - fp16                          : True[0m
[35m[2024-08-05 08:55:59,610] [   DEBUG][0m - fp16_full_eval                : False[0m
[35m[2024-08-05 08:55:59,610] [   DEBUG][0m - fp16_opt_level                : O2[0m
[35m[2024-08-05 08:55:59,610] [   DEBUG][0m - fuse_sequence_parallel_allreduce: False[0m
[35m[2024-08-05 08:55:59,610] [   DEBUG][0m - gradient_accumulation_steps   : 8[0m
[35m[2024-08-05 08:55:59,610] [   DEBUG][0m - greater_is_better             : None[0m
[35m[2024-08-05 08:55:59,611] [   DEBUG][0m - hybrid_parallel_topo_order    : pp_first[0m
[35m[2024-08-05 08:55:59,611] [   DEBUG][0m - ignore_data_skip              : False[0m
[35m[2024-08-05 08:55:59,611] [   DEBUG][0m - ignore_load_lr_and_optim      : False[0m
[35m[2024-08-05 08:55:59,611] [   DEBUG][0m - ignore_save_lr_and_optim      : False[0m
[35m[2024-08-05 08:55:59,611] [   DEBUG][0m - label_names                   : None[0m
[35m[2024-08-05 08:55:59,611] [   DEBUG][0m - lazy_data_processing          : True[0m
[35m[2024-08-05 08:55:59,611] [   DEBUG][0m - learning_rate                 : 0.0003[0m
[35m[2024-08-05 08:55:59,611] [   DEBUG][0m - load_best_model_at_end        : False[0m
[35m[2024-08-05 08:55:59,611] [   DEBUG][0m - load_sharded_model            : False[0m
[35m[2024-08-05 08:55:59,611] [   DEBUG][0m - local_process_index           : 3[0m
[35m[2024-08-05 08:55:59,611] [   DEBUG][0m - local_rank                    : 3[0m
[35m[2024-08-05 08:55:59,611] [   DEBUG][0m - log_level                     : -1[0m
[35m[2024-08-05 08:55:59,611] [   DEBUG][0m - log_level_replica             : -1[0m
[35m[2024-08-05 08:55:59,611] [   DEBUG][0m - log_on_each_node              : True[0m
[35m[2024-08-05 08:55:59,611] [   DEBUG][0m - logging_dir                   : ./unified_checkpoint/checkpoints/llama_lora_ckpts/runs/Aug05_08-55-53_ubuntu[0m
[35m[2024-08-05 08:55:59,611] [   DEBUG][0m - logging_first_step            : False[0m
[35m[2024-08-05 08:55:59,611] [   DEBUG][0m - logging_steps                 : 1[0m
[35m[2024-08-05 08:55:59,611] [   DEBUG][0m - logging_strategy              : IntervalStrategy.STEPS[0m
[35m[2024-08-05 08:55:59,612] [   DEBUG][0m - logical_process_index         : 3[0m
[35m[2024-08-05 08:55:59,612] [   DEBUG][0m - lr_end                        : 1e-07[0m
[35m[2024-08-05 08:55:59,612] [   DEBUG][0m - lr_scheduler_type             : SchedulerType.LINEAR[0m
[35m[2024-08-05 08:55:59,612] [   DEBUG][0m - max_evaluate_steps            : -1[0m
[35m[2024-08-05 08:55:59,612] [   DEBUG][0m - max_grad_norm                 : 1.0[0m
[35m[2024-08-05 08:55:59,612] [   DEBUG][0m - max_steps                     : 15[0m
[35m[2024-08-05 08:55:59,612] [   DEBUG][0m - metric_for_best_model         : None[0m
[35m[2024-08-05 08:55:59,612] [   DEBUG][0m - minimum_eval_times            : None[0m
[35m[2024-08-05 08:55:59,612] [   DEBUG][0m - no_cuda                       : False[0m
[35m[2024-08-05 08:55:59,612] [   DEBUG][0m - no_recompute_layers           : None[0m
[35m[2024-08-05 08:55:59,612] [   DEBUG][0m - num_cycles                    : 0.5[0m
[35m[2024-08-05 08:55:59,612] [   DEBUG][0m - num_train_epochs              : 1.0[0m
[35m[2024-08-05 08:55:59,612] [   DEBUG][0m - optim                         : OptimizerNames.ADAMW[0m
[35m[2024-08-05 08:55:59,612] [   DEBUG][0m - optimizer_name_suffix         : tp03_pp00[0m
[35m[2024-08-05 08:55:59,612] [   DEBUG][0m - output_dir                    : ./unified_checkpoint/checkpoints/llama_lora_ckpts[0m
[35m[2024-08-05 08:55:59,612] [   DEBUG][0m - overwrite_output_dir          : False[0m
[35m[2024-08-05 08:55:59,612] [   DEBUG][0m - past_index                    : -1[0m
[35m[2024-08-05 08:55:59,613] [   DEBUG][0m - per_device_eval_batch_size    : 8[0m
[35m[2024-08-05 08:55:59,613] [   DEBUG][0m - per_device_train_batch_size   : 4[0m
[35m[2024-08-05 08:55:59,613] [   DEBUG][0m - pipeline_parallel_config      : [0m
[35m[2024-08-05 08:55:59,613] [   DEBUG][0m - pipeline_parallel_degree      : 2[0m
[35m[2024-08-05 08:55:59,613] [   DEBUG][0m - pipeline_parallel_rank        : 0[0m
[35m[2024-08-05 08:55:59,613] [   DEBUG][0m - power                         : 1.0[0m
[35m[2024-08-05 08:55:59,613] [   DEBUG][0m - pp_recompute_interval         : 1[0m
[35m[2024-08-05 08:55:59,613] [   DEBUG][0m - prediction_loss_only          : False[0m
[35m[2024-08-05 08:55:59,613] [   DEBUG][0m - process_index                 : 3[0m
[35m[2024-08-05 08:55:59,613] [   DEBUG][0m - recompute                     : True[0m
[35m[2024-08-05 08:55:59,613] [   DEBUG][0m - recompute_granularity         : full[0m
[35m[2024-08-05 08:55:59,613] [   DEBUG][0m - recompute_use_reentrant       : False[0m
[35m[2024-08-05 08:55:59,613] [   DEBUG][0m - remove_unused_columns         : True[0m
[35m[2024-08-05 08:55:59,613] [   DEBUG][0m - report_to                     : ['visualdl'][0m
[35m[2024-08-05 08:55:59,613] [   DEBUG][0m - resume_from_checkpoint        : None[0m
[35m[2024-08-05 08:55:59,613] [   DEBUG][0m - run_name                      : ./unified_checkpoint/checkpoints/llama_lora_ckpts[0m
[35m[2024-08-05 08:55:59,613] [   DEBUG][0m - save_on_each_node             : False[0m
[35m[2024-08-05 08:55:59,613] [   DEBUG][0m - save_sharded_model            : False[0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - save_steps                    : 10[0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - save_strategy                 : IntervalStrategy.STEPS[0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - save_total_limit              : 1[0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - scale_loss                    : 32768[0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - seed                          : 42[0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - sep_parallel_degree           : 1[0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - sequence_parallel             : False[0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - sequence_parallel_config      : [0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - sharding                      : [][0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - sharding_degree               : -1[0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - sharding_parallel_config      : [0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - sharding_parallel_degree      : 1[0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - sharding_parallel_rank        : 0[0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - should_load_dataset           : True[0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - should_load_sharding_stage1_model: False[0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - should_log                    : False[0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - should_save                   : False[0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - should_save_model_state       : True[0m
[35m[2024-08-05 08:55:59,614] [   DEBUG][0m - should_save_sharding_stage1_model: False[0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - skip_memory_metrics           : True[0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - skip_profile_timer            : True[0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - tensor_parallel_config        : [0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - tensor_parallel_degree        : 4[0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - tensor_parallel_output        : False[0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - tensor_parallel_rank          : 3[0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - to_static                     : False[0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - train_batch_size              : 4[0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - unified_checkpoint            : True[0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - unified_checkpoint_config     : [''][0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - use_async_save                : False[0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - use_expert_parallel           : False[0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - use_flash_attention           : False[0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - use_fused_dropout_add         : False[0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - use_fused_linear              : False[0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - use_fused_rms_norm            : False[0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - use_fused_rope                : False[0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - use_hybrid_parallel           : True[0m
[35m[2024-08-05 08:55:59,615] [   DEBUG][0m - virtual_pp_degree             : 1[0m
[35m[2024-08-05 08:55:59,616] [   DEBUG][0m - wandb_api_key                 : None[0m
[35m[2024-08-05 08:55:59,616] [   DEBUG][0m - warmup_ratio                  : 0.0[0m
[35m[2024-08-05 08:55:59,616] [   DEBUG][0m - warmup_steps                  : 30[0m
[35m[2024-08-05 08:55:59,616] [   DEBUG][0m - weight_decay                  : 0.0[0m
[35m[2024-08-05 08:55:59,616] [   DEBUG][0m - weight_name_suffix            : tp03_pp00[0m
[35m[2024-08-05 08:55:59,616] [   DEBUG][0m - world_size                    : 8[0m
[35m[2024-08-05 08:55:59,616] [   DEBUG][0m - [0m
[32m[2024-08-05 08:55:59,616] [    INFO][0m - Starting training from resume_from_checkpoint : ./unified_checkpoint/checkpoints/llama_lora_ckpts/checkpoint-10[0m
ubuntu:2296433:2296827 [3] NCCL INFO Using non-device net plugin version 0
ubuntu:2296433:2296827 [3] NCCL INFO Using network Socket
ubuntu:2296433:2296827 [3] NCCL INFO comm 0x560c1b31f5a0 rank 3 nranks 8 cudaDev 3 nvmlDev 3 busId 57000 commId 0x36a8e8104267bd16 - Init START
ubuntu:2296433:2296827 [3] NCCL INFO Setting affinity for GPU 3 to ffffffff,00000000,ffffffff
ubuntu:2296433:2296827 [3] NCCL INFO NVLS multicast support is not available on dev 3
ubuntu:2296433:2296827 [3] NCCL INFO comm 0x560c1b31f5a0 rank 3 nRanks 8 nNodes 1 localRanks 8 localRank 3 MNNVL 0
ubuntu:2296433:2296827 [3] NCCL INFO Trees [0] 4/-1/-1->3->2 [1] 4/-1/-1->3->2
ubuntu:2296433:2296827 [3] NCCL INFO P2P Chunksize set to 524288
ubuntu:2296433:2296827 [3] NCCL INFO Channel 00/0 : 3[3] -> 1[1] via P2P/CUMEM
ubuntu:2296433:2296827 [3] NCCL INFO Channel 01/0 : 3[3] -> 1[1] via P2P/CUMEM
ubuntu:2296433:2296827 [3] NCCL INFO Connected all rings
ubuntu:2296433:2296827 [3] NCCL INFO Channel 00 : 3[3] -> 4[4] via SHM/direct/direct
ubuntu:2296433:2296827 [3] NCCL INFO Channel 01 : 3[3] -> 4[4] via SHM/direct/direct
ubuntu:2296433:2296827 [3] NCCL INFO Channel 00/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2296433:2296827 [3] NCCL INFO Channel 01/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2296433:2296827 [3] NCCL INFO Connected all trees
ubuntu:2296433:2296827 [3] NCCL INFO NCCL_ALGO set by environment to Tree
ubuntu:2296433:2296827 [3] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 512 | 512
ubuntu:2296433:2296827 [3] NCCL INFO 2 coll channels, 0 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
ubuntu:2296433:2296827 [3] NCCL INFO comm 0x560c1b31f5a0 rank 3 nranks 8 cudaDev 3 nvmlDev 3 busId 57000 commId 0x36a8e8104267bd16 - Init COMPLETE
ubuntu:2296433:2296913 [3] NCCL INFO Using non-device net plugin version 0
ubuntu:2296433:2296913 [3] NCCL INFO Using network Socket
ubuntu:2296433:2296913 [3] NCCL INFO comm 0x560c27fcd350 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 57000 commId 0x8b8004f8edbb0f22 - Init START
ubuntu:2296433:2296913 [3] NCCL INFO Setting affinity for GPU 3 to ffffffff,00000000,ffffffff
ubuntu:2296433:2296913 [3] NCCL INFO NVLS multicast support is not available on dev 3
ubuntu:2296433:2296913 [3] NCCL INFO comm 0x560c27fcd350 rank 3 nRanks 4 nNodes 1 localRanks 4 localRank 3 MNNVL 0
ubuntu:2296433:2296913 [3] NCCL INFO Trees [0] 1/-1/-1->3->0 [1] 0/-1/-1->3->2 [2] -1/-1/-1->3->0 [3] 0/-1/-1->3->-1 [4] 1/-1/-1->3->0 [5] 0/-1/-1->3->2 [6] -1/-1/-1->3->0 [7] 0/-1/-1->3->-1
ubuntu:2296433:2296913 [3] NCCL INFO P2P Chunksize set to 524288
ubuntu:2296433:2296913 [3] NCCL INFO Channel 00/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2296433:2296913 [3] NCCL INFO Channel 01/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2296433:2296913 [3] NCCL INFO Channel 04/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2296433:2296913 [3] NCCL INFO Channel 05/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2296433:2296913 [3] NCCL INFO Channel 02/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2296433:2296913 [3] NCCL INFO Channel 03/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2296433:2296913 [3] NCCL INFO Channel 06/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2296433:2296913 [3] NCCL INFO Channel 07/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2296433:2296913 [3] NCCL INFO Connected all rings
ubuntu:2296433:2296913 [3] NCCL INFO Channel 02/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2296433:2296913 [3] NCCL INFO Channel 03/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2296433:2296913 [3] NCCL INFO Channel 06/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2296433:2296913 [3] NCCL INFO Channel 07/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2296433:2296913 [3] NCCL INFO Channel 00/0 : 3[3] -> 1[1] via P2P/CUMEM
ubuntu:2296433:2296913 [3] NCCL INFO Channel 04/0 : 3[3] -> 1[1] via P2P/CUMEM
ubuntu:2296433:2296913 [3] NCCL INFO Channel 01/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2296433:2296913 [3] NCCL INFO Channel 05/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2296433:2296913 [3] NCCL INFO Connected all trees
ubuntu:2296433:2296913 [3] NCCL INFO NCCL_ALGO set by environment to Tree
ubuntu:2296433:2296913 [3] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
ubuntu:2296433:2296913 [3] NCCL INFO 8 coll channels, 0 collnet channels, 0 nvls channels, 8 p2p channels, 2 p2p channels per peer
ubuntu:2296433:2296913 [3] NCCL INFO comm 0x560c27fcd350 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 57000 commId 0x8b8004f8edbb0f22 - Init COMPLETE
Loading checkpoint shards:   0%|          | 0/8 [00:00<?, ?it/s][32m[2024-08-05 08:56:03,909] [    INFO][0m - Load lora weight successfully[0m
[32m[2024-08-05 08:56:03,974] [    INFO][0m - Load lora weight successfully[0m
Loading checkpoint shards:  25%|██▌       | 2/8 [00:00<00:00, 14.00it/s][32m[2024-08-05 08:56:04,040] [    INFO][0m - Load lora weight successfully[0m
[32m[2024-08-05 08:56:04,106] [    INFO][0m - Load lora weight successfully[0m
Loading checkpoint shards:  50%|█████     | 4/8 [00:00<00:00, 14.66it/s]Loading checkpoint shards: 100%|██████████| 8/8 [00:00<00:00, 29.10it/s]
[32m[2024-08-05 08:56:04,107] [    INFO][0m - Loading model from ./unified_checkpoint/checkpoints/llama_lora_ckpts/checkpoint-10 using unified checkpoint.[0m
[2024-08-05 08:56:04,110] [    INFO] pipeline_parallel.py:235 - dp_comm_overlap False;             sharding_comm_overlap False;             sharding_split_param False;
[2024-08-05 08:56:04,110] [    INFO] pipeline_parallel.py:303 - Pipeline Info -- num_stages: 2, stage_id: 0
[2024-08-05 08:56:04,110] [    INFO] pipeline_parallel.py:308 - start broadcast mp parameters
[2024-08-05 08:56:04,113] [ WARNING] hybrid_parallel_optimizer.py:292 - While using ClipGradByGlobalNorm in TensorParallel, PipelineParallel or Sharding, the grad clip of original optimizer will be changed.
[32m[2024-08-05 08:56:04,131] [    INFO][0m - Loading optimizer and scheduler...[0m
Loading optimizer shards:   0%|          | 0/8 [00:00<?, ?it/s]
Loading master weights shards:   0%|          | 0/8 [00:00<?, ?it/s][ALoading optimizer shards:  12%|█▎        | 1/8 [00:00<00:00,  7.02it/s]Loading optimizer shards:  25%|██▌       | 2/8 [00:00<00:00,  8.37it/s]Loading optimizer shards:  38%|███▊      | 3/8 [00:00<00:00,  7.63it/s]Loading optimizer shards:  50%|█████     | 4/8 [00:00<00:00,  4.29it/s]Loading optimizer shards: 100%|██████████| 8/8 [00:00<00:00, 10.26it/s]

Loading master weights shards:  12%|█▎        | 1/8 [00:00<00:06,  1.14it/s][A
Loading master weights shards:  38%|███▊      | 3/8 [00:01<00:01,  3.29it/s][A
Loading master weights shards:  50%|█████     | 4/8 [00:01<00:00,  4.17it/s][ALoading master weights shards: 100%|██████████| 8/8 [00:01<00:00,  6.65it/s]
[32m[2024-08-05 08:56:06,436] [    INFO][0m - Start broadcast optimizer in data parallel group.[0m
[32m[2024-08-05 08:56:06,485] [    INFO][0m - [timelog] checkpoint loading time: 4.21s (2024-08-05 08:56:06) [0m
[32m[2024-08-05 08:56:06,485] [    INFO][0m - ***** Running training *****[0m
[32m[2024-08-05 08:56:06,485] [    INFO][0m -   Num examples = 114,599[0m
[32m[2024-08-05 08:56:06,485] [    INFO][0m -   Num Epochs = 1[0m
[32m[2024-08-05 08:56:06,485] [    INFO][0m -   Instantaneous batch size per device = 4[0m
[32m[2024-08-05 08:56:06,485] [    INFO][0m -   Total train batch size (w. parallel, distributed & accumulation) = 32[0m
[32m[2024-08-05 08:56:06,485] [    INFO][0m -   Gradient Accumulation steps = 8[0m
[32m[2024-08-05 08:56:06,485] [    INFO][0m -   Total optimization steps = 15[0m
[32m[2024-08-05 08:56:06,485] [    INFO][0m -   Total num train samples = 480[0m
[35m[2024-08-05 08:56:06,486] [   DEBUG][0m -   Number of trainable parameters = 329,728 (per device)[0m
[35m[2024-08-05 08:56:06,488] [   DEBUG][0m -   Number of trainable parameters = 2,637,824 (all devices, roughly)[0m
[32m[2024-08-05 08:56:06,513] [    INFO][0m -   Continuing training from checkpoint, will skip to saved global_step[0m
[32m[2024-08-05 08:56:06,513] [    INFO][0m -   Continuing training from epoch 0[0m
[32m[2024-08-05 08:56:06,513] [    INFO][0m -   Continuing training from global step 10[0m
[32m[2024-08-05 08:56:06,513] [    INFO][0m -   Will skip the first 0 epochs then the first 80 batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` flag to your launch command, but you will resume the training on data already seen by your model.[0m
W0805 08:56:08.615381 2296433 multiply_fwd_func.cc:75] got different data type, run type promotion automatically, this may cause data type been changed.
ubuntu:2296433:2301992 [3] NCCL INFO Using non-device net plugin version 0
ubuntu:2296433:2301992 [3] NCCL INFO Using network Socket
ubuntu:2296433:2301992 [3] NCCL INFO comm 0x560c37310b70 rank 0 nranks 2 cudaDev 3 nvmlDev 3 busId 57000 commId 0x331337e0029dbbb - Init START
ubuntu:2296433:2301992 [3] NCCL INFO Setting affinity for GPU 3 to ffffffff,00000000,ffffffff
ubuntu:2296433:2301992 [3] NCCL INFO comm 0x560c37310b70 rank 0 nRanks 2 nNodes 1 localRanks 2 localRank 0 MNNVL 0
ubuntu:2296433:2301992 [3] NCCL INFO Channel 00/02 :    0   1
ubuntu:2296433:2301992 [3] NCCL INFO Channel 01/02 :    0   1
ubuntu:2296433:2301992 [3] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1
ubuntu:2296433:2301992 [3] NCCL INFO P2P Chunksize set to 131072
ubuntu:2296433:2301992 [3] NCCL INFO Channel 00 : 0[3] -> 1[7] via SHM/direct/direct
ubuntu:2296433:2301992 [3] NCCL INFO Channel 01 : 0[3] -> 1[7] via SHM/direct/direct
ubuntu:2296433:2301992 [3] NCCL INFO Connected all rings
ubuntu:2296433:2301992 [3] NCCL INFO Connected all trees
ubuntu:2296433:2301992 [3] NCCL INFO NCCL_ALGO set by environment to Tree
ubuntu:2296433:2301992 [3] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
ubuntu:2296433:2301992 [3] NCCL INFO 2 coll channels, 0 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
ubuntu:2296433:2301992 [3] NCCL INFO comm 0x560c37310b70 rank 0 nranks 2 cudaDev 3 nvmlDev 3 busId 57000 commId 0x331337e0029dbbb - Init COMPLETE
ubuntu:2296433:2302042 [3] NCCL INFO Channel 00 : 0[3] -> 1[7] via SHM/direct/direct
ubuntu:2296433:2302042 [3] NCCL INFO Channel 01 : 0[3] -> 1[7] via SHM/direct/direct
ubuntu:2296433:2302046 [3] NCCL INFO Channel 00 : 0[3] -> 1[7] via SHM/direct/direct
ubuntu:2296433:2302046 [3] NCCL INFO Channel 01 : 0[3] -> 1[7] via SHM/direct/direct
ubuntu:2296433:2302092 [3] NCCL INFO Using non-device net plugin version 0
ubuntu:2296433:2302092 [3] NCCL INFO Using network Socket
ubuntu:2296433:2302092 [3] NCCL INFO comm 0x560c45c96680 rank 3 nranks 8 cudaDev 3 nvmlDev 3 busId 57000 commId 0xcca97553c6c09fb3 - Init START
ubuntu:2296433:2302092 [3] NCCL INFO Setting affinity for GPU 3 to ffffffff,00000000,ffffffff
ubuntu:2296433:2302092 [3] NCCL INFO NVLS multicast support is not available on dev 3
ubuntu:2296433:2302092 [3] NCCL INFO comm 0x560c45c96680 rank 3 nRanks 8 nNodes 1 localRanks 8 localRank 3 MNNVL 0
ubuntu:2296433:2302092 [3] NCCL INFO Trees [0] 4/-1/-1->3->2 [1] 4/-1/-1->3->2
ubuntu:2296433:2302092 [3] NCCL INFO P2P Chunksize set to 524288
ubuntu:2296433:2302092 [3] NCCL INFO Channel 00/0 : 3[3] -> 1[1] via P2P/CUMEM
ubuntu:2296433:2302092 [3] NCCL INFO Channel 01/0 : 3[3] -> 1[1] via P2P/CUMEM
ubuntu:2296433:2302092 [3] NCCL INFO Connected all rings
ubuntu:2296433:2302092 [3] NCCL INFO Channel 00 : 3[3] -> 4[4] via SHM/direct/direct
ubuntu:2296433:2302092 [3] NCCL INFO Channel 01 : 3[3] -> 4[4] via SHM/direct/direct
ubuntu:2296433:2302092 [3] NCCL INFO Channel 00/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2296433:2302092 [3] NCCL INFO Channel 01/0 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2296433:2302092 [3] NCCL INFO Connected all trees
ubuntu:2296433:2302092 [3] NCCL INFO NCCL_ALGO set by environment to Tree
ubuntu:2296433:2302092 [3] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 512 | 512
ubuntu:2296433:2302092 [3] NCCL INFO 2 coll channels, 0 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
ubuntu:2296433:2302092 [3] NCCL INFO comm 0x560c45c96680 rank 3 nranks 8 cudaDev 3 nvmlDev 3 busId 57000 commId 0xcca97553c6c09fb3 - Init COMPLETE
[32m[2024-08-05 08:56:26,148] [    INFO][0m - 
Training completed. 
[0m
[32m[2024-08-05 08:56:26,150] [    INFO][0m - Saving model checkpoint to ./unified_checkpoint/checkpoints/llama_lora_ckpts[0m
[32m[2024-08-05 08:56:26,314] [    INFO][0m - Unified model tensor parallel weights in shards[0m
ubuntu:2296433:2302304 [3] NCCL INFO Channel 04/1 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2296433:2302304 [3] NCCL INFO Channel 05/1 : 3[3] -> 0[0] via P2P/CUMEM/read
ubuntu:2296433:2302327 [3] NCCL INFO Channel 02/1 : 3[3] -> 1[1] via P2P/CUMEM
ubuntu:2296433:2302327 [3] NCCL INFO Channel 03/1 : 3[3] -> 1[1] via P2P/CUMEM
ubuntu:2296433:2302337 [3] NCCL INFO Channel 06/1 : 3[3] -> 2[2] via P2P/CUMEM
ubuntu:2296433:2302337 [3] NCCL INFO Channel 07/1 : 3[3] -> 2[2] via P2P/CUMEM
[32m[2024-08-05 08:56:26,528] [    INFO][0m - Unified checkpoint: generating sharded_index json files for model weight.[0m
I0805 08:56:28.045898 2296433 process_group_nccl.cc:143] ProcessGroupNCCL destruct 
I0805 08:56:28.048722 2296433 process_group_nccl.cc:143] ProcessGroupNCCL destruct 
I0805 08:56:28.048784 2296433 process_group_nccl.cc:143] ProcessGroupNCCL destruct 
I0805 08:56:28.048803 2296433 process_group_nccl.cc:143] ProcessGroupNCCL destruct 
I0805 08:56:28.048820 2296433 process_group_nccl.cc:143] ProcessGroupNCCL destruct 
