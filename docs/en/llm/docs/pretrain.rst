.. _introduction:
===============================
Introduction to Large Model Pre-training
===============================

The PaddleNLP Large Model Toolkit supports pre-training for large models including LLaMA v1/v2, GPT-3, BaiChuan, Qwen, etc.

Clone the code to start:

.. code-block:: bash

    git clone https://github.com/PaddlePaddle/PaddleNLP.git
    # pip install ./PaddleNLP to use development version
    cd PaddleNLP/llm
    # Enter execution directory


.. _create-dataset:
Data Preparation
-----------------------------

Detailed workflow can be found in the following documentation:

.. toctree::
    :maxdepth: 1

    data/DataPreparation
.. _start_training:
Start Training
-------------------------

To facilitate users in testing and running this model, we provide a processed 100k-doc training sample:

.. code-block:: bash

    # Download llama model data
    wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
    wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx

    # Download gpt model data
    # wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
    # wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz

Organize all preprocessed files into a single directory for training:

.. code-block:: bash

    mkdir data
    mv llama_openwebtext_100k.bin ./data
    mv llama_openwebtext_100k.idx ./data

.. code-block:: bash

    # Compile custom operators (optional)
    cd ../slm/model_zoo/gpt-3/external_ops/ && python3 setup.py install && cd -

    # Pre-training for llama model
    python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./config/llama/pretrain_argument.json

    # Pre-training for Qwen model
    python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./config/qwen/pretrain_argument.json

Notes:

1. It is recommended to use the paddle develop version for training. Requires installation of ``pip install ...`` (remain actual installation command unchanged)
```markdown
pip install fast_dataindex visualdl==2.5.3`` and other missing whl packages.
2. ``use_flash_attention`` requires enabling on A100 machines, recommended to use cuda11.8 environment.
3. ``use_fused_rms_norm`` requires installing custom OPs from `this directory <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/model_zoo/gpt-3/external_ops>`_ via `python setup.py install`. If operators are still not found after installation, additional ``PYTHONPATH`` settings are required.
4. ``continue_training`` indicates loading training from an existing pre-trained model. The initial loss for 7B model is approximately 2.xx, while randomly initialized models start from loss around 11.x and decrease.
5. Current script is the sharding version. For users requiring 4D parallel training (data, sharding, tensor, pipeline parallelism), please refer to ``run_trainer_tp4pp2.sh`` script.
6. During multi-machine training, if all machines use training data files from the same location (e.g., mounted shared storage), specify ``--share_folder true`` to have global rank 0 create cached data. Otherwise, rank 0 on each machine will create cached data independently by default.
7. If the default cache folder ``index-cache/`` exists in the dataset directory, the additionally specified ``--data_cache`` will not take effect, and training will prioritize loading content from the default cache folder.

The pre-training uses PaddleNLP's Trainer module. For related distributed strategy usage, please refer to the `Large Model Trainer Hybrid Parallel Training Tutorial <./llm_trainer.rst>`.

.. _model_capability:
Overview of Distributed Capabilities Supported in Model Pre-training
--------------------------------------

.. csv-table:: Model Capability Summary
    :header: Model,Data Parallelism,Tensor Parallelism,Pipeline Parallelism,Sequence Parallelism,Flash Attention,Selective Recompute,Sharding Stage1 + Recompute,Sharding Stage1 + DP,Stage2 + Recompute,Stage2 + DP,Stage3 + Recompute,Stage3 + DP
    :widths: 5 2 2 2 2 2 2 2 2 2 2 2 2

    ``LLaMA-65B``   ,✅,✅,✅,✅,✅,✅,✅,✅,✅,✅,✅,✅
    ``LLaMA2-70B``  ,✅,✅,✅,✅,✅,✅,✅,✅,✅,✅,✅,✅
    ``BaiChuan-13B``,✅,✅,✅,✅,✅,✅,✅,✅,✅,✅,✅,✅
    ``GPT3``        ,✅,✅,✅,✅,✅,✅,✅,✅,✅,✅,✅,✅
    ``Qwen-7B``     ,✅,✅,✅,⬜,✅,✅,⬜,✅,✅,✅,✅,✅
    ``Qwen-14B``    ,✅,✅,✅,⬜,✅,✅,⬜,✅,✅,✅,✅,✅
    ``OPT 66B``     ,✅,✅,⬜,⬜,❌,🚧,⬜,⬜,⬜,⬜,⬜,⬜
    ``Bloom-176B``
```
``ChatGLM-6B``  ,✅,✅,⬜,⬜,✅,🚧,⬜,⬜,⬜,⬜,⬜,⬜  
``ChatGLM2``    ,✅,✅,⬜,⬜,❌,🚧,⬜,⬜,⬜,⬜,⬜,⬜  
``GLM-130B``    ,✅,✅,⬜,⬜,✅,🚧,⬜,⬜,⬜,⬜,⬜,⬜  

Translation maintained:
1. Technical terms (ChatGLM-6B, ChatGLM2, GLM-130B) preserved in original form
2. Formatting with backticks and commas retained exactly
3. Status symbols (✅, ⬜, ❌, 🚧) kept unchanged
4. Table structure preserved with proper alignment
5. No content modification except direct translation where applicable
.. _model_weight:
Model Weight Support List
-------------------------

The table above shows some model weights. All supported models are as follows:

.. code-block:: text

  * LLaMA Series
    - facebook/llama-7b [English]
    - facebook/llama-13b [English]
    - facebook/llama-65b [English]
    - meta-llama/Llama-2-7b [English]
    - meta-llama/Llama-2-7b-chat [English]
    - meta-llama/Llama-2-13b [English]
    - meta-llama/Llama-2-13b-chat [English]
    - meta-llama/Llama-2-70b [English]
    - baichuan-inc/Baichuan-7B [Chinese]
    - baichuan-inc/Baichuan-13B-Base [Chinese]
    - baichuan-inc/Baichuan-13B-Chat [Chinese]
    - baichuan-inc/Baichuan2-7B-Base [Chinese]
    - baichuan-inc/Baichuan2-7B-Chat [Chinese]
    - baichuan-inc/Baichuan2-13B-Base [Chinese]
    - baichuan-inc/Baichuan2-13B-Chat [Chinese]
    - FlagAlpha/Llama2-Chinese-7b-Chat [Chinese]
    - FlagAlpha/Llama2-Chinese-13b-Chat [Chinese]
    - idea-ccnl/ziya-llama-13b-v1 [Chinese]
    - linly-ai/chinese-llama-2-7b [Chinese]
    - linly-ai/chinese-llama-2-13b [Chinese]
  * ChatGLM Series
    - THUDM/chatglm-6b-v1.1 [Chinese]
    - THUDM/chatglm2-6b [Chinese]
  * BLOOM Series
    - bigscience/bloom-7b1 [English]
    - bigscience/bloomz-7b1 [Multilingual]
    - bigscience/bloomz-7b1-mt [Multilingual]
  * Qwen Series
    - qwen/qwen-7b [Chinese]
    - qwen/qwen-7b-chat [Chinese]
    - qwen/qwen-14b [Chinese]
    - qwen/qwen-14b-chat [Chinese]


.. _model_performance:
Model Pre-training Performance
------------------

The following test results are based on:

Hardware Environment:

- GPU: A100 80G * 8, CUDA 11.8, NCCL 2.15
- CPU: Intel(R) Xeon(R) Platinum 8350C CPU @ 2.60GHz
- Memory: 1 TB

.. code-block:: text

    paddle commit id              : 9b36e53f24ac5f471b20de99e0cc3980f38b44ab
    paddlenlp commit id           : 0b246a609a3062e3c3256d87193b70277b5b07e0


.. csv-table:: Model Performance Test Summary
    :header: Model,Sequence Length,Distributed Strategy,Speed [#]_ [#]_,Memory Usage [#]_,Config File,Test Time
    :widths: 10 2 4 2 2 15 5
``FlagAlpha/Llama2-Chinese-13b-Chat``,4096,``tp2sd4_stage2``,1980.22,64323MB,``./llama/pretrain-flagalpha_llama2_13b-tp2sd4_stage2.json``,2023-11-27 21:42:38  
    ``FlagAlpha/Llama2-Chinese-7b-Chat``,4096,``tp2sd4_stage2``,3744.62,52092MB,``./llama/pretrain-flagalpha_llama2_7b-tp2sd4_stage2.json``,2023-11-27 21:44:57  
    ``baichuan-inc/Baichuan2-13B-Base``,4096,``sd8_stage2``,1354.99,74767MB,``./baichuan/pretrain-baichuan2_13b-sd8_stage2.json``,2023-11-27 21:51:26  
    ``baichuan-inc/Baichuan2-7B-Base``,4096,``tp2sd4_stage2``,3542.45,58363MB,``./baichuan/pretrain-baichuan2_7b-tp2sd4_stage2.json``,2023-11-27 21:53:58  
    ``facebook/llama-13b``,4096,``tp2sd4_stage2``,1969.64,64278MB,``./llama/pretrain-llama_13b-tp2sd4_stage2.json``,2023-11-27 21:58:03  
    ``facebook/llama-7b``,4096,``tp2sd4_stage2``,3754.73,52092MB,``./llama/pretrain-llama_7b-tp2sd4_stage2.json``,2023-11-27 22:00:30  
    ``idea-ccnl/ziya-llama-13b-v1``,4096,``tp2sd4_stage2``,1968.34,63983MB,``./llama/pretrain-ziya_llama_13b-tp2sd4_stage2.json``,2023-11-27 22:04:35  
    ``linly-ai/chinese-llama-2-7b``,4096,``tp2sd4_stage2``,3732.9,51751MB,``./llama/pretrain-linly_llama2_7b-tp2sd4_stage2.json``,2023-11-27 22:06:58  
    ``meta-llama/Llama-2-13b``,4096,``tp2sd4_stage2``,1975.63,64294MB,``./llama/pretrain-llama2_13b-tp2sd4_stage2.json``,2023-11-27 22:11:04  
    ``meta-llama/Llama-2-7b``,4096,``tp2sd4_stage2``,3755.21,52092MB,
``qwen/qwen-7b``                     ,4096,``tp2sd4_stage2``,3607.28,65448MB,``./qwen/pretrain-qwen_7b-tp2sd4_stage2.json``,2023-11-27 22:16:04


..  [#] The unit of speed is ``tokens/card/sec``, representing the number of tokens to be trained per second per GPU card.
..  [#] Speed may fluctuate slightly, e.g., ``facebook/llama-7b`` and ``meta-llama/Llama-2-7b`` share identical training configurations.
..  [#] GPU memory usage is measured in MB using ``max_memory_allocated``. Actual physical memory consumption will be higher, approximately 2-3GB additional.