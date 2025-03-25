# Unified Checkpoint for Large-scale Models in PaddlePaddle

## 1. Background

In the context of large-scale model training, we typically employ multi-GPU distributed training. When saving checkpoints, the obtained model weights are usually stored in sharded format, e.g., split according to tensor parallelism or pipeline parallelism. While this direct storage approach based on distributed strategies is straightforward, it presents the following issues:
* **Inference unfriendly**: Users need to manually merge model weights when using intermediate checkpoints for downstream inference tasks.
* **Limited flexibility for resuming training**: Manual checkpoint processing is often required when changing distributed strategies or adjusting the number training nodes, increasing operational complexity.

To address these issues and reduce user effort, we propose the Unified Checkpoint solution. The core idea is to store model weights and optimizer states in a unified safetensors format, eliminating distinctions between distributed strategies during checkpoint storage. The following sections first introduce the Unified Checkpoint format and usage, then briefly explain its implementation principles.

## 2. Unified Checkpoint Usage Guide

### 2.1 Commands and Configuration Options

- **Usage Example**
```bash
python -u -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    run_pretrain.py \
    --unified_checkpoint 1 \
    --unified_checkpoint_config "async_save"
```

- **Master Switch**
The `unified_checkpoint` parameter controls whether to use Unified Checkpoint format:
```bash
unified_checkpoint: Optional[bool] = field(
    default=False,
    metadata={"help": "Whether to unify hybrid parallel checkpoint."},
)
```

- **Configuration Options**
```bash
unified_checkpoint_config: Optional[str] = field(
    default="",
    metadata={
        "help": (
            "Configs to unify hybrid parallel checkpoint.\n"
            "Following options are supports:\n"
            "- skip_save_model_weight: do not save model weights when the masters weight exist\n"
            "- master_weight_compatible: 1. if the master weights exist, only load when needed\n"
            "                            2. if master weights does not exist, convert model weights to master weights when needed\n"
            "- remove_master_weight, whether save master weight, if master_weights does not exist, convert model weights to master_weights when needed."
            "- async_save: whether to use asynchronous saving."
        )
    },
)
```

Configuration options:
1. `skip_save_model_weight`: Skip saving model weights when the optimizer contains master weights. During restart, master weights will be loaded as model weights. In PaddleNLP, master weights only exist when `fp16_opt_level=O1` for the optimizer.
2. `master_weight_compatible`: Load master weights only when required by the optimizer. If master weights are absent in the checkpoint, use model weights as master weights.
3. `remove_master_weight`: Whether to retain master weights. If `master_weights` are missing in the checkpoint, load model weights as master weights.
4. `async_save`: Enable asynchronous saving to accelerate storage speed.

### 2.2 Unified Checkpoint Storage Format

This section explains the pretrain checkpoint storage format using facebook/llama-7b as an example. For distributed training with TP=4 and PP=2, the original storage format is demonstrated in the following code snippet. Both model parameters and optimizer states are sharded according to the TP and PP parallelism configurations.
```text
-rw-r--r-- 1 root root 1015 Dec 21 11:27 config.json
-rw-r--r-- 1 root root 1.6G Dec 21 11:27 model_state.tp00_pp00.pdparams
-rw-r--r-- 1 root root 1.6G Dec 21 11:27 model_state.tp00_pp01.pdparams
-rw-r--r-- 1 root root 1.6G Dec 21 11:27 model_state.tp01_pp00.pdparams
-rw-r--r-- 1 root root 1.6G Dec 21 11:27 model_state.tp01_pp01.pdparams
-rw-r--r-- 1 root root 1.6G Dec 21 11:27 model_state.tp02_pp00.pdparams
-rw-r--r-- 1 root root 1.6G Dec 21 11:27 model_state.tp02_pp01.pdparams
-rw-r--r-- 1 root root 1.6G Dec 21 11:27 model_state.tp03_pp00.pdparams
-rw-r--r-- 1 root root 1.6G Dec 21 11:27 model_state.tp03_pp01.pdparams
-rw-r--r-- 1 root root 9.5G Dec 21 11:27 optimizer.tp00_pp00.pdopt
-rw-r--r-- 1 root root 9.5G Dec 21 11:27 optimizer.tp00_pp01.pdopt
-rw-r--r-- 1 root root 9.5G Dec 21 11:27 optimizer.tp01_pp00.pdopt
-rw-r--r-- 1 root root 9.5G Dec 21 11:27 optimizer.tp01_pp01.pdopt
-rw-r--r-- 1 root root 9.5G Dec 21 11:27 optimizer.tp02_pp00.pdopt
-rw-r--r-- 1 root root 9.5G Dec 21 11:27 optimizer.tp02_pp01.pdopt
-rw-r--r-- 1 root root 9.5G Dec 21 11:27 optimizer.tp03_pp00.pdopt
-rw-r--r-- 1 root root 9.5G Dec 21 11:27 optimizer.tp03_pp01.pdopt
-rw-r--r-- 1 root root  54K Dec 21 11:27 rng_state_8.pth
-rw-r--r-- 1 root root  317 Dec 21 11:27 scaler.pdparams
-rw-r--r-- 1 root root   50 Dec 21 11:27 scheduler.pdparams
-rw-r--r-- 1 root root 489K Dec 21 11:27 sentencepiece.bpe.model
-rw-r--r-- 1 root root   63 Dec 21 11:27 special_tokens_map.json
-rw-r--r-- 1 root root  207 Dec 21 11:27 tokenizer_config.json
-rw-r--r-- 1 root root 3.1K Dec 21 11:27 trainer_state.json
-rw-r--r-- 1 root root 2.3K Dec 21 11:27 training_args.bin
```
After adopting Unified Checkpoint for unified storage, the new format is as shown in the code snippet below. As can be seen, whether it's model parameters or optimizer parameters, we have adopted the safetensors format for storage, no longer distinguishing between TP and PP strategies; furthermore, we have separated optimizer parameters into optimizer and master_weights (if any), where master_weights themselves are the FP32 versions of the model parameters. Among these, json files such as `model.safetensors.index.json` are used to record the corresponding file locations of the parameters.
```
-rw-r--r-- 1 root root 1015 Dec 21 11:24 config.json
-rw-r--r-- 1 root root 3.1G Dec 21 11:25 master_weights-00001-of-00008.safetensors
-rw-r--r-- 1 root root 3.2G Dec 21 11:25 master_weights-00002-of-00008.safetensors
-rw-r--r-- 1 root root 3.2G Dec 21 11:25 master_weights-00003-of-00008.safetensors
-rw-r--r-- 1 root root 3.2G Dec 21 11:25 master_weights-00004-of-00008.safetensors
-rw-r--r-- 1 root root 3.1G Dec 21 11:25 master_weights-00005-of-00008.safetensors
-rw-r--r-- 1 root root 3.2G Dec 21 11:25 master_weights-00006-of-00008.safetensors
-rw-r--r-- 1 root root 3.1G Dec 21 11:25 master_weights-00007-of-00008.safetensors
-rw-r--r-- 1 root root 3.3G Dec 21 11:25 master_weights-00008-of-00008.safetensors
-rw-r--r-- 1 root root  28K Dec 21 11:25 master_weights.safetensors.index.json
-rw-r--r-- 1 root root 1.6G Dec 21 11:24 model-00001-of-00008.safetensors
-rw-r--r-- 1 root root 1.6G Dec 21 11:24 model-00002-of-00008.safetensors
-rw-r--r-- 1 root root 1.6G Dec 21 11:24 model-00003-of-00008.safetensors
-rw-r--r-- 1 root root 1.6G Dec 21 11:24 model-00004-of-00008.safetensors
-rw-r--r-- 1 root root 1.6G Dec 21 11:24 model-00005-of-00008.safetensors
-rw-r--r-- 1 root root 1.6G Dec 21 11:24 model-00006-of-00008.safetensors
-rw-r--r-- 1 root root 1.6G Dec 21 11:24 model-00007-of-00008.safetensors
-rw-r--r-- 1 root root 1.7G Dec 21 11:24 model-00008-of-00008.safetensors
-rw-r--r-- 1 root root  25K Dec 21 11:24 model.safetensors.index.json
-rw-r--r-- 1 root root 6.2G Dec 21 11:25 optimizer-00001-of-00008.safetensors
-rw-r--r-- 1 root root 6.4G Dec 21 11:25 optimizer-00002-of-00008.safetensors
-rw-r--r-- 1 root root 6.2G Dec 21 11:25 optimizer-00003-of-00008.safetensors
-rw-r--r-- 1 root root 6.4G Dec 21 11:25 optimizer-00004-of-00008.safetensors
-rw-r--r-- 1 root root 6.3G Dec 21 11:25 optimizer-00005-of-00008.safetensors
-rw-r--r-- 1 root root 6.4G Dec 21 11:25 optimizer-00006-of-00008.safetensors
-rw-r--r-- 1 root root 6.3G Dec 21 11:25 optimizer-00007-of-00008.safetensors
-rw-r--r-- 1 root root 6.4G Dec 21 11:25 optimizer-00008-of-00008.safetensors
-rw-r--r-- 1 root root 118K Dec 21 11:25 optimizer.safetensors.index.json
-rw-r--r-- 1 root root  54K Dec 21 11:25 rng_state_8.pth
-rw-r--r-- 1 root root  317 Dec 21 11:25 scaler.pdparams
-rw-r--r-- 1 root root   50 Dec 21 11:25 scheduler.pdparams
-rw-r--r-- 1 root root 489K Dec 21 11:24 sentencepiece.bpe.model
-rw-r--r-- 1 root root   63 Dec 21 11:24 special_tokens_map.json
-rw-r--r-- 1 root root  207 Dec 21 11:24 tokenizer_config.json
-rw-r--r-- 1 root root 3.1K Dec 21 11:25 trainer_state.json
-rw-r--r-- 1 root root 2.3K Dec 21 11:24 training_args.bin
```
[safetensors](https://github.com/huggingface/safetensors) is a new serialization format developed by Hugging Face, designed to simplify and optimize the storage and loading of large, complex tensors. The benefits of using Safetensors are numerous, with key advantages outlined below:
1. **Speed**: Safetensors employs Zero-copy technology for accelerated serialization/deserialization of large tensors.
2. **Size Optimization**: Combines efficient serialization and compression algorithms to reduce tensor size, outperforming other formats (e.g., pickle) in speed and efficiency.
3. **Lazy Loading**: Safetensors enables partial tensor loading - only required portions need to be loaded from files.
4. **Security**: Utilizes checksum mechanisms to prevent data corruption during storage/transmission, ensuring data integrity and reliability.

### 2.3 How to Handle Training Distributed Strategy Changes?

With the unified checkpoint format, when the distributed strategy remains unchanged, we can directly resume training by loading checkpoints. But how to proceed when the distributed strategy changes? We discuss two scenarios:

#### 2.3.1 Same Machine Configuration

When keeping hardware unchanged, strategy changes may include:
- Single-machine training: Transition from TP=8 to TP=4 with Sharding=2
- Dual-machine training: Shift from TP=8/Sharding=2 to PP=8/Sharding=2
- Reducing GPU count while maintaining hardware

In these cases, no checkpoint processing is needed. Simply restart training - the unified checkpoint automatically handles tensor partitioning, communication, etc.

#### 2.3.2 Changing Machine Count (1↔N, N↔M)

Though machine count variations are diverse, users only need to ensure:
*At least one complete checkpoint copy exists across new training machines.* This copy can reside on:
- A single machine (1→N): When expanding from machine A to A+B, ensure A and B collectively contain a full checkpoint.
- Multiple machines (N→M): When scaling from A+B to A+B+C+D, verify all four machines collectively maintain a complete checkpoint.

Users must guarantee participating machines hold a complete checkpoint copy before resuming training.

#### 2.4 Backward Compatibility with Legacy Checkpoints

With the `unified_checkpoint` flag enabled:
1. If legacy format files exist, we load parameters using legacy methods. Subsequent checkpoints will be saved in unified format.
2. If no legacy files exist, we directly use unified format. The system verifies checkpoint completeness across machines and errors if incomplete.

### 2.5 Enabling Checkpoint Compression

1. Checkpoint compression is disabled by default (O0). Users may select from:
- O1: Basic compression (e.g., FP16 casting)
- O2: Aggressive compression (pruning + FP16)
- O3: Experimental compression (novel algorithms)

Note: Higher compression levels may introduce accuracy loss.
`--ckpt_quant_stage` enables compression at O1 or O2 stage, where O1 corresponds to Int8 compression and O2 to Int4 compression. This allows compressing the original float32 optimizer parameters to the dtype specified by `ckpt_quant_stage`.

2. Based on `ckpt_quant_stage` configuration, the `remove_master_weight` field can be added to the `--unified_checkpoint_config` parameter to exclude storage of float32 master weights, further reducing checkpoint size.

- **Usage Example**
```bash
python -u -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    run_pretrain.py \
    --unified_checkpoint 1 \
    --unified_checkpoint_config "async_save remove_master_weight"
    --ckpt_quant_stage O2
```

In this example, both stage O2 optimizer compression and master weight removal are enabled, achieving a total compression rate of 78.5%.
