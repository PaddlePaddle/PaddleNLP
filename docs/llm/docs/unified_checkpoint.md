# 飞桨大模型统一存储文档

## 1. 背景

在大模型背景下，通常我们需要进行多卡分布式的训练，在保存Checkpoint时所得到的模型权重通常是分片放置的，例如根据张量并行、流水线并行进行切分保存。这种根据分布式策略直接存储Checkpoint的方式非常直接明了，但也存在如下的问题：
* 对下游推理不够友好，当用户希望获取中间阶段保存的Checkpoint做下游推理时，需要手动对模型权重进行合并。
* 不利于应对做恢复训练时，可能会面临的分布式策略改变、训练节点数发生变化的情况。用户往往需要手动对Checkpoint进行处理，增加了操作复杂度。

为了最大程度地解决上述的问题，降低用户操作难度，我们提出了大模型统一存储方案——Unified Checkpoint。Unified Checkpoint的核心思想是将模型权重、优化器权重等进行统一safetensors格式存储，在Checkpoint存储时不再对分布式策略进行区分，提高大模型存储的通用性。以下将首先介绍Unified Checkpoint具体存储格式以及如何使用，随后再简要介绍统一存储的实现原理。

## 2. 统一存储 Unified Checkpoint 使用介绍

### 2.1 使用命令与配置项说明

- **使用示例**
``` bash
python -u -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    run_pretrain.py \
    --unified_checkpoint 1 \
    --unified_checkpoint_config "enable_all_options"
```

- **总开关**
`unified_checkpoint`用于控制是否使用Unified Checkpoint存储格式。
``` bash
unified_checkpoint: Optional[bool] = field(
    default=False,
    metadata={"help": "Whether to unify hybrid parallel checkpoint."},
)
```

- **配置项说明**
``` bash
unified_checkpoint_config: Optional[str] = field(
    default="",
    metadata={
        "help": (
            "Configs to unify hybrid parallel checkpoint.\n"
            "Following options are supports:\n"
            "- skip_save_model_weight: do not save model weights when the masters weight exist\n"
            "- master_weight_compatible: 1. if the master weights exist, only load when needed\n"
            "                            2. if master weights does not exist, convert model weights to master weights when needed\n"
            "- enable_all_options: enable all optimization configurations\n"
        )
    },
)
```
介绍如下:
1. skip_save_model_weight：当optimizer具有master weight时，跳过model weight保存，重启时将master weight作为model weight加载。在PaddleNLP中，仅fp16_opt_level=O1时，optimizer不存在master weight。
2. master_weight_compatible：仅当optimizer需要master weight时，才加载master weight; 如果ckpt中不存在master weight，将model weight作为master weight加载。
3. enable_all_options：上述参数均开启。

### 2.2 Unified Checkpoint存储格式介绍

这里以facebook/llama-7b的pretrain checkpoint保存为例进行说明。以TP=4，PP=2的分布式训练为例，原始的存储格式举例如下代码片段。无论是模型参数，异或是优化器参数，均按照TP、PP训练方式进行了分片存储。
```
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

采用Unified Checkpoint进行统一存储后，新格式如下面代码片段。可以看到，无论是模型参数、优化器参数，我们均采用了safetensors格式进行存储，不再区分TP、PP策略；进一步地，我们将优化器参数区分为了optimizer与master_weights（如果有的话），而master_weights本身就是模型参数的FP32版本。其中，`model.safetensors.index.json`等json文件用于记录参数对应所在的文件部分。
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

其中，[safetensors](https://github.com/huggingface/safetensors)是由huggingface开发的一种新序列化格式，旨在简化和精简大型复杂张量的存储和加载。使用Safetensors有很多好处，这里简要列举部分如下：
1. 速度快：Safetensors采用Zero-copy技术进行了速度优化，可以高效处理大型张量的序列化和反序列化。
2. 大小优化：混合使用了有效的序列化和压缩算法，以减少大型张量的大小，相比于其他序列化格式（如pickle），性能更快、更高效。
3. 懒惰加载：Safetensors在加载参数时只需要加载文件中需要的部分张量即可，效率更高。
4. 安全性：为了防止序列化张量在存储或传输过程中出现损坏，Safetensors使用了校验和机制。这保证了额外的安全性，确保存储在Safetensors中的所有数据都准确可靠。

### 2.3 训练分布式策略发生变化时怎么办？

在Unified checkpoint统一存储格式下，当训练分布式策略不变时，我们直接原地加载Checkpoint进行训练即可。那么，当分布式策略发生变化时，应当怎么做？以下区分两种情况进行讨论。

#### 2.3.1 机器不变

在训练机器不变的情况下，进行分布式策略的改变有多种情况，简单举例如下：
* 例如单机训练，希望TP=8转为TP=4、Sharding=2进行训练；
* 例如两机训练时，希望TP=8、Sharding=2转为PP=8、Sharding=2训练；
* 又或者是希望在相同机器的情况下减少参与训练的进程数（GPU卡数）。
在这些情况下，我们都不需要对checkpoint进行处理，只需要进行重启操作即可，Unified checkpoint会自动加载文件并执行相应的Tensor切分、发送接收等操作。

#### 2.3.2 机器数量发生变化，例如1->多，多->1，多->多

尽管机器数量发生变化的情况很多，用户在处理Checkpoint时原则上只需要保证：新的训练机器上至少需要有一份完整的Checkpoint参数，这份完整参数可以放置在同一台机器上，也可以分散放置在多台机器上（如果为多机训练的话）。
* 1->多：例如，原先我们在机器A上训练，Checkpoint存储在A上，接下来想用A、B两台机器同时训练，此时需要确保机器A、B上有一份完整Checkpoint参数即可。
* 多->1：例如，原先我们在两台机器A、B训练，Checkpoint可能分了两台机器进行存储，接下来想只在机器A上进行训练，那么需要将Checkpoint文件完整放置在机器A上。
* 多->多：例如，原先我们在机器A、B上训练，Checkpoint存储在A、B上，接下来想用四台机器(A、B、C、D)训练，此时需要确保参与训练的四台机器上具备一份完整Checkpoint参数即可。
用户只需要确保参与训练的机器上具备一份完整的Checkpoint参数，即可进行重启训练。

#### 2.4 旧格式的Checkpoint如何兼容？

在打开unified_checkpoint开关后，我们会对Checkpoint文件夹中的内容进行检查。
1. 如果文件夹中具备旧格式的参数文件等，我们会按照旧格式的方式进行参数加载，在后续保存新的Checkpoint时会保存成Unified Checkpoint的格式。
2. 如果文件夹中不含旧格式参数文件，则默认采用Unified Checkpoint格式进行加载。我们会检查参与训练的机器中的参数文件是否完整，如完整则直接加载训练，否则会进行报错。
