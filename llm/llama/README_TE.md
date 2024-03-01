# PaddleNLP with Transformer Engine Integration
Transformer Engine (TE) is a library for accelerating Transformer models on NVIDIA GPUs, including using 8-bit floating point (FP8) precision on Hopper GPUs, to provide better performance with lower memory utilization in both training and inference. Refer to [TransformerEngine](https://github.com/NVIDIA/TransformerEngine)

Now TE is integrated into PaddleNLP, and can be used to accelerate the training of LLaMA model.

## Getting started

### Requirements

- NVIDIA Ampere/Hopper GPU
- CUDA >= 12
- Paddle >= 2.5

### Dataset

Refer to [OpenWebText2.md](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/ernie-1.0/preprocess/docs/OpenWebText2.md) to prepare dataset.

### Install TE and PaddleNLP in NGC Paddle container

```bash
docker run -it --rm --gpus=all --net=host \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /path/to/workspace:/workspace \
    -v /path/to/dataset/llama_openwebtext:/dataset \
    nvcr.io/nvidia/paddlepaddle:23.11-py3 bash

cd /workspace
git clone git@github.com:NVIDIA/TransformerEngine.git
cd TransformerEngine
git submodule update --init --recursive
NVTE_FRAMEWORK=paddle pip install .

cd /workspace
git clone git@github.com:PaddlePaddle/PaddleNLP.git
cd PaddleNLP
pip install .

pip install regex tool_helpers visualdl==2.5.3  # requirements

```

### Run LLaMA pretraining interactively

```bash
cd /workspace/PaddleNLP/llm/llama
bash scripts/llama_single_node_interactive.sh <model_name> <tokenizer_name> <batch_size_per_device> <FSDP> <TP> <PP> <VP> <GA> <SP> <max_seqlen> <sharding_stage> <backend> <precision> <recompute> <resume_step> <init_weight> <nsys_profile>
```
`<model_name>`: The model name. For example, meta-llama/Llama-2-7b

`<tokenizer_name>`: The tokenizer name. For example, meta-llama/Llama-2-7b

`<batch_size_per_device>`: The batch size per device.

`<sharing_stage>`: `stage1` or `stage2` or `stage3`.

`<backend>`: `none` or `te` or `pd`. `none` means using PaddleNLP's native transformer. `te` means using TE's transformer. `pd` means using TE's transformer with paddle backend.
Note: we usaually use `te` and `none` backend to compare the performance (and convergence) of TE and PaddleNLP's native transformer. `pd` backend is mainly used to debug or as a reference in the unit test. It may be very slow and use more memory, so it is not recommended to use in real training.

`<precision>`: `bf16` or `fp8`. `bf16` means using bf16. `fp8` means using fp8.

`<recompute>`: `none` or `core_attn` or `full`. `none` means not using recompute. `core_attn` means using recompute with core_attn. `full` means using recompute with full.

`<resume_step>`: The step to resume training. `none`, `auto` or a number. `none` means training from scratch. Default is none. `auto` means automatically find the latest checkpoint in the `output_dir` and resume training from it. A number means resuming training from the checkpoint with the specified step. For example, if there are checkpoints with steps 1000, 2000, 3000 in the `output_dir` and you want to resume training from the 2000th step, you can set `<resume_step>` to 2000.

`<init_weight>`: Path to the initial checkpoint folder. If not set, will not load any checkpoint. Default is None.

`<nsys_profile>`: `true` or `false`. `true` means using [Nsight Systems](https://developer.nvidia.com/nsight-systems) (nsys) to profile. `false` means not using Nsight Systems to profile.


Example:
```bash

bash scripts/llama_single_node_interactive.sh  meta-llama/Llama-2-7b meta-llama/Llama-2-7b 1 8 1 1 1 1 false 4096 stage1 none bf16 full

bash scripts/llama_single_node_interactive.sh  meta-llama/Llama-2-13b meta-llama/Llama-2-13b 1 2 4 1 1 1 true 4096 stage1 te fp8 full

bash scripts/llama_single_node_interactive.sh  meta-llama/Llama-2-7b meta-llama/Llama-2-7b 1 4 2 1 1 16 false 4096 stage2 te fp8 full none /path/to/my_ckpt

```

### Convergence tests

Multi-node training:
```bash
# training from scratch
MBS=1 TP_SIZE=4 PP_SIZE=4 VP_SIZE=1 GA_SIZE=32 FSDP_SIZE=4 SHARDING_STAGE=stage1 BACKEND=te PREC=fp8 RECOMPUTE=full SP=true NSYS=false  MODEL_NAME=meta-llama/Llama-2-70b TOKENIZER_NAME=meta-llama/Llama-2-70b sbatch -N8 llama_multi_node.sub
```

### Checkpoint converter
TE layer has different weight shape/name with PaddleNLP's native transformer. So if we want to keep the initial weight consistent between TE and PaddleNLP's native transformer, we need to convert the checkpoint.
`te_ckpt_converter.py` is a tool to convert the checkpoint between TE and PaddleNLP's native transformer. It can be used as follows:

1. convert paddle checkpoint to TE checkpoint
```bash
python te_llama_ckpt_converter.py --input_ckpt_path <path/to/paddle/ckpt/foler> --output_ckpt_path <path/to/te/ckpt/foler> --mode pd2te
```

2. convert TE checkpoint to paddle checkpoint
```bash
python te_llama_ckpt_converter.py --input_ckpt_path <path/to/te/ckpt/foler> --output_ckpt_path <path/to/paddle/ckpt/foler> --mode te2pd
```

Note:
- This tool can only be used to convert the weights of LLaMA model.
- The model files should be named as `model_state.pdparams` or `model_state.tp00.pdparams`. Other files will be ignored, such as `.pdopt`.
- <path/to/paddle/ckpt/foler> and <path/to/te/ckpt/foler> should be folder, not file.
- The output checkpoint folder can be used as the `<init_weight>` argument in the section [Run LLaMA pretraining interactively](#run-llama-pretraining-interactively).
