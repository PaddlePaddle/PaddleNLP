# PaddlePaddle Large Model Suite Model Fusion Documentation
## 1. Introduction to Large Model Fusion
Model fusion, also known as model merging, is an effective technique that constructs a general-purpose model by combining parameters from multiple independent models with different capabilities, without requiring access to original training data or expensive computations. The closest concept to model fusion is ensemble learning, as both facilitate knowledge integration and transfer. The key difference is that ensemble learning must preserve all individual models and fuse predictions (or outputs) from multiple models during inference, while model fusion directly merges parameters at the model level, resulting in only one final model during inference.

Application scenarios for model fusion:

- **Enhancing Model Capabilities**: Model fusion enables models to handle multi-domain tasks and potentially improves performance in cross-domain scenarios.

- **Mitigating Alignment Costs**: After fine-tuning a pretrained model (SFT) to obtain a fine-tuned model, we typically perform RLHF. While RLHF has proven effective for improving LLMs' user-friendliness, it introduces an "alignment tax" - model performance may degrade after aligning with human preferences. Model fusion can help mitigate this alignment tax.

## 2. Introduction to Large Model Fusion Algorithms
### 2.1 Quick Start
Next, we'll explain how to perform model fusion using a unified script.
#### 2.1.1 Environment Setup

- PaddlePaddle 3.0-beta
- PaddleNLP develop

Clone the code repository to get started:

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git
# pip install ./PaddleNLP (use develop version)
cd PaddleNLP/llm/tools
# Navigate to execution directory
```

#### 2.1.2 Model Fusion

```
python mergekit.py \
    --device cpu \
    --tensor_type np \
    --n_process 2 \
    --merge_method linear \
    --model_path_list ../checkpoints/model1 ../checkpoints/model \
    --output_path ../checkpoints/model_merge
```
### 2.2 Weight Merging Methods
| merge_method   | Weight Sparsity | Weight Merging | Supported Model Count |
|----------------|-----------------|----------------|-----------------------|
| linear         | /               | linear         | >=2                   |
| slerp          | /               | slerp          | =2                    |
| ties           | trim            | ties           | >=2                   |
| della          | magprune        | ties           | >=2                   |
| della_linear   | magprune        | linear         | >=2                   |
| dare_linear    | dare            | linear         | >=2                   |
| dare_ties      | dare            | ties           | >=2                   |

#### Weight Merging Methods Introduction:
- **linear**: Linear weight merging. Optional hyperparameters: `weight_list`, `normalize`.
- **slerp**: Spherical linear interpolation merging, only supports two models. Optional hyperparameter: `slerp_alpha`.
- **ties**: Weight merging method proposed in [TIES](https://arxiv.org/abs/2306.01708) paper, applying sign consistency algorithm to reduce model merging interference. Optional hyperparameters: `weight_list`, `normalize`.

#### Weight Sparsity Methods Introduction:
- **trim**: Sparsity method proposed in [TIES](https://arxiv.org/abs/2306.01708) paper, retains specified proportion of weights with larger absolute values while setting smaller ones to zero. Optional hyperparameters: `reserve_p`, `rescale`.
- **dare**: Sparsity method proposed in [DARE](https://arxiv.org/abs/2311.03099) paper, randomly selects whether to retain original weights or set them to zero based on specified probability. Optional hyperparameters: `reserve_p`, `rescale`.
- **magprune**: Sparsity method proposed in [DELLA](https://arxiv.org/abs/2406.11617) paper, assigns different retention probabilities based on weight magnitudes, randomly selecting whether to retain original weights or set them to zero. Optional hyperparameters: `reserve_p`, `rescale`, `epsilon`.

## 3. Mergekit Parameters Introduction
<summary>&emsp; General Parameters</summary><div>

- `device`: Device type for model merging, supports `"cpu"`, `"gpu"` or `"low_gpu_mem"`, default is `"cpu"`.
- `tensor_type`: Tensor type used in model merging process, supports `"np"` (CPU only) or `"pd"` (supports CPU/GPU), default is `"np"`.
- `n_process`: Number of parallel processes for model merging, default is `1`.
- `merge_prefix`
- `model_prefix`: The prefix name of the model files, e.g. `"model"` or `"master_weights"`, default is `"model"`.
- `merge_method`: Model fusion strategy, supports `"linear"`, `"ties"`, `"slerp"`, `"della_linear"`, `"della"`, `"dare_linear"`, `"dare_ties"`, default is `"linear"`.
- `merge_type`: Type of model fusion process, supports `"linear"`, `"ties"`, `"slerp"`, default is `"linear"`.
- `sparsify_type`: Type of sparsification processing, supports `"trim"`, `"magprune"`, `"dare"`, default is `None`.

</div>

<summary>&emsp; Model Parameters</summary><div>

- `model_path_list`: List of paths or names of models to be fused, must include at least two paths. Default is `None`.
- `base_model_path`: Path or name of the base model, default is `None`.
- `output_path`: Directory path to save the fused model, default is `None`.

</div>

<summary>&emsp; Merge Parameters</summary><div>

- `weight_list`: List of relative (or absolute, if normalize=False) weights for each model during fusion, default is `None`. If not set, weights will be automatically evenly distributed.
- `normalize`: Whether to normalize weights, default is `False`.
- `slerp_alpha`: Slerp interpolation parameter for Slerp method, default is `0.5`.
- `slerp_normalize_eps`: Epsilon value for Slerp normalization process, default is `1e-8`.
- `slerp_dot_threshold`: Slerp dot product threshold. If dot product exceeds this value, linear interpolation will be used, default is `0.9995`.
- `ties_elect_type`: Ties mask processing type, supports `"sum"` or `"count"`, default is `"sum"`.

</div>

<summary>&emsp; Sparsify Parameters</summary><div>

- `rescale`: Whether to rescale weights after sparsification, default is `True`.
- `reserve_p`: Random probability to retain during model sparsification, default is `0.7`.
- `epsilon`: Epsilon value used in `"magprune"` sparsification method, default is `0.14`.

</div>
