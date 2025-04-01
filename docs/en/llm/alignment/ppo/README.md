# PPO && GRPO

PPO (Proximal Policy Optimization) is a reinforcement learning algorithm designed to maximize cumulative rewards through policy optimization. The PPO algorithm combines the advantages of Policy Gradient and TRPO (Trust Region Policy Optimization) by optimizing a "surrogate" objective function using stochastic gradient ascent, enabling minibatch updates rather than performing gradient updates per data sample.

GRPO (Group Relative Policy Optimization) is a variant of PPO. Unlike PPO, GRPO omits the value function estimator. In GRPO, for each state \(s\), the algorithm samples multiple actions \(a_{1}, \dots, a_{G}\) from the current policy \(\pi_{\theta_{t}}\). GRPO then calculates the "group-relative advantage" of these actions relative to other actions within the group, which serves as the basis for policy optimization.

Below is the detailed documentation and examples:

## Environment Dependencies

* Training Environment:
1. Refer to the official Paddle website to install PaddlePaddle-GPU
2. Clone and install PaddleNLP
```shell
git clone https://github.com/PaddlePaddle/PaddleNLP.git
```
3. Install paddlenlp_ops. Refer to PaddleNLP/csrc for installation (required)
```shell
cd your_PaddleNLP_path/csrc
python setup_cuda.py install
```
## Supported Models

|   Model Series   | Model Name                                                                                                                                                                                                                                                                      |
|:----------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   Llama3.1    | meta-llama/Meta-Llama-3.1-8B, meta-llama/Meta-Llama-3.1-8B-Instruct, meta-llama/Meta-Llama-3.1-70B, meta-llama/Meta-Llama-3.1-70B-Instruct, meta-llama/Meta-Llama-3.1-405B, meta-llama/Meta-Llama-3.1-405B-Instruct, meta-llama/Llama-Guard-3-8B                              |
|   Llama3.2    | meta-llama/Llama-3.2-1B, meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-3B, meta-llama/Llama-3.2-3B-Instruct                                                                                                                                                          |
|    Qwen1.5    | Qwen/Qwen1.5-0.5B, Qwen/Qwen1.5-0.5B-Chat, Qwen/Qwen1.5-1.8B, Qwen/Qwen1.5-1.8B-Chat, Qwen/Qwen1.5-4B, Qwen/Qwen1.5-4B-Chat, Qwen/Qwen1.5-7B, Qwen/Qwen1.5-7B-Chat, Qwen/Qwen1.5-14B, Qwen/Qwen1.5-14B-Chat, Qwen/Qwen1.5-32B, Qwen/Qwen1.5-32B-Chat                          |
|     Qwen2     | Qwen/Qwen2-0.5B, Qwen/Qwen2-0.5B-Instruct, Qwen/Qwen2-1.5B, Qwen/Qwen2-1.5B-Instruct, Qwen/Qwen2-7B, Qwen/Qwen2-7B-Instruct, Qwen/Qwen2-72B, Qwen/Qwen2-72B-Instruct, Qwen/Qwen2-57B-A14B, Qwen/Qwen2-57B-A14B-Instruct                                                       |
|  Qwen2-Math   | Qwen/Qwen2-Math-1.5B, Qwen/Qwen2-Math-1.5B-Instruct, Qwen/Qwen2-Math-7B, Qwen/Qwen2-Math-7B-Instruct                                                                                                                                                                          |
|    Qwen2.5    | Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-3B, Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B, Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-14B, Qwen/Qwen2.5-14B-Instruct, Qwen/Qwen2.5-32B, Qwen/Qwen2.5-32B-Instruct, |
| Qwen2.5-Math  | Qwen/Qwen2.5-Math-1.5B, Qwen/Qwen2.5-Math-1.5B-Instruct, Qwen/Qwen2.5-Math-7B, Qwen/Qwen2.5-Math-7B-Instruct                                                                                                                                                                  |
| Qwen2.5-Coder | Qwen/Qwen2.5-Coder-1.5B, Qwen/Qwen2.5-Coder-1.5B-Instruct, Qwen/Qwen2.5-Coder-7B, Qwen/Qwen2.5-Coder-7B-Instruct                                                                                                                                                              |

## Data Protocol

### Field Descriptions

- src (list(str)): User dialogue content, may contain markup elements like [<search-res>];
- tgt (list(str)): System's multi-turn responses except the last round, ordered by dialogue turns, may contain markup elements like [<search>]; Note: len(tgt) == len(src) - 1

### Data Example

```python
{
    "src": [
        "Hi",
        "[<search-res>] The average temperature in Paris during summer is 25°C"
    ],
    "tgt": [
        "Hello! How can I assist you today?",
        "[<search>] According to recent data, the average summer temperature in Paris is around 25°C. This information is based on meteorological records from the past decade."
    ]
}
```
```json
{
    "src": [
        "I need you to help me come up with some creative slogans to capture the market.",
        "The target audience is young people who pursue fashion, individuality and self-expression."
    ],
    "tgt": [
        "Absolutely! I'd be delighted to assist you in creating some creative slogans to promote your new shampoo. Please share some key features of your product, the target audience, and the core message you want to convey. I'll provide several creative slogans based on this information."
    ]
}
```

### PPO & GRPO Data Preparation

```
wget https://paddlenlp.bj.bcebos.com/datasets/examples/ppo-kk.tgz && tar zxf ppo-kk.tgz
```

## Training

### Training Configuration

The configuration files we use are located in `llm/config/llama/ppo_argument.json` and `llm/config/llama/grpo_argument.json`, with detailed parameter explanations as follows:

- `train_task_config`: Training data config, refer to `config/task_ppo.json` as example
- `eval_task_config`: Evaluation data config, refer to `config/task_ppo.json` as example
- `ptx_task_config`: SFT auxiliary data, refer to `config/task_sft.json` as example, default is ""
- `actor_model_name_or_path`: Local model path for actor-model and reference-model in PPO
- `reward_model_name_or_path`: Local model path for reward-model and critic-model in PPO
- `use_fusemt`: Whether to accelerate generation via FustMT, default is True
- `use_flash_attention`: Whether to enable FlashAttention-2, default is False
- `output_dir`: Model parameter save directory
- `max_seq_len`: Maximum length of input data, default is 4096
- `max_dec_len`: Maximum generation length
- `min_dec_len`: Minimum generation length
- `top_p`: Generation decoding hyperparameter
- `temperature`: Generation decoding hyperparameter
- `repetition_penalty`: Generation decoding hyperparameter
- `num_return_sequences`: Generation decoding hyperparameter
- `min_learning_rate`: Minimum learning rate for Actor model
- `critic_learning_rate`: Minimum learning rate for Critic model
- `recompute`: Whether to use recomputation strategy for Actor model, enables training memory saving
- `critic_recompute`: Whether to use recomputation strategy for Critic model, enables training memory saving
- `recompute_granularity`: Granularity of recomputation for Actor model, options: `core_attn` and `full`. `core_attn` is faster but uses more memory, `full` is slower but uses less memory
- `critic_recompute_granularity`: Granularity of recomputation for Critic model, options: `core_attn` and `full`. `core_attn` is faster but uses more memory, `full`...
- `warmup_ratio`: The proportion of total training steps used for linear warmup from 0 to `learning_rate` for the Actor model
- `critic_warmup_ratio`: The proportion of total training steps used for linear warmup from 0 to `critic_learning_rate` for the Critic model
- `lr_scheduler_type`: The learning rate scheduler type to use for the Actor model. (`str`, optional, defaults to `"linear"`)
- `critic_lr_scheduler_type`: The learning rate scheduler type to use for the Critic model. (`str`, optional, defaults to `"linear"`)
- `weight_decay`: The weight decay to apply (excluding all bias and LayerNorm weights) to the Actor model. (`float`, optional, defaults to 0.0)
- `critic_weight_decay`: The weight decay to apply (excluding all bias and LayerNorm weights) to the Critic model. (`float`, optional, defaults to 0.0)
- `max_prompt_len`: Maximum generation length when generating samples. Increasing max_length will prolong generation time and increase memory usage. Note: max_dec_len + max_prompt_len should be less than max_seq_len.
- `per_device_prompt_batch_size`: Batch size for PPO sample generation, equivalent to micro batch size, i.e., global_batch_size = dp (data parallel) * sharding * micro batch size. Increasing batch_size will prolong generation time and increase memory usage.
- `per_device_train_batch_size`: Training batch size, currently set to 1 for optimization purposes. Avoid changing this.
- `per_device_eval_batch_size`: Evaluation batch size.
- `max_steps`: Total number of training steps.
- `eval_steps`: Interval steps for model evaluation.
- `max_evaluate_steps`: Maximum steps for single evaluation run.
- `logging_steps`: Interval steps for training log printing.
- `save_steps`: Interval steps for model parameter saving.
- `weight_decay`: Weight decay value.
- `do_train`: Whether to perform training task.
- `do_eval`: Whether to perform evaluation task.
- `fp16`: Use float16 precision for model training and inference.
- `bf16`: Use bfloat16 precision for model training and inference.
- `fp16_opt_level`: Float16 precision training mode, `O2` indicates pure float16 training.

<!-- ### PPO Training Command

```shell
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_ppo.py llm/config/llama/ppo_argument.json
``` -->

### GRPO Training Command
```shell
cd your_PaddleNLP_path/llm/alignment/ppo
```
```shell
# Start reward server
python reward_server.py
```

```shell
export PYTHONPATH=your_PaddleNLP_path/:$PYTHONPATH
export PYTHONPATH=your_PaddleNLP_path/llm:$PYTHONPATH
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_ppo.py ../../config/qwen/grpo_argument.json
# python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_ppo.py ../../config/llama/grpo_argument.json
```

### Online Monitoring
The output directory specified in `grpo_argument.json` is `"logging_dir": "vdl_log"`. Monitor training progress with:
```shell
visualdl --logdir vdl_log --host 0.0.0.0
```
