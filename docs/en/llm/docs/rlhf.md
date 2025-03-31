# RLHF

Provides code and complete examples for human preference alignment of LLMs using the Proximal Policy Optimization (PPO) reinforcement learning algorithm. Supports **3D distributed parallel training and generation acceleration via prediction optimization during the rollout phase**. The PPO implementation details reference [PKU-Alignment/safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf) (PKU Beaver), supporting common PPO stabilization strategies like reward normalization and pretraining loss. Examples use partial datasets and models provided by PKU-Alignment/safe-rlhf. We will continuously improve and expand to support better performance, lower costs, higher efficiency, and larger-scale RLHF capabilities.

## Quick Start

The project structure is organized as follows:

```
rlhf/
├── configs/               # Configuration files
├── data/                  # Example data
├── docs/                  # Documentation
├── examples/              # Example scripts
├── models/                # Model implementations
├── scripts/               # Utility scripts
├── tests/                 # Test cases
├── train.py               # Main training entry
└── utils/                 # Utility modules
```
```
./alignment
├── ppo                          # PPO training related directory
│   ├── comm_utils.py            # Communication utility Python file
│   ├── data                     # Dataset related directory
│   │   ├── alpaca.py            # Alpaca (raw) dataset Python file
│   │   ├── base.py              # Dataset base classes and utilities
│   │   ├── __init__.py
│   │   ├── preference.py        # Preference dataset Python file
│   │   ├── prompt_only.py       # Prompt-only dataset Python file
│   │   ├── safe_rlhf.py         # safe_rlhf (raw) dataset Python file
│   │   └── supervised.py        # Supervised dataset Python file
│   ├── infer_utils.py           # Generation acceleration utilities
│   ├── models                   # Model related directory
│   │   ├── infer_model_utils.py # Inference acceleration model patches and utilities
│   │   ├── __init__.py
│   │   ├── model_pp.py          # Pipeline parallel model implementation
│   │   ├── pp_model_utils.py    # Pipeline parallel patches and utilities
│   │   ├── ppo_model.py         # PPO model implementation
│   │   ├── ppo_model_utils.py   # PPO loss and model strategy utilities
│   │   ├── score_model.py       # Score model definition
│   │   └── score_model_utils.py # Score model base classes and utilities
│   ├── run_ppo.py               # RLHF training script
│   ├── ppo_trainer.py           # RLHF training executor
│   ├── tests                    # Test related directory
│   │   ├── run_model.py
│   │   └── test_export.py
│   └── trainer_utils.py         # Trainer patches and utilities
├── README.md
└── rm                         # Reward Model training related directory
    ├── models -> ../ppo/models
    ├── run_reward.py            # Reward model training script
    └── reward_trainer.py        # Reward model training executor
```
### Environment Preparation

- Python >= 3.10
- PaddlePaddle >= 2.6.0
- Latest version of PaddleNLP

To enable generation acceleration, [paddlenlp_ops](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/csrc) needs to be installed. Use `git clone https://github.com/PaddlePaddle/PaddleNLP.git` to clone the PaddleNLP repository and add the paths of PaddleNLP/llm and PaddleNLP/llm/alignment/ppo directories to PYTHONPATH (to be improved later). After installing paddlenlp_ops, generation acceleration will be automatically enabled during training (not supported when pipeline parallelism is enabled), otherwise native dynamic graph will be used for generation.

### Data Preparation

PPO training consists of three stages: Supervised Fine-Tuning, Reward Model Fine-Tuning, and RLHF (see Training section below), which involve multiple datasets. The data preparation methods are described below.

#### Supervised Fine-Tuning Data
Same as [LLM Fine-Tuning](finetune.md), refer to corresponding documentation for data preparation.

#### Reward Model Fine-Tuning Data
The Reward Model Fine-Tuning phase requires human preference data. The example uses [PKU-Alignment/PKU-SafeRLHF-30K](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-30K) provided by PKU-Alignment/safe-rlhf. Below is a sample entry, using the `prompt, response_0, response_1, better_response_id` fields to organize preference data (using the helpful annotation from this dataset which separates helpful and harmless annotations).
```json
{
    "prompt": "Is it possible to purchase a complete chess set at a pawn shop?",
    "response_0": "Yes, it is possible to purchase a complete chess set at a pawn shop.",
    "response_1": "Yes, it is possible to purchase a complete chess set at a pawn shop. Pawn shops may carry chess sets that have been either inherited or previously borrowed. There is also a possibility of finding individual chess pieces such as rooks, knights, bishops, queens, and kings. If you cannot find a complete set, you may be able to purchase the various chess pieces separately.",
    "is_response_0_safe": true,
    "is_response_1_safe": true,
    "better_response_id": 1,
    "safer_response_id": 1
}
```

#### RLHF Data
The RLHF phase uses prompt-only data, with optional additional supervised data available for constructing LM loss constraints during RLHF training. The example uses the [PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) dataset (also a human preference dataset, here we only use its prompt field and deduplicate prompts). Additionally, data from [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) is used to construct additional loss terms.

The example datasets mentioned above will be automatically downloaded and cached during training.

#### Custom Data
The data definition revolves around two predefined classes: `RawSample` and `RawDataset`. `RawSample` provides the data sample-level access protocol specification, while `RawDataset` provides the dataset-level access protocol specification. By following the specifications described below, you can use custom data through the predefined `SupervisedDataset`, `PreferenceDataset`, and `PromptOnlyDataset` interfaces required for RLHF training.

To create a custom dataset you need to:
1. Inherit from `RawDataset`, and define the class attribute `NAME` for dataset registration.
2. Implement the `__init__` method (for data loading), `__getitem__` method (to retrieve samples by index and convert them to `RawSample` objects), and `__len__` method (to return dataset size).

Example implementation:

```python
class MyCustomDataset(RawDataset):
    NAME = 'my_dataset'

    def __init__(self):
        # Load data
        self.data = [...]

    def __getitem__(self, index) -> RawSample:
        # Convert data item to RawSample
        return RawSample(...)

    def __len__(self):
        return len(self.data)
```
```python
from datasets import load_dataset
from data import RawDataset, RawSample

class MyRawDataset(RawDataset):
    NAME = 'my-dataset-name'

    def __init__(self, path=None) -> None:
        # Load a dataset from Hugging Face or any other data source
        # self.data = load_dataset(path or 'my-organization/my-dataset')['train']
        self.data = [{
            'col1': 'question',
            'col2': 'answer1',
            'col3': 'answer2',
            'col4': 1,  # score of answer1
            'col5': 2  # score of answer2
        }] * 10  # dummy data for example

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        # Construct a `RawSample` dictionary from your custom dataset item
        return RawSample(
            input=data['col1'],
            answer=data['col2'],
            other_answer=data['col3'],
            better=float(data['col4']) > float(data['col5']),
        )

    def __len__(self) -> int:
        return len(self.data)  # dataset size
```

Where `RawSample` serves as the superset of several data types used throughout the entire RLHF training process, as shown below, which can bridge the sample types required at different training stages. When customizing data:
- For SFT data, use the `(input, answer)` fields of `RawSample`
- For human preference data, use the `(input, answer, other_answer, better)` fields of `RawSample`
- For prompt-only data, use the `(input)` field of `RawSample`
```python
class RawSample(TypedDict, total=False):
    """Raw sample type.

    For SupervisedDataset, should provide (input, answer) or (dialogue).
    For PreferenceDataset, should provide (input, answer, other_answer, better).
    For SafetyPreferenceDataset, should provide (input, answer, other_answer, safer, is_safe, is_other_safe).
    For PromptOnlyDataset, should provide (input).

    When input is a list, it would be processed as a dialogue.
    """

    # Texts
    input: NotRequired[str | list[str]]  # either `input` or `dialogue` should be provided
    """User input text."""
    answer: NotRequired[str]
    """Assistant answer text."""
    other_answer: NotRequired[str]
    """Other assistant answer text via resampling."""
    dialogue: NotRequired[list[str]]  # either `input` or `dialogue` should be provided
    """Dialogue history."""

    # Flags
    better: NotRequired[bool]
    """Whether ``answer`` is better than ``other_answer``."""
    safer: NotRequired[bool]
    """Whether ``answer`` is safer than ``other_answer``."""
    is_safe: NotRequired[bool]
    """Whether ``answer`` is safe."""
    is_other_safe: NotRequired[bool]
    """Whether ``other_answer`` is safe."""
```

Datasets defined as such can be used via predefined interfaces based on `NAME`. Currently supported built-in datasets include `"PKU-SafeRLHF/train", "PKU-SafeRLHF/test", "PKU-SafeRLHF-30K/train", "PKU-SafeRLHF-30K/test", "PKU-SafeRLHF-10K/train", "alpaca"`. Additionally, support is provided for using multiple datasets with specified sampling ratios, allowing us to prepare multiple datasets as needed for each training phase. Example configuration is as follows:
```python
from paddlenlp.transformers import AutoTokenizer
from data import PreferenceDataset

tokenizer = AutoTokenizer.from_pretrained('facebook/llama-7b')
dataset = PreferenceDataset({
    'alpaca': 0.75,
    'my-dataset-name': 0.5
}, tokenizer)
```

### Training

The complete PPO training process consists of 3 stages as shown below (from [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat)):

<p align="center">
  <img src="https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/assets/image/ppo_trainer.png?raw=true" align="middle" width = "600" />
</p>

1. Supervised Fine-Tuning (SFT)

Same as [LLM Fine-Tuning](finetune.md), you can directly refer to the corresponding content for training and use the resulting model.

2. Reward Model Fine-Tuning

Use the `run_reward.py` script to train the reward model according to `rm_argument.json`

```
cd rm
python -u -m paddle.distributed.launch run_reward.py ../../config/llama/rm_argument.json
```

Most parameter explanations in `rm_argument.json` are the same as [LLM Fine-Tuning](finetune.md) and will not be repeated. Some distinct configurations for reward model training include (using default values from PKU-Alignment/PKU-SafeRLHF):

- `normalize_score_during_training`: Whether to normalize rewards during training, default is `False`.
- `normalizer_type`: Method to compute mean and variance when using normalizer, options: `"RunningMeanStd", "ExponentialMovingAverage"`.
- `normalizer_momentum`: Momentum specified when using `ExponentialMovingAverage` normalizer, default is `0.9`.
- `loss_type`: Use token-level or sequence-level loss for reward model training, options: `"token-wise", "sequence-wise"`, default is `"sequence-wise"`.
- `regularization`: Regularization coefficient for rewards in reward model training objective, default is `0.001`.
3. RLHF:

The RLHF phase requires four models: actor model, reference model, critic model, and reward model. The actor-model/reference-model is initialized/frozen using the SFT model; the critic-model/reward-model is initialized/frozen using the reward model (note that if LoRA is used for SFT, merge LoRA weights first). Here we use the SFT model ([PKU-Alignment/alpaca-7b-reproduced](https://huggingface.co/PKU-Alignment/alpaca-7b-reproduced)) and reward model ([PKU-Alignment/beaver-7b-v1.0-reward](https://huggingface.co/PKU-Alignment/beaver-7b-v1.0-reward), note this model focuses only on helpfulness not safety) provided by PKU-Alignment/PKU-SafeRLHF as examples, and perform RLHF training using the `run_ppo.py` script according to `ppo_argument.json`.

```
# Type promotion warnings are temporarily suppressed via loglevel, to be fixed later
cd ppo
PYTHONPATH=../../ GLOG_minloglevel=2 python -u -m paddle.distributed.launch run_ppo.py ../../config/llama/ppo_argument.json
```

Most parameters in `ppo_argument.json` share the same definitions as [LLM Fine-tuning](finetune.md), which will not be reiterated. Key parameters and their explanations are provided below (using default values from PKU-Alignment/PKU-SafeRLHF):

- `train_datasets`: Specifies training datasets using the `NAME` attribute registered during dataset definition.
- `eval_datasets`: Specifies evaluation datasets using the `NAME` attribute registered during dataset definition.
- `ptx_datasets`: Specifies datasets for ptx-loss using the `NAME` attribute registered during dataset definition. ptx-loss will not be used if not provided.
- `actor_model_name_or_path`: Model name or path used to initialize/freeze the actor-model/reference-model.
- `reward_model_name_or_path`: Model name or path for the reward-model.
- `reward_critic_model_name_or_path`: Model name or path for the critic-model. If not provided, `reward_model_name_or_path` will be used to initialize the critic-model.
- `per_device_prompt_batch_size`: Batch size per device for prompt-only dataset loading during rollout generation in training.
- `per_device_train_batch_size`: Batch size per device for generation and training based on prompts.
- `num_return_sequences`: Number of responses generated per prompt during generation.
`GenerationConfig.num_return_sequences`, all responses will be used for training.
- `temperature`: The `temperature` used in generation sampling, i.e., `GenerationConfig.temperature`.
- `top_p`: The top-p-filtering threshold used in generation sampling, i.e., `GenerationConfig.top_p`.
- `repetition_penalty`: The length penalty coefficient used in generation sampling, i.e., `GenerationConfig.repetition_penalty`.
- `update_iters`: The number of times generated data is used.
- `kl_coeff`: The coefficient for KL-Penalty on the reward.
- `clip_range_score`: The threshold for clipping the reward.
- `clip_range_value`: When the difference between the new value from the critic model (value function) and the old value in the Experience Buffer exceeds this range, clipping will be applied.
- `clip_range_ratio`: Clips the ratio of new probabilities to old probabilities from the Experience Buffer to the range `(1-clip_range_ratio, 1+clip_range_ratio)` (PPO-Clip).
- `ptx_coeff`: The coefficient for the pretraining loss term ptx-loss.

Additionally, all [supported parameters from `TrainingArguments`](https://paddlenlp.readthedocs.io/zh/latest/trainer.html#trainingarguments) will be reused for training both the actor-model and critic-model (e.g., `sharding_stage`), except that separate parameters `critic_learning_rate/critic_weight_decay/critic_lr_scheduler_type/critic_warmup_ratio/critic_recompute` are provided to specify individual configurations for critic-model training. The checkpoints for actor-model and critic-model will be saved separately in the policy and value folders under the directory specified by `output_dir`.

Furthermore, to support higher performance and larger-scale RLHF training, the following special parameter configurations are provided and can be used as needed:
- `use_fusemt`: After installing paddlenlp_ops, generation acceleration will be enabled during rollout generation (not supported when pipeline parallelism is enabled). This setting can disable generation acceleration.
- `eval_mode`: Supports being empty or set to "single"/"tensor_parallel". Typically, can be set to "tensor_parallel" when using pipeline parallelism training, enabling the use of non-pipeline-parallel models and generation acceleration during the rollout generation phase.
- `offload_level`
Support setting to "freeze_model", "optimizer", "train_model" or combinations (space-separated), which respectively indicate offloading/reloading of reward+reference frozen models, actor+critic training models' optimizer states and model parameters. This is used to immediately offload model/optimizer after their usage in different stages and reload corresponding parameter weights next time to save GPU memory.

Note: When using pipeline parallelism (pipeline_parallel_degree >1), it's recommended to set `dataloader_drop_last` to true to avoid issues caused by varying batch sizes.

### Inference

After training completes, you can directly use the checkpoints under the policy folder in the `output_dir` specified directory for inference according to the instructions in the [LLM Inference](predict/inference.md) section. Please refer to that section.

## Acknowledge

We acknowledge the excellent design and implementation of [PKU-Alignment/safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf) (PKU Beaver), and extend our gratitude to its authors.

## References
- Zheng R, Dou S, Gao S, et al. Secrets of rlhf in large language models part i: Ppo[J]. arXiv preprint arXiv:2307.04964, 2023.
- Dai J, Pan X, Sun R, et al. Safe rlhf: Safe reinforcement learning from human feedback[J]. arXiv preprint arXiv:2310.12773, 2023.
