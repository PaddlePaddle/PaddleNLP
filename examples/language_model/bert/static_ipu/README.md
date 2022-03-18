# Paddle-BERT with Graphcore IPUs

## Overview

This project enabled BERT-Base pre-training and SQuAD fine-tuning task using [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) on Graphcore [IPU-POD16](https://www.graphcore.ai/products/mk2/ipu-pod16).

## File Structure

| File                     | Description                                                        |
| ------------------------ | ------------------------------------------------------------------ |
| `README.md`              | How to run the model.                                              |
| `run_pretrain.py`        | The algorithm script to run pretraining tasks (phase1 and phase2). |
| `run_squad.py`           | The algorithm script to run SQuAD finetune and validation task.    |
| `modeling.py`            | The algorithm script to build the Bert-Base model.                 |
| `dataset_ipu.py`         | The algorithm script to load input data in pretraining.            |
| `run_pretrain.sh`        | Test script to run pretrain phase 1.                               |
| `run_pretrain_phase2.sh` | Test script to run pretrain phase 2.                               |
| `run_squad.sh`           | Test script to run SQuAD finetune.                                 |
| `run_squad_infer.sh`     | Test script to run SQuAD validation.                               |
| `LICENSE`                | The license of Apache.                                             |

## Dataset

1. Pretraining dataset

   Wikipedia dataset is used to do pretraining. Please refer to the Wikipedia dataset generator provided by [Nvidia](https://github.com/NVIDIA/DeepLearningExamples.git) to generate pretraining dataset.

   The sequence length used in pretraining phase1 and phase2 are: 128 and 384. Following steps are provided for dataset generation.

   ```
   # Code base：https://github.com/NVIDIA/DeepLearningExamples/tree/88eb3cff2f03dad85035621d041e23a14345999e/TensorFlow/LanguageModeling/BERT
   git clone https://github.com/NVIDIA/DeepLearningExamples.git
   git checkout 88eb3cff2f03dad85035621d041e23a14345999e

   cd DeepLearningExamples/TensorFlow/LanguageModeling/BERT

   bash scripts/docker/build.sh

   cd data/

   # Modified the parameters `--max_seq_length 512` to `--max_seq_length 384` at line 68, `--max_predictions_per_seq 80` to `--max_predictions_per_seq 56` at line 69.
   vim create_datasets_from_start.sh

   cd ../

   # Use NV's docker to download and generate tfrecord. This may requires GPU available. Removing `--gpus $NV_VISIBLE_DEVICES` in data_download.sh to avoid GPU requirements.
   bash scripts/data_download.sh wiki_only
   ```

2. SQuAD 1.1 dataset

   ```
   curl --create-dirs -L https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -o data/squad/train-v1.1.json

   curl --create-dirs -L https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -o data/squad/dev-v1.1.json
   ```

## Quick Start Guide

### 1）Prepare Project Environment

PaddlePaddle with IPU implementation, which is provided by Graphcore, is required by this application. User can either download the released package or build it from source.

#### Install PaddlePaddle IPU Package

The released PaddlePaddle IPU package can be downloaded from https://github.com/graphcore/Paddle/releases/tag/bert-base-v0.1.

#### Build PaddlePaddle From Source

- Create Docker container

```
git clone -b bert_base_sdk_2.3.0 https://github.com/graphcore/Paddle.git

cd Paddle

# build docker image
docker build -t paddlepaddle/paddle:dev-ipu-2.3.0 -f tools/dockerfile/Dockerfile.ipu .

# create container
# The ipuof.conf is required here.
docker run --ulimit memlock=-1:-1 --net=host --cap-add=IPC_LOCK \
--device=/dev/infiniband/ --ipc=host --name paddle-ipu-dev \
-v ${HOST_IPUOF_PATH}:/ipuof \
-e IPUOF_CONFIG_PATH=/ipuof/ipu.conf \
-it paddlepaddle/paddle:dev-ipu-2.3.0 bash
```

All of later processes are required to be executed in the container.

- Compile and installation

```
git clone -b bert_base_sdk_2.3.0 https://github.com/graphcore/Paddle.git

cd Paddle

cmake -DPYTHON_EXECUTABLE=/usr/bin/python \
-DWITH_PYTHON=ON -DWITH_IPU=ON -DPOPLAR_DIR=/opt/poplar \
-DPOPART_DIR=/opt/popart -G "Unix Makefiles" -H`pwd` -B`pwd`/build

cmake --build `pwd`/build --config Release --target paddle_python -j$(nproc)

pip3.7 install -U build/python/dist/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl
```

### 2) Execution

- Run pretraining phase1 (sequence_length = 128)

```
./run_pretrain.sh
```

- Run pretraining phase2 (sequence_length = 384)

```
./run_pretrain_phase2.sh
```

- Run SQuAD finetune task

```
./run_squad.sh
```

- Run SQuAD validation

```
./run_squad_infer.sh
```

#### Parameters

- `task` The type of the NLP model.
- `input_files` The directory of the input data.
- `output_dir` The directory of the trained models.
- `is_training` Training or inference.
- `seq_len` The sequence length.
- `vocab_size` Size of the vocabulary.
- `max_predictions_per_seq` The max number of the masked token each sentence.
- `max_position_embeddings` The length of the input mask.
- `num_hidden_layers` The number of encoder layers.
- `hidden_size` The size of the hidden state of the transformer layers size.
- `ignore_index` The ignore index for the masked position.
- `hidden_dropout_prob` The dropout probability for fully connected layer in embedding and encoder
- `attention_probs_dropout_prob` The dropout probability for attention layer in encoder.
- `learning_rate` The learning rate for training.
- `weight_decay` The weight decay.
- `beta1` The Adam/Lamb beta1 value
- `beta2` The Adam/Lamb beta2 value
- `adam_epsilon` Epsilon for Adam optimizer.
- `max_steps` The max training steps.
- `warmup_steps` The warmup steps used to update learning rate with lr_schedule.
- `scale_loss` The loss scaling.
- `accl1_type` set accl1 type to FLOAT or FLOAT16
- `accl2_type` set accl2 type to FLOAT or FLOAT16
- `weight_decay_mode` decay or l2 regularization
- `optimizer_state_offchip` The store location of the optimizer tensors
- `logging_steps` The gap steps of logging.
- `save_steps` Save the paddle model every n steps.
- `epochs` the iteration of the whole dataset.
- `batch_size` total batch size (= batches_per_step \* num_replica \* grad_acc_factor \* micro_batch_size).
- `micro_batch_size` The batch size of the IPU graph.
- `batches_per_step` The number of batches per step with pipelining.
- `seed` The random seed.
- `num_ipus` The number of IPUs.
- `ipu_enable_fp16` Enable FP16 or not.
- `num_replica` The number of the graph replication.
- `enable_grad_acc` Enable gradiant accumulation or not.
- `grad_acc_factor` Update the weights every n batches.
- `available_mem_proportion` The available proportion of memory used by conv or matmul.
- `shuffle` Shuffle Dataset.
- `wandb` Enable logging to Weights and Biases.
- `enable_load_params` Load paddle params or not.
- `tf_checkpoint` Path to Tensorflow Checkpoint to initialise the model.

## Result

| Task   | Metric   | Result  |
| ------ | -------- | ------- |
| Phase1 | MLM Loss | 1.6064  |
|        | NSP Loss | 0.0272  |
|        | MLM Acc  | 0.6689  |
|        | NSP Acc  | 0.9897  |
|        | tput     | 11700   |
| Phase2 | MLM Loss | 1.5029  |
|        | NSP Loss | 0.02444 |
|        | MLM Acc  | 0.68555 |
|        | NSP Acc  | 0.99121 |
|        | tput     | 3470    |
| SQuAD  | EM       | 79.9053 |
|        | F1       | 87.6396 |
