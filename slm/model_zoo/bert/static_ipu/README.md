# Paddle-BERT with Graphcore IPUs

## Overview

This project enabled BERT-Base pre-training and SQuAD fine-tuning task using [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) on Graphcore [IPU-POD16](https://www.graphcore.ai/products/mk2/ipu-pod16).

## File Structure

| File              | Description                                                        |
| ----------------- | ------------------------------------------------------------------ |
| `README.md`       | How to run the model.                                              |
| `run_pretrain.py` | The algorithm script to run pretraining tasks (phase1 and phase2). |
| `run_squad.py`    | The algorithm script to run SQuAD finetune and validation task.    |
| `modeling.py`     | The algorithm script to build the Bert-Base model.                 |
| `dataset_ipu.py`  | The algorithm script to load input data in pretraining.            |
| `custom_ops/`     | The folder contains custom ops that will be used.                  |
| `scripts/`        | The folder contains scripts for model running.                     |

## Dataset

- Pretraining dataset

   Wikipedia dataset is used to do pretraining. Please refer to the Wikipedia dataset generator provided by [Nvidia](https://github.com/NVIDIA/DeepLearningExamples.git) to generate pretraining dataset.

   The sequence length used in pretraining phase1 and phase2 are: 128 and 384. Following steps are provided for dataset generation.

   ```bash
   # Here we use a specific commmit, the latest commit should also be fine.
   git clone https://github.com/NVIDIA/DeepLearningExamples.git
   git checkout 88eb3cff2f03dad85035621d041e23a14345999e

   cd DeepLearningExamples/PyTorch/LanguageModeling/BERT

   # Modified the parameters `--max_seq_length 512` to `--max_seq_length 384` at line 50 and
   # `--max_predictions_per_seq 80` to `--max_predictions_per_seq 56` at line 51.
   vim data/create_datasets_from_start.sh

   # Build docker image
   bash scripts/docker/build.sh

   # Use NV's docker to download and generate hdf5 file. This may requires GPU available.
   # You can Remove `--gpus $NV_VISIBLE_DEVICES` to avoid GPU requirements.
   bash scripts/docker/launch.sh

   # generate dataset with wiki_only
   bash data/create_datasets_from_start.sh wiki_only
   ```

- SQuAD v1.1 dataset

   SQuAD v1.1 dataset will be downloaded automatically. You don't have to download manually.


## Quick Start Guide

### Prepare Project Environment

- Create docker image

```bash
# clone paddle repo
git clone https://github.com/paddlepaddle/Paddle.git -b release/2.3
cd Paddle

# build docker image
docker build -t paddlepaddle/paddle:latest-dev-ipu -f tools/dockerfile/Dockerfile.ipu .
```

- Create docker container

```bash
# clone paddlenlp repo
git clone https://github.com/paddlepaddle/paddlenlp.git
cd paddlenlp/examples/language_model/bert/static_ipu

# create docker container
# the ipuof configuration file need to be pre-generated and mounted to docker container
# the environment variable IPUOF_CONFIG_PATH should point to the ipuof configuration file
# more information on ipuof configuration is available at https://docs.graphcore.ai/projects/vipu-admin/en/latest/cli_reference.html?highlight=ipuof#ipuof-configuration-file
docker run --ulimit memlock=-1:-1 --net=host --cap-add=IPC_LOCK \
--device=/dev/infiniband/ --ipc=host \
--name paddle-bert-base \
-v ${IPUOF_CONFIG_PATH}:/ipu.conf \
-e IPUOF_CONFIG_PATH=/ipu.conf \
-v ${PWD}:/workdir \
-w /home -it paddlepaddle/paddle:latest-dev-ipu bash
```

All of later processes are required to be executed in the container.

- Compile and installation

```bash
# clone paddle repo
git clone https://github.com/paddlepaddle/Paddle.git -b release/2.3
cd Paddle

mkdir build && cd build

# run cmake
cmake .. -DWITH_IPU=ON -DWITH_PYTHON=ON -DPY_VERSION=3.7 -DWITH_MKL=ON \
         -DPOPLAR_DIR=/opt/poplar -DPOPART_DIR=/opt/popart -DCMAKE_BUILD_TYPE=Release

# compile
make paddle_python -j$(nproc)

# install paddle package
pip install -U python/dist/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl

# go to workdir
cd /workdir
```

### Execution

- Run pretraining phase1 (sequence_length = 128)

```bash
# pod16
# takes about 11.3 hours
bash scripts/pod16/run_pretrain.sh

# pod4
# takes about 11.3 * 4 hours
bash scripts/pod4/run_pretrain.sh
```

- Run pretraining phase2 (sequence_length = 384)

```bash
# pod16
# takes about 3 hours
bash scripts/pod16/run_pretrain_phase2.sh

# pod4
# takes about 3 * 4 hours
bash scripts/pod4/run_pretrain_phase2.sh
```

- Run SQuAD finetune task

```bash
# pod16
bash scripts/pod16/run_squad.sh

# pod4
bash scripts/pod4/run_squad.sh
```

- Run SQuAD validation

```bash
# pod16
bash scripts/pod16/run_squad_infer.sh

# pod4
bash scripts/pod4/run_squad_infer.sh
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
- `enable_engine_caching` Enable engine caching or not.
- `enable_load_params` Load paddle params or not.
- `tf_checkpoint` Path to Tensorflow Checkpoint to initialise the model.

## Result

For a POD16 platform:

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
