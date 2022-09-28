# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import pickle
import random
import time

import numpy as np
import paddle
import paddle.optimizer
import paddle.static
from paddlenlp.transformers import LinearDecayWithWarmup
from scipy.stats import truncnorm

from dataset_ipu import PretrainingHDF5DataLoader
from modeling import (BertModel, DeviceScope, IpuBertConfig,
                      IpuBertPretrainingMLMAccAndLoss,
                      IpuBertPretrainingMLMHeads,
                      IpuBertPretrainingNSPAccAndLoss,
                      IpuBertPretrainingNSPHeads)
from utils import load_custom_ops, parse_args, ProgressBar, ProgressFunc


def set_seed(seed):
    """
    Use the same data seed(for data shuffle) for all procs to guarantee data
    consistency after sharding.
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def create_data_holder(args):
    bs = args.micro_batch_size
    indices = paddle.static.data(name="indices",
                                 shape=[bs * args.seq_len],
                                 dtype="int32")
    segments = paddle.static.data(name="segments",
                                  shape=[bs * args.seq_len],
                                  dtype="int32")
    positions = paddle.static.data(name="positions",
                                   shape=[bs * args.seq_len],
                                   dtype="int32")
    mask_tokens_mask_idx = paddle.static.data(name="mask_tokens_mask_idx",
                                              shape=[bs, 1],
                                              dtype="int32")
    sequence_mask_idx = paddle.static.data(name="sequence_mask_idx",
                                           shape=[bs, 1],
                                           dtype="int32")
    masked_lm_ids = paddle.static.data(name="masked_lm_ids",
                                       shape=[bs, args.max_predictions_per_seq],
                                       dtype="int32")
    next_sentence_labels = paddle.static.data(name="next_sentence_labels",
                                              shape=[bs],
                                              dtype="int32")
    return [
        indices, segments, positions, mask_tokens_mask_idx, sequence_mask_idx,
        masked_lm_ids, next_sentence_labels
    ]


def reset_program_state_dict(state_dict, mean=0, scale=0.02):
    """
    Initialize the parameter from the bert config, and set the parameter by
    reseting the state dict."
    """
    new_state_dict = dict()
    for n, p in state_dict.items():
        if  n.endswith('_moment1_0') or n.endswith('_moment2_0') \
            or n.endswith('_beta2_pow_acc_0') or n.endswith('_beta1_pow_acc_0'):
            continue
        if 'learning_rate' in n:
            continue

        dtype_str = "float32"
        if p._dtype == paddle.float64:
            dtype_str = "float64"

        if "layer_norm" in n and n.endswith('.w_0'):
            new_state_dict[n] = np.ones(p.shape()).astype(dtype_str)
            continue

        if n.endswith('.b_0'):
            new_state_dict[n] = np.zeros(p.shape()).astype(dtype_str)
        else:
            new_state_dict[n] = truncnorm.rvs(-2,
                                              2,
                                              loc=mean,
                                              scale=scale,
                                              size=p.shape()).astype(dtype_str)
    return new_state_dict


def create_ipu_strategy(args):
    ipu_strategy = paddle.static.IpuStrategy()
    options = {
        'is_training': args.is_training,
        'enable_manual_shard': True,
        'enable_pipelining': True,
        'batches_per_step': args.batches_per_step,
        'micro_batch_size': args.micro_batch_size,
        'loss_scaling': args.scale_loss,
        'enable_replicated_graphs': True,
        'replicated_graph_count': args.num_replica,
        'num_ipus': args.num_ipus * args.num_replica,
        'enable_gradient_accumulation': args.enable_grad_acc,
        'accumulation_factor': args.grad_acc_factor,
        'auto_recomputation': 3,
        'enable_half_partial': True,
        'available_memory_proportion': args.available_mem_proportion,
        'enable_stochastic_rounding': True,
        'max_weight_norm': 65504.0,
        'default_prefetch_buffering_depth': 3,
        'rearrange_anchors_on_host': False,
        'enable_fp16': args.ipu_enable_fp16,
        'random_seed': args.seed,
        'use_no_bias_optimizer': True,
        'enable_prefetch_datastreams': True,
        'enable_outlining': True,
        'subgraph_copying_strategy': 1,  # JustInTime
        'outline_threshold': 10.0,
        'disable_grad_accumulation_tensor_streams': True,
        'schedule_non_weight_update_gradient_consumers_early': True,
        'cache_path': 'paddle_cache',
        'enable_floating_point_checks': False,
        'accl1_type': args.accl1_type,
        'accl2_type': args.accl2_type,
        'weight_decay_mode': args.weight_decay_mode,
    }

    if not args.optimizer_state_offchip:
        options['location_optimizer'] = {
            'on_chip': 1,  # popart::TensorStorage::OnChip
            'use_replicated_tensor_sharding':
            1,  # popart::ReplicatedTensorSharding::On
        }

    # use popart::AccumulateOuterFragmentSchedule::OverlapMemoryOptimized
    # excludedVirtualGraphs = [0]
    options['accumulate_outer_fragment'] = {3: [0]}

    options['convolution_options'] = {"partialsType": "half"}
    options['engine_options'] = {
        "opt.useAutoloader": "true",
        "target.syncReplicasIndependently": "true",
        "exchange.streamBufferOverlap": "hostRearrangeOnly",
    }

    options['enable_engine_caching'] = args.enable_engine_caching

    options['compilation_progress_logger'] = ProgressFunc

    ipu_strategy.set_options(options)

    # enable custom patterns
    ipu_strategy.enable_pattern('DisableAttnDropoutBwdPattern')

    return ipu_strategy


def main(args):
    paddle.enable_static()
    place = paddle.set_device('ipu')
    set_seed(args.seed)
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    # The sharding of encoder layers
    if args.num_hidden_layers == 12:
        attn_index = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        ff_index = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    else:
        raise Exception("Only support num_hidden_layers = 12")

    bert_config = {
        k: getattr(args, k)
        for k in IpuBertConfig._fields if hasattr(args, k)
    }
    bert_config['embeddings_scope'] = DeviceScope(0, 0, "Embedding")
    bert_config['attn_scopes'] = [
        DeviceScope(attn_index[i], attn_index[i])
        for i in range(args.num_hidden_layers)
    ]
    bert_config['ff_scopes'] = [
        DeviceScope(ff_index[i], ff_index[i])
        for i in range(args.num_hidden_layers)
    ]
    bert_config['mlm_scope'] = DeviceScope(0, args.num_ipus, "MLM")
    bert_config['nsp_scope'] = DeviceScope(0, args.num_ipus, "NSP")
    bert_config['layers_per_ipu'] = [4, 4, 4]

    config = IpuBertConfig(**bert_config)

    # custom_ops
    custom_ops = load_custom_ops()

    # Load the training dataset
    logging.info("Loading dataset")
    input_files = [
        os.path.join(args.input_files, f) for f in os.listdir(args.input_files)
        if os.path.isfile(os.path.join(args.input_files, f)) and "training" in f
    ]
    input_files.sort()

    dataset = PretrainingHDF5DataLoader(
        input_files=input_files,
        max_seq_length=args.seq_len,
        max_mask_tokens=args.max_predictions_per_seq,
        batch_size=args.batch_size,
        shuffle=args.shuffle)
    logging.info(f"dataset length: {len(dataset)}")
    total_samples = dataset.total_samples
    logging.info("total samples: %d, total batch_size: %d, max steps: %d" %
                 (total_samples, args.batch_size, args.max_steps))

    logging.info("Building Model")

    [
        indices, segments, positions, mask_tokens_mask_idx, sequence_mask_idx,
        masked_lm_ids, next_sentence_labels
    ] = create_data_holder(args)

    # Encoder Layers
    bert_model = BertModel(config, custom_ops)
    encoders, word_embedding = bert_model(
        indices, segments, positions, [mask_tokens_mask_idx, sequence_mask_idx])

    # PretrainingHeads
    mlm_heads = IpuBertPretrainingMLMHeads(args.hidden_size, args.vocab_size,
                                           args.max_position_embeddings,
                                           args.max_predictions_per_seq,
                                           args.seq_len)
    nsp_heads = IpuBertPretrainingNSPHeads(args.hidden_size,
                                           args.max_predictions_per_seq,
                                           args.seq_len)

    # AccAndLoss
    nsp_criterion = IpuBertPretrainingNSPAccAndLoss(args.micro_batch_size,
                                                    args.ignore_index,
                                                    custom_ops)
    mlm_criterion = IpuBertPretrainingMLMAccAndLoss(args.micro_batch_size,
                                                    args.ignore_index,
                                                    custom_ops)

    with config.nsp_scope:
        nsp_out = nsp_heads(encoders)
        nsp_acc, nsp_loss = nsp_criterion(nsp_out, next_sentence_labels)

    with config.mlm_scope:
        mlm_out = mlm_heads(encoders, word_embedding)
        mlm_acc, mlm_loss, = mlm_criterion(mlm_out, masked_lm_ids)
        total_loss = mlm_loss + nsp_loss

    # lr_scheduler
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, args.max_steps,
                                         args.warmup_steps)
    # optimizer
    optimizer = paddle.optimizer.Lamb(learning_rate=lr_scheduler,
                                      lamb_weight_decay=args.weight_decay,
                                      beta1=args.beta1,
                                      beta2=args.beta2,
                                      epsilon=args.adam_epsilon)
    optimizer.minimize(total_loss)

    # Static executor
    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    # Set initial weights
    state_dict = main_program.state_dict()
    reset_state_dict = reset_program_state_dict(state_dict)
    paddle.static.set_program_state(main_program, reset_state_dict)

    if args.enable_load_params:
        logging.info(f'loading weights from: {args.load_params_path}')
        if not args.load_params_path.endswith('pdparams'):
            raise Exception('need pdparams file')
        with open(args.load_params_path, 'rb') as file:
            params = pickle.load(file)
        paddle.static.set_program_state(main_program, params)

    # Create ipu_strategy
    ipu_strategy = create_ipu_strategy(args)

    feed_list = [
        "indices",
        "segments",
        "positions",
        "mask_tokens_mask_idx",
        "sequence_mask_idx",
        "masked_lm_ids",
        "next_sentence_labels",
    ]
    fetch_list = [mlm_acc.name, mlm_loss.name, nsp_acc.name, nsp_loss.name]

    # Compile program for IPU
    ipu_compiler = paddle.static.IpuCompiledProgram(main_program,
                                                    ipu_strategy=ipu_strategy)
    logging.info(f'start compiling, please wait some minutes')
    cur_time = time.time()
    main_program = ipu_compiler.compile(feed_list, fetch_list)
    time_cost = time.time() - cur_time
    logging.info(f'finish compiling! time cost: {time_cost}')

    batch_start = time.time()
    global_step = 0
    for batch in dataset:
        global_step += 1
        epoch = global_step * args.batch_size // total_samples
        read_cost = time.time() - batch_start

        feed = {
            "indices": batch[0],
            "segments": batch[1],
            "positions": batch[2],
            "mask_tokens_mask_idx": batch[3],
            "sequence_mask_idx": batch[4],
            "masked_lm_ids": batch[5],
            "next_sentence_labels": batch[6],
        }
        lr_scheduler.step()

        train_start = time.time()
        loss_return = exe.run(main_program,
                              feed=feed,
                              fetch_list=fetch_list,
                              use_program_cache=True)
        train_cost = time.time() - train_start
        total_cost = time.time() - batch_start
        tput = args.batch_size / total_cost

        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "global_step": global_step,
                "loss/MLM": np.mean(loss_return[1]),
                "loss/NSP": np.mean(loss_return[3]),
                "accuracy/MLM": np.mean(loss_return[0]),
                "accuracy/NSP": np.mean(loss_return[2]),
                "latency/read": read_cost,
                "latency/train": train_cost,
                "latency/e2e": total_cost,
                "throughput": tput,
                "learning_rate": lr_scheduler(),
            })

        if global_step % args.logging_steps == 0:
            logging.info({
                "epoch": epoch,
                "global_step": global_step,
                "loss/MLM": np.mean(loss_return[1]),
                "loss/NSP": np.mean(loss_return[3]),
                "accuracy/MLM": np.mean(loss_return[0]),
                "accuracy/NSP": np.mean(loss_return[2]),
                "latency/read": read_cost,
                "latency/train": train_cost,
                "latency/e2e": total_cost,
                "throughput": tput,
                "learning_rate": lr_scheduler(),
            })

        if global_step % args.save_steps == 0:
            ipu_compiler._backend.weights_to_host()
            paddle.static.save(
                main_program.org_program,
                os.path.join(args.output_dir, 'step_{}'.format(global_step)))

        if global_step >= args.max_steps:
            ipu_compiler._backend.weights_to_host()
            paddle.static.save(
                main_program.org_program,
                os.path.join(args.output_dir,
                             'final_step_{}'.format(global_step)))
            dataset.release()
            del dataset
            return

        batch_start = time.time()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt='%Y-%m-%d %H:%M:%S %a')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        import wandb
        wandb.init(project="paddle-base-bert",
                   settings=wandb.Settings(console='off'),
                   name='paddle-base-bert')
        wandb_config = vars(args)
        wandb_config["global_batch_size"] = args.batch_size
        wandb.config.update(args)

    logging.info(args)
    main(args)
    logging.info("program finished")
