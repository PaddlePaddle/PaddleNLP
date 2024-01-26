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
import sys
import pickle
import random
import numpy as np
import paddle
import paddle.optimizer
import paddle.static
from pathlib import Path
from args import load_custom_ops, parse_args
from paddlenlp.transformers import LinearDecayWithWarmup
from scipy.stats import truncnorm
from dataset_ipu import PretrainingHDF5DataLoader

root_folder = str(Path(__file__).parent.parent.parent.absolute())
root_folder = "../../../paddlenlp"
sys.path.insert(0, root_folder)
from trainer import IPUTrainer
from IPU.modeling import (
    IpuBertPretrainingMLMAccAndLoss,
    IpuBertPretrainingMLMHeads, IpuBertPretrainingNSPAccAndLoss,
    IpuBertPretrainingNSPHeads)
from IPU.utils import (
    AutoModel,
    AutoConfig,
    AutoPipeline
)

class PretrainIPUTrainer(IPUTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        input_files = [
            os.path.join(self.args.input_files, f) for f in os.listdir(self.args.input_files)
            if os.path.isfile(os.path.join(self.args.input_files, f)) and "training" in f
        ]
        input_files.sort()

        return PretrainingHDF5DataLoader(
            input_files=input_files,
            max_seq_length=self.args.seq_len,
            max_mask_tokens=self.args.max_predictions_per_seq,
            batch_size=self.args.batch_size,
            shuffle=self.args.shuffle)

    def log(self):
        logging.info({
            "global_step": self.global_step,
            "loss/MLM": np.mean(self.loss[1]),
            "loss/NSP": np.mean(self.loss[3]),
            "accuracy/MLM": np.mean(self.loss[0]),
            "accuracy/NSP": np.mean(self.loss[2]),
            "latency/read": self.read_cost,
            "latency/train": self.train_cost,
            "latency/e2e": self.total_cost,
            "throughput": self.tput,
            "learning_rate": self.lr_scheduler(),
        })


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
    indices = paddle.static.data(
        name="indices", shape=[bs * args.seq_len], dtype="int32")
    segments = paddle.static.data(
        name="segments", shape=[bs * args.seq_len], dtype="int32")
    positions = paddle.static.data(
        name="positions", shape=[bs * args.seq_len], dtype="int32")
    mask_tokens_mask_idx = paddle.static.data(
        name="mask_tokens_mask_idx", shape=[bs, 1], dtype="int32")
    sequence_mask_idx = paddle.static.data(
        name="sequence_mask_idx", shape=[bs, 1], dtype="int32")
    masked_lm_ids = paddle.static.data(
        name="masked_lm_ids",
        shape=[bs, args.max_predictions_per_seq],
        dtype="int32")
    next_sentence_labels = paddle.static.data(
        name="next_sentence_labels", shape=[bs], dtype="int32")
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


def main(args):
    paddle.enable_static()
    place = paddle.set_device('ipu')
    set_seed(args.seed)
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()
     
    # custom_ops
    custom_ops = load_custom_ops()

    logging.info("Building Model")

    [
        indices, segments, positions, mask_tokens_mask_idx, sequence_mask_idx,
        masked_lm_ids, next_sentence_labels
    ] = create_data_holder(args)

    # Encoder Layers
    config = AutoConfig(args)
    bert_model = AutoModel(args, config, custom_ops)
    AutoPipeline(args, config, bert_model)
    encoders, word_embedding = bert_model(
        indices, segments, positions,
        [mask_tokens_mask_idx, sequence_mask_idx])
    
    # PretrainingHeads
    mlm_heads = IpuBertPretrainingMLMHeads(
        args.hidden_size, args.vocab_size, args.max_position_embeddings,
        args.max_predictions_per_seq, args.seq_len)
    nsp_heads = IpuBertPretrainingNSPHeads(
        args.hidden_size, args.max_predictions_per_seq, args.seq_len)

    # AccAndLoss
    nsp_criterion = IpuBertPretrainingNSPAccAndLoss(
        args.micro_batch_size, args.ignore_index, custom_ops)
    mlm_criterion = IpuBertPretrainingMLMAccAndLoss(
        args.micro_batch_size, args.ignore_index, custom_ops)

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
    optimizer = paddle.optimizer.Lamb(
        learning_rate=lr_scheduler,
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

    amp_list = paddle.static.amp.CustomOpLists()
    amp_list.unsupported_list = {}
    to_fp16_var_names = paddle.static.amp.cast_model_to_fp16(
        main_program, amp_list, use_fp16_guard=False)
    paddle.static.amp.cast_parameters_to_fp16(
        paddle.CPUPlace(),
        main_program,
        to_fp16_var_names=to_fp16_var_names)

    if args.enable_load_params:
        logging.info(f'loading weights from: {args.load_params_path}')
        if not args.load_params_path.endswith('pdparams'):
            raise Exception('need pdparams file')
        with open(args.load_params_path, 'rb') as file:
            params = pickle.load(file)
        paddle.static.set_program_state(main_program, params)

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

    # Initialize Trainer
    trainer = PretrainIPUTrainer(
        args = args,
        exe = exe,
        tensor_list = [feed_list, fetch_list],
        program = [main_program, startup_program],
        optimizers = [optimizer, lr_scheduler]
    )

    trainer.train()
    trainer.save_model()

    trainer.train_dataloader.release()
    del trainer.train_dataloader


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S %a')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        import wandb
        wandb.init(
            project="paddle-base-bert",
            settings=wandb.Settings(console='off'),
            name='paddle-base-bert')
        wandb_config = vars(args)
        wandb_config["global_batch_size"] = args.batch_size
        wandb.config.update(args)

    logging.info(args)
    main(args)
    logging.info("program finished")
